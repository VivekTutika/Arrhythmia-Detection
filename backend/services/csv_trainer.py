"""
CSV-Based Training & Analysis Pipeline
Handles:
  - Record-based CSV dataset splitting
  - Improved MLP model training on tabular ECG features
  - Batch CSV inference with majority voting
  - Output format compatible with existing EDF analysis results

Accuracy improvements (v2):
  - RobustScaler   — outlier-robust feature normalization
  - Class-weighted CrossEntropyLoss — handles severe MIT-BIH imbalance (N >> S,V,F,Q)
  - Deeper residual MLP (512→256→128→64) with skip connections
  - AdamW + CosineAnnealingLR — better generalization, smooth LR decay
  - Best-model checkpointing by val_acc
  - Gradient clipping to stabilize training
"""
import os
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Records designated for test split (per project spec)
# -------------------------------------------------------------------
TEST_RECORDS = [101, 200, 207, 209, 213, 222, 228]

# -------------------------------------------------------------------
# Class label mapping  →  human-readable diagnosis names
# -------------------------------------------------------------------
LABEL_TO_DIAGNOSIS = {
    'N': 'Normal Sinus Rhythm',
    'S': 'Supraventricular Arrhythmia',
    'V': 'Ventricular Arrhythmia',
    'F': 'Fusion Beat',
    'Q': 'Unknown / Unclassifiable',
}


def get_diagnosis_name(label):
    """Map a raw class label to a human-readable diagnosis string."""
    if isinstance(label, str):
        return LABEL_TO_DIAGNOSIS.get(label.strip().upper(), str(label))
    return str(label)


# -------------------------------------------------------------------
# Improved MLP Model — Residual Blocks for better gradient flow
# -------------------------------------------------------------------
class ResidualBlock(nn.Module):
    """BN → Linear → BN → ReLU → Dropout → Linear + skip projection."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.4):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.SiLU(),                          # Swish — smoother than ReLU
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )
        # Project input dimension if it differs
        self.skip = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else nn.Identity()
        self.out_bn = nn.BatchNorm1d(out_dim)

    def forward(self, x):
        return F.silu(self.out_bn(self.block(x) + self.skip(x)))


class ECGMorph_MLP(nn.Module):
    """
    Ultra-Shallow Residual MLP for tabular ECG morphology features.
    Architecture: Input + Gaussian Noise -> 64 -> 32 -> 16 -> num_classes.
    """

    def __init__(self, input_dim: int, num_classes: int, noise_std: float = 0.2):
        super().__init__()
        self.noise_std = noise_std
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(0.5), # Extreme dropout to crush memorization space
        )
        self.res1 = ResidualBlock(64, 32, dropout=0.5)
        self.res2 = ResidualBlock(32, 16, dropout=0.5)
        self.head = nn.Sequential(
            nn.BatchNorm1d(16),
            nn.Dropout(0.5),
            nn.Linear(16, num_classes),
        )

    def forward(self, x):
        # Apply tabular data augmentation via Gaussian Noise (training only)
        if self.training and self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
            
        x = self.input_proj(x)
        x = self.res1(x)
        x = self.res2(x)
        return self.head(x)


# -------------------------------------------------------------------
# 1. CSV Dataset Splitting
# -------------------------------------------------------------------
def split_csv_by_record(dataset_path: str, output_dir: str | None = None) -> dict:
    """
    Split a CSV dataset into train/test sets using record-based splitting.

    Args:
        dataset_path: Absolute path to the source CSV file.
        output_dir:   Directory where train.csv / test.csv will be written.
                      Defaults to <project_root>/Dataset/CSV/splits/

    Returns:
        dict with keys: train_file, test_file, message,
                        train_rows, test_rows, total_rows
    """
    try:
        if not os.path.exists(dataset_path):
            return {'success': False, 'error': f'File not found: {dataset_path}'}

        df = pd.read_csv(dataset_path)

        if 'record' not in df.columns:
            return {'success': False, 'error': "CSV must contain a 'record' column"}

        # Determine output directory
        if output_dir is None:
            project_root = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            output_dir = os.path.join(project_root, 'Dataset', 'CSV', 'splits')

        os.makedirs(output_dir, exist_ok=True)

        # Record-based split
        train_df = df[~df['record'].isin(TEST_RECORDS)].copy()
        test_df  = df[df['record'].isin(TEST_RECORDS)].copy()

        train_file = os.path.join(output_dir, 'train.csv')
        test_file  = os.path.join(output_dir, 'test.csv')

        train_df.to_csv(train_file, index=False)
        test_df.to_csv(test_file, index=False)

        logger.info(
            f"CSV split complete — train: {len(train_df)} rows, test: {len(test_df)} rows"
        )

        return {
            'success':    True,
            'train_file': train_file,
            'test_file':  test_file,
            'train_rows': int(len(train_df)),
            'test_rows':  int(len(test_df)),
            'total_rows': int(len(df)),
            'message':    'Split successful',
        }

    except Exception as e:
        logger.error(f"CSV split error: {e}")
        return {'success': False, 'error': str(e)}


# -------------------------------------------------------------------
# 2. CSV Model Training  (v2 — improved accuracy)
# -------------------------------------------------------------------
def train_csv_model(
    train_csv_path: str,
    test_csv_path: str,
    epochs: int = 100,
    batch_size: int = 128,
    learning_rate: float = 5e-4,
    models_dir: str | None = None,
    images_dir: str | None = None,
    progress_callback=None,
    stop_event=None,
) -> dict:
    """
    Train a residual MLP on the tabular ECG CSV dataset.

    Key improvements over v1:
      - RobustScaler (outlier-robust normalisation)
      - Class-weighted CrossEntropyLoss (fixes N-class dominance)
      - Deeper residual MLP (512→256→128→64) with SiLU activations
      - AdamW optimizer + CosineAnnealingLR scheduler
      - Gradient clipping (max_norm=1.0)
      - Best-model checkpoint saved by val_acc (not just last epoch)

    Args:
        train_csv_path:    Path to train.csv
        test_csv_path:     Path to test.csv
        epochs:            Training epochs (default 100 for better convergence)
        batch_size:        Mini-batch size (128 for stable class-weighted gradients)
        learning_rate:     Initial AdamW lr
        models_dir:        Directory to save model_csv.pth + scaler.pkl
        images_dir:        Directory to save training plots
        progress_callback: fn(epoch, total_epochs, tr_loss, tr_acc, val_loss, val_acc)
        stop_event:        threading.Event to abort training loop

    Returns:
        dict with keys: success, metrics, model_path, scaler_path, image_files
    """
    try:
        if not os.path.exists(train_csv_path):
            return {'success': False, 'error': f'train.csv not found: {train_csv_path}'}
        if not os.path.exists(test_csv_path):
            return {'success': False, 'error': f'test.csv not found: {test_csv_path}'}

        # ---------- Load data ----------
        train_df = pd.read_csv(train_csv_path)
        test_df  = pd.read_csv(test_csv_path)

        # Drop metadata; split features / labels
        X_train_raw = train_df.drop(columns=['record', 'type'], errors='ignore').values.astype(np.float32)
        y_train_raw = train_df['type'].values

        X_test_raw  = test_df.drop(columns=['record', 'type'], errors='ignore').values.astype(np.float32)
        y_test_raw  = test_df['type'].values

        # ---------- Label encoding ----------
        le = LabelEncoder()
        y_train = le.fit_transform(y_train_raw)
        y_test  = le.transform(y_test_raw)
        num_classes = len(le.classes_)
        input_dim   = X_train_raw.shape[1]

        logger.info(f"Classes: {list(le.classes_)} | Input dim: {input_dim} | num_classes: {num_classes}")

        # ---------- Class-weight computation (key for imbalanced data) ----------
        raw_class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.arange(num_classes),
            y=y_train,
        )
        # Smooth weights with sqrt and cap them to avoid extremely large loss penalty
        # on rare classes, which practically forces the network to overfit on a handful of examples.
        class_weights_np = np.sqrt(raw_class_weights)
        class_weights_np = np.clip(class_weights_np, a_min=None, a_max=4.0)
        logger.info(f"Smoothed & Capped class weights: {dict(zip(le.classes_, class_weights_np.round(3)))}")

        # ---------- Feature scaling — RobustScaler (outlier-robust) ----------
        scaler = RobustScaler()
        X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
        X_test  = scaler.transform(X_test_raw).astype(np.float32)

        # Replace any remaining NaN / Inf after scaling
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_test  = np.nan_to_num(X_test,  nan=0.0, posinf=0.0, neginf=0.0)

        # ---------- Persist scaler + label encoder ----------
        if models_dir is None:
            models_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models'
            )
        if images_dir is None:
            images_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'images'
            )
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(images_dir, exist_ok=True)

        scaler_path = os.path.join(models_dir, 'scaler.pkl')
        le_path     = os.path.join(models_dir, 'label_encoder.pkl')
        joblib.dump(scaler, scaler_path)
        joblib.dump(le, le_path)

        # ---------- Build PyTorch datasets ----------
        X_tr_t = torch.FloatTensor(X_train)
        y_tr_t = torch.LongTensor(y_train)
        X_te_t = torch.FloatTensor(X_test)
        y_te_t = torch.LongTensor(y_test)

        train_loader = DataLoader(
            TensorDataset(X_tr_t, y_tr_t),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,          # keep batch-norm happy
        )
        test_loader = DataLoader(
            TensorDataset(X_te_t, y_te_t),
            batch_size=batch_size * 2,
        )

        # ---------- Model, loss, optimizer, scheduler ----------
        device       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model        = ECGMorph_MLP(input_dim, num_classes).to(device)
        class_weights_t = torch.FloatTensor(class_weights_np).to(device)

        # Label smoothing removed to lower the mathematical loss floor closer to 0.5 (user preference)
        criterion  = nn.CrossEntropyLoss(weight=class_weights_t)
        # Stronger weight decay + balanced learning rate
        optimizer  = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=5e-2)
        # Natively locks onto validation peaks to stop stuttering
        scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)

        logger.info(f"Model params: {sum(p.numel() for p in model.parameters()):,} | Device: {device}")

        # ---------- Training loop with best-model checkpointing ----------
        history            = []
        model_path         = os.path.join(models_dir, 'model_csv.pth')
        best_val_acc       = -1.0
        best_state_dict    = None
        patience_counter   = 0
        patience_limit     = 12

        for epoch in range(1, epochs + 1):
            # Stop signal
            if stop_event and stop_event.is_set():
                logger.info("CSV training stopped by user at epoch %d", epoch)
                break

            # ── Train ──
            model.train()
            tr_loss_sum, tr_correct, tr_total = 0.0, 0, 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                
                logits = model(xb)
                loss   = criterion(logits, yb)
                tr_correct += (logits.argmax(dim=1) == yb).sum().item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # gradient clip
                optimizer.step()
                
                tr_loss_sum += loss.item() * len(xb)
                tr_total    += len(xb)

            tr_loss = tr_loss_sum / tr_total
            tr_acc  = tr_correct  / tr_total * 100

            # ── Validate ──
            model.eval()
            val_loss_sum, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for xb, yb in test_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    logits  = model(xb)
                    loss    = criterion(logits, yb)
                    val_loss_sum += loss.item() * len(xb)
                    val_correct  += (logits.argmax(dim=1) == yb).sum().item()
                    val_total    += len(xb)

            val_loss = val_loss_sum / val_total
            val_acc  = val_correct  / val_total * 100

            # Best-model checkpoint & Early Stopping
            if val_acc > best_val_acc:
                best_val_acc    = val_acc
                best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
                logger.info(f"  ↑ New best val_acc={best_val_acc:.2f}% at epoch {epoch}")
            else:
                patience_counter += 1

            ep_record = {
                'epoch':      epoch,
                'train_loss': round(float(tr_loss), 4),
                'train_acc':  round(float(tr_acc),  2),
                'val_loss':   round(float(val_loss), 4),
                'val_acc':    round(float(val_acc),  2),
            }
            history.append(ep_record)

            if progress_callback:
                progress_callback(epoch, epochs, tr_loss, tr_acc, val_loss, val_acc)

            current_lr = optimizer.param_groups[0]['lr']
            logger.info(
                f"Epoch {epoch}/{epochs} | "
                f"TrLoss={tr_loss:.4f} TrAcc={tr_acc:.1f}% | "
                f"ValLoss={val_loss:.4f} ValAcc={val_acc:.1f}% | "
                f"LR={current_lr:.2e}"
            )

            # Step the plateau scheduler strictly on val_acc
            scheduler.step(val_acc)

            if patience_counter >= patience_limit:
                logger.info(f"Early stopping triggered at epoch {epoch} (No improvement for {patience_limit} epochs)")
                break

        # ---------- Save best checkpoint (not last epoch) ----------
        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)
            logger.info(f"Restored best model checkpoint (val_acc={best_val_acc:.2f}%)")

        torch.save(
            {
                'model_state_dict': model.state_dict(),
                'input_dim':        input_dim,
                'num_classes':      num_classes,
                'classes':          list(le.classes_),
                'best_val_acc':     best_val_acc,
            },
            model_path,
        )

        # ---------- Final evaluation on best checkpoint ----------
        model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                all_preds.extend(model(xb).argmax(dim=1).cpu().numpy())
                all_true.extend(yb.numpy())

        all_preds = np.array(all_preds)
        all_true  = np.array(all_true)

        eval_metrics = {
            'accuracy':  float(accuracy_score(all_true, all_preds)),
            'precision': float(precision_score(all_true, all_preds, average='weighted', zero_division=0)),
            'recall':    float(recall_score(all_true, all_preds, average='weighted', zero_division=0)),
            'f1':        float(f1_score(all_true, all_preds, average='weighted', zero_division=0)),
        }

        logger.info(f"CSV training complete. Best val_acc={best_val_acc:.2f}% | Final eval: {eval_metrics}")

        # ---------- Generate training plots ----------
        image_files = []
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

            # --- Training history ---
            hist_path   = os.path.join(images_dir, 'training_history_csv.png')
            epochs_list = [h['epoch'] for h in history]
            fig, axes   = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle('CSV / MLP Training History (v2)', fontsize=14, fontweight='bold')

            axes[0].plot(epochs_list, [h['train_loss'] for h in history], label='Train Loss', color='#6366f1', linewidth=2)
            axes[0].plot(epochs_list, [h['val_loss']   for h in history], label='Val Loss',   color='#f59e0b', linewidth=2, linestyle='--')
            axes[0].set_title('Loss'); axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss (weighted CE)')
            axes[0].legend(); axes[0].grid(alpha=0.3)

            axes[1].plot(epochs_list, [h['train_acc'] for h in history], label='Train Acc', color='#10b981', linewidth=2)
            axes[1].plot(epochs_list, [h['val_acc']   for h in history], label='Val Acc',   color='#ef4444', linewidth=2, linestyle='--')
            axes[1].axhline(best_val_acc, color='#6366f1', linestyle=':', linewidth=1.5, label=f'Best Val {best_val_acc:.1f}%')
            axes[1].set_title('Accuracy'); axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy (%)')
            axes[1].legend(); axes[1].grid(alpha=0.3)

            plt.tight_layout()
            plt.savefig(hist_path, dpi=120, bbox_inches='tight')
            plt.close(fig)
            image_files.append(hist_path)

            # --- Confusion matrix ---
            cm_path      = os.path.join(images_dir, 'confusion_matrix_csv.png')
            class_labels = list(le.classes_)
            cm    = confusion_matrix(all_true, all_preds)
            fig2, ax2 = plt.subplots(figsize=(max(6, len(class_labels) * 1.4), max(5, len(class_labels) * 1.2)))
            disp  = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
            disp.plot(ax=ax2, colorbar=True, cmap='Blues')
            ax2.set_title(f'CSV / MLP Confusion Matrix — Best Val Acc {best_val_acc:.1f}%', fontsize=13, fontweight='bold')
            plt.tight_layout()
            plt.savefig(cm_path, dpi=120, bbox_inches='tight')
            plt.close(fig2)
            image_files.append(cm_path)

            logger.info(f"Saved CSV training plots: {hist_path}, {cm_path}")

        except Exception as plot_err:
            logger.warning(f"Could not generate CSV training plots: {plot_err}")

        return {
            'success':      True,
            'model_path':   model_path,
            'scaler_path':  scaler_path,
            'history':      history,
            'metrics':      eval_metrics,
            'classes':      list(le.classes_),
            'image_files':  image_files,
            'best_val_acc': best_val_acc,
        }

    except Exception as e:
        logger.error(f"CSV training error: {e}", exc_info=True)
        return {'success': False, 'error': str(e)}


# -------------------------------------------------------------------
# 3. CSV Batch Analysis (inference)
# -------------------------------------------------------------------
def analyze_csv_file(
    csv_path: str,
    models_dir: str | None = None,
) -> dict:
    """
    Run batch inference on a CSV file using the saved MLP + scaler.

    Args:
        csv_path:   Path to the uploaded CSV file (multiple rows = beats)
        models_dir: Directory containing model_csv.pth + scaler.pkl

    Returns:
        dict compatible with existing EDF analysis result format:
          primary_diagnosis, confidence, is_normal, predictions, segments_analyzed
    """
    try:
        if models_dir is None:
            models_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models'
            )

        model_path  = os.path.join(models_dir, 'model_csv.pth')
        scaler_path = os.path.join(models_dir, 'scaler.pkl')
        le_path     = os.path.join(models_dir, 'label_encoder.pkl')

        if not os.path.exists(model_path):
            return {'success': False, 'error': 'CSV model not found. Train the CSV model first (model_csv.pth).'}
        if not os.path.exists(scaler_path):
            return {'success': False, 'error': 'Scaler not found. Train the CSV model first (scaler.pkl).'}
        if not os.path.exists(le_path):
            return {'success': False, 'error': 'Label encoder not found. Train the CSV model first (label_encoder.pkl).'}

        # ---------- Load artifacts ----------
        scaler    = joblib.load(scaler_path)
        le        = joblib.load(le_path)

        checkpoint = torch.load(model_path, map_location='cpu')
        input_dim  = checkpoint['input_dim']
        num_classes = checkpoint['num_classes']
        classes    = checkpoint.get('classes', list(le.classes_))

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model  = ECGMorph_MLP(input_dim, num_classes).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # ---------- Load and prepare CSV ----------
        df = pd.read_csv(csv_path)

        # Drop metadata columns if present; keep only feature columns
        cols_to_drop = [c for c in ['record', 'type'] if c in df.columns]
        X_raw = df.drop(columns=cols_to_drop).values.astype(np.float32)

        if X_raw.shape[1] != input_dim:
            return {
                'success': False,
                'error': (
                    f"Feature dimension mismatch: CSV has {X_raw.shape[1]} features "
                    f"but model expects {input_dim}."
                ),
            }

        X_scaled = scaler.transform(X_raw).astype(np.float32)
        
        # Consistent NaNs/Infs handling
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        X_tensor = torch.FloatTensor(X_scaled).to(device)

        # ---------- Batch inference ----------
        batch_size = 256
        all_preds = []
        with torch.no_grad():
            for i in range(0, len(X_tensor), batch_size):
                batch  = X_tensor[i:i + batch_size]
                logits = model(batch)
                preds  = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)

        all_preds = np.array(all_preds)
        total_samples = len(all_preds)

        # ---------- Majority voting + distribution ----------
        unique_idxs, counts = np.unique(all_preds, return_counts=True)
        majority_idx   = unique_idxs[np.argmax(counts)]
        majority_count = int(np.max(counts))
        confidence     = round(majority_count / total_samples * 100, 1)

        # Map encoded class index → raw label → diagnosis name
        majority_raw_label = le.inverse_transform([majority_idx])[0]
        primary_diagnosis  = get_diagnosis_name(majority_raw_label)
        is_normal          = majority_raw_label.strip().upper() == 'N'

        # Build per-class prediction dict (diagnosis name → count)
        predictions = {}
        for idx, cnt in zip(unique_idxs, counts):
            raw_label = le.inverse_transform([idx])[0]
            diag_name = get_diagnosis_name(raw_label)
            predictions[diag_name] = int(cnt)

        logger.info(
            f"CSV inference complete: {total_samples} beats → "
            f"{primary_diagnosis} ({confidence}%)"
        )

        return {
            'success':           True,
            'primary_diagnosis': primary_diagnosis,
            'confidence':        confidence,
            'is_normal':         is_normal,
            'predictions':       predictions,
            'segments_analyzed': total_samples,
            'classes':           classes,
        }

    except Exception as e:
        logger.error(f"CSV analysis error: {e}", exc_info=True)
        return {'success': False, 'error': str(e)}
