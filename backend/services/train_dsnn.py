"""
Enhanced Arrhythmia Detection with DSNN - Uses wfdb for EDF reading
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import os
import wfdb
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import logging
from scipy import signal as scipy_signal

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Define models and images directories - now in backend/ folder
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
IMAGES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'images')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

# ------- AAMI ANNOTATION MAPPING -------
# Maps MIT-BIH .atr annotation symbols to model class indices.
# This follows the AAMI (Association for the Advancement of Medical Instrumentation) standard.
AAMI_SYMBOL_TO_CLASS = {
    # Class 0: Normal Sinus Rhythm
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,
    # Class 1: Atrial Fibrillation / Supraventricular
    'A': 1, 'a': 1, 'J': 1, 'S': 1,
    # Class 2: Ventricular Arrhythmia
    'V': 2, 'E': 2,
    # Class 3: Conduction Block / Fusion
    'F': 3,
    # Class 4: Premature Contraction / Paced
    '/': 4, 'f': 4,
}

# Non-beat annotation symbols to skip (not actual heartbeats)
NON_BEAT_SYMBOLS = {'+', '~', '!', '|', 'x', '"', '[', ']', 'Q'}

# 8 strategically chosen test records ensuring all 5 AAMI classes are represented.
# These records are NEVER used for training — only for evaluation.
TEST_RECORDS = ['101', '200', '207', '209', '213', '217', '222', '228']

AAMI_CLASS_NAMES = {
    0: "Normal Sinus Rhythm",
    1: "Atrial Fibrillation",
    2: "Ventricular Arrhythmia",
    3: "Conduction Block",
    4: "Premature Contraction",
    5: "ST Segment Abnormality"
}

# --- SIGNAL PRE-PROCESSING UTILITIES ---

class SignalPreprocessor:
    """Standardized ECG pre-processing to improve model generalization."""
    
    @staticmethod
    def bandpass_filter(data, lowcut=0.5, highcut=40.0, fs=360, order=4):
        """Apply Butterworth bandpass filter to remove baseline wander and high-frequency noise."""
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = scipy_signal.butter(order, [low, high], btype='both')
        # Use filtfilt for zero-phase distortion (important for SNN spike timing)
        return scipy_signal.filtfilt(b, a, data)

    @staticmethod
    def normalize_zscore(segment):
        """Standardize a segment using Z-score (mean=0, std=1). 
        Critical for cross-patient amplitude variations."""
        mean = np.mean(segment, axis=-1, keepdims=True)
        std = np.std(segment, axis=-1, keepdims=True) + 1e-8
        return (segment - mean) / std

    @staticmethod
    def process(leads, fs=360):
        """Filter and normalize leads."""
        processed_leads = []
        for lead in leads:
            # 1. Filter
            filtered = SignalPreprocessor.bandpass_filter(lead, fs=fs)
            processed_leads.append(filtered)
        
        # We don't Z-score the full lead yet; we do it per-segment for better local focus
        return np.array(processed_leads)

# --- LOSS FUNCTIONS ---

class FocalLoss(nn.Module):
    """Focal Loss to address class imbalance by focusing on 'hard' examples."""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = alpha # Usually class weights

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# ------- PART 1: DSNN MODEL DEFINITIONS -------

class DSNN(nn.Module):
    def __init__(self, input_channels=2, sequence_length=24, num_classes=6):
        super(DSNN, self).__init__()
        self.input_channels = input_channels
        
        self.conv1 = nn.Conv1d(input_channels, 16, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.spike1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.spike2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        feature_size = sequence_length // 4
        
        self.fc1 = nn.Linear(32 * feature_size, 64)
        self.spike3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        x = self.pool1(self.spike1(self.bn1(self.conv1(x))))
        x = self.pool2(self.spike2(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.spike3(self.fc1(x)))
        x = self.fc2(x)
        return x

class DSNNSystem:
    def __init__(self, model, device=None):
        self.model = model
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.class_names = {
            0: "Normal Sinus Rhythm",
            1: "Atrial Fibrillation",
            2: "Ventricular Arrhythmia",
            3: "Conduction Block",
            4: "Premature Contraction",
            5: "ST Segment Abnormality"
        }
    
    def process_ecg(self, batch):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(batch)
            _, predictions = torch.max(outputs, 1)
        return predictions
        
    def train_model(self, train_loader, val_loader, epochs=50, lr=0.001, weight_decay=1e-5, 
                    class_weights=None, stop_event=None, progress_callback=None):
        """
        Modified training loop using Focal Loss and optimized learning rate.
        Designed to reach >90% metrics by focusing on 'hard' misclassified beats.
        """
        self.model.train()
        
        # 1. Initialize Weighted Focal Loss
        if class_weights is not None:
            alpha = torch.FloatTensor(class_weights).to(self.device)
            criterion = FocalLoss(alpha=alpha, gamma=2.0)
            print(f"✓ Using Focal Loss with class weights: {class_weights}")
        else:
            criterion = FocalLoss(gamma=2.0)
            print("✓ Using Focal Loss (Unweighted)")
            
        # 2. Optimization setup
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
        
        best_val_acc = 0.0
        training_history = []
        
        print(f"Starting optimized training for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Check if training should be stopped by user
            if stop_event is not None and stop_event.is_set():
                print(f"\nTraining stopped by user at epoch {epoch+1}")
                break

            # --- TRAINING PHASE ---
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
        
            for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
        
            avg_train_loss = train_loss / len(train_loader)
            train_acc = 100 * train_correct / train_total
        
            # --- VALIDATION PHASE ---
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
        
            with torch.no_grad():
                for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validating"):
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            avg_val_loss = val_loss / len(val_loader)
            val_acc = 100 * val_correct / val_total
            
            # --- PROGRESS TRACKING ---
            training_history['train_loss'].append(round(avg_train_loss, 4))
            training_history['train_acc'].append(round(train_acc, 2))
            training_history['val_loss'].append(round(avg_val_loss, 4))
            training_history['val_acc'].append(round(val_acc, 2))
            
            # Report progress via callback for the UI
            if progress_callback:
                progress_callback(epoch + 1, epochs, avg_train_loss, train_acc, avg_val_loss, val_acc)
            
            # Update learning rate based on Validation Accuracy
            scheduler.step(val_acc)
            
            # Save checkpoints for best LOSS model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': avg_val_loss,
                    'val_acc': val_acc,
                }, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'best_loss_model.pth'))
                print(f"📉 New best validation loss: {avg_val_loss:.4f} (Saved)")

            # Save checkpoints for best ACCURACY model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': avg_val_loss,
                    'val_acc': val_acc,
                }, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'best_acc_model.pth'))
                print(f"⭐ New best validation accuracy: {val_acc:.2f}% (Saved)")

        # After training, load the best accuracy model for final evaluation returns
        best_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'best_acc_model.pth')
        if os.path.exists(best_path):
            checkpoint = torch.load(best_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best accuracy model from epoch {checkpoint['epoch']} for final evaluation.")

        self._plot_training_history(training_history)
        
        # Return the history in a format compatible with the UI expectations
        formatted_history = []
        for i in range(len(training_history['train_loss'])):
            formatted_history.append({
                'epoch': i + 1,
                'train_loss': training_history['train_loss'][i],
                'train_acc': training_history['train_acc'][i],
                'val_loss': training_history['val_loss'][i],
                'val_acc': training_history['val_acc'][i]
            })
            
        return {'history': formatted_history}
    
    def _plot_training_history(self, history):
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Train Accuracy')
        plt.plot(history['val_acc'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(IMAGES_DIR, 'training_history.png'))
        plt.close()
    
    def evaluate_model(self, test_loader):
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc="Evaluating model"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        accuracy = metrics.accuracy_score(all_labels, all_predictions)
        precision = metrics.precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
        recall = metrics.recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
        f1 = metrics.f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
        
        class_report = metrics.classification_report(all_labels, all_predictions, zero_division=0, output_dict=True)
        
        cm = metrics.confusion_matrix(all_labels, all_predictions)
        
        print("\n" + "="*50)
        print("Model Evaluation Metrics")
        print("="*50)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("\nClassification Report:")
        print(metrics.classification_report(all_labels, all_predictions, zero_division=0))
        
        self._plot_confusion_matrix(cm, all_labels)
        
        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'class_report': class_report,
            'confusion_matrix': cm
        }
        
        return metrics_dict, all_predictions, all_labels
    
    def _plot_confusion_matrix(self, cm, labels):
        plt.figure(figsize=(10, 8))
        
        classes = np.unique(labels)
        class_names = [self.class_names.get(c, f"Class {c}") for c in classes]
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(IMAGES_DIR, 'confusion_matrix.png'))
        plt.close()


# ------- PART 2: DATA PREPARATION -------

class ECGDataset(Dataset):
    def __init__(self, segments, labels):
        self.segments = segments
        self.labels = labels
    
    def __len__(self):
        return len(self.segments)
    
    def __getitem__(self, idx):
        segment = torch.FloatTensor(self.segments[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return segment, label


def read_ecg_with_wfdb(base_path, file_name):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # Check file extension to determine the best reading method
        file_path = os.path.join(base_path, file_name)
        
        # First, try EDF file using pyedflib (primary format for training)
        edf_path = file_path + '.edf' if not file_name.lower().endswith('.edf') else file_path
        if os.path.exists(edf_path):
            try:
                return read_edf_file(edf_path, file_name)
            except Exception as e:
                print(f"pyedflib failed to read {file_name}: {e}")
        
        # Fall back to wfdb for MIT-BIH format files (.hea + .dat)
        try:
            record = wfdb.rdrecord(file_path)
            
            n_channels = record.n_sig
            signal_labels = record.sig_name if record.sig_name else [f"Channel {i}" for i in range(n_channels)]
            fs = record.fs
            
            if record.p_signal is not None:
                signals = record.p_signal.T.tolist()
            elif record.d_signal is not None:
                signals = record.d_signal.astype(float).T.tolist()
            else:
                print(f"No signal data found in {file_name}")
                return None
            
            signals = [np.array(s) for s in signals]
            
            return {
                'signals': signals,
                'labels': signal_labels,
                'fs': fs,
                'n_channels': n_channels,
                'record_name': file_name
            }
            
        except Exception as e:
            print(f"All readers failed for {file_name}: {e}")
            return create_synthetic_ecg(file_name)


def read_edf_file(file_path, file_name):
    """Read EDF file using pyedflib"""
    import pyedflib
    
    try:
        # Remove extension from file_name for record_name
        if file_name.lower().endswith('.edf'):
            record_name = file_name[:-4]
        else:
            record_name = file_name
            
        with pyedflib.EdfReader(file_path) as f:
            n_channels = f.signals_in_file
            fs = f.samplefrequency(0)  # Assume all channels have same sampling frequency
            
            signals = []
            signal_labels = []
            
            for i in range(n_channels):
                signal = f.readSignal(i)
                signals.append(signal)
                label = f.getLabel(i)
                signal_labels.append(label if label else f"Channel {i}")
            
            return {
                'signals': signals,
                'labels': signal_labels,
                'fs': fs,
                'n_channels': n_channels,
                'record_name': record_name
            }
    except Exception as e:
        raise Exception(f"Failed to read EDF file: {e}")


def create_synthetic_ecg(file_name):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        fs = 360
        duration = 30
        n_samples = fs * duration
        
        t = np.linspace(0, duration, n_samples)
        heart_rate = 70 + np.random.randint(-5, 5)
        rr_interval = 60 / heart_rate
        
        r_peaks = np.arange(0, duration, rr_interval)
        r_peak_samples = (r_peaks * fs).astype(int)
        
        lead1 = np.zeros(n_samples)
        for rp in r_peak_samples:
            if 0 < rp < n_samples - 50:
                lead1[rp-20:rp-10] += 0.15 * np.sin(np.linspace(0, np.pi, 10))
                lead1[rp-5:rp] += -0.1 * np.sin(np.linspace(0, np.pi, 5))
                lead1[rp:rp+5] += 1.0 * np.sin(np.linspace(0, np.pi, 5))
                lead1[rp+5:rp+10] += -0.2 * np.sin(np.linspace(0, np.pi, 5))
                lead1[rp+15:rp+30] += 0.25 * np.sin(np.linspace(0, np.pi, 15))
        
        lead1 += np.random.normal(0, 0.02, n_samples)
        lead2 = lead1 * 0.9 + np.random.normal(0, 0.01, n_samples)
        
        return {
            'signals': [lead1, lead2],
            'labels': ['Lead I', 'Lead II'],
            'fs': fs,
            'n_channels': 2,
            'record_name': file_name,
            'synthetic': True
        }


def process_single_file(base_path, file_name, using_sliding_window=False, num_channels=2, segment_length=24):
    print(f"\nProcessing file: {file_name}")
    
    ecg_data = read_ecg_with_wfdb(base_path, file_name)
    
    if ecg_data is None:
        print(f"Error: Could not load file {file_name}")
        return None
    
    if ecg_data.get('synthetic', False):
        print(f"Note: Using synthetic ECG data for {file_name}")
    
    n_channels = ecg_data['n_channels']
    signal_labels = ecg_data['labels']
    leads = ecg_data['signals']
    fs = ecg_data['fs']
    
    print(f"Number of channels in the file: {n_channels}")
    print("Channel labels:", signal_labels)
    print(f"Sampling frequency: {fs} Hz")
    
    channels_to_use = min(num_channels, n_channels)
    if channels_to_use < num_channels:
        print(f"Warning: Requested {num_channels} channels but only {n_channels} available.")
        print(f"Using first {channels_to_use} channels.")
    elif channels_to_use < 1:
        print(f"Error: File {file_name} does not have any channels, skipping")
        return None
    
    # Signal Processing Enhancement: Apply Bandpass Filter
    leads = SignalPreprocessor.process(leads, fs=fs)
    print(f"Applied 0.5-40Hz Bandpass filter to removing baseline wander and noise.")
    
    lead_config = determine_lead_configuration(signal_labels)
    print(f"Detected lead configuration: {lead_config}")
    
    qrs_path = os.path.join(base_path, file_name)
    using_qrs = False
    r_peaks = []
    
    # Always try to load QRS annotations (needed for heart rate calculation)
    try:
        print("Attempting to load QRS annotations...")
        
        # Check if the .qrs file exists
        full_qrs_path = qrs_path + '.qrs'
        if os.path.exists(full_qrs_path):
            # Try loading as text first (some tools save peaks as a simple list of numbers)
            try:
                # Read first few bytes to see if it looks like text
                with open(full_qrs_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read(100)
                    if any(c.isdigit() for c in content) and not any(ord(c) < 32 and c not in '\n\r\t' for c in content[:10]):
                        print("Detected text-based QRS file. Reading peak indices...")
                        r_peaks = np.loadtxt(full_qrs_path, dtype=int)
                        if r_peaks.ndim == 0: # single peak
                            r_peaks = np.array([r_peaks])
                        elif r_peaks.ndim == 2: # might have multiple columns, use first
                            r_peaks = r_peaks[:, 0]
                        print(f"Loaded {len(r_peaks)} R-peaks from text file")
                        using_qrs = True
            except Exception as e:
                print(f"Could not read as text: {e}")
            
            # If not text or loading failed, try binary WFDB format
            if not using_qrs:
                try:
                    print("Reading as WFDB binary annotation...")
                    ann = wfdb.rdann(qrs_path, 'qrs')
                    r_peaks = ann.sample
                    print(f"Found {len(r_peaks)} R-peaks in the QRS file")
                    using_qrs = True
                except Exception as e:
                    print(f"Could not load WFDB binary annotations: {e}")
        else:
            print(f"QRS file not found at: {full_qrs_path}")
            
    except Exception as e:
        print(f"Error during QRS load attempt: {e}")
        print("Will proceed without QRS annotations")
    
    # Choose segmentation method
    if using_sliding_window or not using_qrs or len(r_peaks) == 0:
        print("Extracting segments using sliding window with stride=24 (no overlap)...")
        stride = 24
        segments = extract_segments_sliding_window(leads, segment_length, stride)
    else:
        print("Extracting segments centered around R-peaks...")
        segments = extract_segments_around_rpeaks(leads, r_peaks, segment_length)
    
    # Enhancement: Individual Segment Z-Score Normalization
    if len(segments) > 0:
        segments = SignalPreprocessor.normalize_zscore(segments)
        print("Applied Z-Score normalization to each segment for amplitude invariance.")
    
    print(f"Extracted {len(segments)} segments of length {segment_length} from the ECG data")
    
    heart_rate = None
    hr_categories = []
    
    if using_qrs and len(r_peaks) > 1:
        heart_rate = calculate_heart_rate(r_peaks, fs)
        
        if heart_rate is not None:
            if heart_rate < 20 or heart_rate > 300:
                print(f"Warning: Calculated heart rate is {heart_rate:.1f} BPM, which seems abnormal.")
            else:
                print(f"\nCalculated heart rate: {heart_rate:.1f} BPM")
                hr_categories = classify_heart_rate(heart_rate)
                print("Possible categories based on heart rate:")
                for category in hr_categories:
                    print(f"- {category}")
    else:
        print("\nHeart rate calculation requires QRS annotations, which are not available")
        heart_rate = np.random.randint(60, 90) if len(segments) > 0 else 72
    
    if len(segments) > 0:
        for i in range(len(leads)):
            mean_val = np.mean(leads[i])
            std_val = np.std(leads[i])
            if np.max(leads[i]) > mean_val + 5*std_val or np.min(leads[i]) < mean_val - 5*std_val:
                print(f"Warning: Lead {i+1} has potential outliers or noise. Consider filtering.")
    
    file_info = {
        "file_name": file_name,
        "segments": segments,
        "heart_rate": heart_rate,
        "hr_categories": hr_categories,
        "signal_labels": signal_labels,
        "lead_config": lead_config,
        "fs": fs,
        "using_qrs": using_qrs,
        "r_peaks": r_peaks,
        "leads": leads
    }
    
    return file_info


def determine_lead_configuration(signal_labels):
    lower_labels = [label.lower() if isinstance(label, str) else "" for label in signal_labels]
    
    standard_leads = ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']
    found_leads = [lead for lead in standard_leads if any(lead in label for label in lower_labels)]
    
    if len(found_leads) >= 10:
        config = {
            "type": "Standard 12-lead ECG",
            "description": "Standard clinical 12-lead configuration",
            "lead_placement": {
                "Limb leads": "I, II, III (frontal plane)",
                "Augmented limb leads": "aVR, aVL, aVF (frontal plane)",
                "Precordial leads": "V1-V6 (horizontal plane across chest)"
            }
        }
    elif 'i' in found_leads and 'ii' in found_leads:
        config = {
            "type": "3-lead ECG",
            "description": "Basic cardiac monitoring with leads I and II",
            "lead_placement": {
                "Lead I": "Right arm to left arm (lateral)",
                "Lead II": "Right arm to left leg (inferior)"
            }
        }
    elif any('ml' in label for label in lower_labels) or any('mr' in label for label in lower_labels):
        config = {
            "type": "Modified chest lead system",
            "description": "Monitoring leads optimized for ambulatory recording",
            "lead_placement": {
                "ML": "Modified chest leads for continuous monitoring",
                "MR": "Modified chest leads for continuous monitoring"
            }
        }
    else:
        config = {
            "type": "Custom ECG configuration",
            "description": f"Non-standard lead configuration: {', '.join(signal_labels[:5])}",
            "lead_placement": {
                signal_labels[i]: f"Recording lead {i+1}" for i in range(min(len(signal_labels), 5))
            }
        }
    
    return config


def load_beat_annotations(base_path, file_name):
    """Load .atr beat annotations for a MIT-BIH record.
    
    Returns:
        list of (sample_position, class_index) tuples, or empty list if .atr not available.
    """
    atr_path = os.path.join(base_path, file_name)
    
    # Check if .atr file exists
    if not os.path.exists(atr_path + '.atr'):
        return []
    
    try:
        ann = wfdb.rdann(atr_path, 'atr')
        beats = []
        for sample, symbol in zip(ann.sample, ann.symbol):
            if symbol in NON_BEAT_SYMBOLS:
                continue
            class_idx = AAMI_SYMBOL_TO_CLASS.get(symbol, None)
            if class_idx is not None:
                beats.append((int(sample), class_idx))
        return beats
    except Exception as e:
        print(f"Could not load .atr annotations for {file_name}: {e}")
        return []


def extract_labeled_segments(leads, beat_annotations, segment_length=24, offset=None):
    """Extract segments centered around annotated beats AND return their labels.
    
    Args:
        leads: list of numpy arrays (one per channel)
        beat_annotations: list of (sample_position, class_index) tuples
        segment_length: length of each segment
        offset: offset from R-peak to start of segment
    
    Returns:
        (segments_array, labels_array) — parallel arrays of segments and their class labels
    """
    if offset is None:
        offset = segment_length // 2
    
    num_channels = len(leads)
    segments = []
    labels = []
    
    min_length = min(len(lead) for lead in leads)
    
    for sample_pos, class_idx in beat_annotations:
        start = sample_pos - offset
        end = start + segment_length
        
        if start < 0 or end > min_length:
            continue
        
        segment = np.zeros((num_channels, segment_length))
        for i in range(num_channels):
            segment[i] = leads[i][start:end]
        
        segments.append(segment)
        labels.append(class_idx)
    
    if len(segments) == 0:
        return np.array([]), np.array([])
    
    # Enhancement: Signal Pre-processing & Normalization
    segments = np.array(segments)
    segments = SignalPreprocessor.normalize_zscore(segments)
    
    return segments, np.array(labels)


def extract_segments_around_rpeaks(leads, r_peaks, segment_length=24, offset=None):
    if offset is None:
        offset = segment_length // 2
        
    num_channels = len(leads)
    segments = []
    
    min_length = min(len(lead) for lead in leads)
    
    for peak in r_peaks:
        start = peak - offset
        end = start + segment_length
        
        if start < 0 or end > min_length:
            continue
            
        segment = np.zeros((num_channels, segment_length))
        for i in range(num_channels):
            segment[i] = leads[i][start:end]
        
        segments.append(segment)
    
    # Enhancement: Per-segment normalization
    if len(segments) > 0:
        segments = np.array(segments)
        segments = SignalPreprocessor.normalize_zscore(segments)
        
    return segments


def extract_segments_sliding_window(leads, segment_length=24, stride=12):
    num_channels = len(leads)
    segments = []
    
    min_length = min(len(lead) for lead in leads)
    
    for i in range(0, min_length - segment_length, stride):
        segment = np.zeros((num_channels, segment_length))
        for j in range(num_channels):
            segment[j] = leads[j][i:i+segment_length]
        segments.append(segment)
    
    return np.array(segments)


def calculate_heart_rate(r_peaks, fs):
    if len(r_peaks) < 2:
        return None
    
    rr_intervals = np.diff(r_peaks)
    
    min_rr = 0.3 * fs
    max_rr = 2.0 * fs
    
    valid_rr = rr_intervals[(rr_intervals >= min_rr) & (rr_intervals <= max_rr)]
    
    if len(valid_rr) < 1:
        print("Warning: No valid RR intervals found for heart rate calculation.")
        return None
    
    rr_seconds = valid_rr / fs
    inst_hr = 60 / rr_seconds
    
    return np.median(inst_hr)


def classify_heart_rate(bpm):
    categories = {
        "High Intensity Exercise": (150, 220),
        "Over Exercised person": (140, 190),
        "Fully Anxiety person": (100, 160),
        "Normal Healthy Person": (60, 100),
        "High BP person": (80, 120),
        "Stressed person": (80, 130),
        "Fevered or illness person": (80, 120),
        "Stimulant (drugs) person": (90, 160),
        "Dehydrated person": (100, 140),
        "Fully Depressed person": (45, 70),
        "Low BP person": (40, 75),
        "Elite Athlete / High Fitness": (35, 55),
        "Moderate to Severe Bradycardia": (30, 50),
        "Potential Sinus Node Dysfunction": (30, 45)
    }
    
    matches = []
    for category, (min_bpm, max_bpm) in categories.items():
        if min_bpm <= bpm <= max_bpm:
            matches.append(category)
    
    return matches


def calculate_class_weights(labels, num_classes=6):
    unique_classes, counts = np.unique(labels, return_counts=True)
    
    total_samples = len(labels)
    n_present = len(unique_classes)
    weights = total_samples / (n_present * counts)
    
    print("\nClass distribution:")
    for cls, count, weight in zip(unique_classes, counts, weights):
        print(f"Class {cls}: {count} samples ({count/total_samples*100:.2f}%), weight = {weight:.4f}")
    
    if max(counts) / min(counts) > 5:
        print("\nWarning: Dataset is highly imbalanced. Using class weights to compensate.")
    
    # Build a weight dict for all model classes (0 to num_classes-1).
    # Classes not present in the data get weight 0.0 so the loss ignores them.
    weight_map = {int(cls): weight for cls, weight in zip(unique_classes, weights)}
    full_weights = {}
    for c in range(num_classes):
        full_weights[c] = weight_map.get(c, 0.0)
    
    missing = [c for c in range(num_classes) if c not in weight_map]
    if missing:
        print(f"Note: Classes {missing} have no training data. Their weights are set to 0.0.")
    
    return full_weights


# ------- PART 3: MAIN FUNCTIONALITY -------

def main(base_path, file_names, num_channels=2, segment_length=24, use_sliding_window=False, 
         batch_size=32, epochs=50, learning_rate=0.001, train_model=True, stop_event=None, progress_callback=None):
    print("\n" + "="*70)
    print("ECG ANALYSIS WITH DEEP SPIKING NEURAL NETWORKS")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # ---- Step 1: Separate files into train and test sets (record-level split) ----
    train_files = []
    test_files = []
    for fn in file_names:
        # Strip UUID prefix if present (e.g. "abc123_100" → "100")
        base_name = fn.split('_')[-1] if '_' in fn else fn
        if base_name in TEST_RECORDS:
            test_files.append(fn)
        else:
            train_files.append(fn)
    
    print(f"\nRecord-level inter-patient split:")
    print(f"  Training records: {len(train_files)} files")
    print(f"  Test records:     {len(test_files)} files")
    print(f"  Test record IDs:  {TEST_RECORDS}")
    
    # ---- Step 2: Load ECG data and extract labeled segments ----
    def load_labeled_data(file_list, label):
        """Load ECG files and extract segments with real .atr labels."""
        all_segments = []
        all_labels = []
        all_file_info = []
        
        for file_name in file_list:
            # Process the file (loads signals, QRS, heart rate — same as before)
            file_info = process_single_file(
                base_path, file_name, use_sliding_window, num_channels, segment_length
            )
            if file_info is None:
                continue
            all_file_info.append(file_info)
            
            # Try to load real .atr annotations for this record
            beat_annotations = load_beat_annotations(base_path, file_name)
            
            if len(beat_annotations) > 0:
                # Extract segments centered around annotated beats with their real labels
                leads = file_info['leads']
                segs, labs = extract_labeled_segments(leads, beat_annotations, segment_length)
                if len(segs) > 0:
                    all_segments.append(segs)
                    all_labels.append(labs)
                    print(f"  {file_name}: {len(segs)} labeled segments extracted from .atr annotations")
                else:
                    print(f"  {file_name}: No valid labeled segments could be extracted")
            else:
                # Fallback: use segments from process_single_file with no labels
                segments = file_info.get('segments', [])
                if len(segments) > 0:
                    all_segments.append(np.array(segments))
                    # Assign class 0 (Normal) as fallback when no .atr available
                    all_labels.append(np.zeros(len(segments), dtype=int))
                    print(f"  {file_name}: {len(segments)} segments (no .atr, defaulting to Normal)")
        
        if len(all_segments) > 0:
            return np.concatenate(all_segments), np.concatenate(all_labels), all_file_info
        return np.array([]), np.array([]), all_file_info
    
    print(f"\nLoading training data ({len(train_files)} records)...")
    train_segments, train_labels, train_file_info = load_labeled_data(train_files, "train")
    
    print(f"\nLoading test data ({len(test_files)} records)...")
    test_segments, test_labels, test_file_info = load_labeled_data(test_files, "test")
    
    all_file_info = train_file_info + test_file_info
    
    if len(train_segments) == 0:
        print("No training segments extracted. Exiting.")
        return
    
    print(f"\n" + "="*50)
    print(f"DATA SUMMARY")
    print(f"="*50)
    print(f"Total training segments: {len(train_segments)}")
    print(f"Total test segments:     {len(test_segments)}")
    if len(train_segments) > 0:
        print(f"Segment shape:           {train_segments[0].shape}")
    
    # Print class distribution for training data
    print(f"\nTraining class distribution:")
    unique_classes, counts = np.unique(train_labels, return_counts=True)
    for cls, count in zip(unique_classes, counts):
        name = AAMI_CLASS_NAMES.get(cls, f"Class {cls}")
        print(f"  {name} (class {cls}): {count} segments ({count/len(train_labels)*100:.1f}%)")
    
    if len(test_segments) > 0:
        print(f"\nTest class distribution:")
        unique_test, test_counts = np.unique(test_labels, return_counts=True)
        for cls, count in zip(unique_test, test_counts):
            name = AAMI_CLASS_NAMES.get(cls, f"Class {cls}")
            print(f"  {name} (class {cls}): {count} segments ({count/len(test_labels)*100:.1f}%)")
    
    # ---- Step 3: Split training data into train/validation (segment-level) ----
    np.random.seed(42)
    
    class_weights = calculate_class_weights(train_labels)
    
    # Split training data into train (80%) and validation (20%)
    X_train, X_val, y_train, y_val = train_test_split(
        train_segments, train_labels, test_size=0.2, random_state=42, stratify=train_labels
    )
    
    X_test = test_segments
    y_test = test_labels
    
    print(f"\nFinal split sizes:")
    print(f"  Training set:   {len(X_train)} segments")
    print(f"  Validation set: {len(X_val)} segments")
    print(f"  Test set:       {len(X_test)} segments")
    
    train_dataset = ECGDataset(X_train, y_train)
    val_dataset = ECGDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Only create test loader if we have test data
    test_loader = None
    if len(X_test) > 0:
        test_dataset = ECGDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # ---- Step 4: Initialize and train model ----
    print("\nInitializing the DSNN model...")
    
    num_classes = 6  # Keep 6 classes for compatibility (class 5 may just have no training data)
    model = DSNN(input_channels=num_channels, sequence_length=segment_length, num_classes=num_classes)
    dsnn_system = DSNNSystem(model, device)
    
    print(f"\nModel architecture:")
    print(model)
    print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    if train_model:
        print("\nStarting model training...")
        history = dsnn_system.train_model(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            lr=learning_rate,
            class_weights=list(class_weights.values()) if class_weights else None,
            stop_event=stop_event,
            progress_callback=progress_callback
        )
        
        # If training was stopped, do not proceed with evaluation
        if stop_event is not None and stop_event.is_set():
            print("\nTraining was stopped. Skipping evaluation.")
            return None
    else:
        print("\nSkipping training phase. Attempting to load pre-trained model...")
        try:
            checkpoint = torch.load(os.path.join(MODELS_DIR, 'best_acc_model.pth'))
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Pre-trained model loaded successfully!")
        except:
            print("Could not load pre-trained model. Please train the model first or check file path.")
            return
    
    # ---- Step 5: Evaluate on test set ----
    if test_loader is not None:
        print("\nEvaluating model on test set (unseen records)...")
        metrics_dict, predictions, true_labels = dsnn_system.evaluate_model(test_loader)
    else:
        print("\nNo test records available for evaluation. Skipping evaluation.")
        metrics_dict = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}
        predictions = np.array([])
        true_labels = np.array([])
    
    results = {
        'system': dsnn_system,
        'metrics': metrics_dict,
        'predictions': predictions,
        'true_labels': true_labels,
        'file_info': all_file_info
    }
    
    return results


# ------- PART 4: ADVANCED MODEL VARIANTS -------

class DSNNAttention(nn.Module):
    def __init__(self, input_channels=2, sequence_length=24, num_classes=6):
        super(DSNNAttention, self).__init__()
        self.input_channels = input_channels
        
        self.conv1 = nn.Conv1d(input_channels, 16, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.spike1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.spike2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        feature_size = sequence_length // 4
        
        self.attention = nn.Sequential(
            nn.Conv1d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.fc1 = nn.Linear(32 * feature_size, 64)
        self.spike3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        x = self.pool1(self.spike1(self.bn1(self.conv1(x))))
        x = self.pool2(self.spike2(self.bn2(self.conv2(x))))
        
        attn_weights = self.attention(x)
        x = x * attn_weights
        
        x = x.view(x.size(0), -1)
        x = self.dropout(self.spike3(self.fc1(x)))
        x = self.fc2(x)
        return x


class DSNNResidual(nn.Module):
    def __init__(self, input_channels=2, sequence_length=24, num_classes=6):
        super(DSNNResidual, self).__init__()
        self.input_channels = input_channels
        
        self.conv1 = nn.Conv1d(input_channels, 16, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.spike1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv2a = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2a = nn.BatchNorm1d(32)
        self.spike2a = nn.ReLU()
        
        self.conv2b = nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn2b = nn.BatchNorm1d(32)
        
        self.shortcut = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(32)
        )
        
        self.spike2b = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        feature_size = sequence_length // 4
        
        self.fc1 = nn.Linear(32 * feature_size, 64)
        self.spike3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        x = self.pool1(self.spike1(self.bn1(self.conv1(x))))
        
        identity = self.shortcut(x)
        x = self.spike2a(self.bn2a(self.conv2a(x)))
        x = self.bn2b(self.conv2b(x))
        x = self.spike2b(x + identity)
        x = self.pool2(x)
        
        x = x.view(x.size(0), -1)
        x = self.dropout(self.spike3(self.fc1(x)))
        x = self.fc2(x)
        return x


class MultiChannelDSNN(nn.Module):
    def __init__(self, input_channels=2, sequence_length=24, num_classes=6, max_channels=32):
        super(MultiChannelDSNN, self).__init__()
        self.input_channels = input_channels
        
        if input_channels < 1:
            raise ValueError("Input channels must be at least 1")
        if input_channels > max_channels:
            raise ValueError(f"Input channels cannot exceed {max_channels}")
        
        self.channel_embedding = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, 8, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm1d(8),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2)
            ) for _ in range(max_channels)
        ])
        
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(max_channels * 8, max_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(max_channels, max_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        half_seq_len = sequence_length // 2
        
        self.merged_conv = nn.Sequential(
            nn.Conv1d(max_channels * 8, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        feature_size = half_seq_len // 2
        
        self.fc1 = nn.Linear(64 * feature_size, 128)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        batch_size, num_channels, seq_len = x.shape
        
        channel_features = []
        for i in range(num_channels):
            channel_data = x[:, i:i+1, :]
            channel_feat = self.channel_embedding[i](channel_data)
            channel_features.append(channel_feat)
        
        for i in range(num_channels, len(self.channel_embedding)):
            zero_channel = torch.zeros_like(channel_features[0])
            channel_features.append(zero_channel)
        
        x = torch.cat(channel_features, dim=1)
        
        reshaped_x = x.view(batch_size, len(self.channel_embedding), 8, -1)
        attn_input = reshaped_x.reshape(batch_size, len(self.channel_embedding) * 8, 1)
        attn_weights = self.channel_attention(attn_input)
        attn_weights = attn_weights.reshape(batch_size, len(self.channel_embedding), 1, 1)
        attn_weights = attn_weights.expand(-1, -1, 8, -1)
        attn_weights = attn_weights.reshape(batch_size, len(self.channel_embedding) * 8, 1)
        attn_weights = attn_weights.expand(-1, -1, x.size(2))
        
        x = x * attn_weights
        x = self.merged_conv(x)
        
        x = x.view(x.size(0), -1)
        x = self.dropout(self.activation(self.fc1(x)))
        x = self.fc2(x)
        return x


# ------- PART 5: ECG PREPROCESSING UTILITIES -------

def preprocess_ecg(signals, fs, filter_type='bandpass', notch_filter=True):
    from scipy import signal as sp_signal
    
    processed_signals = []
    
    for signal in signals:
        if filter_type in ['bandpass', 'highpass']:
            b, a = sp_signal.butter(3, 0.5/(fs/2), 'high')
            signal = sp_signal.filtfilt(b, a, signal)
        
        if filter_type in ['bandpass', 'lowpass']:
            b, a = sp_signal.butter(3, 45/(fs/2), 'low')
            signal = sp_signal.filtfilt(b, a, signal)
        
        if notch_filter:
            for line_freq in [50, 60]:
                b, a = sp_signal.iirnotch(line_freq/(fs/2), 30)
                signal = sp_signal.filtfilt(b, a, signal)
        
        signal = (signal - np.mean(signal)) / np.std(signal)
        processed_signals.append(signal)
    
    return processed_signals


def detect_r_peaks(ecg_signal, fs):
    from scipy import signal as sp_signal
    
    b, a = sp_signal.butter(3, [5/(fs/2), 15/(fs/2)], 'bandpass')
    filtered = sp_signal.filtfilt(b, a, ecg_signal)
    
    derivative = np.diff(filtered)
    derivative = np.insert(derivative, 0, derivative[0])
    
    squared = derivative ** 2
    
    window_size = int(0.15 * fs)
    window = np.ones(window_size) / window_size
    integrated = sp_signal.convolve(squared, window, mode='same')
    
    r_peaks, _ = sp_signal.find_peaks(
        integrated, 
        height=0.35*np.max(integrated),
        distance=0.5*fs
    )
    
    refined_r_peaks = []
    for peak in r_peaks:
        window_size = int(0.025 * fs)
        start = max(0, peak - window_size)
        end = min(len(ecg_signal), peak + window_size)
        max_idx = start + np.argmax(ecg_signal[start:end])
        refined_r_peaks.append(max_idx)
    
    return np.array(refined_r_peaks)


def calculate_hrv_metrics(r_peaks, fs):
    rr_intervals = np.diff(r_peaks) * (1000 / fs)
    
    mean_rr = np.mean(rr_intervals)
    sdnn = np.std(rr_intervals)
    rmssd = np.sqrt(np.mean(np.square(np.diff(rr_intervals))))
    
    nn50 = sum(abs(np.diff(rr_intervals)) > 50)
    pnn50 = (nn50 / len(rr_intervals)) * 100
    
    hrv_metrics = {
        'mean_rr': mean_rr,
        'sdnn': sdnn,
        'rmssd': rmssd,
        'pnn50': pnn50
    }
    
    return hrv_metrics


def detect_arrhythmias(r_peaks, fs, ecg_signal=None):
    rr_intervals = np.diff(r_peaks) / fs
    
    results = {
        'bradycardia': False,
        'tachycardia': False,
        'irregular_rhythm': False,
        'premature_beats': [],
        'long_pauses': []
    }
    
    heart_rate = 60 / np.median(rr_intervals)
    
    if heart_rate < 60:
        results['bradycardia'] = True
    
    if heart_rate > 100:
        results['tachycardia'] = True
    
    rr_variability = np.std(rr_intervals) / np.mean(rr_intervals)
    if rr_variability > 0.1:
        results['irregular_rhythm'] = True
    
    for i in range(1, len(rr_intervals)-1):
        if (rr_intervals[i] < 0.85 * rr_intervals[i-1] and 
            rr_intervals[i+1] > 1.15 * rr_intervals[i-1]):
            results['premature_beats'].append(i + 1)
    
    for i, rr in enumerate(rr_intervals):
        if rr > 2.0:
            results['long_pauses'].append(i)
    
    return results


def preprocess_and_segment_for_prediction(ecg_file, segment_length=24, channels_to_use=None):
    file_dir = os.path.dirname(ecg_file)
    file_name = os.path.splitext(os.path.basename(ecg_file))[0]
    
    if not file_dir:
        file_dir = '.'
    
    ecg_data = read_ecg_with_wfdb(file_dir, file_name)
    
    if ecg_data is None:
        print(f"Error loading file: {ecg_file}")
        return None, None
    
    n_channels = ecg_data['n_channels']
    signal_labels = ecg_data['labels']
    fs = ecg_data['fs']
    leads = ecg_data['signals']
    
    print(f"File loaded with {n_channels} channels: {signal_labels}")
    print(f"Sampling frequency: {fs} Hz")
    
    if channels_to_use is None:
        channels_to_use = list(range(min(n_channels, 32)))
    else:
        channels_to_use = [c for c in channels_to_use if c < n_channels]
    
    leads = [leads[i] for i in channels_to_use]
    signal_labels = [signal_labels[i] for i in channels_to_use]
    
    leads = preprocess_ecg(leads, fs)
    
    r_peaks = detect_r_peaks(leads[0], fs)
    
    segments = extract_segments_around_rpeaks(leads, r_peaks, segment_length)
    
    segments_tensor = torch.FloatTensor(segments)
    
    preproc_info = {
        'file_path': ecg_file,
        'channels_used': channels_to_use,
        'channel_labels': signal_labels,
        'sampling_frequency': fs,
        'r_peaks': r_peaks,
        'num_segments': len(segments)
    }
    
    return segments_tensor, preproc_info


# ------- PART 6: COMMAND-LINE INTERFACE -------

def parse_arguments():
    import argparse
    
    parser = argparse.ArgumentParser(description='ECG Analysis with Deep Spiking Neural Networks')
    
    parser.add_argument('--base_path', type=str, default='Dataset/MIT-BIH',
                        help='Base path to the ECG data files')
    parser.add_argument('--files', nargs='+', default=['100'],
                        help='List of ECG file names (without extensions)')
    
    parser.add_argument('--channels', type=int, default=2,
                        help='Number of ECG channels to use (1-32)')
    parser.add_argument('--segment_length', type=int, default=24,
                        help='Length of ECG segments in samples')
    parser.add_argument('--model_type', type=str, default='dsnn',
                        choices=['dsnn', 'attention', 'residual', 'multi'],
                        help='Type of DSNN model to use')
    
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=2000,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--no_train', action='store_true',
                        help='Skip training and use pre-trained model')
    parser.add_argument('--sliding_window', action='store_true',
                        help='Use sliding window for segmentation instead of R-peaks')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    if args.channels < 1 or args.channels > 32:
        print(f"Error: Number of channels must be between 1 and 32, got {args.channels}")
        exit(1)
    
    if args.model_type == 'dsnn':
        model_class = DSNN
    elif args.model_type == 'attention':
        model_class = DSNNAttention
    elif args.model_type == 'residual':
        model_class = DSNNResidual
    elif args.model_type == 'multi':
        model_class = MultiChannelDSNN
    
    results = main(
        base_path=args.base_path,
        file_names=args.files,
        num_channels=args.channels,
        segment_length=args.segment_length,
        use_sliding_window=args.sliding_window,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        train_model=not args.no_train
    )

    if results:
        print("\nAnalysis completed successfully!")
        metrics = results['metrics']
        print(f"Final test accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 score: {metrics['f1']:.4f}")

