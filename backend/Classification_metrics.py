import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import torch
from collections import defaultdict

class ClassificationMetrics:
    """
    A comprehensive class for calculating and visualizing classification metrics
    for multi-class classification problems, with special handling for binary tasks.
    """
    
    def __init__(self, num_classes=6):
        """
        Initialize the metrics calculator with the number of classes.
        
        Args:
            num_classes (int): Number of classes in the classification problem
        """
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        """Reset all metrics for a new evaluation run."""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self.class_names = [f"Class {i}" for i in range(self.num_classes)]
        self.pred_probas = []
        self.true_labels = []
    
    def update(self, y_true, y_pred, pred_proba=None):
        """
        Update metrics with a new batch of predictions.
        
        Args:
            y_true: Ground truth labels (numpy array or torch tensor)
            y_pred: Predicted labels (numpy array or torch tensor)
            pred_proba: Prediction probabilities (optional, for ROC-AUC calculation)
        """
        # Convert tensors to numpy if needed
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()
        if isinstance(pred_proba, torch.Tensor) and pred_proba is not None:
            pred_proba = pred_proba.cpu().numpy()
            
        # Update confusion matrix
        for i in range(len(y_true)):
            self.confusion_matrix[y_true[i], y_pred[i]] += 1
        
        # Store probabilities for ROC-AUC calculation
        if pred_proba is not None:
            self.pred_probas.extend(pred_proba)
            self.true_labels.extend(y_true)
    
    def set_class_names(self, class_names):
        """
        Set custom class names for better visualization.
        
        Args:
            class_names (list): List of class names
        """
        if len(class_names) != self.num_classes:
            raise ValueError(f"Expected {self.num_classes} class names, got {len(class_names)}")
        self.class_names = class_names
    
    def get_metrics(self):
        """
        Calculate all classification metrics.
        
        Returns:
            dict: Dictionary containing all calculated metrics
        """
        metrics = {}
        
        # Per-class metrics
        per_class_metrics = defaultdict(dict)
        
        # Calculate metrics for each class
        for i in range(self.num_classes):
            # True positives, false positives, false negatives, true negatives
            tp = self.confusion_matrix[i, i]
            fp = np.sum(self.confusion_matrix[:, i]) - tp
            fn = np.sum(self.confusion_matrix[i, :]) - tp
            tn = np.sum(self.confusion_matrix) - tp - fp - fn
            
            # Precision, recall, specificity, F1 score
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Store in per-class metrics
            per_class_metrics[self.class_names[i]] = {
                'precision': precision,
                'recall': recall,
                'specificity': specificity,
                'f1': f1,
                'support': tp + fn
            }
        
        # Overall metrics (macro and weighted)
        precision_list = [m['precision'] for m in per_class_metrics.values()]
        recall_list = [m['recall'] for m in per_class_metrics.values()]
        f1_list = [m['f1'] for m in per_class_metrics.values()]
        specificity_list = [m['specificity'] for m in per_class_metrics.values()]
        support_list = [m['support'] for m in per_class_metrics.values()]
        
        # Macro average (unweighted mean)
        metrics['macro_precision'] = np.mean(precision_list)
        metrics['macro_recall'] = np.mean(recall_list)
        metrics['macro_f1'] = np.mean(f1_list)
        metrics['macro_specificity'] = np.mean(specificity_list)
        
        # Weighted average (weighted by support)
        total_support = sum(support_list)
        if total_support > 0:
            weights = [s/total_support for s in support_list]
            metrics['weighted_precision'] = np.sum(np.multiply(precision_list, weights))
            metrics['weighted_recall'] = np.sum(np.multiply(recall_list, weights))
            metrics['weighted_f1'] = np.sum(np.multiply(f1_list, weights))
            metrics['weighted_specificity'] = np.sum(np.multiply(specificity_list, weights))
        
        # Overall accuracy
        metrics['accuracy'] = np.trace(self.confusion_matrix) / np.sum(self.confusion_matrix)
        
        # Per-class metrics
        metrics['per_class'] = dict(per_class_metrics)
        
        # Calculate ROC-AUC if probabilities are available
        if self.pred_probas and self.true_labels:
            metrics['roc_auc'] = self.calculate_roc_auc()
        
        return metrics
    
    def calculate_roc_auc(self):
        """
        Calculate the ROC-AUC score for each class (one-vs-rest).
        
        Returns:
            dict: ROC-AUC scores for each class and the macro average
        """
        true_labels = np.array(self.true_labels)
        pred_probas = np.array(self.pred_probas)
        
        # One-vs-Rest ROC-AUC for each class
        roc_auc_dict = {}
        
        for i in range(self.num_classes):
            # One-hot encode for current class
            y_true_binary = (true_labels == i).astype(int)
            
            # Get probabilities for current class
            y_score = pred_probas[:, i]
            
            # Calculate ROC curve and AUC
            try:
                fpr, tpr, _ = roc_curve(y_true_binary, y_score)
                roc_auc = auc(fpr, tpr)
                roc_auc_dict[self.class_names[i]] = roc_auc
            except Exception as e:
                print(f"Error calculating ROC-AUC for {self.class_names[i]}: {e}")
                roc_auc_dict[self.class_names[i]] = float('nan')
        
        # Calculate macro average ROC-AUC
        valid_aucs = [auc_val for auc_val in roc_auc_dict.values() if not np.isnan(auc_val)]
        if valid_aucs:
            roc_auc_dict['macro_average'] = np.mean(valid_aucs)
        else:
            roc_auc_dict['macro_average'] = float('nan')
            
        return roc_auc_dict
    
    def plot_confusion_matrix(self, normalize=True, figsize=(10, 8)):
        """
        Plot the confusion matrix.
        
        Args:
            normalize (bool): Whether to normalize the confusion matrix
            figsize (tuple): Figure size
            
        Returns:
            matplotlib.figure.Figure: The confusion matrix figure
        """
        cm = self.confusion_matrix.copy()
        
        if normalize:
            cm_sum = cm.sum(axis=1)[:, np.newaxis]
            cm = np.divide(cm, cm_sum, out=np.zeros_like(cm, dtype=float), where=cm_sum!=0) * 100
            
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(cm, cmap='Blues')
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        if normalize:
            cbar.set_label('Percentage (%)')
        else:
            cbar.set_label('Count')
            
        # Set labels
        ax.set_xticks(np.arange(self.num_classes))
        ax.set_yticks(np.arange(self.num_classes))
        ax.set_xticklabels(self.class_names)
        ax.set_yticklabels(self.class_names)
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Loop over data dimensions and create text annotations.
        thresh = cm.max() / 2.
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                text = f"{cm[i, j]:.1f}" if normalize else f"{cm[i, j]}"
                ax.text(j, i, text,
                         ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black")
                
        ax.set_ylabel('True label')
        ax.set_xlabel('Predicted label')
        ax.set_title('Confusion Matrix')
        fig.tight_layout()
        
        return fig
    
    def plot_roc_curves(self, figsize=(10, 8)):
        """
        Plot ROC curves for each class.
        
        Args:
            figsize (tuple): Figure size
            
        Returns:
            matplotlib.figure.Figure: The ROC curves figure
        """
        if not self.pred_probas or not self.true_labels:
            raise ValueError("No prediction probabilities available for ROC curve plotting")
            
        true_labels = np.array(self.true_labels)
        pred_probas = np.array(self.pred_probas)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        for i in range(self.num_classes):
            # One-hot encode for current class
            y_true_binary = (true_labels == i).astype(int)
            
            # Get probabilities for current class
            y_score = pred_probas[:, i]
            
            # Calculate ROC curve and AUC
            try:
                fpr, tpr, _ = roc_curve(y_true_binary, y_score)
                roc_auc = auc(fpr, tpr)
                
                # Plot ROC curve
                ax.plot(fpr, tpr, lw=2,
                        label=f'{self.class_names[i]} (AUC = {roc_auc:.2f})')
            except Exception as e:
                print(f"Error plotting ROC curve for {self.class_names[i]}: {e}")
        
        # Plot random guess line
        ax.plot([0, 1], [0, 1], 'k--', lw=2)
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curves')
        ax.legend(loc="lower right")
        
        return fig
    
    def print_report(self):
        """
        Print a classification report similar to sklearn's classification_report.
        """
        metrics = self.get_metrics()
        
        # Print header
        print("Classification Report:")
        print("-" * 80)
        print(f"{'Class':<20} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Specificity':<12} {'Support':<8}")
        print("-" * 80)
        
        # Print per-class metrics
        for class_name, class_metrics in metrics['per_class'].items():
            print(f"{class_name:<20} "
                  f"{class_metrics['precision']:.4f}     "
                  f"{class_metrics['recall']:.4f}     "
                  f"{class_metrics['f1']:.4f}      "
                  f"{class_metrics['specificity']:.4f}       "
                  f"{int(class_metrics['support']):<8}")
            
        # Print average metrics
        print("-" * 80)
        print(f"{'Macro Avg':<20} "
              f"{metrics['macro_precision']:.4f}     "
              f"{metrics['macro_recall']:.4f}     "
              f"{metrics['macro_f1']:.4f}      "
              f"{metrics['macro_specificity']:.4f}       "
              f"{np.sum(self.confusion_matrix):<8}")
        
        print(f"{'Weighted Avg':<20} "
              f"{metrics.get('weighted_precision', 0):.4f}     "
              f"{metrics.get('weighted_recall', 0):.4f}     "
              f"{metrics.get('weighted_f1', 0):.4f}      "
              f"{metrics.get('weighted_specificity', 0):.4f}       "
              f"{np.sum(self.confusion_matrix):<8}")
        
        print("-" * 80)
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        
        # Print ROC-AUC if available
        if 'roc_auc' in metrics:
            print("-" * 80)
            print("ROC-AUC Scores:")
            for class_name, auc_score in metrics['roc_auc'].items():
                if class_name != 'macro_average':
                    print(f"{class_name:<20} {auc_score:.4f}")
            print("-" * 80)
            print(f"{'Macro ROC-AUC':<20} {metrics['roc_auc'].get('macro_average', 0):.4f}")


def evaluate_dsnn_model(model, data_loader, device, num_classes=6, class_names=None):
    """
    Evaluate a DSNN model on a test dataset and return comprehensive metrics.
    
    Args:
        model: The DSNN model to evaluate
        data_loader: DataLoader containing the test dataset
        device: Device to run evaluation on (cpu or cuda)
        num_classes: Number of classes in the classification problem
        class_names: Optional list of class names for better visualization
        
    Returns:
        metrics: Dictionary containing all calculated metrics
        cm_fig: Confusion matrix figure
        roc_fig: ROC curves figure (if probabilities are available)
    """
    # Initialize metrics calculator
    metrics_calc = ClassificationMetrics(num_classes=num_classes)
    
    # Set class names if provided
    if class_names is not None:
        metrics_calc.set_class_names(class_names)
    
    # Set model to evaluation mode
    model.eval()
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            output = model(data)
            
            # Get predictions
            _, predicted = torch.max(output, 1)
            
            # Update metrics
            metrics_calc.update(target.cpu().numpy(), predicted.cpu().numpy(), output.cpu().numpy())
    
    # Calculate and print report
    metrics_calc.print_report()
    
    # Generate figures
    cm_fig = metrics_calc.plot_confusion_matrix()
    
    # Try to generate ROC curves if probabilities are available
    try:
        roc_fig = metrics_calc.plot_roc_curves()
    except Exception:
        roc_fig = None
    
    # Get metrics dictionary
    metrics = metrics_calc.get_metrics()
    
    return metrics, cm_fig, roc_fig


# Example usage with your DSNN model
def evaluate_example():
    """
    Example showing how to use the evaluation code with your DSNN model.
    """
    # Assuming you have:
    # - model: Your trained DSNN model
    # - test_loader: DataLoader with your test dataset
    # - device: 'cuda' or 'cpu'
    
    # Define class names for ECG classification
    class_names = [
        "Normal Sinus Rhythm",
        "Atrial Fibrillation",
        "Ventricular Arrhythmia",
        "Conduction Block",
        "ST Segment Abnormality",
        "Other Abnormality"
    ]
    
    # Evaluate model
    metrics, cm_fig, roc_fig = evaluate_dsnn_model(
        model=model,  # Your DSNN model
        data_loader=test_loader,  # Your test data loader
        device=device,  # 'cuda' or 'cpu'
        num_classes=6,
        class_names=class_names
    )
    
    # Save figures
    cm_fig.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    if roc_fig:
        roc_fig.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
    
    # Print overall metrics
    print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1 Score: {metrics['macro_f1']:.4f}")
    
    # You can add the metrics to your PDF report
    report_metrics = {
        'accuracy': metrics['accuracy'],
        'macro_f1': metrics['macro_f1'],
        'macro_precision': metrics['macro_precision'],
        'macro_recall': metrics['macro_recall'],
        'roc_auc': metrics.get('roc_auc', {}).get('macro_average', 0)
    }
    
    # Return metrics for later use
    return metrics, report_metrics
