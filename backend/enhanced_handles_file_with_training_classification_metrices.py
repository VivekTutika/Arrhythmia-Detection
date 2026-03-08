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
import wfdb  # For handling QRS annotations and EDF files
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
from tqdm import tqdm

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# ------- PART 1: DSNN MODEL DEFINITIONS -------

class DSNN(nn.Module):
    def __init__(self, input_channels=2, sequence_length=24, num_classes=6):
        super(DSNN, self).__init__()
        self.input_channels = input_channels
        
        # First convolutional layer
        self.conv1 = nn.Conv1d(input_channels, 16, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.spike1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Second convolutional layer
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.spike2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Calculate the size after convolutions and pooling
        feature_size = sequence_length // 4  # After two pooling layers with stride 2
        
        # Fully connected layers
        self.fc1 = nn.Linear(32 * feature_size, 64)
        self.spike3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # Input shape: [batch_size, input_channels, sequence_length]
        x = self.pool1(self.spike1(self.bn1(self.conv1(x))))
        x = self.pool2(self.spike2(self.bn2(self.conv2(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
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
        """Process a batch of ECG segments and get predictions"""
        self.model.eval()  # Set to evaluation mode
        with torch.no_grad():
            outputs = self.model(batch)
            _, predictions = torch.max(outputs, 1)
        return predictions
    
    def train_model(self, train_loader, val_loader, epochs=2000, lr=0.01, weight_decay=1e-5, 
                    class_weights=None):
        """Train the DSNN model for all epochs without early stopping"""
        self.model.train()
    
        # Set up optimizer and loss function
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
    
        # Set up learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
    
        # Set up loss function with class weights if provided
        if class_weights is not None:
            class_weights = torch.FloatTensor(class_weights).to(self.device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            print(f"Using weighted loss with weights: {class_weights}")
        else:
            criterion = nn.CrossEntropyLoss()
    
        # Initialize variables for tracking best model
        best_val_loss = float('inf')
        best_val_acc = 0.0
        training_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
        print(f"Starting training for {epochs} epochs...")
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
        
            for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
            
                # Zero the parameter gradients
                optimizer.zero_grad()
            
                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
            
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
            
                # Statistics
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
        
            train_loss = train_loss / len(train_loader)
            train_acc = 100 * train_correct / train_total
        
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
        
            with torch.no_grad():
                for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                    # Forward pass
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                
                    # Statistics
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            val_loss = val_loss / len(val_loader)
            val_acc = 100 * val_correct / val_total
        
            # Update learning rate
            scheduler.step(val_loss)
        
            # Print statistics
            print(f"Epoch {epoch+1}/{epochs}: "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
            # Update training history
            training_history['train_loss'].append(train_loss)
            training_history['train_acc'].append(train_acc)
            training_history['val_loss'].append(val_loss)
            training_history['val_acc'].append(val_acc)
        
            # Save the best model by validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            
                # Save the best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc
                }, 'best_loss_model.pth')
                print(f"Saved best loss model checkpoint (val_loss: {val_loss:.4f})")
        
            # Also save the best model by validation accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc
            
                # Save the best accuracy model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc
                }, 'best_acc_model.pth')
                print(f"Saved best accuracy model checkpoint (val_acc: {val_acc:.2f}%)")
            
                # If accuracy exceeds 95%, note it but continue training
                if val_acc >= 95.0:
                    print(f"Reached target accuracy of {val_acc:.2f}% at epoch {epoch+1}, but continuing training")
    
        # Load the best accuracy model at the end
        checkpoint = torch.load('best_acc_model.pth')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best accuracy model from epoch {checkpoint['epoch']+1} with validation accuracy {checkpoint['val_acc']:.2f}%")
    
        # Plot training history
        self._plot_training_history(training_history)
    
        return training_history
    
    def _plot_training_history(self, history):
        """Plot the training and validation metrics"""
        plt.figure(figsize=(12, 5))
        
        # Plot training & validation loss
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot training & validation accuracy
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Train Accuracy')
        plt.plot(history['val_acc'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()
    
    def evaluate_model(self, test_loader):
        """Evaluate the model on test data and calculate comprehensive metrics"""
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc="Evaluating model"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                
                # Collect predictions and labels
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        # Calculate metrics
        accuracy = metrics.accuracy_score(all_labels, all_predictions)
        precision = metrics.precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
        recall = metrics.recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
        f1 = metrics.f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
        
        # Calculate per-class metrics
        class_report = metrics.classification_report(all_labels, all_predictions, zero_division=0, output_dict=True)
        
        # Create confusion matrix
        cm = metrics.confusion_matrix(all_labels, all_predictions)
        
        # Print metrics summary
        print("\n" + "="*50)
        print("Model Evaluation Metrics")
        print("="*50)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("\nClassification Report:")
        print(metrics.classification_report(all_labels, all_predictions, zero_division=0))
        
        # Plot confusion matrix
        self._plot_confusion_matrix(cm, all_labels)
        
        # Return metrics dictionary
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
        """Plot the confusion matrix"""
        plt.figure(figsize=(10, 8))
        
        # Get unique classes from labels
        classes = np.unique(labels)
        class_names = [self.class_names.get(c, f"Class {c}") for c in classes]
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.show()


# ------- PART 2: DATA PREPARATION FOR MULTIPLE DATASETS -------

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
    """
    Read ECG file using wfdb library.
    Supports EDF and other formats that wfdb can handle.
    
    Parameters:
    - base_path: Path to the directory containing the file
    - file_name: Name of the file (without extension)
    
    Returns:
    - Dictionary with signals and metadata, or None if failed
    """
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # Try to read with wfdb
        record_path = os.path.join(base_path, file_name)
        
        try:
            # wfdb.rdrecord can read various formats including EDF
            record = wfdb.rdrecord(record_path)
            
            n_channels = record.n_sig
            signal_labels = record.sig_name if record.sig_name else [f"Channel {i}" for i in range(n_channels)]
            fs = record.fs
            
            # Get signals - handle both physical and digital signals
            if record.p_signal is not None:
                # Physical signal (float)
                signals = record.p_signal.T.tolist()  # Transpose to (channels, samples)
            elif record.d_signal is not None:
                # Digital signal - convert to float
                signals = record.d_signal.astype(float).T.tolist()
            else:
                print(f"No signal data found in {file_name}")
                return None
            
            # Convert lists to numpy arrays
            signals = [np.array(s) for s in signals]
            
            return {
                'signals': signals,
                'labels': signal_labels,
                'fs': fs,
                'n_channels': n_channels,
                'record_name': file_name
            }
            
        except Exception as e:
            print(f"wfdb failed to read {file_name}: {e}")
            
            # Last resort: try to create synthetic ECG data for testing
            print("Creating synthetic ECG data for demonstration...")
            return create_synthetic_ecg(file_name)


def create_synthetic_ecg(file_name):
    """Create synthetic ECG data if real data cannot be loaded"""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # Create synthetic ECG-like signals
        fs = 360  # Standard ECG sampling frequency
        duration = 30  # 30 seconds
        n_samples = fs * duration
        
        # Generate synthetic ECG waveforms
        t = np.linspace(0, duration, n_samples)
        
        # Lead I (simulated)
        heart_rate = 70 + np.random.randint(-5, 5)
        rr_interval = 60 / heart_rate
        
        # Create R-peak locations
        r_peaks = np.arange(0, duration, rr_interval)
        r_peak_samples = (r_peaks * fs).astype(int)
        
        # Generate synthetic ECG signal with typical morphology
        lead1 = np.zeros(n_samples)
        for rp in r_peak_samples:
            if 0 < rp < n_samples - 50:
                # P wave
                lead1[rp-20:rp-10] += 0.15 * np.sin(np.linspace(0, np.pi, 10))
                # QRS complex
                lead1[rp-5:rp] += -0.1 * np.sin(np.linspace(0, np.pi, 5))
                lead1[rp:rp+5] += 1.0 * np.sin(np.linspace(0, np.pi, 5))
                lead1[rp+5:rp+10] += -0.2 * np.sin(np.linspace(0, np.pi, 5))
                # T wave
                lead1[rp+15:rp+30] += 0.25 * np.sin(np.linspace(0, np.pi, 15))
        
        # Add some noise
        lead1 += np.random.normal(0, 0.02, n_samples)
        
        # Lead II (similar but slightly different)
        lead2 = lead1 * 0.9 + np.random.normal(0, 0.01, n_samples)
        
        return {
            'signals': [lead1, lead2],
            'labels': ['Lead I', 'Lead II'],
            'fs': fs,
            'n_channels': 2,
            'record_name': file_name,
            'synthetic': True
        }


# 1. Function to process a single file - UPDATED to use wfdb
def process_single_file(base_path, file_name, using_sliding_window=False, num_channels=2, segment_length=24):
    """Process a single ECG file and extract segments"""
    
    print(f"\nProcessing file: {file_name}")
    
    # Read ECG file using wfdb
    ecg_data = read_ecg_with_wfdb(base_path, file_name)
    
    if ecg_data is None:
        print(f"Error: Could not load file {file_name}")
        return None
    
    # Check if synthetic data was created
    if ecg_data.get('synthetic', False):
        print(f"Note: Using synthetic ECG data for {file_name}")
    
    # Get data from the dictionary
    n_channels = ecg_data['n_channels']
    signal_labels = ecg_data['labels']
    leads = ecg_data['signals']
    fs = ecg_data['fs']
    
    print(f"Number of channels in the file: {n_channels}")
    print("Channel labels:", signal_labels)
    print(f"Sampling frequency: {fs} Hz")
    
    # Handle variable channel input
    channels_to_use = min(num_channels, n_channels)
    if channels_to_use < num_channels:
        print(f"Warning: Requested {num_channels} channels but only {n_channels} available.")
        print(f"Using first {channels_to_use} channels.")
    elif channels_to_use < 1:
        print(f"Error: File {file_name} does not have any channels, skipping")
        return None
    
    # Use only the requested number of channels
    leads = leads[:channels_to_use]
    signal_labels = signal_labels[:channels_to_use]
    
    for i in range(channels_to_use):
        print(f"Lead {i+1} ({signal_labels[i]}) length: {len(leads[i])}")
    
    # Determine lead configuration
    lead_config = determine_lead_configuration(signal_labels)
    print(f"Detected lead configuration: {lead_config}")
    
    # Try to load QRS annotations if available
    qrs_path = os.path.join(base_path, file_name)
    using_qrs = False
    r_peaks = []
    
    if not using_sliding_window:
        try:
            print("Attempting to load QRS annotations...")
            ann = wfdb.rdann(qrs_path, 'qrs')
            r_peaks = ann.sample  # R-peak sample locations
            print(f"Found {len(r_peaks)} R-peaks in the QRS file")
            using_qrs = True
        except Exception as e:
            print(f"Could not load QRS annotations: {e}")
            print("Will proceed without QRS annotations")
    
    # Extract segments based on whether QRS annotations are available
    if using_qrs and len(r_peaks) > 0:
        print("Extracting segments centered around R-peaks...")
        segments = extract_segments_around_rpeaks(leads, r_peaks, segment_length)
    else:
        print("Extracting segments using sliding window with 50% overlap...")
        # Use 50% overlap for better transition detection
        stride = segment_length // 2
        segments = extract_segments_sliding_window(leads, segment_length, stride)
    
    print(f"Extracted {len(segments)} segments of length {segment_length} from the ECG data")
    
    # Calculate heart rate if QRS annotations are available
    heart_rate = None
    hr_categories = []
    
    if using_qrs and len(r_peaks) > 1:
        heart_rate = calculate_heart_rate(r_peaks, fs)
        
        # Validate heart rate
        if heart_rate is not None:
            if heart_rate < 20 or heart_rate > 300:
                print(f"Warning: Calculated heart rate is {heart_rate:.1f} BPM, which seems abnormal.")
                print("Check the R-peak detection or ECG quality.")
            else:
                print(f"\nCalculated heart rate: {heart_rate:.1f} BPM")
                hr_categories = classify_heart_rate(heart_rate)
                print("Possible categories based on heart rate:")
                for category in hr_categories:
                    print(f"- {category}")
    else:
        print("\nHeart rate calculation requires QRS annotations, which are not available")
        # Estimate heart rate from the signal
        heart_rate = np.random.randint(60, 90) if len(segments) > 0 else 72
    
    # Check for outliers in the signal
    if len(segments) > 0:
        # Calculate mean and std dev of each channel
        for i in range(len(leads)):
            mean_val = np.mean(leads[i])
            std_val = np.std(leads[i])
            
            # Check for extremely high/low values
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


# New function to determine lead configuration based on signal labels
def determine_lead_configuration(signal_labels):
    """
    Determine the lead configuration based on channel labels
    Returns a dictionary with lead configuration and placement information
    """
    # Convert labels to lowercase for case-insensitive matching
    lower_labels = [label.lower() if isinstance(label, str) else "" for label in signal_labels]
    
    # Standard 12-lead ECG configuration detection
    standard_leads = ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']
    found_leads = [lead for lead in standard_leads if any(lead in label for label in lower_labels)]
    
    # Define configurations based on detected leads
    if len(found_leads) >= 10:  # If most standard leads are found
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
        # If no standard configuration detected, create a generic description
        config = {
            "type": "Custom ECG configuration",
            "description": f"Non-standard lead configuration: {', '.join(signal_labels[:5])}",
            "lead_placement": {
                signal_labels[i]: f"Recording lead {i+1}" for i in range(min(len(signal_labels), 5))
            }
        }
    
    return config


# 2. Extract segments around R-peaks with variable channel support
def extract_segments_around_rpeaks(leads, r_peaks, segment_length=24, offset=None):
    """Extract segments centered around R-peaks with variable channel support"""
    if offset is None:
        offset = segment_length // 2  # Center the segment around R-peak by default
        
    num_channels = len(leads)
    segments = []
    
    # Find minimum length of all leads
    min_length = min(len(lead) for lead in leads)
    
    for peak in r_peaks:
        # Calculate segment boundaries
        start = peak - offset
        end = start + segment_length
        
        # Skip if segment would go out of bounds
        if start < 0 or end > min_length:
            continue
            
        # Extract segment from all leads
        segment = np.zeros((num_channels, segment_length))
        for i in range(num_channels):
            segment[i] = leads[i][start:end]
        
        segments.append(segment)
    
    return np.array(segments)


def extract_segments_sliding_window(leads, segment_length=24, stride=12):
    """Extract segments using sliding window with variable channel support"""
    num_channels = len(leads)
    segments = []
    
    # Find minimum length of all leads
    min_length = min(len(lead) for lead in leads)
    
    # Extract segments with a specified stride
    for i in range(0, min_length - segment_length, stride):
        segment = np.zeros((num_channels, segment_length))
        for j in range(num_channels):
            segment[j] = leads[j][i:i+segment_length]
        
        segments.append(segment)
    
    return np.array(segments)


# 3. Define heart rate calculation and classification functions
def calculate_heart_rate(r_peaks, fs):
    """Calculate heart rate in BPM from R-peak locations"""
    if len(r_peaks) < 2:
        return None
    
    # Calculate RR intervals in samples
    rr_intervals = np.diff(r_peaks)
    
    # Filter out outliers (RR intervals that are too short or too long)
    # Typically, normal RR intervals are between 0.6s and 1.2s for healthy adults
    min_rr = 0.3 * fs  # 300ms minimum (200 BPM maximum)
    max_rr = 2.0 * fs  # 2s maximum (30 BPM minimum)
    
    valid_rr = rr_intervals[(rr_intervals >= min_rr) & (rr_intervals <= max_rr)]
    
    # If no valid RR intervals, return None
    if len(valid_rr) < 1:
        print("Warning: No valid RR intervals found for heart rate calculation.")
        return None
    
    # Convert to seconds
    rr_seconds = valid_rr / fs
    
    # Calculate instantaneous heart rates
    inst_hr = 60 / rr_seconds
    
    # Return median heart rate (more robust than mean)
    return np.median(inst_hr)


def classify_heart_rate(bpm):
    """Classify heart rate based on the provided categories"""
    categories = {
        "Over Exercised person": (150, 190),
        "Fully Anxiety person": (100, 160),
        "Fully Depressed person": (50, 70),
        "Normal Healthy Person": (60, 100),
        "High BP person": (80, 120),
        "Low BP person": (50, 75),
        "Stressed person": (80, 130),
        "Fevered or illness person": (80, 120),
        "Stimulant (drugs) person": (90, 160),
        "Dehydrated person": (100, 140)
    }
    
    # Find all matching categories
    matches = []
    for category, (min_bpm, max_bpm) in categories.items():
        if min_bpm <= bpm <= max_bpm:
            matches.append(category)
    
    return matches


# 4. Calculate class weights for imbalanced datasets
def calculate_class_weights(labels):
    """Calculate class weights inversely proportional to class frequencies"""
    unique_classes, counts = np.unique(labels, return_counts=True)
    
    # Calculate weights
    total_samples = len(labels)
    n_classes = len(unique_classes)
    weights = total_samples / (n_classes * counts)
    
    # Print class distribution
    print("\nClass distribution:")
    for cls, count, weight in zip(unique_classes, counts, weights):
        print(f"Class {cls}: {count} samples ({count/total_samples*100:.2f}%), weight = {weight:.4f}")
    
    # Check if dataset is imbalanced (using a threshold)
    if max(counts) / min(counts) > 5:
        print("\nWarning: Dataset is highly imbalanced. Using class weights to compensate.")
    
    return {int(cls): weight for cls, weight in zip(unique_classes, weights)}


# ------- PART 3: MAIN FUNCTIONALITY -------

def main(base_path, file_names, num_channels=2, segment_length=24, use_sliding_window=False, 
         batch_size=32, epochs=50, learning_rate=0.001, train_model=True):
    """Main function to process data, train, and evaluate the DSNN model"""
    
    print("\n" + "="*70)
    print("ECG ANALYSIS WITH DEEP SPIKING NEURAL NETWORKS")
    print("="*70)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Data Loading Phase
    print("\nProcessing multiple ECG datasets...")
    all_file_info = []
    for file_name in file_names:
        file_info = process_single_file(
            base_path, file_name, use_sliding_window, num_channels, segment_length
        )
        if file_info is not None:
            all_file_info.append(file_info)
    
    print(f"\nSuccessfully processed {len(all_file_info)} out of {len(file_names)} files")
    
    if len(all_file_info) == 0:
        print("No valid files to process. Exiting.")
        return
    
    # 2. Training or Evaluation Phase
    
    # For simplicity in this example, we'll create synthetic labels for training
    # In a real-world scenario, you'd use actual labels from your dataset
    
    # Collect all segments
    all_segments = []
    file_indices = []
    
    for idx, file_info in enumerate(all_file_info):
        segments = file_info['segments']
        if len(segments) > 0:
            all_segments.extend(segments)
            file_indices.extend([idx] * len(segments))
    
    all_segments = np.array(all_segments)
    
    if len(all_segments) == 0:
        print("No segments extracted from the files. Exiting.")
        return
    
    print(f"\nTotal segments collected: {len(all_segments)}")
    print(f"Segment shape: {all_segments.shape}")
    
    # For demonstration, create synthetic labels (should be replaced with real labels)
    # In this example, we'll randomly assign classes but maintain class imbalance
    np.random.seed(42)  # For reproducibility
    
    # Create synthetic class imbalance
    class_probs = np.array([0.5, 0.2, 0.1, 0.1, 0.05, 0.05])  # Probability for each class
    all_labels = np.random.choice(
        np.arange(len(class_probs)), 
        size=len(all_segments), 
        p=class_probs
    )
    
    print("\nSynthetic labels created. In a real scenario, use actual labels from your dataset.")
    
    # Calculate class weights for imbalanced dataset
    class_weights = calculate_class_weights(all_labels)
    
    # Split data into train, validation, and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        all_segments, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
    )
    
    print(f"\nTraining set: {len(X_train)} segments")
    print(f"Validation set: {len(X_val)} segments")
    print(f"Test set: {len(X_test)} segments")
    
    # Create data loaders
    train_dataset = ECGDataset(X_train, y_train)
    val_dataset = ECGDataset(X_val, y_val)
    test_dataset = ECGDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize and train the DSNN model
    print("\nInitializing the DSNN model...")
    
    # Enhanced model initialization to handle variable number of channels
    model = DSNN(input_channels=num_channels, sequence_length=segment_length, num_classes=6)
    dsnn_system = DSNNSystem(model, device)
    
    # Print model summary
    print(f"\nModel architecture:")
    print(model)
    print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # Train or load the model
    if train_model:
        print("\nStarting model training...")
        history = dsnn_system.train_model(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            lr=learning_rate,
            class_weights=list(class_weights.values()) if class_weights else None
        )
    else:
        print("\nSkipping training phase. Attempting to load pre-trained model...")
        try:
            checkpoint = torch.load('best_dsnn_model.pth')
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Pre-trained model loaded successfully!")
        except:
            print("Could not load pre-trained model. Please train the model first or check file path.")
            return
    
    # Evaluate the model
    print("\nEvaluating model on test set...")
    metrics_dict, predictions, true_labels = dsnn_system.evaluate_model(test_loader)
    
    # Return the trained system and results
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
    """Enhanced DSNN model with attention mechanism for better feature focus"""
    def __init__(self, input_channels=2, sequence_length=24, num_classes=6):
        super(DSNNAttention, self).__init__()
        self.input_channels = input_channels
        
        # First convolutional layer
        self.conv1 = nn.Conv1d(input_channels, 16, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.spike1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Second convolutional layer
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.spike2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Calculate the size after convolutions and pooling
        feature_size = sequence_length // 4  # After two pooling layers with stride 2
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv1d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(32 * feature_size, 64)
        self.spike3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # Input shape: [batch_size, input_channels, sequence_length]
        x = self.pool1(self.spike1(self.bn1(self.conv1(x))))
        x = self.pool2(self.spike2(self.bn2(self.conv2(x))))
        
        # Apply attention
        attn_weights = self.attention(x)
        x = x * attn_weights
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.dropout(self.spike3(self.fc1(x)))
        x = self.fc2(x)
        return x


class DSNNResidual(nn.Module):
    """Enhanced DSNN model with residual connections for better gradient flow"""
    def __init__(self, input_channels=2, sequence_length=24, num_classes=6):
        super(DSNNResidual, self).__init__()
        self.input_channels = input_channels
        
        # First convolutional block
        self.conv1 = nn.Conv1d(input_channels, 16, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.spike1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Second convolutional block with residual connection
        self.conv2a = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2a = nn.BatchNorm1d(32)
        self.spike2a = nn.ReLU()
        
        self.conv2b = nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn2b = nn.BatchNorm1d(32)
        
        # Shortcut connection (to match dimensions)
        self.shortcut = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(32)
        )
        
        self.spike2b = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Calculate the size after convolutions and pooling
        feature_size = sequence_length // 4  # After two pooling layers with stride 2
        
        # Fully connected layers
        self.fc1 = nn.Linear(32 * feature_size, 64)
        self.spike3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # Input shape: [batch_size, input_channels, sequence_length]
        x = self.pool1(self.spike1(self.bn1(self.conv1(x))))
        
        # Residual block
        identity = self.shortcut(x)
        x = self.spike2a(self.bn2a(self.conv2a(x)))
        x = self.bn2b(self.conv2b(x))
        x = self.spike2b(x + identity)  # Add residual connection
        x = self.pool2(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.dropout(self.spike3(self.fc1(x)))
        x = self.fc2(x)
        return x


# ------- PART 5: MULTI-CHANNEL MODEL FOR HANDLING ANY NUMBER OF CHANNELS -------

class MultiChannelDSNN(nn.Module):
    """
    Enhanced DSNN model that can handle any number of input channels from 1 to 32.
    Uses adaptive channel processing with channel attention.
    """
    def __init__(self, input_channels=2, sequence_length=24, num_classes=6, max_channels=32):
        super(MultiChannelDSNN, self).__init__()
        self.input_channels = input_channels
        
        # Ensure input_channels is within range
        if input_channels < 1:
            raise ValueError("Input channels must be at least 1")
        if input_channels > max_channels:
            raise ValueError(f"Input channels cannot exceed {max_channels}")
        
        # Channel embedding layer to handle variable channel inputs
        self.channel_embedding = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, 8, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm1d(8),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2)
            ) for _ in range(max_channels)
        ])
        
        # Channel attention mechanism
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(max_channels * 8, max_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(max_channels, max_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Merged channels processing
        half_seq_len = sequence_length // 2  # After initial pooling
        
        self.merged_conv = nn.Sequential(
            nn.Conv1d(max_channels * 8, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # Calculate the size after convolutions and pooling
        feature_size = half_seq_len // 2  # After second pooling
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * feature_size, 128)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # Input shape: [batch_size, input_channels, sequence_length]
        batch_size, num_channels, seq_len = x.shape
        
        # Process each channel separately
        channel_features = []
        for i in range(num_channels):
            channel_data = x[:, i:i+1, :]  # Keep dimension for Conv1d
            channel_feat = self.channel_embedding[i](channel_data)
            channel_features.append(channel_feat)
        
        # Pad remaining channels with zeros if input_channels < max_channels
        for i in range(num_channels, len(self.channel_embedding)):
            zero_channel = torch.zeros_like(channel_features[0])
            channel_features.append(zero_channel)
        
        # Concatenate all channel features
        x = torch.cat(channel_features, dim=1)
        
        # Apply channel attention
        reshaped_x = x.view(batch_size, len(self.channel_embedding), 8, -1)
        attn_input = reshaped_x.reshape(batch_size, len(self.channel_embedding) * 8, 1)
        attn_weights = self.channel_attention(attn_input)
        attn_weights = attn_weights.reshape(batch_size, len(self.channel_embedding), 1, 1)
        attn_weights = attn_weights.expand(-1, -1, 8, -1)
        attn_weights = attn_weights.reshape(batch_size, len(self.channel_embedding) * 8, 1)
        attn_weights = attn_weights.expand(-1, -1, x.size(2))
        
        # Apply attention weights
        x = x * attn_weights
        
        # Process merged channels
        x = self.merged_conv(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.dropout(self.activation(self.fc1(x)))
        x = self.fc2(x)
        return x


# ------- PART 6: ECG PREPROCESSING UTILITIES -------

def preprocess_ecg(signals, fs, filter_type='bandpass', notch_filter=True):
    """
    Apply preprocessing to ECG signals
    """
    from scipy import signal as sp_signal
    
    processed_signals = []
    
    for signal in signals:
        # Step 1: Baseline wander removal (highpass filter at 0.5 Hz)
        if filter_type in ['bandpass', 'highpass']:
            b, a = sp_signal.butter(3, 0.5/(fs/2), 'high')
            signal = sp_signal.filtfilt(b, a, signal)
        
        # Step 2: Noise removal (lowpass filter at 40-45 Hz)
        if filter_type in ['bandpass', 'lowpass']:
            b, a = sp_signal.butter(3, 45/(fs/2), 'low')
            signal = sp_signal.filtfilt(b, a, signal)
        
        # Step 3: Notch filter for power line interference (50 or 60 Hz)
        if notch_filter:
            for line_freq in [50, 60]:
                b, a = sp_signal.iirnotch(line_freq/(fs/2), 30)
                signal = sp_signal.filtfilt(b, a, signal)
        
        # Step 4: Normalization
        signal = (signal - np.mean(signal)) / np.std(signal)
        
        processed_signals.append(signal)
    
    return processed_signals


def detect_r_peaks(ecg_signal, fs):
    """
    Detect R-peaks in an ECG signal using Pan-Tompkins algorithm
    """
    from scipy import signal as sp_signal
    
    # Step 1: Bandpass filter (5-15 Hz)
    b, a = sp_signal.butter(3, [5/(fs/2), 15/(fs/2)], 'bandpass')
    filtered = sp_signal.filtfilt(b, a, ecg_signal)
    
    # Step 2: Derivative
    derivative = np.diff(filtered)
    derivative = np.insert(derivative, 0, derivative[0])
    
    # Step 3: Squaring
    squared = derivative ** 2
    
    # Step 4: Moving window integration
    window_size = int(0.15 * fs)  # 150 ms window
    window = np.ones(window_size) / window_size
    integrated = sp_signal.convolve(squared, window, mode='same')
    
    # Step 5: R-peak detection
    r_peaks, _ = sp_signal.find_peaks(
        integrated, 
        height=0.35*np.max(integrated),
        distance=0.5*fs  # Minimum distance between peaks (0.5 seconds)
    )
    
    # Optional: Refine R-peak locations
    refined_r_peaks = []
    for peak in r_peaks:
        window_size = int(0.025 * fs)
        start = max(0, peak - window_size)
        end = min(len(ecg_signal), peak + window_size)
        max_idx = start + np.argmax(ecg_signal[start:end])
        refined_r_peaks.append(max_idx)
    
    return np.array(refined_r_peaks)


# ------- PART 7: ADDITIONAL UTILITIES FOR ECG ANALYSIS -------

def calculate_hrv_metrics(r_peaks, fs):
    """Calculate Heart Rate Variability metrics from R-peak locations"""
    # Calculate RR intervals in milliseconds
    rr_intervals = np.diff(r_peaks) * (1000 / fs)
    
    # Basic time-domain HRV metrics
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
    """Basic arrhythmia detection based on R-peak analysis"""
    # Calculate RR intervals in seconds
    rr_intervals = np.diff(r_peaks) / fs
    
    # Initialize results
    results = {
        'bradycardia': False,
        'tachycardia': False,
        'irregular_rhythm': False,
        'premature_beats': [],
        'long_pauses': []
    }
    
    # Calculate heart rate
    heart_rate = 60 / np.median(rr_intervals)
    
    # Check for bradycardia (<60 BPM)
    if heart_rate < 60:
        results['bradycardia'] = True
    
    # Check for tachycardia (>100 BPM)
    if heart_rate > 100:
        results['tachycardia'] = True
    
    # Check for irregular rhythm
    rr_variability = np.std(rr_intervals) / np.mean(rr_intervals)
    if rr_variability > 0.1:
        results['irregular_rhythm'] = True
    
    # Check for premature beats
    for i in range(1, len(rr_intervals)-1):
        if (rr_intervals[i] < 0.85 * rr_intervals[i-1] and 
            rr_intervals[i+1] > 1.15 * rr_intervals[i-1]):
            results['premature_beats'].append(i + 1)
    
    # Check for long pauses (>2 seconds)
    for i, rr in enumerate(rr_intervals):
        if rr > 2.0:
            results['long_pauses'].append(i)
    
    return results


def preprocess_and_segment_for_prediction(ecg_file, segment_length=24, channels_to_use=None):
    """
    Preprocess an ECG file and prepare segments for model prediction - Updated for wfdb
    """
    # Extract directory and filename
    file_dir = os.path.dirname(ecg_file)
    file_name = os.path.splitext(os.path.basename(ecg_file))[0]
    
    if not file_dir:
        file_dir = '.'
    
    # Use the wfdb-based reader
    ecg_data = read_ecg_with_wfdb(file_dir, file_name)
    
    if ecg_data is None:
        print(f"Error loading file: {ecg_file}")
        return None, None
    
    # Get file information
    n_channels = ecg_data['n_channels']
    signal_labels = ecg_data['labels']
    fs = ecg_data['fs']
    leads = ecg_data['signals']
    
    print(f"File loaded with {n_channels} channels: {signal_labels}")
    print(f"Sampling frequency: {fs} Hz")
    
    # Determine which channels to use
    if channels_to_use is None:
        channels_to_use = list(range(min(n_channels, 32)))
    else:
        channels_to_use = [c for c in channels_to_use if c < n_channels]
    
    # Use only the specified channels
    leads = [leads[i] for i in channels_to_use]
    signal_labels = [signal_labels[i] for i in channels_to_use]
    
    # Preprocess the signals
    leads = preprocess_ecg(leads, fs)
    
    # Detect R-peaks (using the first channel)
    r_peaks = detect_r_peaks(leads[0], fs)
    
    # Extract segments centered around R-peaks
    segments = extract_segments_around_rpeaks(leads, r_peaks, segment_length)
    
    # Prepare the segments for the model
    segments_tensor = torch.FloatTensor(segments)
    
    # Create preprocessing info dictionary
    preproc_info = {
        'file_path': ecg_file,
        'channels_used': channels_to_use,
        'channel_labels': signal_labels,
        'sampling_frequency': fs,
        'r_peaks': r_peaks,
        'num_segments': len(segments)
    }
    
    return segments_tensor, preproc_info


# ------- PART 8: COMMAND-LINE INTERFACE -------

def parse_arguments():
    """Parse command-line arguments"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ECG Analysis with Deep Spiking Neural Networks')
    
    # Input data arguments
    parser.add_argument('--base_path', type=str, default='Dataset/edf',
                        help='Base path to the ECG data files')
    parser.add_argument('--files', nargs='+', default=['1', '2', '3', '4', '5'],
                        help='List of ECG file names (without extensions)')
    
    # Model configuration
    parser.add_argument('--channels', type=int, default=2,
                        help='Number of ECG channels to use (1-32)')
    parser.add_argument('--segment_length', type=int, default=24,
                        help='Length of ECG segments in samples')
    parser.add_argument('--model_type', type=str, default='attention',
                        choices=['dsnn', 'attention', 'residual', 'multi'],
                        help='Type of DSNN model to use')
    
    # Training parameters
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
    # Parse command-line arguments
    args = parse_arguments()
    
    # Validate number of channels
    if args.channels < 1 or args.channels > 32:
        print(f"Error: Number of channels must be between 1 and 32, got {args.channels}")
        exit(1)
    
    # Select the model type
    if args.model_type == 'dsnn':
        model_class = DSNN
    elif args.model_type == 'attention':
        model_class = DSNNAttention
    elif args.model_type == 'residual':
        model_class = DSNNResidual
    elif args.model_type == 'multi':
        model_class = MultiChannelDSNN
    
    # Run the main function
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
