#visulaization part
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

def visualize_multichannel_attention(model, data_loader, num_examples=2, device='cpu'):
    """
    Visualize channel attention weights from the MultiChannelDSNN model
    
    Parameters:
    -----------
    model : MultiChannelDSNN
        The trained multichannel model
    data_loader : DataLoader
        DataLoader containing ECG examples
    num_examples : int
        Number of examples to visualize
    device : str
        Device to run the model on ('cpu' or 'cuda')
    """
    # Set model to evaluation mode
    model.eval()
    
    # Get a batch of data
    iterator = iter(data_loader)
    batch = next(iterator)
    inputs, labels = batch
    inputs = inputs.to(device)
    
    # Limit to the number of examples we want
    inputs = inputs[:num_examples]
    labels = labels[:num_examples]
    
    # Forward pass to get channel attention weights
    with torch.no_grad():
        batch_size, num_channels, seq_len = inputs.shape
        
        # Process each channel separately to get channel features
        channel_features = []
        for i in range(num_channels):
            channel_data = inputs[:, i:i+1, :]
            channel_feat = model.channel_embedding[i](channel_data)
            channel_features.append(channel_feat)
        
        # Pad remaining channels with zeros if input_channels < max_channels
        for i in range(num_channels, len(model.channel_embedding)):
            zero_channel = torch.zeros_like(channel_features[0])
            channel_features.append(zero_channel)
        
        # Concatenate all channel features
        x = torch.cat(channel_features, dim=1)
        
        # Apply channel attention
        reshaped_x = x.view(batch_size, len(model.channel_embedding), 8, -1)
        attn_input = reshaped_x.reshape(batch_size, len(model.channel_embedding) * 8, 1)
        attn_weights = model.channel_attention(attn_input)
        
        # Extract attention weights for each channel
        channel_attn = attn_weights.reshape(batch_size, len(model.channel_embedding), 1)
        channel_attn = channel_attn.squeeze().cpu().numpy()
    
    # Convert inputs to numpy for plotting
    inputs = inputs.cpu().numpy()
    
    # Create a figure
    fig = plt.figure(figsize=(15, 6*num_examples))
    gs = GridSpec(num_examples, 2, width_ratios=[3, 1])
    
    for i in range(num_examples):
        # 1. Plot original ECG signals from all channels
        ax1 = fig.add_subplot(gs[i, 0])
        
        for ch in range(num_channels):
            ax1.plot(inputs[i, ch, :], label=f'Channel {ch+1}')
        
        ax1.set_title(f'Example {i+1}: Multi-channel ECG (Class {labels[i].item()})')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Amplitude')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Plot channel attention weights
        ax2 = fig.add_subplot(gs[i, 1])
        
        # Only plot attention for active channels
        attention_data = channel_attn[i, :num_channels]
        channel_names = [f'Channel {j+1}' for j in range(num_channels)]
        
        # Create horizontal bar plot
        bars = ax2.barh(channel_names, attention_data, color=plt.cm.viridis(attention_data/np.max(attention_data)))
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax2.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                     f'{width:.3f}', va='center')
        
        ax2.set_title('Channel Attention Weights')
        ax2.set_xlabel('Attention Weight')
        ax2.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig

def visualize_preprocessed_ecg(raw_signal, processed_signal, fs, title="ECG Preprocessing Results"):
    """
    Visualize raw vs preprocessed ECG signal
    
    Parameters:
    -----------
    raw_signal : array-like
        Original ECG signal
    processed_signal : array-like
        Preprocessed ECG signal
    fs : float
        Sampling frequency in Hz
    title : str
        Plot title
    """
    # Create time axis in seconds
    time = np.arange(len(raw_signal)) / fs
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot raw signal
    axes[0].plot(time, raw_signal, color='#3498db')
    axes[0].set_title("Raw ECG Signal")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, linestyle='--', alpha=0.7)
    
    # Plot processed signal
    axes[1].plot(time, processed_signal, color='#2ecc71')
    axes[1].set_title("Preprocessed ECG Signal")
    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(True, linestyle='--', alpha=0.7)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    return fig

def visualize_r_peaks(ecg_signal, r_peaks, fs, title="R-Peak Detection"):
    """
    Visualize detected R-peaks on an ECG signal
    
    Parameters:
    -----------
    ecg_signal : array-like
        ECG signal
    r_peaks : array-like
        Indices of detected R-peaks
    fs : float
        Sampling frequency in Hz
    title : str
        Plot title
    """
    # Create time axis in seconds
    time = np.arange(len(ecg_signal)) / fs
    r_peak_times = r_peaks / fs
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot the ECG signal
    ax.plot(time, ecg_signal, color='#3498db', label='ECG Signal')
    
    # Plot R-peaks
    ax.scatter(r_peak_times, ecg_signal[r_peaks], color='red', marker='o', s=50, label='R-peaks')
    
    # Add vertical lines at R-peaks
    for peak_time in r_peak_times:
        ax.axvline(x=peak_time, color='red', linestyle='--', alpha=0.3)
    
    # Calculate heart rate
    rr_intervals = np.diff(r_peaks) / fs  # in seconds
    hr = 60 / np.mean(rr_intervals)  # in bpm
    
    # Add heart rate info
    ax.text(0.02, 0.95, f'Average Heart Rate: {hr:.1f} BPM', 
            transform=ax.transAxes, fontsize=12, 
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Label axes
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Amplitude')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    return fig

def visualize_hrv(r_peaks, fs, title="Heart Rate Variability Analysis"):
    """
    Visualize heart rate variability metrics and RR interval distribution
    
    Parameters:
    -----------
    r_peaks : array-like
        Indices of detected R-peaks
    fs : float
        Sampling frequency in Hz
    title : str
        Plot title
    """
    # Calculate RR intervals in milliseconds
    rr_intervals = np.diff(r_peaks) * (1000 / fs)
    
    # Calculate HRV metrics
    hrv_metrics = calculate_hrv_metrics(r_peaks, fs)
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: RR intervals over time
    axes[0].plot(rr_intervals, marker='o', markersize=4, linestyle='-', color='#3498db')
    axes[0].set_title("RR Intervals Over Time")
    axes[0].set_xlabel("Beat number")
    axes[0].set_ylabel("RR Interval (ms)")
    axes[0].grid(True, linestyle='--', alpha=0.7)
    
    # Add a horizontal line for the mean RR interval
    axes[0].axhline(y=hrv_metrics["mean_rr"], color='r', linestyle='--', label=f'Mean: {hrv_metrics["mean_rr"]:.1f} ms')
    axes[0].legend()
    
    # Plot 2: Histogram of RR intervals with kernel density estimate
    sns.histplot(rr_intervals, kde=True, bins=20, color='#3498db', ax=axes[1])
    axes[1].set_title("Distribution of RR Intervals")
    axes[1].set_xlabel("RR Interval (ms)")
    axes[1].set_ylabel("Frequency")
    axes[1].grid(True, linestyle='--', alpha=0.7)
    
    # Add HRV metrics text box
    textstr = '\n'.join((
        f'Mean RR: {hrv_metrics["mean_rr"]:.1f} ms',
        f'SDNN: {hrv_metrics["sdnn"]:.1f} ms',
        f'RMSSD: {hrv_metrics["rmssd"]:.1f} ms',
        f'pNN50: {hrv_metrics["pnn50"]:.1f}%'
    ))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    axes[1].text(0.05, 0.95, textstr, transform=axes[1].transAxes, fontsize=10,
        verticalalignment='top', bbox=props)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    return fig

def visualize_arrhythmia_detection(ecg_signal, r_peaks, fs, arrhythmia_results, title="Arrhythmia Detection"):
    """
    Visualize ECG with detected arrhythmias
    
    Parameters:
    -----------
    ecg_signal : array-like
        ECG signal
    r_peaks : array-like
        Indices of detected R-peaks
    fs : float
        Sampling frequency in Hz
    arrhythmia_results : dict
        Results from detect_arrhythmias function
    title : str
        Plot title
    """
    # Create time axis in seconds
    time = np.arange(len(ecg_signal)) / fs
    r_peak_times = r_peaks / fs
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot ECG signal
    ax.plot(time, ecg_signal, color='#3498db', label='ECG Signal')
    
    # Plot all R-peaks
    ax.scatter(r_peak_times, ecg_signal[r_peaks], color='black', marker='o', s=30, alpha=0.5)
    
    # Define colors for different arrhythmia types
    colors = {
        'premature_beats': '#e74c3c',  # Red
        'long_pauses': '#9b59b6'       # Purple
    }
    
    # Plot premature beats
    if arrhythmia_results['premature_beats']:
        premature_indices = np.array(arrhythmia_results['premature_beats'])
        premature_times = r_peak_times[premature_indices]
        ax.scatter(premature_times, ecg_signal[r_peaks[premature_indices]], 
                  color=colors['premature_beats'], marker='*', s=200, 
                  label='Premature Beat')
    
    # Plot long pauses
    if arrhythmia_results['long_pauses']:
        pause_indices = np.array(arrhythmia_results['long_pauses'])
        # For pauses, highlight the interval between beats
        for idx in pause_indices:
            start_time = r_peak_times[idx]
            end_time = r_peak_times[idx+1] if idx+1 < len(r_peak_times) else time[-1]
            ax.axvspan(start_time, end_time, color=colors['long_pauses'], alpha=0.3)
        ax.plot([], [], color=colors['long_pauses'], alpha=0.3, linewidth=10, label='Long Pause')
    
    # Add rhythm status text
    rhythm_status = []
    if arrhythmia_results['bradycardia']:
        rhythm_status.append("BRADYCARDIA")
    if arrhythmia_results['tachycardia']:
        rhythm_status.append("TACHYCARDIA")
    if arrhythmia_results['irregular_rhythm']:
        rhythm_status.append("IRREGULAR RHYTHM")
    
    if rhythm_status:
        status_text = " & ".join(rhythm_status)
        ax.text(0.5, 0.97, status_text, transform=ax.transAxes, fontsize=14,
                color='red', ha='center', va='top',
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Calculate heart rate
    rr_intervals = np.diff(r_peaks) / fs  # in seconds
    hr = 60 / np.mean(rr_intervals)  # in bpm
    
    # Add heart rate info
    ax.text(0.02, 0.97, f'Heart Rate: {hr:.1f} BPM', 
            transform=ax.transAxes, fontsize=12, va='top',
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Summary of findings
    summary_text = [
        f"Premature Beats: {len(arrhythmia_results['premature_beats'])}",
        f"Long Pauses: {len(arrhythmia_results['long_pauses'])}"
    ]
    textstr = '\n'.join(summary_text)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.05, textstr, transform=ax.transAxes, fontsize=12, 
            va='bottom', bbox=props)
    
    # Label axes and add legend
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Amplitude')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    return fig

def visualize_ecg_segments(segments, labels=None, num_segments=4, title="ECG Segments"):
    """
    Visualize extracted ECG segments
    
    Parameters:
    -----------
    segments : array-like
        ECG segments with shape (n_segments, n_channels, segment_length)
    labels : array-like, optional
        Class labels for each segment
    num_segments : int
        Number of segments to visualize
    title : str
        Plot title
    """
    # Limit to the requested number of segments
    if num_segments > segments.shape[0]:
        num_segments = segments.shape[0]
    
    # Get number of channels
    num_channels = segments.shape[1]
    
    # Create a figure with subplots
    fig, axes = plt.subplots(num_segments, 1, figsize=(12, 3*num_segments))
    if num_segments == 1:
        axes = [axes]
    
    # Plot each segment
    for i in range(num_segments):
        ax = axes[i]
        segment = segments[i]
        
        # Plot each channel
        for ch in range(num_channels):
            ax.plot(segment[ch, :], label=f'Channel {ch+1}')
        
        # Add segment label if available
        if labels is not None:
            segment_title = f"Segment {i+1} (Class {labels[i]})"
        else:
            segment_title = f"Segment {i+1}"
        
        ax.set_title(segment_title)
        ax.set_xlabel('Time')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    return fig

# Function to integrate all visualizations
def visualize_ecg_pipeline(ecg_signal, fs, segment_length=24, preprocess=True, heartrate_category=True):
    """
    Comprehensive visualization of the entire ECG processing pipeline
    
    Parameters:
    -----------
    ecg_signal : array-like or list of arrays
        Single or multi-channel ECG signal
    fs : float
        Sampling frequency in Hz
    segment_length : int
        Length of segments for analysis in seconds
    preprocess : bool
        Whether to preprocess the signal
    heartrate_category : bool
        Whether to classify heart rate category
    """
    results = {}
    
    # Convert to multi-channel format if necessary
    if isinstance(ecg_signal, list):
        signals = ecg_signal
    else:
        signals = [ecg_signal]
    
    num_channels = len(signals)
    
    # Initialize the plots list
    plots = []
    
    # 1. Preprocessing
    if preprocess:
        processed_signals = preprocess_ecg(signals, fs)
        
        # Visualize for first channel
        preproc_fig = visualize_preprocessed_ecg(
            signals[0], processed_signals[0], fs, 
            title=f"ECG Preprocessing (Channel 1 of {num_channels})"
        )
        plots.append(('preprocessing', preproc_fig))
    else:
        processed_signals = signals
    
    # Use first channel for rhythm analysis (typically lead II)
    analysis_signal = processed_signals[0]
    
    # 2. R-peak detection
    r_peaks = detect_r_peaks(analysis_signal, fs)
    rpeak_fig = visualize_r_peaks(analysis_signal, r_peaks, fs)
    plots.append(('r_peaks', rpeak_fig))
    
    # 3. HRV analysis
    if len(r_peaks) > 1:  # Need at least 2 peaks for HRV
        hrv_fig = visualize_hrv(r_peaks, fs)
        plots.append(('hrv', hrv_fig))
        
        # Calculate average heart rate
        rr_intervals = np.diff(r_peaks) / fs  # in seconds
        heart_rate = 60 / np.mean(rr_intervals)  # in bpm
        
        # 4. Heart rate classification
        if heartrate_category:
            categories = classify_heart_rate(heart_rate)
            
            # Create simple visualization of heart rate categories
            hr_fig, ax = plt.subplots(figsize=(10, 6))
            y_pos = np.arange(len(categories))
            ax.barh(y_pos, [1] * len(categories), align='center', 
                   color=plt.cm.viridis(np.linspace(0, 1, len(categories))))
            ax.set_yticks(y_pos)
            ax.set_yticklabels(categories)
            ax.invert_yaxis()  # Put the first category at the top
            ax.set_xlabel('Heart Rate Classification')
            ax.set_title(f'Heart Rate: {heart_rate:.1f} BPM - Potential Categories')
            
            # Remove x ticks
            ax.set_xticks([])
            
            # Add heart rate to the right of each bar
            for i, category in enumerate(categories):
                ax.text(1.01, i, f"{heart_rate:.1f} BPM", va='center')
            
            plt.tight_layout()
            plots.append(('hr_category', hr_fig))
        
        # 5. Arrhythmia detection
        arrhythmia_results = detect_arrhythmias(r_peaks, fs, analysis_signal)
        arr_fig = visualize_arrhythmia_detection(analysis_signal, r_peaks, fs, arrhythmia_results)
        plots.append(('arrhythmia', arr_fig))
    
    # 6. Generate sample segments
    # For simplicity, create segments manually here
    segment_samples = int(segment_length * fs)
    num_segments = min(4, len(analysis_signal) // segment_samples)
    
    # Initialize array for segments
    segments = np.zeros((num_segments, num_channels, segment_samples))
    
    # Extract segments
    for i in range(num_segments):
        start = i * segment_samples
        end = start + segment_samples
        
        if end <= len(analysis_signal):
            for ch in range(num_channels):
                if ch < len(processed_signals):
                    current_signal = processed_signals[ch]
                    if len(current_signal) >= end:
                        segments[i, ch, :] = current_signal[start:end]
    
    # Visualize segments
    segment_fig = visualize_ecg_segments(segments, num_segments=num_segments)
    plots.append(('segments', segment_fig))
    
    # Save all plots
    import os
    os.makedirs('ecg_analysis', exist_ok=True)
    
    for name, fig in plots:
        fig.savefig(f'ecg_analysis/{name}.png', dpi=300, bbox_inches='tight')
    
    print(f"Saved {len(plots)} visualizations to 'ecg_analysis' directory")
    
    return plots

# Function to enhance the main pipeline with advanced visualizations
def enhance_main_with_visualizations(base_path, file_names, **kwargs):
    """
    Enhanced main function with comprehensive visualizations
    
    This function wraps the original main() function and adds visualization capabilities
    """
    # Call the original main function
    results = main(base_path, file_names, **kwargs)
    
    if results is None:
        print("No results to visualize.")
        return
    
    # Check if model type is MultiChannelDSNN for specific visualizations
    model = results['system'].model
    device = results['system'].device
    
    print("\nGenerating advanced visualizations...")
    
    # Create visualizations directory
    import os
    os.makedirs('visualizations', exist_ok=True)
    
    # 1. Always visualize class distribution
    fig_dist = visualize_class_distribution(results['true_labels'])
    fig_dist.savefig('visualizations/class_distribution.png', dpi=300, bbox_inches='tight')
    
    # 2. Visualize attention weights based on model type
    if isinstance(model, DSNNAttention):
        print("Visualizing DSNNAttention weights...")
        fig_attn = visualize_attention_weights(model, results['system'].test_loader, num_examples=3, device=device)
        fig_attn.savefig('visualizations/attention_weights.png', dpi=300, bbox_inches='tight')
        
        fig_heatmap = visualize_attention_heatmap(model, results['system'].test_loader, num_examples=3, device=device)
        fig_heatmap.savefig('visualizations/attention_heatmap.png', dpi=300, bbox_inches='tight')
    
    # 3. Visualize multichannel attention if applicable
    if isinstance(model, MultiChannelDSNN):
        print("Visualizing MultiChannelDSNN attention weights...")
        fig_multichannel = visualize_multichannel_attention(model, results['system'].test_loader, device=device)
        fig_multichannel.savefig('visualizations/multichannel_attention.png', dpi=300, bbox_inches='tight')
    
    # 4. Extract a sample ECG from the dataset for pipeline visualization
    try:
        # Try to get a sample from the first file info
        if 'file_info' in results and results['file_info']:
            file_info = results['file_info'][0]
            if 'data' in file_info:
                sample_ecg = file_info['data']
                fs = file_info.get('sampling_rate', 250)  # Default to 250 Hz if not specified
                
                print("Visualizing complete ECG analysis pipeline...")
                visualize_ecg_pipeline(sample_ecg, fs)
            else:
                print("No ECG data found in file_info")
    except Exception as e:
        print(f"Error visualizing ECG pipeline: {e}")
    
    print("Advanced visualizations completed!")
    return results

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def visualize_raw_ecg(leads, signal_labels, channels_used, fs, save_path=None):
    """
    Visualize raw ECG signals
    
    Parameters:
    - leads: List of ECG leads/channels
    - signal_labels: Labels for each signal channel
    - channels_used: Indices of channels being visualized
    - fs: Sampling frequency in Hz
    - save_path: Path to save the figure (if None, figure is displayed)
    """
    plt.figure(figsize=(15, 10))
    time = np.arange(len(leads[0])) / fs
    
    for i, lead in enumerate(leads):
        plt.subplot(len(leads), 1, i+1)
        plt.plot(time, lead)
        plt.title(f"Raw ECG - Channel {i} ({signal_labels[channels_used[i]]})")
        plt.ylabel("Amplitude")
    
    plt.xlabel("Time (s)")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Raw signals visualization saved to {save_path}")
        plt.close()
    else:
        plt.show()


def visualize_preprocessed_ecg(leads, signal_labels, channels_used, fs, save_path=None):
    """
    Visualize preprocessed ECG signals
    
    Parameters:
    - leads: List of preprocessed ECG leads/channels
    - signal_labels: Labels for each signal channel
    - channels_used: Indices of channels being visualized
    - fs: Sampling frequency in Hz
    - save_path: Path to save the figure (if None, figure is displayed)
    """
    plt.figure(figsize=(15, 10))
    time = np.arange(len(leads[0])) / fs
    
    for i, lead in enumerate(leads):
        plt.subplot(len(leads), 1, i+1)
        plt.plot(time, lead)
        plt.title(f"Preprocessed ECG - Channel {i} ({signal_labels[channels_used[i]]})")
        plt.ylabel("Amplitude")
    
    plt.xlabel("Time (s)")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Preprocessed signals visualization saved to {save_path}")
        plt.close()
    else:
        plt.show()


def visualize_r_peaks(lead, r_peaks, fs, save_path=None):
    """
    Visualize R-peaks detection on an ECG lead
    
    Parameters:
    - lead: ECG lead signal to visualize
    - r_peaks: Indices of detected R-peaks
    - fs: Sampling frequency in Hz
    - save_path: Path to save the figure (if None, figure is displayed)
    """
    plt.figure(figsize=(15, 5))
    time = np.arange(len(lead)) / fs
    
    # Plot the entire signal
    plt.plot(time, lead)
    
    # Plot R-peaks
    plt.scatter(r_peaks/fs, [lead[r] for r in r_peaks], color='red', label='R-peaks')
    
    plt.title("R-peaks Detection")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
        print(f"R-peaks visualization saved to {save_path}")
        plt.close()
    else:
        plt.show()


def visualize_segments(segments, channels_used=None, n_segments=5, save_path=None):
    """
    Visualize ECG segments extracted around R-peaks
    
    Parameters:
    - segments: Array of segmented ECG data
    - channels_used: Number of channels in each segment
    - n_segments: Number of segments to visualize
    - save_path: Path to save the figure (if None, figure is displayed)
    """
    if len(segments) == 0:
        print("No segments to visualize")
        return
    
    n_segments_to_show = min(n_segments, len(segments))
    n_channels = segments.shape[1] if channels_used is None else len(channels_used)
    
    plt.figure(figsize=(15, 10))
    for i in range(n_segments_to_show):
        plt.subplot(n_segments_to_show, 1, i+1)
        for ch in range(n_channels):
            plt.plot(segments[i][ch], label=f"Channel {ch}")
        plt.title(f"Segment {i+1} around R-peak")
        plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Segments visualization saved to {save_path}")
        plt.close()
    else:
        plt.show()


def visualize_training_history(history, save_path=None):
    """
    Visualize training and validation metrics over epochs
    
    Parameters:
    - history: Dictionary containing training history with keys:
               'train_acc', 'val_acc', 'train_loss', 'val_loss'
    - save_path: Path to save the figure (if None, figure is displayed)
    """
    plt.figure(figsize=(15, 10))
    
    # Plot training & validation accuracy
    plt.subplot(2, 1, 1)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    # Plot training & validation loss
    plt.subplot(2, 1, 2)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Training history visualization saved to {save_path}")
        plt.close()
    else:
        plt.show()


def visualize_confusion_matrix(y_true, y_pred, class_names=None, save_path=None):
    """
    Visualize confusion matrix of model predictions
    
    Parameters:
    - y_true: Ground truth labels
    - y_pred: Predicted labels
    - class_names: Names of the classes (optional)
    - save_path: Path to save the figure (if None, figure is displayed)
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    if class_names:
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix visualization saved to {save_path}")
        plt.close()
    else:
        plt.show()


def visualize_prediction_distribution(y_pred, y_true=None, save_path=None):
    """
    Visualize distribution of model predictions
    
    Parameters:
    - y_pred: Predicted labels
    - y_true: Ground truth labels (optional, used to determine bins)
    - save_path: Path to save the figure (if None, figure is displayed)
    """
    if y_true is not None:
        bins = len(set(y_true))
    else:
        bins = len(set(y_pred))
    
    plt.figure(figsize=(10, 6))
    plt.hist(y_pred, bins=bins, alpha=0.7)
    plt.title("Distribution of Predictions")
    plt.xlabel("Class")
    plt.ylabel("Count")
    
    if save_path:
        plt.savefig(save_path)
        print(f"Prediction distribution saved to {save_path}")
        plt.close()
    else:
        plt.show()


def visualize_performance_metrics(metrics, save_path=None):
    """
    Visualize model performance metrics
    
    Parameters:
    - metrics: Dictionary of metric names and values
    - save_path: Path to save the figure (if None, figure is displayed)
    """
    plt.figure(figsize=(10, 6))
    metrics_names = list(metrics.keys())
    metrics_values = [metrics[k] for k in metrics_names]
    
    plt.bar(metrics_names, metrics_values)
    plt.title("Model Performance Metrics")
    plt.ylabel("Score")
    plt.ylim([0, 1])
    
    if save_path:
        plt.savefig(save_path)
        print(f"Performance metrics visualization saved to {save_path}")
        plt.close()
    else:
        plt.show()


def visualize_spike_trains(spike_data, time_steps, neuron_indices=None, save_path=None):
    """
    Visualize spike trains from spiking neural network
    
    Parameters:
    - spike_data: Binary matrix where 1 indicates a spike (shape: neurons x time)
    - time_steps: Number of time steps to visualize
    - neuron_indices: Specific neurons to visualize (None for all)
    - save_path: Path to save the figure (if None, figure is displayed)
    """
    if neuron_indices is None:
        neuron_indices = range(spike_data.shape[0])
    
    plt.figure(figsize=(15, 8))
    
    for i, neuron_idx in enumerate(neuron_indices):
        spike_times = np.where(spike_data[neuron_idx, :time_steps] == 1)[0]
        plt.scatter(spike_times, [i] * len(spike_times), marker='|', color='black', s=100)
    
    plt.title("Spike Trains")
    plt.xlabel("Time Step")
    plt.ylabel("Neuron Index")
    plt.yticks(range(len(neuron_indices)), [f"Neuron {i}" for i in neuron_indices])
    plt.xlim(0, time_steps)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Spike trains visualization saved to {save_path}")
        plt.close()
    else:
        plt.show()


def visualize_activation_heatmap(activations, layer_name, save_path=None):
    """
    Visualize activations of a network layer as a heatmap
    
    Parameters:
    - activations: Activation values from a layer (neurons x time steps)
    - layer_name: Name of the layer being visualized
    - save_path: Path to save the figure (if None, figure is displayed)
    """
    plt.figure(figsize=(12, 8))
    
    plt.imshow(activations, aspect='auto', cmap='viridis')
    plt.colorbar(label='Activation Value')
    plt.title(f"Activation Heatmap - {layer_name}")
    plt.xlabel("Time Step")
    plt.ylabel("Neuron")
    
    if save_path:
        plt.savefig(save_path)
        print(f"Activation heatmap visualization saved to {save_path}")
        plt.close()
    else:
        plt.show()


def visualize_frequency_spectrum(ecg_signal, fs, save_path=None):
    """
    Visualize frequency spectrum of an ECG signal
    
    Parameters:
    - ecg_signal: Single channel ECG time series
    - fs: Sampling frequency in Hz
    - save_path: Path to save the figure (if None, figure is displayed)
    """
    # Compute FFT
    n = len(ecg_signal)
    yf = np.fft.rfft(ecg_signal)
    xf = np.fft.rfftfreq(n, 1/fs)
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(xf, np.abs(yf))
    plt.title("Frequency Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid(True)
    
    # Focus on relevant frequency range for ECG (typically up to 40-50 Hz)
    plt.xlim(0, min(50, max(xf)))
    
    if save_path:
        plt.savefig(save_path)
        print(f"Frequency spectrum visualization saved to {save_path}")
        plt.close()
    else:
        plt.show()

