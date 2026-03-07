import pyedflib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def read_edf_file(file_path):
    """
    Read an EDF file and return its signals and header information
    """
    try:
        # Open the EDF file
        f = pyedflib.EdfReader(file_path)
        
        # Get number of signals
        n_signals = f.signals_in_file
        signal_labels = f.getSignalLabels()
        
        # Read all signals
        signals = []
        for i in range(n_signals):
            signals.append(f.readSignal(i))
            
        # Get sampling frequencies
        sample_rates = f.getSampleFrequencies()
        
        # Get start time
        start_time = f.getStartdatetime()
        
        return {
            'signals': signals,
            'labels': signal_labels,
            'sample_rates': sample_rates,
            'start_time': start_time,
            'duration': f.getFileDuration()
        }
    
    finally:
        # Always close the file
        if 'f' in locals():
            f.close()

def read_qrs_annotations(qrs_file):
    """
    Read QRS annotations file if available
    Returns list of R-peak positions
    """
    try:
        with open(qrs_file, 'r') as f:
            # Read R-peak positions
            # Format may vary depending on your specific QRS file format
            annotations = f.readlines()
            return [float(line.strip()) for line in annotations if line.strip()]
    except Exception as e:
        print(f"Could not read QRS file: {e}")
        return None

def plot_ecg_with_qrs(data, qrs_points=None, duration=10):
    """
    Plot ECG signal with QRS annotations if available
    duration: seconds of signal to plot
    """
    plt.figure(figsize=(15, 5))
    
    # Get the first ECG signal
    signal = data['signals'][0]
    sample_rate = data['sample_rates'][0]
    
    # Plot only first 'duration' seconds
    samples_to_plot = int(duration * sample_rate)
    time = np.arange(samples_to_plot) / sample_rate
    
    plt.plot(time, signal[:samples_to_plot], 'b-', label='ECG Signal')
    
    if qrs_points is not None:
        # Plot R-peaks
        qrs_times = [p for p in qrs_points if p <= duration]
        qrs_amplitudes = [signal[int(p * sample_rate)] for p in qrs_times]
        plt.plot(qrs_times, qrs_amplitudes, 'ro', label='R-peaks')
    
    plt.title(f'ECG Signal from {data["start_time"]}')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude (μV)')
    plt.grid(True)
    plt.legend()
    plt.show()

def analyze_ecg(edf_file, qrs_file=None):
    """
    Main function to analyze ECG data
    """
    # Read EDF file
    print(f"Reading EDF file: {edf_file}")
    ecg_data = read_edf_file(edf_file)
    
    # Read QRS annotations if available
    qrs_points = None
    if qrs_file and Path(qrs_file).exists():
        print(f"Reading QRS annotations: {qrs_file}")
        qrs_points = read_qrs_annotations(qrs_file)
    
    # Print basic information
    print("\nSignal Information:")
    print(f"Number of signals: {len(ecg_data['signals'])}")
    print(f"Signal labels: {ecg_data['labels']}")
    print(f"Recording duration: {ecg_data['duration']} seconds")
    print(f"Sampling rates: {ecg_data['sample_rates']} Hz")
    
    # Plot the ECG signal
    plot_ecg_with_qrs(ecg_data, qrs_points)
    
    return ecg_data

# Example usage
if __name__ == "__main__":
    # Replace with your actual file paths
    edf_file = "r01.edf"
    qrs_file = "r01.edf.qrs"
    
    ecg_data = analyze_ecg(edf_file, qrs_file)
