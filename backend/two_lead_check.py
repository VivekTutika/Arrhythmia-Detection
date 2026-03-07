import numpy as np
import matplotlib.pyplot as plt
import pyedflib  # For EDF files
import os

# Step 1: Load the EDF file
file_path = "D:/DSNN/data/edf/1.edf"  # Replace with your actual file path

try:
    # Open the EDF file
    f = pyedflib.EdfReader(file_path)
    
    # Get basic information
    n_channels = f.signals_in_file
    print(f"Number of channels/signals in the file: {n_channels}")
    
    # Get signal labels (which often indicate lead types)
    signal_labels = f.getSignalLabels()
    print("Signal labels:", signal_labels)
    
    # Check if there are at least 2 channels for two leads
    if n_channels >= 2:
        print(f"The dataset has {n_channels} channels which can be represent as leads and the algoritm will take any two signals randomly most probably first two channels.")
        
        # Read the first two signals/channels
        signal_1 = f.readSignal(0)
        signal_2 = f.readSignal(1)
        
        # Plot the first 2000 samples of both signals
        plt.figure(figsize=(12, 6))
        plt.subplot(5, 1, 1)
        plt.plot(signal_1[:2000])
        plt.title(f"Channel 1: {signal_labels[0]}")
        
        plt.subplot(5, 1, 2)
        plt.plot(signal_2[:2000])
        plt.title(f"Channel 2: {signal_labels[1]}")

        plt.subplot(5, 1, 3)
        plt.plot(signal_2[:2000])
        plt.title(f"Channel 3: {signal_labels[2]}")

        plt.subplot(5, 1, 4)
        plt.plot(signal_2[:2000])
        plt.title(f"Channel 4: {signal_labels[3]}")

        plt.subplot(5, 1, 5)
        plt.plot(signal_2[:2000])
        plt.title(f"Channel 5: {signal_labels[4]}")
        plt.tight_layout()
        plt.show()
        
        # Check if these look like typical ECG leads
        # (Basic check - real validation would be more complex)
        def looks_like_ecg(signal):
            """Basic check if data resembles ECG patterns"""
            # Check for variability
            if np.std(signal) < 0.01:
                return False
            # In real applications, you would check for QRS complexes, 
            # typical ECG frequency components, etc.
            return True
        
        print(f"Channel 1 resembles ECG: {looks_like_ecg(signal_1)}")
        print(f"Channel 2 resembles ECG: {looks_like_ecg(signal_2)}")
    else:
        print("The dataset has fewer than 2 channels")
    
    # Close the file
    f.close()

except Exception as e:
    print(f"Error loading EDF file: {e}")
    
# For QRS files (if needed)
# QRS files typically contain annotations of R-peaks rather than the raw signal
# You would need a specific library or parser for your particular QRS format
# If your .qrs format is from PhysioNet, you might use wfdb:

try:
    import wfdb
    
    # Remove the extension to get the record name
    record_name = os.path.splitext(file_path)[0]
    
    # Read the annotations
    ann = wfdb.rdann(record_name, 'qrs')
    
    print("QRS annotations loaded successfully")
    print(f"Number of QRS annotations: {len(ann.sample)}")
    
    # This doesn't directly tell you about leads, but confirms
    # that QRS annotations exist for the record
    
except ImportError:
    print("wfdb library not installed. Install with: pip install wfdb")
except Exception as e:
    print(f"Error loading QRS file: {e}")
