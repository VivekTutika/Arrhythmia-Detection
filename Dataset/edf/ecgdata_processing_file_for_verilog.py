import pyedflib
import numpy as np
import os

# Output file to store processed data
output_file = 'processed_ecg_data.txt'

# Process all five pairs of files
for i in range(1, 6):  # Assuming files are numbered 1-5
    edf_file = f'{i}.edf'
    qrs_file = f'{i}.qrs'
    
    # Read EDF file
    f = pyedflib.EdfReader(edf_file)
    n_channels = f.signals_in_file
    
    # Assuming the first two channels are ECG leads
    lead1 = f.readSignal(0)
    lead2 = f.readSignal(1) if n_channels > 1 else np.zeros_like(lead1)
    
    # Scale to 12-bit range (0-4095)
    # You'll need to adjust this based on your specific data range
    lead1_min, lead1_max = np.min(lead1), np.max(lead1)
    lead2_min, lead2_max = np.min(lead2), np.max(lead2)
    
    lead1_scaled = np.round(((lead1 - lead1_min) / (lead1_max - lead1_min)) * 4095).astype(int)
    lead2_scaled = np.round(((lead2 - lead2_min) / (lead2_max - lead2_min)) * 4095).astype(int)
    
    # You may also want to read the QRS file for annotations
    # This depends on the format of your .qrs files
    # For now, we'll just process the raw ECG data
    
    # Write to output file
    with open(output_file, 'a') as out:
        for l1, l2 in zip(lead1_scaled, lead2_scaled):
            out.write(f"{l1} {l2}\n")
    
    f.close()
    
print(f"Data processing complete. Output saved to {output_file}")
