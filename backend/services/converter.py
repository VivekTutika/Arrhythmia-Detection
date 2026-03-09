"""
MIT-BIH to EDF/QRS Converter
Batch conversion of MIT-BIH dataset files to EDF format and QRS annotations.
"""
import os
import numpy as np
import wfdb
from pathlib import Path


# Base paths
MITBIH_BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'Dataset', 'MIT-BIH')
EDF_OUTPUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'Dataset', 'edf')
QRS_OUTPUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'Dataset', 'qrs')


def get_mitbih_files():
    """
    Get list of all MIT-BIH record names (without extension).
    Returns list of file base names that have .hea and .dat files.
    """
    mitbih_path = Path(MITBIH_BASE)
    
    # Get all .hea files
    hea_files = list(mitbih_path.glob('*.hea'))
    
    # Extract base names (without extension)
    record_names = []
    for hea_file in hea_files:
        base_name = hea_file.stem
        # Check if corresponding .dat file exists
        dat_file = hea_file.with_suffix('.dat')
        if dat_file.exists():
            record_names.append(base_name)
    
    return sorted(record_names)


def convert_dat_hea_to_edf(record_name, mitbih_base=None, edf_output=None):
    """
    Convert a single MIT-BIH record (.hea + .dat) to EDF format.
    
    Args:
        record_name: Base name of the MIT-BIH record (e.g., '100')
        mitbih_base: Path to MIT-BIH dataset folder
        edf_output: Path to output EDF folder
    
    Returns:
        dict: Status with 'success', 'message', and 'output_file' keys
    """
    if mitbih_base is None:
        mitbih_base = MITBIH_BASE
    if edf_output is None:
        edf_output = EDF_OUTPUT
    
    try:
        # Read the record using wfdb
        record_path = os.path.join(mitbih_base, record_name)
        record = wfdb.rdrecord(record_path)
        
        # Create output directory if not exists
        os.makedirs(edf_output, exist_ok=True)
        
        output_file = os.path.join(edf_output, f"{record_name}.edf")
        
        # Always use the basic EDF writer - it's more reliable
        _write_basic_edf(record, output_file)
        
        return {
            'success': True,
            'message': f"Successfully converted {record_name} to EDF",
            'output_file': output_file
        }
        
    except Exception as e:
        return {
            'success': False,
            'message': f"Error converting {record_name}: {str(e)}",
            'output_file': None
        }


def _write_basic_edf(record, output_file):
    """
    Write a basic EDF file from wfdb record without edfio.
    This is a simplified implementation for compatibility.
    """
    # Get signal data
    try:
        if record.p_signal is not None:
            signals = record.p_signal.astype(np.float32)
        else:
            signals = record.d_signal.astype(np.float32)
    except Exception as e:
        raise Exception(f"Error getting signal data: {e}")
    
    num_signals = signals.shape[1] if signals.ndim > 1 else 1
    num_samples = signals.shape[0]
    
    # EDF header size is 256 bytes + num_signals * 256 bytes
    header_size = 256 + num_signals * 256
    
    # Handle single channel case
    if num_signals == 1:
        signals = signals.reshape(-1, 1)
    
    # Calculate min/max for each channel
    physical_mins = np.min(signals, axis=0)
    physical_maxs = np.max(signals, axis=0)
    
    # Ensure non-zero range
    for i in range(num_signals):
        if physical_maxs[i] == physical_mins[i]:
            physical_maxs[i] = physical_mins[i] + 1
    
    digital_mins = np.iinfo(np.int16).min * np.ones(num_signals)
    digital_maxs = np.iinfo(np.int16).max * np.ones(num_signals)
    
    # Get signal names - handle ALL possible types
    sig_names = []
    try:
        if record.sig_name is not None:
            # Convert to list in case it's a numpy array or other iterable
            sig_name_iterable = list(record.sig_name) if hasattr(record.sig_name, '__iter__') else [record.sig_name]
            for name in sig_name_iterable:
                # Handle bytes
                if isinstance(name, bytes):
                    sig_names.append(name.decode('latin-1').strip('\x00'))
                # Handle numpy bytes
                elif hasattr(name, 'item'):  # numpy bytes
                    try:
                        sig_names.append(name.item().decode('latin-1').strip('\x00'))
                    except:
                        sig_names.append(str(name))
                # Handle string
                elif isinstance(name, str):
                    sig_names.append(name.strip('\x00'))
                # Handle anything else
                else:
                    sig_names.append(str(name))
    except Exception as e:
        raise Exception(f"Error processing sig_name: {e}. Type: {type(record.sig_name)}")
    
    # Pad with defaults if needed
    while len(sig_names) < num_signals:
        sig_names.append(f"Channel {len(sig_names)}")
    
    # Get units - handle ALL possible types
    units = []
    try:
        if record.units is not None:
            # Convert to list in case it's a numpy array or other iterable
            units_iterable = list(record.units) if hasattr(record.units, '__iter__') else [record.units]
            for unit in units_iterable:
                # Handle bytes
                if isinstance(unit, bytes):
                    units.append(unit.decode('latin-1').strip('\x00'))
                # Handle numpy bytes
                elif hasattr(unit, 'item'):  # numpy bytes
                    try:
                        units.append(unit.item().decode('latin-1').strip('\x00'))
                    except:
                        units.append(str(unit))
                # Handle string
                elif isinstance(unit, str):
                    units.append(unit.strip('\x00'))
                # Handle anything else
                else:
                    units.append(str(unit))
    except Exception as e:
        raise Exception(f"Error processing units: {e}. Type: {type(record.units)}")
    
    # Pad with defaults if needed
    while len(units) < num_signals:
        units.append("mV")
    
    # Write EDF file
    with open(output_file, 'wb') as f:
        # Main header (256 bytes)
        f.write(b'0       ')  # EDF magic number
        f.write(b' ' * 80)    # Patient ID (space-padded)
        f.write(b' ' * 80)    # Recording info
        f.write(b'01.01.00'.encode('latin-1'))  # Start date
        f.write(b'00.00.00'.encode('latin-1'))  # Start time
        header_bytes = str(header_size).encode('latin-1').ljust(8)
        f.write(header_bytes)
        f.write(b' ' * 44)    # Reserved
        f.write(b'1'.encode('latin-1').ljust(8))  # Num data records (1 for simplicity)
        f.write(str(num_samples).encode('latin-1').ljust(8))  # Data record duration (samples)
        f.write(str(num_signals).encode('latin-1').ljust(4))  # Num signals
        
        # Signal headers (256 bytes each)
        for i in range(num_signals):
            # Label (16 bytes) - encode FIRST, then pad
            label = str(sig_names[i]) if i < len(sig_names) else f"Channel {i}"
            label_bytes = label[:16].encode('latin-1').ljust(16)
            f.write(label_bytes)
            
            # Transducer (80 bytes)
            f.write(b' ' * 80)
            
            # Physical dimension (8 bytes) - encode FIRST, then pad
            unit = str(units[i]) if i < len(units) else 'mV'
            unit_bytes = unit[:8].encode('latin-1').ljust(8)
            f.write(unit_bytes)
            
            # Physical min (8 bytes) - encode FIRST, then pad
            phys_min = str(physical_mins[i])[:8].encode('latin-1').ljust(8)
            f.write(phys_min)
            
            # Physical max (8 bytes) - encode FIRST, then pad
            phys_max = str(physical_maxs[i])[:8].encode('latin-1').ljust(8)
            f.write(phys_max)
            
            # Digital min (8 bytes) - encode FIRST, then pad
            dig_min = str(int(digital_mins[i]))[:8].encode('latin-1').ljust(8)
            f.write(dig_min)
            
            # Digital max (8 bytes) - encode FIRST, then pad
            dig_max = str(int(digital_maxs[i]))[:8].encode('latin-1').ljust(8)
            f.write(dig_max)
            
            # Prefiltering (80 bytes)
            f.write(b' ' * 80)
            
            # Num samples in each data record (8 bytes) - encode FIRST, then pad
            num_samp = str(num_samples)[:8].encode('latin-1').ljust(8)
            f.write(num_samp)
        
        # Write signal data
        # Convert to 16-bit digital values
        for i in range(num_signals):
            digital_values = ((signals[:, i] - physical_mins[i]) / 
                             (physical_maxs[i] - physical_mins[i]) * 
                             (digital_maxs[i] - digital_mins[i]) + digital_mins[i])
            digital_values = np.clip(digital_values, digital_mins[i], digital_maxs[i]).astype(np.int16)
            f.write(digital_values.tobytes())


def convert_atr_to_qrs(record_name, mitbih_base=None, qrs_output=None):
    """
    Convert a single MIT-BIH annotation (.atr) file to QRS format.
    
    Args:
        record_name: Base name of the MIT-BIH record (e.g., '100')
        mitbih_base: Path to MIT-BIH dataset folder
        qrs_output: Path to output QRS folder
    
    Returns:
        dict: Status with 'success', 'message', and 'output_file' keys
    """
    if mitbih_base is None:
        mitbih_base = MITBIH_BASE
    if qrs_output is None:
        qrs_output = QRS_OUTPUT
    
    try:
        # Read the annotation file using wfdb
        annotation_path = os.path.join(mitbih_base, record_name)
        annotation = wfdb.rdann(annotation_path, 'atr')
        
        # Create output directory if not exists
        os.makedirs(qrs_output, exist_ok=True)
        
        output_file = os.path.join(qrs_output, f"{record_name}.qrs")
        
        # Write QRS peak sample locations
        # QRS format: sample number of each R-peak (one per line)
        with open(output_file, 'w') as f:
            for sample in annotation.sample:
                f.write(f"{sample}\n")
        
        return {
            'success': True,
            'message': f"Successfully converted {record_name} annotations to QRS",
            'output_file': output_file,
            'num_peaks': len(annotation.sample)
        }
        
    except Exception as e:
        return {
            'success': False,
            'message': f"Error converting {record_name} annotations: {str(e)}",
            'output_file': None,
            'num_peaks': 0
        }


def convert_all_mitbih_files():
    """
    Batch convert all MIT-BIH files to EDF and QRS format.
    
    Returns:
        dict: Summary of conversion results with 'edf' and 'qrs' statistics
    """
    record_names = get_mitbih_files()
    
    results = {
        'total_records': len(record_names),
        'edf': {'success': 0, 'failed': 0, 'files': []},
        'qrs': {'success': 0, 'failed': 0, 'files': []}
    }
    
    print(f"Found {len(record_names)} MIT-BIH records to convert")
    print("-" * 50)
    
    for record_name in record_names:
        print(f"Processing: {record_name}")
        
        # Convert to EDF
        edf_result = convert_dat_hea_to_edf(record_name)
        if edf_result['success']:
            results['edf']['success'] += 1
            results['edf']['files'].append(edf_result['output_file'])
        else:
            results['edf']['failed'] += 1
            print(f"  EDF: FAILED - {edf_result['message']}")
        
        # Convert to QRS
        qrs_result = convert_atr_to_qrs(record_name)
        if qrs_result['success']:
            results['qrs']['success'] += 1
            results['qrs']['files'].append(qrs_result['output_file'])
            print(f"  QRS: OK ({qrs_result['num_peaks']} peaks)")
        else:
            results['qrs']['failed'] += 1
            print(f"  QRS: FAILED - {qrs_result['message']}")
    
    print("-" * 50)
    print(f"Conversion complete!")
    print(f"  EDF: {results['edf']['success']} success, {results['edf']['failed']} failed")
    print(f"  QRS: {results['qrs']['success']} success, {results['qrs']['failed']} failed")
    
    return results


def convert_single_record(record_name):
    """
    Convert a single MIT-BIH record to EDF and QRS format.
    Convenience function for converting one file at a time.
    
    Args:
        record_name: Base name of the MIT-BIH record (e.g., '100')
    
    Returns:
        dict: Combined results for EDF and QRS conversion
    """
    edf_result = convert_dat_hea_to_edf(record_name)
    qrs_result = convert_atr_to_qrs(record_name)
    
    return {
        'edf': edf_result,
        'qrs': qrs_result,
        'overall_success': edf_result['success'] and qrs_result['success']
    }


if __name__ == "__main__":
    # Run conversion when script is executed directly
    print("MIT-BIH to EDF/QRS Converter")
    print("=" * 50)
    
    # Get list of files first
    records = get_mitbih_files()
    print(f"Found {len(records)} records: {records[:5]}... (showing first 5)")
    print()
    
    # Run batch conversion
    results = convert_all_mitbih_files()

