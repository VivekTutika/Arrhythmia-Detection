"""
MIT-BIH to EDF/QRS Converter
Batch conversion of MIT-BIH dataset files to EDF format and QRS annotations.
Uses wfdb for reading MIT-BIH files and pyedflib for writing EDF files.
"""
import os
import numpy as np
import wfdb
import pyedflib
from datetime import datetime
from pathlib import Path


# Base paths - output to MIT-BIH folder (same as source files)
MITBIH_BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'Dataset', 'MIT-BIH')
OUTPUT_BASE = MITBIH_BASE  # Both EDF and QRS go to MIT-BIH folder


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


def _safe_str(value):
    """Safely convert any value to string"""
    if value is None:
        return ""
    # If already a string
    if isinstance(value, str):
        return str(value).strip()
    # If bytes
    if isinstance(value, bytes):
        try:
            return value.decode('utf-8').strip()
        except:
            return str(value)
    # If numpy bytes or other numpy types
    if hasattr(value, 'item'):
        try:
            return str(value.item())
        except:
            return str(value)
    # If numpy array
    if hasattr(value, 'tolist'):
        return str(value.tolist())
    # Anything else
    return str(value)


def convert_dat_hea_to_edf(record_name, mitbih_base=None, output_base=None):
    """
    Convert a single MIT-BIH record (.hea + .dat) to EDF format.
    Uses wfdb for reading and pyedflib for writing EDF.
    
    Args:
        record_name: Base name of the MIT-BIH record (e.g., '100')
        mitbih_base: Path to MIT-BIH dataset folder
        output_base: Path to output folder (defaults to MIT-BIH folder)
    
    Returns:
        dict: Status with 'success', 'message', and 'output_file' keys
    """
    if mitbih_base is None:
        mitbih_base = MITBIH_BASE
    if output_base is None:
        output_base = OUTPUT_BASE
    
    try:
        # Read the record using wfdb
        record_path = os.path.join(mitbih_base, record_name)
        record = wfdb.rdrecord(record_path)
        
        # Create output directory if not exists (same as MIT-BIH folder)
        os.makedirs(output_base, exist_ok=True)
        
        output_file = os.path.join(output_base, f"{record_name}.edf")
        
        # Get signal data
        if record.p_signal is not None:
            signals = record.p_signal
        else:
            signals = record.d_signal.astype(np.float32)
        
        # Get number of signals
        num_signals = signals.shape[1] if signals.ndim > 1 else 1
        
        # Handle single channel case
        if signals.ndim == 1:
            signals = signals.reshape(-1, 1)
        
        # Get sampling frequency
        fs = int(record.fs)
        
        # Get signal names and units
        sig_names = record.sig_name if record.sig_name else []
        units = record.units if record.units else []
        
        # Create EDF file writer with correct file type
        file_type = pyedflib.FILETYPE_EDF
        f = pyedflib.EdfWriter(output_file, num_signals, file_type)
        
        # Set file info
        f.setPatientName('')
        f.setPatientCode('')
        f.setGender(0)
        f.setBirthdate(datetime(2024, 1, 1))
        f.setStartdatetime(datetime(2024, 1, 1, 0, 0, 0))
        
        # Set channel info and write signals
        for i in range(num_signals):
            # Get signal name
            if i < len(sig_names):
                label = _safe_str(sig_names[i])[:16]
            else:
                label = f"Channel {i+1}"
            
            # Get unit
            if i < len(units):
                unit = _safe_str(units[i])[:8]
            else:
                unit = "mV"
            
            # Get physical min/max
            physical_min = float(np.min(signals[:, i]))
            physical_max = float(np.max(signals[:, i]))
            
            # Ensure non-zero range
            if physical_max == physical_min:
                physical_max = physical_min + 1
            
            # Set channel properties
            f.setLabel(i, label)
            f.setPhysicalDimension(i, unit)
            f.setPhysicalMinimum(i, physical_min)
            f.setPhysicalMaximum(i, physical_max)
            f.setDigitalMinimum(i, -32768)
            f.setDigitalMaximum(i, 32767)
            f.setPrefilter(i, '')
            f.setTransducer(i, '')
            
            # Write signal data
            f.writePhysicalSamples(signals[:, i])
        
        # Close the file
        f.close()
        
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


def convert_atr_to_qrs(record_name, mitbih_base=None, output_base=None):
    """
    Convert a single MIT-BIH annotation (.atr) file to QRS format.
    
    Args:
        record_name: Base name of the MIT-BIH record (e.g., '100')
        mitbih_base: Path to MIT-BIH dataset folder
        output_base: Path to output folder (defaults to MIT-BIH folder)
    
    Returns:
        dict: Status with 'success', 'message', and 'output_file' keys
    """
    if mitbih_base is None:
        mitbih_base = MITBIH_BASE
    if output_base is None:
        output_base = OUTPUT_BASE
    
    try:
        # Read the annotation file using wfdb
        annotation_path = os.path.join(mitbih_base, record_name)
        annotation = wfdb.rdann(annotation_path, 'atr')
        
        # Create output directory if not exists (same as MIT-BIH folder)
        os.makedirs(output_base, exist_ok=True)
        
        output_file = os.path.join(output_base, f"{record_name}.qrs")
        
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
        
        # Convert to EDF (output to MIT-BIH folder)
        edf_result = convert_dat_hea_to_edf(record_name, output_base=OUTPUT_BASE)
        if edf_result['success']:
            results['edf']['success'] += 1
            results['edf']['files'].append(edf_result['output_file'])
            print(f"  EDF: OK")
        else:
            results['edf']['failed'] += 1
            print(f"  EDF: FAILED - {edf_result['message']}")
        
        # Convert to QRS (output to MIT-BIH folder)
        qrs_result = convert_atr_to_qrs(record_name, output_base=OUTPUT_BASE)
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
    edf_result = convert_dat_hea_to_edf(record_name, output_base=OUTPUT_BASE)
    qrs_result = convert_atr_to_qrs(record_name, output_base=OUTPUT_BASE)
    
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

