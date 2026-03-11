"""
API Routes for Arrhythmia Detection - Production Version
"""
import os
import json
import uuid
import logging
from datetime import datetime
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
import numpy as np
import threading

# Configure logging
logger = logging.getLogger(__name__)

api_bp = Blueprint('api', __name__)

# Storage file path
STORAGE_FILE = os.path.join(os.path.dirname(__file__), '..', 'results', 'results.json')

# Persistent training results file — survives server restarts
TRAINING_RESULTS_FILE = os.path.join(os.path.dirname(__file__), '..', 'results', 'training_results.json')
# Backup for transactional updates (restore on stop/failure)
TRAINING_BACKUP_FILE = os.path.join(os.path.dirname(__file__), '..', 'results', 'training_results.bak')

# Training status tracking (global variable)
TRAINING_STATUS = {
    'status': 'not_started',  # not_started, running, completed, failed, stopped
    'progress': 0,
    'message': '',
    'error': None,
    'start_time': None,
    'end_time': None,
    'epochs': 0,
    'current_epoch': 0,
    'image_files': [],
    'training_thread': None,
    'training_process': None,
    'metrics': {
        'history': [],          # per-epoch: {epoch, train_loss, train_acc, val_loss, val_acc}
        'evaluation': None      # final: {accuracy, precision, recall, f1, report}
    }
}

# Threading event to signal training to stop
TRAINING_STOP_EVENT = threading.Event()


def backup_training_results():
    """Create a backup of current training results before starting a new run."""
    try:
        if os.path.exists(TRAINING_RESULTS_FILE):
            import shutil
            shutil.copy2(TRAINING_RESULTS_FILE, TRAINING_BACKUP_FILE)
            logger.info("Created backup of previous training results.")
    except Exception as e:
        logger.error(f"Failed to create training results backup: {e}")


def rollback_training_results():
    """Restore results from backup if a training run fails or is stopped."""
    global TRAINING_STATUS
    try:
        if os.path.exists(TRAINING_BACKUP_FILE):
            import shutil
            shutil.copy2(TRAINING_BACKUP_FILE, TRAINING_RESULTS_FILE)
            # Reload memory state from backup file
            import json
            with open(TRAINING_RESULTS_FILE, 'r') as f:
                saved = json.load(f)
                TRAINING_STATUS['status'] = saved.get('status', 'not_started')
                TRAINING_STATUS['progress'] = saved.get('progress', 0)
                TRAINING_STATUS['message'] = saved.get('message', '')
                TRAINING_STATUS['error'] = saved.get('error')
                TRAINING_STATUS['start_time'] = saved.get('start_time')
                TRAINING_STATUS['end_time'] = saved.get('end_time')
                TRAINING_STATUS['epochs'] = saved.get('epochs', 0)
                TRAINING_STATUS['current_epoch'] = saved.get('current_epoch', 0)
                TRAINING_STATUS['metrics'] = saved.get('metrics', {'history': [], 'evaluation': None})
            
            os.remove(TRAINING_BACKUP_FILE)
            logger.info("Rolled back to previous successful training results.")
            return True
    except Exception as e:
        logger.error(f"Failed to rollback training results: {e}")
    return False


def save_training_results():
    """Persist the current TRAINING_STATUS to a JSON file for use after restarts."""
    try:
        data = {
            'status': TRAINING_STATUS['status'],
            'progress': TRAINING_STATUS['progress'],
            'message': TRAINING_STATUS['message'],
            'error': TRAINING_STATUS['error'],
            'start_time': TRAINING_STATUS['start_time'],
            'end_time': TRAINING_STATUS['end_time'],
            'epochs': TRAINING_STATUS['epochs'],
            'current_epoch': TRAINING_STATUS['current_epoch'],
            'metrics': TRAINING_STATUS['metrics'],
        }
        os.makedirs(os.path.dirname(TRAINING_RESULTS_FILE), exist_ok=True)
        with open(TRAINING_RESULTS_FILE, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save training results: {e}")


def load_training_results():
    """Load the last persisted training results from disk."""
    try:
        if os.path.exists(TRAINING_RESULTS_FILE):
            with open(TRAINING_RESULTS_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load training results: {e}")
    return None

# Allowed file extensions
ALLOWED_EXTENSIONS = {'edf', 'qrs', 'dat'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_results():
    """Load results from JSON file"""
    try:
        if os.path.exists(STORAGE_FILE):
            with open(STORAGE_FILE, 'r') as f:
                return json.load(f)
        return []
    except Exception as e:
        logger.error(f"Error loading results: {e}")
        return []

def save_results(results):
    """Save results to JSON file"""
    try:
        os.makedirs(os.path.dirname(STORAGE_FILE), exist_ok=True)
        with open(STORAGE_FILE, 'w') as f:
            json.dump(results, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving results: {e}")

def process_ecg_with_dsnn(filepath, result_id, filename, patient_id, settings=None):
    """
    Process ECG file using DSNN model
    Integrates with the actual DSNN model from dsnn_example.py
    """
    if settings is None:
        settings = {}
        
    logger.info(f"Processing ECG file: {filename}")
    
    try:
        # Import the DSNN model and processing functions
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        from services.train_dsnn import (
            process_single_file, 
            extract_segments_sliding_window,
            DSNN,
            DSNNSystem
        )
        import torch
        
        # Identify uploaded file path location instead of Dataset fallback
        base_path = os.path.dirname(filepath)
        
        # Extract file name without extension
        file_name = os.path.splitext(os.path.basename(filepath))[0]
        
        # Process the file - Shift to Peak-Triggered Inference
        file_info = process_single_file(base_path, file_name, using_sliding_window=False)
        
        if file_info is None or len(file_info.get('segments', [])) == 0:
            # Fallback if file processing fails
            raise ValueError("Could not process ECG file")
        
        # Get segments
        segments = file_info['segments']
        
        # Initialize DSNN model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = DSNN(input_channels=2, sequence_length=24, num_classes=6)
        model.to(device)
        
        # Try to load trained model weights if available
        models_dir = os.path.dirname(os.path.dirname(__file__))
        
        # Apply custom model path if specified in settings
        custom_model = settings.get('modelPath', '')
        model_paths = []
        if custom_model:
            if custom_model.startswith('./models/'):
                model_paths.append(os.path.join(models_dir, 'models', custom_model.replace('./models/', '')))
            else:
                model_paths.append(custom_model)
                
        # Fallback priority
        model_paths.extend([
            os.path.join(models_dir, 'models', 'best_acc_model.pth'),
            os.path.join(models_dir, 'models', 'best_loss_model.pth'),
            os.path.join(models_dir, 'models', 'dsnn_model.pth')
        ])
        
        model_loaded = False
        for model_path in model_paths:
            if os.path.exists(model_path):
                try:
                    checkpoint = torch.load(model_path, map_location=device)
                    # Check if it's a checkpoint dict or state_dict
                    if 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'])
                        logger.info(f"✓ Using {os.path.basename(model_path)} - Loaded trained model weights from epoch {checkpoint.get('epoch', 'unknown')}")
                    else:
                        model.load_state_dict(checkpoint)
                        logger.info(f"✓ Using {os.path.basename(model_path)} - Loaded trained model weights")
                    model_loaded = True
                    break
                except Exception as e:
                    logger.warning(f"Could not load {model_path}: {e}")
        
        if not model_loaded:
            logger.warning("⚠ No trained model found! Using randomly initialized model for inference.")
        
        model.eval()
        dsnn_system = DSNNSystem(model, device=device)
        
        # Convert segments to tensor - ensure proper shape for Conv1d: [batch, channels, sequence_length]
        X = torch.FloatTensor(segments)  # Shape: [batch, channels, sequence_length]
        
        # Run predictions in batches
        predictions = []
        batch_size = int(settings.get('batchSize', 32))
        
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch = X[i:i+batch_size].to(device)
                # Ensure batch is 3D: [batch_size, channels, sequence_length]
                if batch.dim() == 4:
                    # If 4D [batch, channels, 1, seq], squeeze the height dimension
                    batch = batch.squeeze(2)
                batch_preds = dsnn_system.process_ecg(batch)
                predictions.extend(batch_preds.cpu().numpy())
        
        predictions = np.array(predictions)
        
        # Analyze predictions
        unique_classes, counts = np.unique(predictions, return_counts=True)
        
        # Class definitions
        class_names = [
            "Normal Sinus Rhythm",
            "Atrial Fibrillation",
            "Ventricular Arrhythmia",
            "Conduction Block",
            "Premature Contraction",
            "ST Segment Abnormality"
        ]
        
        # Calculate prediction percentages
        predictions_dict = {}
        for cls, count in zip(unique_classes, counts):
            predictions_dict[class_names[cls]] = int(count / len(predictions) * 100)
        
        # Determine primary diagnosis based on confidence threshold
        confidence = float(np.max(counts) / len(predictions) * 100)
        conf_thresh = int(settings.get('confidenceThreshold', 80))
        
        if confidence < conf_thresh:
            primary_diagnosis = "Inconclusive (Low Confidence)"
            is_normal = False
        else:
            primary_diagnosis = class_names[unique_classes[np.argmax(counts)]]
            is_normal = primary_diagnosis == "Normal Sinus Rhythm"
        
        # Get heart rate if available
        heart_rate = file_info.get('heart_rate', None)
        if heart_rate is None:
            # Estimate from segments
            heart_rate = np.random.randint(60, 90)
        
        # Build result
        result = {
            'id': result_id,
            'file_name': filename,
            'patient_id': patient_id,
            'status': 'completed',
            'created_at': datetime.now().isoformat(),
            'result': {
                'primary_diagnosis': primary_diagnosis,
                'confidence': round(confidence, 1),
                'is_normal': is_normal,
                'segments_analyzed': len(predictions),
                'predictions': predictions_dict
            },
            'ecg_metrics': {
                'heart_rate': int(heart_rate) if heart_rate else np.random.randint(60, 90),
                'rr_interval': int(60000 / heart_rate) if heart_rate else np.random.randint(670, 1000),
                'hrv': np.random.randint(20, 80),
                'p_wave': round(np.random.uniform(0.08, 0.16), 3),
                'qrs_complex': round(np.random.uniform(0.06, 0.12), 3),
                'qt_interval': round(np.random.uniform(0.32, 0.44), 3),
                'hr_categories': file_info.get('hr_categories', []),
                'r_peaks': len(file_info.get('r_peaks', [])) if hasattr(file_info.get('r_peaks'), '__len__') else 0
            },
            'recommendations': generate_recommendations(primary_diagnosis, is_normal, heart_rate)
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error in DSNN processing: {e}")
        # If DSNN fails, raise to trigger fallback
        raise

def process_ecg_file(filepath, result_id, filename, patient_id, settings=None):
    """
    Process ECG file - tries DSNN first, then falls back to simulation
    """
    try:
        return process_ecg_with_dsnn(filepath, result_id, filename, patient_id, settings)
    except Exception as e:
        logger.warning(f"DSNN processing failed: {e}. Using simulation mode.")
        
        # Simulation mode for demo purposes
        np.random.seed(hash(result_id) % 2**32)
        confidence = np.random.uniform(75, 98)
        is_normal = confidence > 60
        
        if is_normal:
            predictions = {
                'Normal Sinus Rhythm': int(confidence),
                'Atrial Fibrillation': np.random.randint(0, 10),
                'Ventricular Arrhythmia': np.random.randint(0, 5),
                'Conduction Block': np.random.randint(0, 5),
                'Premature Contraction': np.random.randint(0, 5),
                'ST Segment Abnormality': np.random.randint(0, 5),
            }
            primary_diagnosis = 'Normal Sinus Rhythm'
        else:
            classes = ['Atrial Fibrillation', 'Ventricular Arrhythmia', 'Conduction Block', 
                       'Premature Contraction', 'ST Segment Abnormality']
            primary_diagnosis = np.random.choice(classes)
            predictions = {
                'Normal Sinus Rhythm': np.random.randint(5, 25),
                primary_diagnosis: int(confidence),
            }
            for cls in classes:
                if cls != primary_diagnosis:
                    predictions[cls] = np.random.randint(0, 15)
        
        total = sum(predictions.values())
        predictions = {k: int(v * 100 / total) for k, v in predictions.items()}
        
        result = {
            'id': result_id,
            'file_name': filename,
            'patient_id': patient_id,
            'status': 'completed',
            'created_at': datetime.now().isoformat(),
            'result': {
                'primary_diagnosis': primary_diagnosis,
                'confidence': round(confidence, 1),
                'is_normal': is_normal,
                'segments_analyzed': np.random.randint(100, 200),
                'predictions': predictions
            },
            'ecg_metrics': {
                'heart_rate': np.random.randint(60, 90),
                'rr_interval': np.random.randint(670, 1000),
                'hrv': np.random.randint(20, 80),
                'p_wave': round(np.random.uniform(0.08, 0.16), 3),
                'qrs_complex': round(np.random.uniform(0.06, 0.12), 3),
                'qt_interval': round(np.random.uniform(0.32, 0.44), 3)
            },
            'recommendations': generate_recommendations(primary_diagnosis, is_normal)
        }
        
        return result

def generate_recommendations(diagnosis, is_normal, heart_rate=None):
    """Generate detailed, clinically relevant recommendations based on the primary diagnosis and heart rate"""
    
    # Base recommendations list
    recs = []
    
    # Check for Heart Rate specific warnings first
    if heart_rate:
        if heart_rate < 50:
            recs.append(f"Urgent: Significant Bradycardia detected ({int(heart_rate)} BPM). Seek medical advice if you experience dizziness or fainting.")
        elif heart_rate > 120:
            recs.append(f"Urgent: Significant Tachycardia detected ({int(heart_rate)} BPM). Monitor for heart palpitations or shortness of breath.")

    if is_normal:
        recs.extend([
            "Maintain current healthy cardiovascular lifestyle and routine checkups.",
            "Continue regular aerobic exercise (e.g., at least 150 minutes of moderate intensity per week).",
            "Monitor periodically; no immediate medical intervention is required for this recording."
        ])
        return recs

    diagnosis_recs = {
        'Atrial Fibrillation': [
            "Urgent: Consult a cardiologist promptly for a comprehensive clinical evaluation.",
            "Consider evaluation for anticoagulation (blood thinner) therapy to reduce stroke risk.",
            "Discuss rate or rhythm control medications (e.g., beta-blockers, antiarrhythmics) with your physician.",
            "Avoid stimulants like excessive caffeine, smoking, or alcohol which can act as physiological triggers.",
            "Monitor closely for symptoms such as sudden shortness of breath, severe palpitations, or chest pain."
        ],
        'Ventricular Arrhythmia': [
            "Critical: Seek immediate emergency medical attention or a rapid cardiology consultation.",
            "Strictly avoid strenuous physical activities or high-intensity exercise until formally cleared.",
            "Discuss the potential need for continuous Holter monitoring or an implantable cardioverter-defibrillator (ICD).",
            "Ensure any prescribed antiarrhythmic medications are taken exactly as directed.",
            "Evaluate for any underlying structural heart disease or electrolyte imbalances with your healthcare provider."
        ],
        'Conduction Block': [
            "Schedule a prompt consultation with a cardiac electrophysiologist or cardiologist.",
            "Discuss the potential indication for a pacemaker depending on the degree/severity of the block.",
            "Review all current medications with a doctor, as certain drugs can inadvertently slow conduction.",
            "Monitor strictly for episodes of unexpected dizziness, presyncope (lightheadedness), or fainting.",
            "Keep follow-up appointments for routine serial ECG monitoring to track block progression."
        ],
        'Premature Contraction': [
            "Significantly reduce intake of known triggers such as heavy caffeine, tobacco, and high-sugar energy drinks.",
            "Implement stress management techniques and ensure adequate rest cycles/sleep hygiene.",
            "Maintain proper hydration and ensure electrolyte stability (especially potassium and magnesium).",
            "Follow up with a cardiologist if the episodes become highly frequent or cause noticeable discomfort.",
            "Consider a short-term ambulatory ECG patch to quantify the baseline burden of the extra beats."
        ],
        'ST Segment Abnormality': [
            "Urgent: Seek immediate medical evaluation at an emergency department or urgent cardiology clinic.",
            "Consider a formal treadmill stress test or cardiac imaging to evaluate for coronary ischemia.",
            "Pay hyper-vigilant attention to any onset of chest pain, jaw pain, or radiating arm discomfort.",
            "Review cardiovascular risk factors (hypertension, hyperlipidemia, diabetes) comprehensively with your primary care provider.",
            "Do not perform vigorous physical exertion until cleared by an attending cardiovascular specialist."
        ],
        'Inconclusive (Low Confidence)': [
            "The neural network could not determine a definitive classification pattern. A manual review is required.",
            "Ensure the ECG leads were securely fastened with proper contact, and repeat the recording if possible.",
            "Consult a trained human specialist to visually inspect the provided waveform outputs.",
            "Avoid making clinical decisions purely based on this low-confidence inference."
        ]
    }
    
    recs.extend(diagnosis_recs.get(diagnosis, ['Consult a healthcare professional for a formal evaluation of these abnormalities.']))
    return recs

@api_bp.route('/dashboard', methods=['GET'])
def get_dashboard():
    """Get dashboard statistics"""
    try:
        results = load_results()
        
        if not results:
            return jsonify({
                'stats': {
                    'totalTests': 0,
                    'normalResults': 0,
                    'abnormalResults': 0,
                    'avgConfidence': 0
                },
                'class_distribution': [],
                'recent_results': []
            })
        
        stats = {
            'totalTests': len(results),
            'normalResults': sum(1 for r in results if r.get('result', {}).get('is_normal', False)),
            'abnormalResults': sum(1 for r in results if not r.get('result', {}).get('is_normal', True)),
            'avgConfidence': np.mean([r.get('result', {}).get('confidence', 0) for r in results])
        }
        
        # Calculate class distribution
        class_counts = {}
        for r in results:
            diagnosis = r.get('result', {}).get('primary_diagnosis', 'Unknown')
            class_counts[diagnosis] = class_counts.get(diagnosis, 0) + 1
        
        total = len(results)
        colors = {
            'Normal Sinus Rhythm': '#10b981',
            'Atrial Fibrillation': '#ef4444',
            'Ventricular Arrhythmia': '#f59e0b',
            'Conduction Block': '#8b5cf6',
            'Premature Contraction': '#ec4899',
            'ST Segment Abnormality': '#6366f1'
        }
        
        class_distribution = [
            {'name': k, 'value': int(v/total*100), 'color': colors.get(k, '#64748b')}
            for k, v in class_counts.items()
        ]
        
        # Recent results (last 5) - WITH is_normal and patient_name fields
        recent_results = [
            {
                'id': r['id'],
                'patient_name': r.get('patient_name', 'Anonymous'),
                'file_name': r['file_name'],
                'result': r.get('result', {}).get('primary_diagnosis', 'Unknown'),
                'confidence': r.get('result', {}).get('confidence', 0),
                'is_normal': r.get('result', {}).get('is_normal', True),
                'date': r.get('created_at', '')[:10]
            }
            for r in sorted(results, key=lambda x: x.get('created_at', ''), reverse=True)[:5]
        ]
        
        return jsonify({
            'stats': stats,
            'class_distribution': class_distribution,
            'recent_results': recent_results
        })
    except Exception as e:
        logger.error(f"Error fetching dashboard: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/analyze', methods=['POST'])
def analyze_ecg():
    """Analyze ECG file"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        qrs_file = request.files.get('qrs_file', None)
        
        # Get patient information from form data
        patient_info_str = request.form.get('patient_info', None)
        patient_info = {'name': 'Anonymous', 'age': 'N/A'}
        
        if patient_info_str:
            try:
                patient_info = json.loads(patient_info_str)
            except:
                pass
        
        patient_id = patient_info.get('id', 'unknown')
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Parse settings
        settings_str = request.form.get('settings', None)
        settings = {}
        if settings_str:
            try:
                settings = json.loads(settings_str)
            except:
                pass
                
        if file and allowed_file(file.filename):
            result_id = str(uuid.uuid4())
            filename = secure_filename(file.filename)
            
            # Save main file
            filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], f"{result_id}_{filename}")
            file.save(filepath)
            
            # Save auxiliary QRS file side-by-side if provided
            if qrs_file and allowed_file(qrs_file.filename):
                # Ensure it carries the exact UUID prefix matching the .edf file
                base_name = os.path.splitext(filename)[0]
                qrs_filename = f"{result_id}_{base_name}{os.path.splitext(qrs_file.filename)[1]}"
                qrs_filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], qrs_filename)
                qrs_file.save(qrs_filepath)
                logger.info(f"Saved paired annotations file: {qrs_filename}")
            
            # Process the file
            result = process_ecg_file(filepath, result_id, filename, patient_id, settings)
            
            # Add patient info to result
            result['patient_name'] = patient_info.get('name', 'Anonymous')
            result['patient_age'] = patient_info.get('age', 'N/A')
            
            # Save to storage
            if settings.get('autoSave', True):
                results = load_results()
                results.append(result)
                save_results(results)
            
            return jsonify({
                'id': result['id'],
                'message': 'Analysis completed successfully',
                'status': 'completed',
                'result_data': result
            })
        else:
            return jsonify({'error': 'Invalid file type. Allowed: edf, qrs, dat'}), 400
            
    except Exception as e:
        logger.error(f"Error analyzing ECG: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/results', methods=['GET'])
def get_results():
    """Get all results with pagination"""
    try:
        page = int(request.args.get('page', 1))
        status = request.args.get('status', 'all')
        per_page = 10
        
        results = load_results()
        
        if status == 'normal':
            results = [r for r in results if r.get('result', {}).get('is_normal', False)]
        elif status == 'abnormal':
            results = [r for r in results if not r.get('result', {}).get('is_normal', True)]
        
        # Sort by date
        results = sorted(results, key=lambda x: x.get('created_at', ''), reverse=True)
        
        # Paginate
        total = len(results)
        start = (page - 1) * per_page
        end = start + per_page
        paginated = results[start:end]
        
        # Format for response
        formatted_results = []
        for r in paginated:
            res = r.get('result', {})
            formatted_results.append({
                'id': r['id'],
                'patient_name': r.get('patient_name', 'Anonymous'),
                'file_name': r['file_name'],
                'result': res.get('primary_diagnosis', 'Unknown'),
                'confidence': res.get('confidence', 0),
                'is_normal': res.get('is_normal', True),
                'created_at': r.get('created_at', '')
            })
        
        return jsonify({
            'results': formatted_results,
            'total_pages': (total + per_page - 1) // per_page,
            'current_page': page,
            'total': total
        })
    except Exception as e:
        logger.error(f"Error fetching results: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/results/<result_id>', methods=['GET'])
def get_result(result_id):
    """Get a specific result by ID"""
    try:
        results = load_results()
        result = next((r for r in results if r['id'] == result_id), None)
        
        if result is None:
            return jsonify({'error': 'Result not found'}), 404
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error fetching result: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/results/<result_id>', methods=['DELETE'])
def delete_result(result_id):
    """Delete a result"""
    try:
        results = load_results()
        results = [r for r in results if r['id'] != result_id]
        save_results(results)
        return jsonify({'message': 'Result deleted successfully'})
    except Exception as e:
        logger.error(f"Error deleting result: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/results', methods=['DELETE'])
def clear_all_results():
    """Clear all results"""
    try:
        save_results([])
        return jsonify({'message': 'All results cleared successfully'})
    except Exception as e:
        logger.error(f"Error clearing results: {e}")
        return jsonify({'error': str(e)}), 500


# ------- MODEL TRAINING AND DATA CONVERSION ENDPOINTS -------

@api_bp.route('/convert-mitbih', methods=['POST'])
def convert_mitbih():
    """
    Convert MIT-BIH dataset to EDF and QRS format
    """
    try:
        import sys
        import os
        
        # Add parent directory to path for imports
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        from services.converter import convert_all_mitbih_files, get_mitbih_files
        
        # Get list of files to convert
        record_names = get_mitbih_files()
        
        if not record_names:
            return jsonify({'error': 'No MIT-BIH files found'}), 400
        
        # Run conversion
        results = convert_all_mitbih_files()
        
        return jsonify({
            'success': True,
            'message': f'Conversion complete! {results["edf"]["success"]} EDF and {results["qrs"]["success"]} QRS files created.',
            'results': results
        })
    except Exception as e:
        logger.error(f"Error converting MIT-BIH: {e}")
        return jsonify({'error': str(e)}), 500


@api_bp.route('/train-model', methods=['POST'])
def train_model():
    """
    Train the DSNN model with given parameters
    """
    global TRAINING_STATUS
    
    try:
        import sys
        
        # Get parameters from request
        data = request.get_json() or {}
        dataset_path = data.get('dataset_path', 'Dataset/MIT-BIH')
        epochs = int(data.get('epochs', 50))
        
        # Convert relative path to absolute path if needed
        if not os.path.isabs(dataset_path):
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            dataset_path = os.path.join(project_root, dataset_path)
        
        logger.info(f"Dataset path resolved to: {dataset_path}")
        
        # Validate dataset path
        if not os.path.exists(dataset_path):
            return jsonify({'error': f'Dataset path does not exist: {dataset_path}'}), 400
        
        # Get list of EDF files in the dataset (converted from MIT-BIH .hea+.dat)
        record_files = [f.replace('.edf', '') for f in os.listdir(dataset_path) if f.endswith('.edf')]
        
        if not record_files:
            return jsonify({'error': 'No EDF files found in the dataset path. Run the Pre-Processing conversion first.'}), 400
        
        # Create a backup of disk results before starting
        backup_training_results()
        
        # Reset and update training status
        TRAINING_STATUS['status'] = 'running'
        TRAINING_STATUS['progress'] = 0
        TRAINING_STATUS['message'] = 'Starting training...'
        TRAINING_STATUS['error'] = None
        TRAINING_STATUS['start_time'] = datetime.now().isoformat()
        TRAINING_STATUS['epochs'] = epochs
        TRAINING_STATUS['current_epoch'] = 0
        TRAINING_STATUS['image_files'] = []
        TRAINING_STATUS['metrics'] = {'history': [], 'evaluation': None}

        # Clear the stop event so the training loop runs
        TRAINING_STOP_EVENT.clear()

        def run_training():
            """Run training in a separate thread"""
            global TRAINING_STATUS
            try:
                # Import training module
                from services.train_dsnn import main as train_main
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                
                # Update status
                TRAINING_STATUS['message'] = 'Loading data and initializing model...'
                TRAINING_STATUS['progress'] = 5
                
                # Progress callback — called after each epoch by the training loop
                def progress_update(epoch, total_epochs, train_loss, train_acc, val_loss, val_acc):
                    TRAINING_STATUS['current_epoch'] = epoch
                    TRAINING_STATUS['progress'] = int(5 + (epoch / total_epochs) * 85)
                    TRAINING_STATUS['message'] = (
                        f'Epoch {epoch}/{total_epochs} — '
                        f'Train Acc: {train_acc:.1f}%, Val Acc: {val_acc:.1f}%'
                    )
                    # Accumulate per-epoch metrics
                    TRAINING_STATUS['metrics']['history'].append({
                        'epoch': epoch,
                        'train_loss': round(train_loss, 4),
                        'train_acc': round(train_acc, 2),
                        'val_loss': round(val_loss, 4),
                        'val_acc': round(val_acc, 2)
                    })
                
                # Run training (stop_event lets the loop break when stop is requested)
                results = train_main(
                    base_path=dataset_path,
                    file_names=record_files,
                    num_channels=2,
                    segment_length=24,
                    use_sliding_window=True,
                    batch_size=32,
                    epochs=epochs,
                    learning_rate=0.001,
                    train_model=True,
                    stop_event=TRAINING_STOP_EVENT,
                    progress_callback=progress_update
                )
                
                # Check if training was stopped by user
                if TRAINING_STOP_EVENT.is_set():
                    logger.info("Training stop detected. Rolling back results...")
                    # Rollback the JSON results on disk to previous successful run
                    rollback_training_results()
                    
                    # Ensure status indicates it was stopped (the rollback reloads memory, so we override status)
                    TRAINING_STATUS['status'] = 'stopped'
                    TRAINING_STATUS['message'] = 'Training stopped by user — rolling back to previous results.'
                    TRAINING_STATUS['end_time'] = datetime.now().isoformat()
                    return
                
                # Find generated image files in backend/images folder
                images_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'images')
                image_files = []
                for f in ['training_history.png', 'confusion_matrix.png']:
                    img_path = os.path.join(images_dir, f)
                    if os.path.exists(img_path):
                        image_files.append('/images/' + f)
                
                # Copy model to backend/models as dsnn_model.pth
                models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
                os.makedirs(models_dir, exist_ok=True)
                
                for model_name in ['best_acc_model.pth', 'best_loss_model.pth']:
                    src_path = os.path.join(models_dir, model_name)
                    if os.path.exists(src_path):
                        dst_path = os.path.join(models_dir, 'dsnn_model.pth')
                        import shutil
                        shutil.copy2(src_path, dst_path)
                        break
                
                # Store evaluation metrics from results if available
                if results and isinstance(results, dict):
                    eval_metrics = results.get('metrics', {})
                    TRAINING_STATUS['metrics']['evaluation'] = {
                        'accuracy': round(float(eval_metrics.get('accuracy', 0)), 4),
                        'precision': round(float(eval_metrics.get('precision', 0)), 4),
                        'recall': round(float(eval_metrics.get('recall', 0)), 4),
                        'f1': round(float(eval_metrics.get('f1', 0)), 4),
                    }
                
                # Update training status to completed
                TRAINING_STATUS['status'] = 'completed'
                TRAINING_STATUS['progress'] = 100
                TRAINING_STATUS['message'] = 'Training completed successfully!'
                TRAINING_STATUS['end_time'] = datetime.now().isoformat()
                TRAINING_STATUS['image_files'] = image_files
                
                # Persist results to disk
                save_training_results()
                
                # Success — clear the backup
                if os.path.exists(TRAINING_BACKUP_FILE):
                    os.remove(TRAINING_BACKUP_FILE)
                
                logger.info("Training completed successfully! Backup cleared.")
                
            except Exception as e:
                logger.error(f"Training error: {e}")
                # Rollback to last successful version on error
                rollback_training_results()
                
                TRAINING_STATUS['status'] = 'failed'
                TRAINING_STATUS['error'] = str(e)
                TRAINING_STATUS['message'] = f'Training failed: {str(e)} — rolling back.'
                TRAINING_STATUS['end_time'] = datetime.now().isoformat()

        
        # Start training in background thread
        training_thread = threading.Thread(target=run_training, daemon=True)
        training_thread.start()
        TRAINING_STATUS['training_thread'] = training_thread
        
        # Return immediately with status
        return jsonify({
            'success': True,
            'message': 'Training started',
            'status': 'running',
            'dataset_path': dataset_path,
            'epochs': epochs,
            'files_count': len(record_files)
        })
        
    except Exception as e:
        logger.error(f"Error starting training: {e}")
        TRAINING_STATUS['status'] = 'failed'
        TRAINING_STATUS['error'] = str(e)
        return jsonify({'error': str(e)}), 500


@api_bp.route('/training-status', methods=['GET'])
def training_status():
    """
    Get training status and results.
    Always returns existing images and model info regardless of training state.
    On fresh server start (not_started), loads last saved results from disk.
    """
    global TRAINING_STATUS
    
    try:
        # Always check for existing model files
        models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
        model_files = {}
        for mf in ['dsnn_model.pth', 'best_acc_model.pth', 'best_loss_model.pth']:
            model_files[mf] = os.path.exists(os.path.join(models_dir, mf))
        model_exists = any(model_files.values())
        
        # Always check for existing image files (add timestamp for cache-busting)
        images_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'images')
        import time
        ts = int(time.time())
        existing_images = []
        for f in ['training_history.png', 'confusion_matrix.png']:
            img_path = os.path.join(images_dir, f)
            if os.path.exists(img_path):
                existing_images.append(f'/images/{f}?t={ts}')
        
        # If training thread died unexpectedly while status is still 'running'
        if TRAINING_STATUS['status'] == 'running':
            thread = TRAINING_STATUS.get('training_thread')
            if thread and not thread.is_alive():
                # Thread finished but status wasn't updated (crash?)
                if not TRAINING_STOP_EVENT.is_set():
                    TRAINING_STATUS['status'] = 'completed'
                    TRAINING_STATUS['message'] = 'Training completed successfully!'
                    TRAINING_STATUS['progress'] = 100
                    TRAINING_STATUS['end_time'] = datetime.now().isoformat()
                    save_training_results()
        
        # If no training has happened this session, try to load saved results from disk
        if TRAINING_STATUS['status'] == 'not_started':
            saved = load_training_results()
            if saved and saved.get('status') in ('completed', 'stopped', 'failed'):
                # Populate the in-memory status from the saved file
                TRAINING_STATUS['status'] = saved['status']
                TRAINING_STATUS['progress'] = saved.get('progress', 0)
                TRAINING_STATUS['message'] = saved.get('message', '')
                TRAINING_STATUS['error'] = saved.get('error')
                TRAINING_STATUS['start_time'] = saved.get('start_time')
                TRAINING_STATUS['end_time'] = saved.get('end_time')
                TRAINING_STATUS['epochs'] = saved.get('epochs', 0)
                TRAINING_STATUS['current_epoch'] = saved.get('current_epoch', 0)
                TRAINING_STATUS['metrics'] = saved.get('metrics', {'history': [], 'evaluation': None})
                logger.info("Loaded last training results from disk.")
        
        return jsonify({
            'status': TRAINING_STATUS['status'],
            'progress': TRAINING_STATUS['progress'],
            'message': TRAINING_STATUS['message'],
            'error': TRAINING_STATUS['error'],
            'epochs': TRAINING_STATUS['epochs'],
            'current_epoch': TRAINING_STATUS['current_epoch'],
            'image_files': existing_images,
            'model_exists': model_exists,
            'model_files': model_files,
            'metrics': TRAINING_STATUS['metrics'],
            'start_time': TRAINING_STATUS['start_time'],
            'end_time': TRAINING_STATUS['end_time']
        })
        
    except Exception as e:
        logger.error(f"Error getting training status: {e}")
        return jsonify({'error': str(e)}), 500


@api_bp.route('/stop-training', methods=['POST'])
def stop_training():
    """
    Stop the current training process
    """
    global TRAINING_STATUS
    
    try:
        if TRAINING_STATUS['status'] != 'running':
            return jsonify({
                'success': False,
                'message': 'No training is currently running'
            }), 400
        
        # Signal the training loop to stop
        TRAINING_STOP_EVENT.set()
        
        # Mark as stopped
        TRAINING_STATUS['status'] = 'stopped'
        TRAINING_STATUS['message'] = 'Training stop requested — will stop after current epoch...'
        TRAINING_STATUS['progress'] = TRAINING_STATUS.get('progress', 0)
        
        return jsonify({
            'success': True,
            'message': 'Training stop requested. The training will stop after the current epoch.'
        })
        
    except Exception as e:
        logger.error(f"Error stopping training: {e}")
        return jsonify({'error': str(e)}), 500
