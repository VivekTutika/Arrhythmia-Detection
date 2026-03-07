"""
API Routes for Arrhythmia Detection
"""
import os
import uuid
import logging
from datetime import datetime
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

api_bp = Blueprint('api', __name__)

# Mock results storage (in production, use a database)
results_db = []

# Allowed file extensions
ALLOWED_EXTENSIONS = {'edf', 'qrs'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@api_bp.route('/dashboard', methods=['GET'])
def get_dashboard():
    """Get dashboard statistics"""
    try:
        # In production, fetch from database
        stats = {
            'total_tests': len(results_db),
            'normal_results': sum(1 for r in results_db if r.get('result', {}).get('is_normal', False)),
            'abnormal_results': sum(1 for r in results_db if not r.get('result', {}).get('is_normal', True)),
            'avg_confidence': np.mean([r.get('result', {}).get('confidence', 0) for r in results_db]) if results_db else 0
        }
        
        # Mock data if empty
        if stats['total_tests'] == 0:
            stats = {
                'totalTests': 24,
                'normalResults': 18,
                'abnormalResults': 6,
                'avgConfidence': 87.5
            }
        
        # Class distribution
        class_distribution = [
            {'name': 'Normal', 'value': 75, 'color': '#10b981'},
            {'name': 'Atrial Fibrillation', 'value': 10, 'color': '#ef4444'},
            {'name': 'Ventricular', 'value': 5, 'color': '#f59e0b'},
            {'name': 'Other', 'value': 10, 'color': '#64748b'},
        ]
        
        # Recent results
        recent_results = [
            {'id': 1, 'file_name': 'patient_001.edf', 'result': 'Normal', 'confidence': 92.5, 'date': '2024-01-15'},
            {'id': 2, 'file_name': 'patient_002.edf', 'result': 'Atrial Fibrillation', 'confidence': 88.3, 'date': '2024-01-14'},
            {'id': 3, 'file_name': 'patient_003.edf', 'result': 'Normal', 'confidence': 95.1, 'date': '2024-01-14'},
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
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        patient_id = request.form.get('patient_id', 'unknown')
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            # Generate unique ID
            result_id = str(uuid.uuid4())
            filename = secure_filename(file.filename)
            
            # Save file
            filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], f"{result_id}_{filename}")
            file.save(filepath)
            
            # Process the file (placeholder - integrate with DSNN model)
            result = process_ecg_file(filepath, result_id, filename, patient_id)
            
            # Save to results
            results_db.append(result)
            
            return jsonify({
                'id': result['id'],
                'message': 'Analysis completed successfully',
                'status': 'completed'
            })
        else:
            return jsonify({'error': 'Invalid file type'}), 400
            
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
        
        filtered_results = results_db.copy()
        
        if status == 'normal':
            filtered_results = [r for r in filtered_results if r.get('result', {}).get('is_normal', False)]
        elif status == 'abnormal':
            filtered_results = [r for r in filtered_results if not r.get('result', {}).get('is_normal', True)]
        
        # Mock data if empty
        if len(filtered_results) == 0:
            filtered_results = [
                {'id': 1, 'file_name': 'patient_001.edf', 'result': {'primary_diagnosis': 'Normal Sinus Rhythm', 'is_normal': True, 'confidence': 92.5}, 'created_at': '2024-01-15T10:30:00Z'},
                {'id': 2, 'file_name': 'patient_002.edf', 'result': {'primary_diagnosis': 'Atrial Fibrillation', 'is_normal': False, 'confidence': 88.3}, 'created_at': '2024-01-14T14:20:00Z'},
                {'id': 3, 'file_name': 'patient_003.edf', 'result': {'primary_diagnosis': 'Normal Sinus Rhythm', 'is_normal': True, 'confidence': 95.1}, 'created_at': '2024-01-14T09:15:00Z'},
                {'id': 4, 'file_name': 'patient_004.edf', 'result': {'primary_diagnosis': 'Ventricular Arrhythmia', 'is_normal': False, 'confidence': 78.9}, 'created_at': '2024-01-13T16:45:00Z'},
                {'id': 5, 'file_name': 'patient_005.edf', 'result': {'primary_diagnosis': 'Conduction Block', 'is_normal': False, 'confidence': 84.2}, 'created_at': '2024-01-12T11:00:00Z'},
            ]
        
        # Format for response
        formatted_results = []
        for r in filtered_results:
            res = r.get('result', {})
            formatted_results.append({
                'id': r['id'],
                'file_name': r['file_name'],
                'result': res.get('primary_diagnosis', 'Unknown'),
                'confidence': res.get('confidence', 0),
                'is_normal': res.get('is_normal', True),
                'created_at': r.get('created_at', datetime.now().isoformat())
            })
        
        return jsonify({
            'results': formatted_results,
            'total_pages': 3,
            'current_page': page
        })
    except Exception as e:
        logger.error(f"Error fetching results: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/results/<result_id>', methods=['GET'])
def get_result(result_id):
    """Get a specific result by ID"""
    try:
        # Try to find in database
        result = next((r for r in results_db if r['id'] == result_id), None)
        
        # Mock data if not found
        if result is None:
            result = {
                'id': result_id,
                'file_name': 'patient_sample.edf',
                'status': 'completed',
                'created_at': '2024-01-15T10:30:00Z',
                'result': {
                    'primary_diagnosis': 'Normal Sinus Rhythm',
                    'confidence': 92.5,
                    'is_normal': True,
                    'segments_analyzed': 156,
                    'predictions': {
                        'Normal Sinus Rhythm': 85,
                        'Atrial Fibrillation': 5,
                        'Ventricular Arrhythmia': 3,
                        'Conduction Block': 2,
                        'Premature Contraction': 3,
                        'ST Segment Abnormality': 2
                    }
                },
                'ecg_metrics': {
                    'heart_rate': 72,
                    'rr_interval': 833,
                    'hrv': 45,
                    'p_wave': 0.12,
                    'qrs_complex': 0.08,
                    'qt_interval': 0.38
                },
                'recommendations': [
                    'Continue regular cardiac checkups',
                    'Maintain healthy lifestyle',
                    'No immediate medical intervention required'
                ]
            }
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error fetching result: {e}")
        return jsonify({'error': str(e)}), 500

@api_bp.route('/results/<result_id>', methods=['DELETE'])
def delete_result(result_id):
    """Delete a result"""
    try:
        global results_db
        results_db = [r for r in results_db if r['id'] != result_id]
        return jsonify({'message': 'Result deleted successfully'})
    except Exception as e:
        logger.error(f"Error deleting result: {e}")
        return jsonify({'error': str(e)}), 500

def process_ecg_file(filepath, result_id, filename, patient_id):
    """
    Process ECG file using DSNN model
    Placeholder for actual model inference
    """
    logger.info(f"Processing ECG file: {filename}")
    
    # This is a placeholder - integrate with your actual DSNN model
    # In production, you would:
    # 1. Load the EDF file
    # 2. Extract segments
    # 3. Run through DSNN model
    # 4. Generate predictions
    
    # Mock prediction result
    np.random.seed(hash(result_id) % 2**32)
    confidence = np.random.uniform(75, 98)
    is_normal = confidence > 60
    
    # Class distribution
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
        # Fill remaining
        for cls in classes:
            if cls != primary_diagnosis:
                predictions[cls] = np.random.randint(0, 15)
    
    # Normalize predictions to sum to 100
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

def generate_recommendations(diagnosis, is_normal):
    """Generate recommendations based on diagnosis"""
    if is_normal:
        return [
            'Continue regular cardiac checkups',
            'Maintain healthy lifestyle',
            'No immediate medical intervention required'
        ]
    else:
        recommendations = {
            'Atrial Fibrillation': [
                'Consult a cardiologist promptly',
                'Consider anticoagulation therapy',
                'Monitor heart rate regularly'
            ],
            'Ventricular Arrhythmia': [
                'Seek immediate medical attention',
                'Avoid strenuous activities',
                'Consider wearable cardiac monitor'
            ],
            'Conduction Block': [
                'Consult a cardiologist',
                'Consider pacemaker evaluation',
                'Regular monitoring required'
            ],
            'Premature Contraction': [
                'Reduce caffeine and alcohol intake',
                'Manage stress levels',
                'Follow up with cardiologist if frequent'
            ],
            'ST Segment Abnormality': [
                'Seek immediate medical evaluation',
                'Consider stress test',
                'Monitor for chest pain'
            ]
        }
        return recommendations.get(diagnosis, ['Consult a healthcare professional'])


