"""
Arrhythmia Detection Web Application - Backend API
Flask Application Entry Point
"""
import os
import sys
import logging
from datetime import datetime
from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_cors import CORS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'arrhythmia-detection-secret-key-2024')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['RESULTS_FOLDER'] = os.path.join(os.path.dirname(__file__), 'results')
app.config['IMAGES_FOLDER'] = os.path.join(os.path.dirname(__file__), 'images')

# Enable CORS
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
os.makedirs(app.config['IMAGES_FOLDER'], exist_ok=True)

# Import routes
from routes.api import api_bp
from routes.web import web_bp

# Register Blueprints
app.register_blueprint(web_bp, url_prefix='/')
app.register_blueprint(api_bp, url_prefix='/api')

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

# Route to serve training images
@app.route('/images/<filename>')
def serve_image(filename):
    """Serve training visualization images"""
    return send_from_directory(app.config['IMAGES_FOLDER'], filename)

if __name__ == '__main__':
    logger.info("Starting Arrhythmia Detection Web Application...")
    logger.info(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    logger.info(f"Results folder: {app.config['RESULTS_FOLDER']}")
    
    # Run the app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )


