"""
Web Routes for Arrhythmia Detection
Serves the React frontend
"""
import os
import logging
from flask import Blueprint, send_from_directory, render_template_string

logger = logging.getLogger(__name__)

web_bp = Blueprint('web', __name__)

# Get the frontend build path
FRONTEND_BUILD_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'frontend', 'dist')

@web_bp.route('/')
def index():
    """Serve the main React app"""
    # Check if frontend is built
    index_path = os.path.join(FRONTEND_BUILD_PATH, 'index.html')
    
    if os.path.exists(index_path):
        return send_from_directory(FRONTEND_BUILD_PATH, 'index.html')
    else:
        # Return a simple placeholder during development
        return """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Arrhythmia Detection - Web App</title>
            <style>
                body {
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    min-height: 100vh;
                    margin: 0;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                }
                .container {
                    text-align: center;
                    color: white;
                    padding: 40px;
                }
                h1 { font-size: 2.5rem; margin-bottom: 1rem; }
                p { font-size: 1.2rem; opacity: 0.9; }
                .btn {
                    display: inline-block;
                    margin-top: 20px;
                    padding: 12px 24px;
                    background: white;
                    color: #667eea;
                    text-decoration: none;
                    border-radius: 8px;
                    font-weight: 600;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🏥 Arrhythmia Detection</h1>
                <p>Early Detection of Arrhythmia using Deep Spiking Neural Network</p>
                <p>API Server Running - Build frontend for full experience</p>
            </div>
        </body>
        </html>
        """

@web_bp.route('/<path:filename>')
def serve_static(filename):
    """Serve static files from frontend build"""
    return send_from_directory(FRONTEND_BUILD_PATH, filename)


