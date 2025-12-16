"""
Oral Disease Detection - Flask Inference Server (Alternative)
=============================================================
REST API for oral disease prediction using Flask.

DISCLAIMER: This AI provides preliminary screening only and is not a medical diagnosis.
Always consult a qualified dental professional for proper diagnosis and treatment.

Usage:
    python inference_flask.py
    
    # Or with gunicorn for production:
    gunicorn -w 4 -b 0.0.0.0:8000 inference_flask:app
"""

import os
import io
import json
import logging
from datetime import datetime
from pathlib import Path

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

from predict import OralDiseasePredictor, DISCLAIMER

# ============================================
# LOGGING SETUP
# ============================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================
# FLASK APP
# ============================================
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp'}

# Global predictor
predictor = None

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model():
    """Load the prediction model."""
    global predictor
    try:
        logger.info("Loading oral disease detection model...")
        predictor = OralDiseasePredictor()
        logger.info("Model loaded successfully!")
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False

# ============================================
# API ENDPOINTS
# ============================================

@app.route('/', methods=['GET'])
def root():
    """Root endpoint - API information."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor is not None,
        'version': '1.0.0',
        'endpoints': {
            'predict': 'POST /predict',
            'health': 'GET /health',
            'classes': 'GET /classes'
        },
        'disclaimer': DISCLAIMER
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy' if predictor else 'degraded',
        'model_loaded': predictor is not None,
        'version': '1.0.0',
        'disclaimer': DISCLAIMER
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict oral disease from uploaded image.
    
    Request:
        - Form data with 'file' field containing image
        
    Response:
        {
            "success": true,
            "disease": "Gingivitis",
            "confidence": 0.87,
            "severity": "High",
            "all_predictions": [...],
            "disclaimer": "...",
            "timestamp": "..."
        }
    """
    # Check if model is loaded
    if predictor is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded. Please try again later.'
        }), 503
    
    # Check if file is present
    if 'file' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No file provided. Please upload an image.'
        }), 400
    
    file = request.files['file']
    
    # Check if file is selected
    if file.filename == '':
        return jsonify({
            'success': False,
            'error': 'No file selected.'
        }), 400
    
    # Validate file type
    if not allowed_file(file.filename):
        return jsonify({
            'success': False,
            'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'
        }), 400
    
    try:
        # Read file contents
        contents = file.read()
        logger.info(f"Processing image: {file.filename} ({len(contents)} bytes)")
        
        # Run prediction
        result = predictor.predict_from_bytes(contents)
        
        # Return response
        return jsonify({
            'success': True,
            'disease': result['disease'],
            'confidence': result['confidence'],
            'severity': result['severity'],
            'all_predictions': result['all_predictions'],
            'disclaimer': DISCLAIMER,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({
            'success': False,
            'error': f'Prediction failed: {str(e)}'
        }), 500

@app.route('/classes', methods=['GET'])
def get_classes():
    """Get list of detectable disease classes."""
    if predictor is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded'
        }), 503
    
    return jsonify({
        'success': True,
        'classes': list(predictor.class_labels.values()),
        'count': predictor.num_classes
    })

# ============================================
# ERROR HANDLERS
# ============================================

@app.errorhandler(413)
def too_large(e):
    return jsonify({
        'success': False,
        'error': 'File too large. Maximum size is 10MB.'
    }), 413

@app.errorhandler(500)
def server_error(e):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

# ============================================
# MAIN
# ============================================
if __name__ == '__main__':
    # Load model on startup
    load_model()
    
    # Run server
    app.run(
        host='0.0.0.0',
        port=8000,
        debug=True
    )
