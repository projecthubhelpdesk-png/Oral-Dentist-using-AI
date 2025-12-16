"""
Integration Example - Connecting ML Model to Oral Care AI Backend
================================================================

This script shows how to integrate the ML model with the PHP backend.
The PHP backend can call this service via HTTP or you can run it as a subprocess.

Option 1: HTTP API (Recommended)
    - Run inference_api.py as a separate service
    - PHP calls http://localhost:8000/predict
    
Option 2: Direct Python Call
    - PHP uses exec() to call this script
    - Pass image path as argument
"""

import sys
import json
import base64
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from predict import predict_oral_disease, OralDiseasePredictor

def analyze_from_path(image_path: str) -> dict:
    """
    Analyze image from file path.
    Called by PHP: exec("python integration_example.py path /path/to/image.jpg")
    """
    result = predict_oral_disease(image_path)
    return result

def analyze_from_base64(base64_data: str) -> dict:
    """
    Analyze image from base64 encoded data.
    Called by PHP: exec("python integration_example.py base64 <base64_string>")
    """
    import base64
    image_bytes = base64.b64decode(base64_data)
    result = predict_oral_disease(image_bytes)
    return result

def format_for_php(result: dict) -> dict:
    """
    Format result for PHP backend consumption.
    Matches the expected format in AIAnalysisService.php
    """
    # Map disease to finding type
    disease_mapping = {
        'Calculus': 'calculus_buildup',
        'Caries': 'tooth_decay',
        'Gingivitis': 'gum_inflammation',
        'Hypodontia': 'missing_teeth',
        'Mouth Ulcer': 'oral_ulcer',
        'Tooth Discoloration': 'discoloration'
    }
    
    # Map severity
    severity_mapping = {
        'Low': 'mild',
        'Medium': 'moderate',
        'High': 'severe'
    }
    
    finding_type = disease_mapping.get(result['disease'], 'unknown')
    severity = severity_mapping.get(result['severity'], 'moderate')
    
    # Build findings array
    findings = [{
        'type': finding_type,
        'severity': severity,
        'location': 'detected_region',
        'confidence': result['confidence'],
        'description': f"AI detected {result['disease']} with {result['confidence']*100:.1f}% confidence"
    }]
    
    # Add secondary findings if confidence is high enough
    for pred in result['all_predictions'][1:3]:
        if pred['confidence'] > 0.1:
            findings.append({
                'type': disease_mapping.get(pred['disease'], 'unknown'),
                'severity': 'mild',
                'location': 'secondary_region',
                'confidence': pred['confidence'],
                'description': f"Possible {pred['disease']}"
            })
    
    # Calculate overall score (inverse of severity)
    score_mapping = {'Low': 85, 'Medium': 65, 'High': 45}
    overall_score = score_mapping.get(result['severity'], 70)
    
    # Adjust score based on confidence
    overall_score = overall_score + (1 - result['confidence']) * 20
    
    # Build recommendations
    recommendations = get_recommendations(result['disease'], result['severity'])
    
    return {
        'overall_score': round(overall_score, 2),
        'confidence_score': result['confidence'],
        'findings': findings,
        'risk_areas': [{
            'x': 100,
            'y': 100,
            'width': 200,
            'height': 150,
            'label': finding_type
        }],
        'recommendations': recommendations,
        'model_type': 'oral_disease_rgb',
        'model_version': '1.0.0',
        'disclaimer': result['disclaimer']
    }

def get_recommendations(disease: str, severity: str) -> list:
    """Get recommendations based on disease and severity."""
    recommendations = {
        'Calculus': [
            'Schedule professional dental cleaning',
            'Improve daily brushing technique',
            'Use tartar-control toothpaste',
            'Floss daily to prevent buildup'
        ],
        'Caries': [
            'Visit dentist immediately for treatment',
            'Reduce sugar intake',
            'Use fluoride toothpaste',
            'Consider dental sealants'
        ],
        'Gingivitis': [
            'Brush teeth twice daily with soft brush',
            'Floss daily',
            'Use antiseptic mouthwash',
            'Schedule dental checkup'
        ],
        'Hypodontia': [
            'Consult with orthodontist',
            'Discuss treatment options',
            'Regular dental monitoring'
        ],
        'Mouth Ulcer': [
            'Avoid spicy and acidic foods',
            'Use saltwater rinse',
            'Apply topical treatments',
            'Consult doctor if persists over 2 weeks'
        ],
        'Tooth Discoloration': [
            'Professional teeth cleaning',
            'Reduce staining foods/drinks',
            'Consider whitening treatments',
            'Maintain good oral hygiene'
        ]
    }
    
    base_recs = recommendations.get(disease, ['Consult a dental professional'])
    
    if severity == 'High':
        base_recs.insert(0, '⚠️ URGENT: Schedule dental appointment immediately')
    
    return base_recs

# ============================================
# CLI INTERFACE
# ============================================
if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(json.dumps({
            'error': 'Usage: python integration_example.py <mode> <data>',
            'modes': ['path', 'base64']
        }))
        sys.exit(1)
    
    mode = sys.argv[1]
    data = sys.argv[2]
    
    try:
        if mode == 'path':
            result = analyze_from_path(data)
        elif mode == 'base64':
            result = analyze_from_base64(data)
        else:
            print(json.dumps({'error': f'Unknown mode: {mode}'}))
            sys.exit(1)
        
        # Format for PHP
        formatted = format_for_php(result)
        
        # Output JSON (PHP will capture this)
        print(json.dumps(formatted, indent=2))
        
    except Exception as e:
        print(json.dumps({
            'error': str(e),
            'success': False
        }))
        sys.exit(1)
