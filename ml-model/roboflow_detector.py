"""
Roboflow API Dental Detection
=============================

Uses Roboflow's hosted inference API for dental detection.
No local training required - uses pre-trained models.

Setup:
1. Get free API key from https://app.roboflow.com/
2. Set ROBOFLOW_API_KEY environment variable or pass to constructor
"""

import os
import json
import base64
import requests
from pathlib import Path
from typing import Dict, Union, List
from PIL import Image
import io

class RoboflowDentalDetector:
    """
    Dental detection using Roboflow's hosted API.
    Uses pre-trained dental models for accurate detection.
    """
    
    # Public dental models on Roboflow (no API key needed for some)
    PUBLIC_MODELS = {
        'dental-disease': {
            'workspace': 'dental-disease-detection',
            'project': 'dental-disease-detection',
            'version': 1,
            'classes': ['cavity', 'calculus', 'gingivitis', 'healthy']
        },
        'teeth-detection': {
            'workspace': 'teeth-detection',
            'project': 'teeth-detection',
            'version': 1,
            'classes': ['tooth', 'cavity', 'filling']
        }
    }
    
    def __init__(self, api_key: str = None, model_id: str = None):
        """
        Initialize Roboflow detector.
        
        Args:
            api_key: Roboflow API key (get free at roboflow.com)
            model_id: Specific model ID to use (workspace/project/version)
        """
        self.api_key = api_key or os.getenv('ROBOFLOW_API_KEY')
        self.model_id = model_id
        self.base_url = "https://detect.roboflow.com"
        
        if not self.api_key:
            print("⚠️ No Roboflow API key provided.")
            print("Get free key at: https://app.roboflow.com/")
            print("Set via: ROBOFLOW_API_KEY environment variable")
    
    def detect(self, image: Union[str, bytes, Image.Image], 
               confidence: float = 0.25) -> Dict:
        """
        Detect dental conditions in image.
        
        Args:
            image: Image path, bytes, or PIL Image
            confidence: Minimum confidence threshold
            
        Returns:
            Detection results with bounding boxes and classes
        """
        if not self.api_key:
            return self._mock_detection(image)
        
        # Convert image to base64
        image_b64 = self._image_to_base64(image)
        
        # Build API URL
        # Format: https://detect.roboflow.com/{workspace}/{project}/{version}
        if self.model_id:
            url = f"{self.base_url}/{self.model_id}"
        else:
            # Use default dental model
            url = f"{self.base_url}/dental-disease-detection/1"
        
        # Make request
        params = {
            'api_key': self.api_key,
            'confidence': confidence,
        }
        
        try:
            response = requests.post(
                url,
                params=params,
                data=image_b64,
                headers={'Content-Type': 'application/x-www-form-urlencoded'}
            )
            
            if response.status_code == 200:
                return self._process_response(response.json())
            else:
                print(f"API Error: {response.status_code} - {response.text}")
                return self._mock_detection(image)
                
        except Exception as e:
            print(f"Request failed: {e}")
            return self._mock_detection(image)
    
    def _image_to_base64(self, image: Union[str, bytes, Image.Image]) -> str:
        """Convert image to base64 string."""
        if isinstance(image, str):
            with open(image, 'rb') as f:
                image_bytes = f.read()
        elif isinstance(image, bytes):
            image_bytes = image
        elif isinstance(image, Image.Image):
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG')
            image_bytes = buffer.getvalue()
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        return base64.b64encode(image_bytes).decode('utf-8')
    
    def _process_response(self, response: Dict) -> Dict:
        """Process Roboflow API response."""
        predictions = response.get('predictions', [])
        
        detections = []
        condition_counts = {}
        
        for pred in predictions:
            class_name = pred.get('class', 'unknown')
            confidence = pred.get('confidence', 0)
            
            # Map to our condition names
            condition = self._map_class(class_name)
            
            detections.append({
                'class': condition,
                'confidence': confidence,
                'bbox': {
                    'x': pred.get('x', 0),
                    'y': pred.get('y', 0),
                    'width': pred.get('width', 0),
                    'height': pred.get('height', 0)
                },
                'original_class': class_name
            })
            
            # Count conditions
            if condition not in condition_counts:
                condition_counts[condition] = 0
            condition_counts[condition] += 1
        
        # Determine primary condition
        if detections:
            primary = max(detections, key=lambda x: x['confidence'])
            primary_condition = primary['class']
            primary_confidence = primary['confidence']
        else:
            primary_condition = 'healthy'
            primary_confidence = 0.8
        
        # Calculate severity
        severity = self._calculate_severity(detections, condition_counts)
        
        # Calculate health score
        health_score = self._calculate_health_score(detections, condition_counts)
        
        return {
            'success': True,
            'primary_condition': primary_condition,
            'confidence': primary_confidence,
            'severity': severity,
            'health_score': health_score,
            'detections': detections,
            'condition_counts': condition_counts,
            'total_issues': len([d for d in detections if d['class'] != 'healthy'])
        }
    
    def _map_class(self, class_name: str) -> str:
        """Map Roboflow class names to our standard names."""
        mapping = {
            'cavity': 'Caries',
            'caries': 'Caries',
            'decay': 'Caries',
            'calculus': 'Calculus',
            'tartar': 'Calculus',
            'plaque': 'Calculus',
            'gingivitis': 'Gingivitis',
            'gum_disease': 'Gingivitis',
            'inflammation': 'Gingivitis',
            'healthy': 'Healthy',
            'tooth': 'Healthy',
            'normal': 'Healthy',
            'ulcer': 'Mouth_Ulcer',
            'mouth_ulcer': 'Mouth_Ulcer',
        }
        return mapping.get(class_name.lower(), class_name)
    
    def _calculate_severity(self, detections: List, counts: Dict) -> str:
        """Calculate overall severity."""
        if not detections:
            return 'Low'
        
        # Count serious conditions
        serious = counts.get('Caries', 0) + counts.get('Calculus', 0)
        moderate = counts.get('Gingivitis', 0)
        
        # Check max confidence
        max_conf = max(d['confidence'] for d in detections) if detections else 0
        
        if serious >= 3 or (serious >= 1 and max_conf > 0.8):
            return 'High'
        elif serious >= 1 or moderate >= 2 or max_conf > 0.6:
            return 'Medium'
        else:
            return 'Low'
    
    def _calculate_health_score(self, detections: List, counts: Dict) -> float:
        """Calculate health score (0-100)."""
        if not detections:
            return 85.0  # No detections = likely healthy
        
        # Start with base score
        score = 100.0
        
        # Deduct for each condition
        deductions = {
            'Caries': 25,
            'Calculus': 20,
            'Gingivitis': 15,
            'Mouth_Ulcer': 10,
        }
        
        for condition, count in counts.items():
            if condition in deductions:
                # Deduct more for multiple instances
                score -= deductions[condition] * min(count, 3)
        
        # Deduct based on confidence
        for det in detections:
            if det['class'] != 'Healthy':
                score -= det['confidence'] * 5
        
        return max(5.0, min(100.0, score))
    
    def _mock_detection(self, image) -> Dict:
        """Mock detection when API is not available."""
        print("⚠️ Using mock detection (no API key)")
        
        # Use visual analyzer as fallback
        try:
            from visual_analyzer import VisualDentalAnalyzer
            analyzer = VisualDentalAnalyzer()
            visual_result = analyzer.analyze(image)
            
            return {
                'success': True,
                'primary_condition': visual_result['primary_condition'],
                'confidence': visual_result['confidence'],
                'severity': visual_result['severity'],
                'health_score': visual_result['overall_health_score'],
                'detections': [],
                'condition_counts': {},
                'total_issues': 0,
                'note': 'Using visual analysis (no Roboflow API key)'
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'No API key and visual analysis failed: {e}',
                'primary_condition': 'Unknown',
                'confidence': 0,
                'severity': 'Unknown',
                'health_score': 50
            }


def test_roboflow_detection(image_path: str = None, api_key: str = None):
    """Test Roboflow detection."""
    print("🦷 Testing Roboflow Dental Detection")
    print("=" * 50)
    
    # Find test image
    if not image_path:
        test_paths = [
            '../backend-php/storage/scans/sample1.jpg',
            'test_image.jpg',
        ]
        for p in test_paths:
            if Path(p).exists():
                image_path = p
                break
    
    if not image_path or not Path(image_path).exists():
        print("❌ No test image found")
        return
    
    print(f"Image: {image_path}")
    
    # Create detector
    detector = RoboflowDentalDetector(api_key=api_key)
    
    # Run detection
    result = detector.detect(image_path)
    
    print(f"\n📊 Results:")
    print(f"   Primary: {result['primary_condition']}")
    print(f"   Confidence: {result['confidence']:.1%}")
    print(f"   Severity: {result['severity']}")
    print(f"   Health Score: {result['health_score']:.1f}")
    print(f"   Total Issues: {result.get('total_issues', 0)}")
    
    if result.get('detections'):
        print(f"\n🔍 Detections:")
        for det in result['detections'][:5]:
            print(f"   - {det['class']}: {det['confidence']:.1%}")
    
    return result


if __name__ == '__main__':
    import sys
    
    api_key = sys.argv[1] if len(sys.argv) > 1 else None
    image_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    test_roboflow_detection(image_path, api_key)
