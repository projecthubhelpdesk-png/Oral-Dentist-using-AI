"""
Roboflow Dental Analysis Integration
====================================
Uses Roboflow's trained model for detecting gums, teeth, and plaques.
This provides more accurate detection than local models.
"""

import requests
import base64
import io
import json
from typing import Dict, Union, List, Optional
from pathlib import Path
from PIL import Image
import numpy as np


class RoboflowDentalAnalyzer:
    """
    Integrates with Roboflow API for dental analysis.
    Detects: gums, teeth, plaques, and other dental conditions.
    """
    
    def __init__(self, api_key: str = "2XzxgkyUiHDNIIHz6NO5"):
        self.api_key = api_key
        self.workflow_url = "https://serverless.roboflow.com/aman-t9quf/workflows/find-gums-teeth-and-plaques"
        
    def analyze(self, image: Union[str, bytes, Image.Image, np.ndarray]) -> Dict:
        """
        Analyze dental image using Roboflow API.
        
        Args:
            image: Image as file path, bytes, PIL Image, or numpy array
            
        Returns:
            Dict with detection results and health assessment
        """
        # Convert image to base64
        image_base64 = self._image_to_base64(image)
        
        # Call Roboflow API
        try:
            response = requests.post(
                self.workflow_url,
                headers={'Content-Type': 'application/json'},
                json={
                    'api_key': self.api_key,
                    'inputs': {
                        'image': {
                            'type': 'base64',
                            'value': image_base64
                        }
                    }
                },
                timeout=30
            )
            
            if response.status_code != 200:
                print(f"Roboflow API error: {response.status_code} - {response.text}")
                return self._fallback_result("API error")
            
            result = response.json()
            return self._process_roboflow_result(result)
            
        except requests.exceptions.Timeout:
            print("Roboflow API timeout")
            return self._fallback_result("API timeout")
        except Exception as e:
            print(f"Roboflow API error: {e}")
            return self._fallback_result(str(e))
    
    def _image_to_base64(self, image: Union[str, bytes, Image.Image, np.ndarray]) -> str:
        """Convert image to base64 string."""
        if isinstance(image, (str, Path)):
            with open(image, 'rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')
        elif isinstance(image, bytes):
            return base64.b64encode(image).decode('utf-8')
        elif isinstance(image, Image.Image):
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG')
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(image)
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG')
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
    
    def _process_roboflow_result(self, result: Dict) -> Dict:
        """Process Roboflow API response into our format."""
        detections = []
        plaque_count = 0
        teeth_count = 0
        gum_issues = 0
        total_plaque_area = 0
        total_teeth_area = 0
        
        # Parse detections from Roboflow response
        outputs = result.get('outputs', [])
        
        for output in outputs:
            predictions = output.get('predictions', [])
            if isinstance(predictions, dict):
                predictions = predictions.get('predictions', [])
            
            for pred in predictions:
                class_name = pred.get('class', '').lower()
                confidence = pred.get('confidence', 0)
                x = pred.get('x', 0)
                y = pred.get('y', 0)
                width = pred.get('width', 0)
                height = pred.get('height', 0)
                
                detection = {
                    'class': class_name,
                    'confidence': confidence,
                    'bbox': [x - width/2, y - height/2, width, height],
                    'area': width * height
                }
                detections.append(detection)
                
                # Count by type
                if 'plaque' in class_name or 'tartar' in class_name or 'calculus' in class_name:
                    plaque_count += 1
                    total_plaque_area += width * height
                elif 'tooth' in class_name or 'teeth' in class_name:
                    teeth_count += 1
                    total_teeth_area += width * height
                elif 'gum' in class_name:
                    if confidence > 0.5:  # Only count confident gum detections
                        gum_issues += 1
        
        # Calculate health metrics
        health_assessment = self._calculate_health(
            plaque_count, teeth_count, gum_issues,
            total_plaque_area, total_teeth_area, detections
        )
        
        return {
            'success': True,
            'source': 'roboflow',
            'detections': detections,
            'counts': {
                'plaque': plaque_count,
                'teeth': teeth_count,
                'gum_issues': gum_issues,
                'total': len(detections)
            },
            'health_assessment': health_assessment,
            'raw_response': result
        }
    
    def _calculate_health(self, plaque_count: int, teeth_count: int, 
                          gum_issues: int, plaque_area: float, 
                          teeth_area: float, detections: List) -> Dict:
        """Calculate health score and assessment from detections."""
        
        # Start with base score
        health_score = 100
        severity = 'Low'
        issues = []
        
        # Plaque penalty - EXTREMELY AGGRESSIVE
        if plaque_count > 0:
            # Each plaque detection is VERY serious
            plaque_penalty = plaque_count * 25  # 25 points per plaque area
            health_score -= plaque_penalty
            
            if plaque_count >= 3:
                severity = 'High'
                issues.append(f"SEVERE plaque buildup detected ({plaque_count} areas)")
                health_score = min(health_score, 10)  # Cap at 10%
            elif plaque_count >= 2:
                severity = 'High'
                issues.append(f"Significant plaque buildup ({plaque_count} areas)")
                health_score = min(health_score, 15)  # Cap at 15%
            elif plaque_count >= 1:
                severity = 'Medium'
                issues.append(f"Plaque detected ({plaque_count} area)")
                health_score = min(health_score, 30)  # Cap at 30%
        
        # Plaque area ratio penalty
        if teeth_area > 0 and plaque_area > 0:
            plaque_ratio = plaque_area / teeth_area
            if plaque_ratio > 0.2:  # More than 20% coverage
                health_score -= 40
                severity = 'High'
                issues.append("Heavy plaque coverage on teeth")
                health_score = min(health_score, 8)  # Cap at 8%
            elif plaque_ratio > 0.1:
                health_score -= 25
                severity = 'High'
                issues.append("Moderate plaque coverage")
                health_score = min(health_score, 15)
        
        # Gum issues penalty - MORE AGGRESSIVE
        if gum_issues > 0:
            health_score -= gum_issues * 20
            if gum_issues >= 2:
                severity = 'High'
                issues.append(f"Multiple gum issues detected ({gum_issues})")
                health_score = min(health_score, 12)
            else:
                severity = 'Medium' if severity == 'Low' else severity
                issues.append("Gum issues detected")
                health_score = min(health_score, 25)
        
        # Check for other concerning detections
        for det in detections:
            class_name = det['class'].lower()
            conf = det['confidence']
            
            if 'cavity' in class_name or 'caries' in class_name or 'decay' in class_name:
                health_score -= 35
                severity = 'High'
                issues.append(f"Cavity/decay detected (confidence: {conf:.0%})")
                health_score = min(health_score, 10)  # Cap at 10%
            
            if 'inflammation' in class_name or 'gingivitis' in class_name:
                health_score -= 30
                severity = 'High'
                issues.append(f"Gum inflammation detected")
                health_score = min(health_score, 15)
            
            if 'tartar' in class_name or 'calculus' in class_name:
                health_score -= 30
                severity = 'High'
                issues.append(f"Tartar/calculus buildup")
                health_score = min(health_score, 12)
            
            if 'blood' in class_name or 'bleeding' in class_name:
                health_score -= 40
                severity = 'High'
                issues.append(f"Bleeding detected")
                health_score = min(health_score, 8)  # Cap at 8%
        
        # Ensure score is in valid range
        health_score = max(5, min(100, health_score))
        
        # Determine primary condition
        if not issues:
            primary_condition = 'Healthy'
            severity = 'Low'
        elif any('cavity' in i.lower() or 'decay' in i.lower() for i in issues):
            primary_condition = 'Caries'
        elif any('tartar' in i.lower() or 'calculus' in i.lower() or 'plaque' in i.lower() for i in issues):
            primary_condition = 'Calculus'
        elif any('gum' in i.lower() or 'inflammation' in i.lower() or 'bleeding' in i.lower() for i in issues):
            primary_condition = 'Gingivitis'
        else:
            primary_condition = 'Oral Issues Detected'
        
        return {
            'health_score': health_score,
            'severity': severity,
            'primary_condition': primary_condition,
            'issues': issues,
            'requires_attention': health_score < 50,
            'urgent': health_score < 20
        }
    
    def _fallback_result(self, error: str) -> Dict:
        """Return fallback result when API fails."""
        return {
            'success': False,
            'source': 'roboflow',
            'error': error,
            'detections': [],
            'counts': {'plaque': 0, 'teeth': 0, 'gum_issues': 0, 'total': 0},
            'health_assessment': {
                'health_score': None,
                'severity': None,
                'primary_condition': None,
                'issues': [f"Analysis failed: {error}"],
                'requires_attention': True,
                'urgent': False
            }
        }


# Convenience function
def analyze_with_roboflow(image) -> Dict:
    """Analyze dental image using Roboflow API."""
    analyzer = RoboflowDentalAnalyzer()
    return analyzer.analyze(image)
