"""
Oral Disease Detection - Prediction Module
==========================================
Reusable prediction function for oral disease detection.

DISCLAIMER: This AI provides preliminary screening only and is not a medical diagnosis.
Always consult a qualified dental professional for proper diagnosis and treatment.
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, Union, Optional
from PIL import Image
import io

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras


# ============================================
# CONFIGURATION
# ============================================
MODEL_PATH = Path(__file__).parent / 'models' / 'oral_disease_rgb_model.h5'
LABELS_PATH = Path(__file__).parent / 'config' / 'class_labels.json'
IMAGE_SIZE = (224, 224)

# Severity mapping based on confidence
SEVERITY_THRESHOLDS = {
    'low': 0.5,
    'medium': 0.75,
    'high': 1.0
}

# Disease severity mapping (some diseases are inherently more severe)
DISEASE_SEVERITY_WEIGHT = {
    'caries': 1.2,
    'gingivitis': 1.1,
    'ulcers': 1.3,
    'calculus': 1.0,
    'hypodontia': 1.1,
    'tooth discoloration': 0.8,
    'healthy': 0.0
}

# Medical disclaimer
DISCLAIMER = "This AI provides preliminary screening only and is not a medical diagnosis. Always consult a qualified dental professional."


class OralDiseasePredictor:
    """
    Oral Disease Detection Predictor
    
    Usage:
        predictor = OralDiseasePredictor()
        result = predictor.predict(image_path)
        # or
        result = predictor.predict_from_bytes(image_bytes)
    """
    
    def __init__(self, model_path: str = None, labels_path: str = None):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to the trained model (.h5 file)
            labels_path: Path to class labels JSON file
        """
        self.model_path = Path(model_path) if model_path else MODEL_PATH
        self.labels_path = Path(labels_path) if labels_path else LABELS_PATH
        
        self.model = None
        self.class_labels = None
        self.num_classes = None
        self.image_size = IMAGE_SIZE
        
        self._load_model()
        self._load_labels()
    
    def _load_model(self):
        """Load the trained model."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at: {self.model_path}")
        
        print(f"Loading model from: {self.model_path}")
        self.model = keras.models.load_model(str(self.model_path))
        print("Model loaded successfully!")
    
    def _load_labels(self):
        """Load class labels from JSON."""
        if not self.labels_path.exists():
            raise FileNotFoundError(f"Labels file not found at: {self.labels_path}")
        
        with open(self.labels_path, 'r') as f:
            data = json.load(f)
        
        self.class_labels = data['class_labels']
        self.num_classes = data['num_classes']
        self.recommendations = data.get('recommendations', {})
        
        if 'image_size' in data:
            self.image_size = tuple(data['image_size'])
        
        print(f"Loaded {self.num_classes} class labels")
    
    def preprocess_image(self, image: Union[str, Path, bytes, Image.Image, np.ndarray]) -> np.ndarray:
        """
        Preprocess image for prediction.
        
        Args:
            image: Can be file path, bytes, PIL Image, or numpy array
            
        Returns:
            Preprocessed image array ready for prediction
        """
        # Handle different input types
        if isinstance(image, (str, Path)):
            img = Image.open(image)
        elif isinstance(image, bytes):
            img = Image.open(io.BytesIO(image))
        elif isinstance(image, Image.Image):
            img = image
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(image)
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize
        img = img.resize(self.image_size, Image.Resampling.LANCZOS)
        
        # Convert to array and normalize
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def _calculate_severity(self, disease: str, confidence: float, all_predictions: list = None) -> str:
        """
        Calculate severity based on confidence, disease type, and multi-condition detection.
        
        Args:
            disease: Detected disease name
            confidence: Model confidence score
            all_predictions: All prediction scores (optional, for multi-condition detection)
            
        Returns:
            Severity level: 'Low', 'Medium', or 'High'
        """
        # Get disease weight (default to 1.0 if not found)
        disease_lower = disease.lower()
        weight = DISEASE_SEVERITY_WEIGHT.get(disease_lower, 1.0)
        
        # Adjust confidence with disease weight
        adjusted_confidence = min(confidence * weight, 1.0)
        
        # IMPROVED: Check for multiple conditions (indicates more severe overall state)
        multi_condition_boost = 0
        if all_predictions:
            # Count how many conditions have significant confidence
            significant_conditions = sum(1 for p in all_predictions 
                                        if p['confidence'] > 0.1 and p['disease'].lower() != 'healthy')
            if significant_conditions >= 2:
                multi_condition_boost = 0.2  # Boost severity for multiple conditions
            if significant_conditions >= 3:
                multi_condition_boost = 0.35
        
        adjusted_confidence = min(adjusted_confidence + multi_condition_boost, 1.0)
        
        # IMPROVED: More aggressive severity thresholds
        # Any dental disease detection should be taken seriously
        if adjusted_confidence < 0.35:  # Was 0.5
            return 'Low'
        elif adjusted_confidence < 0.55:  # Was 0.75
            return 'Medium'
        else:
            return 'High'
    
    def predict(self, image: Union[str, Path, bytes, Image.Image, np.ndarray]) -> Dict:
        """
        Predict oral disease from image.
        
        Args:
            image: Input image (file path, bytes, PIL Image, or numpy array)
            
        Returns:
            Dictionary with prediction results:
            {
                "disease": "Gingivitis",
                "confidence": 0.87,
                "severity": "High",
                "all_predictions": [...],
                "disclaimer": "..."
            }
        """
        # Preprocess
        img_array = self.preprocess_image(image)
        
        # Predict with ML model
        predictions = self.model.predict(img_array, verbose=0)[0]
        
        # Get ML predictions sorted by confidence
        ml_predictions = []
        for idx, conf in enumerate(predictions):
            ml_predictions.append({
                'disease': self.class_labels[str(idx)],
                'confidence': float(conf)
            })
        ml_predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        # ENHANCED: Run visual analysis to supplement ML predictions
        try:
            from visual_analyzer import VisualDentalAnalyzer
            visual_analyzer = VisualDentalAnalyzer()
            visual_result = visual_analyzer.analyze(image)
            
            # Combine ML and visual analysis
            combined_result = self._combine_predictions(ml_predictions, visual_result)
            
            top_disease = combined_result['disease']
            top_confidence = combined_result['confidence']
            all_predictions = combined_result['all_predictions']
            severity = combined_result['severity']
            health_score = combined_result.get('health_score', 50)
            
        except Exception as e:
            # Fallback to ML-only if visual analysis fails
            print(f"Visual analysis failed, using ML only: {e}")
            top_idx = np.argmax(predictions)
            top_confidence = float(predictions[top_idx])
            top_disease = self.class_labels[str(top_idx)]
            all_predictions = ml_predictions
            severity = self._calculate_severity(top_disease, top_confidence, all_predictions)
            health_score = None
        
        # Get recommendations for detected disease
        disease_info = self.recommendations.get(top_disease, {})
        recommendations = disease_info.get('recommendations', ['Consult a dental professional'])
        description = disease_info.get('description', f'{top_disease} detected')
        
        result = {
            'disease': top_disease,
            'confidence': round(top_confidence, 4),
            'severity': severity,
            'description': description,
            'recommendations': recommendations,
            'all_predictions': all_predictions,
            'disclaimer': DISCLAIMER
        }
        
        if health_score is not None:
            result['visual_health_score'] = health_score
        
        # ROBOFLOW ANALYSIS: Use Roboflow API for better detection
        roboflow_result = None
        try:
            from roboflow_analyzer import RoboflowDentalAnalyzer
            roboflow = RoboflowDentalAnalyzer()
            roboflow_result = roboflow.analyze(image)
            
            if roboflow_result.get('success'):
                rf_health = roboflow_result['health_assessment']
                rf_score = rf_health.get('health_score')
                rf_severity = rf_health.get('severity')
                rf_condition = rf_health.get('primary_condition')
                
                # Use Roboflow results if they indicate worse condition
                if rf_score is not None:
                    if rf_score < (health_score or 100):
                        health_score = rf_score
                        result['visual_health_score'] = rf_score
                    
                    # Override severity if Roboflow says it's worse
                    if rf_severity == 'High' and severity != 'High':
                        severity = 'High'
                        result['severity'] = 'High'
                    elif rf_severity == 'Medium' and severity == 'Low':
                        severity = 'Medium'
                        result['severity'] = 'Medium'
                    
                    # Add Roboflow issues to result
                    result['roboflow_detections'] = roboflow_result.get('counts', {})
                    result['roboflow_issues'] = rf_health.get('issues', [])
                    
                    # If Roboflow detected plaque/tartar, ensure we report it
                    if roboflow_result['counts'].get('plaque', 0) > 0:
                        if rf_condition == 'Calculus' and top_disease != 'Calculus':
                            # Roboflow detected tartar but ML didn't - trust Roboflow
                            result['disease'] = 'Calculus'
                            result['roboflow_override'] = True
                
                print(f"Roboflow analysis: score={rf_score}, severity={rf_severity}, condition={rf_condition}")
        except Exception as e:
            print(f"Roboflow analysis failed: {e}")
        
        # CLINICAL SAFETY: Apply safety pipeline for medical-grade results
        try:
            from clinical_safety import ClinicalSafetyPipeline
            safety_pipeline = ClinicalSafetyPipeline()
            safe_result = safety_pipeline.process(image, result)
            
            # Update result with safety-checked values
            result['severity'] = safe_result.get('severity', severity)
            result['confidence'] = safe_result.get('confidence', result['confidence'])
            result['visual_health_score'] = safe_result.get('health_score', health_score)
            result['safety_flags'] = safe_result.get('safety_flags', [])
            result['visual_risk_level'] = safe_result.get('visual_risk_level', 'LOW')
            
            # Use the LOWEST score from all sources
            final_score = result['visual_health_score']
            if roboflow_result and roboflow_result.get('success'):
                rf_score = roboflow_result['health_assessment'].get('health_score')
                if rf_score is not None:
                    final_score = min(final_score, rf_score)
            result['visual_health_score'] = final_score
            
            if safe_result.get('urgent_warning'):
                result['urgent_warning'] = safe_result['urgent_warning']
                # Add urgent warning to recommendations
                result['recommendations'] = [safe_result['urgent_warning']] + result['recommendations']
            
        except Exception as e:
            print(f"Clinical safety check failed: {e}")
        
        return result
    
    def _combine_predictions(self, ml_predictions: list, visual_result: Dict) -> Dict:
        """
        Combine ML model predictions with visual analysis for more accurate results.
        Visual analysis can override ML when there's strong visual evidence.
        """
        visual_scores = visual_result['condition_scores']
        visual_health = visual_result['overall_health_score']
        
        # Map visual conditions to disease names
        visual_to_disease = {
            'tartar_buildup': 'Calculus',
            'decay_cavities': 'Caries',
            'gum_inflammation': 'Gingivitis',
            'healthy_teeth': 'Healthy'
        }
        
        # Get ML top prediction
        ml_top = ml_predictions[0]
        ml_disease = ml_top['disease']
        ml_confidence = ml_top['confidence']
        
        # Get visual top condition
        visual_top_key = max(visual_scores, key=visual_scores.get)
        visual_top_score = visual_scores[visual_top_key]
        visual_disease = visual_to_disease.get(visual_top_key, ml_disease)
        
        # Decision logic: Trust visual analysis more when it has strong signals
        # Visual analysis is better at detecting tartar (yellow/brown) and decay (dark spots)
        
        # Check for strong visual indicators that should override ML
        tartar_score = visual_scores['tartar_buildup']
        decay_score = visual_scores['decay_cavities']
        healthy_score = visual_scores['healthy_teeth']
        
        # If visual shows strong tartar but ML says something else
        if tartar_score >= 0.4 and ml_disease != 'Calculus':
            # Override to Calculus if visual evidence is strong
            final_disease = 'Calculus'
            final_confidence = max(tartar_score, ml_confidence * 0.5)
        # If visual shows strong decay but ML says something else
        elif decay_score >= 0.4 and ml_disease != 'Caries':
            final_disease = 'Caries'
            final_confidence = max(decay_score, ml_confidence * 0.5)
        # If visual shows healthy teeth but ML detects disease
        elif healthy_score >= 0.6 and visual_health >= 70:
            # Trust visual - teeth look healthy
            if ml_confidence < 0.5:
                final_disease = 'Healthy'
                final_confidence = healthy_score
            else:
                # ML is confident, keep ML prediction but reduce confidence
                final_disease = ml_disease
                final_confidence = ml_confidence * 0.7
        else:
            # Use weighted combination
            # If ML and visual agree, boost confidence
            if visual_disease == ml_disease:
                final_disease = ml_disease
                final_confidence = min(1.0, (ml_confidence + visual_top_score) / 1.5)
            else:
                # Disagreement - use the one with higher confidence
                if visual_top_score > ml_confidence:
                    final_disease = visual_disease
                    final_confidence = visual_top_score
                else:
                    final_disease = ml_disease
                    final_confidence = ml_confidence
        
        # Build combined predictions list
        combined_predictions = []
        disease_scores = {}
        
        # Add ML predictions
        for pred in ml_predictions:
            disease_scores[pred['disease']] = pred['confidence'] * 0.5
        
        # Add visual predictions
        for visual_key, score in visual_scores.items():
            disease = visual_to_disease.get(visual_key)
            if disease:
                if disease in disease_scores:
                    disease_scores[disease] = (disease_scores[disease] + score * 0.5)
                else:
                    disease_scores[disease] = score * 0.5
        
        # Sort by combined score
        for disease, score in sorted(disease_scores.items(), key=lambda x: x[1], reverse=True):
            combined_predictions.append({
                'disease': disease,
                'confidence': round(min(1.0, score), 4)
            })
        
        # Calculate severity based on visual indicators
        if tartar_score >= 0.5 or decay_score >= 0.5:
            severity = 'High'
        elif tartar_score >= 0.3 or decay_score >= 0.3:
            severity = 'Medium'
        elif visual_health < 50:
            severity = 'Medium'
        else:
            severity = 'Low'
        
        # Override severity if health score is very low
        if visual_health < 30:
            severity = 'High'
        elif visual_health < 50:
            severity = 'Medium' if severity == 'Low' else severity
        
        return {
            'disease': final_disease,
            'confidence': round(final_confidence, 4),
            'severity': severity,
            'all_predictions': combined_predictions,
            'health_score': visual_health
        }
    
    def predict_from_bytes(self, image_bytes: bytes) -> Dict:
        """
        Predict from image bytes (useful for API endpoints).
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Prediction results dictionary
        """
        return self.predict(image_bytes)
    
    def predict_batch(self, images: list) -> list:
        """
        Predict on multiple images.
        
        Args:
            images: List of images (paths, bytes, or arrays)
            
        Returns:
            List of prediction results
        """
        results = []
        for image in images:
            try:
                result = self.predict(image)
                results.append(result)
            except Exception as e:
                results.append({
                    'error': str(e),
                    'disease': None,
                    'confidence': 0,
                    'severity': None
                })
        return results


# ============================================
# STANDALONE PREDICTION FUNCTION
# ============================================
_predictor = None

def get_predictor() -> OralDiseasePredictor:
    """Get or create singleton predictor instance."""
    global _predictor
    if _predictor is None:
        _predictor = OralDiseasePredictor()
    return _predictor


def predict_oral_disease(image: Union[str, Path, bytes, Image.Image, np.ndarray]) -> Dict:
    """
    Standalone prediction function.
    
    Args:
        image: Input image (file path, bytes, PIL Image, or numpy array)
        
    Returns:
        Prediction results:
        {
            "disease": "Gingivitis",
            "confidence": 0.87,
            "severity": "High",
            "all_predictions": [...],
            "disclaimer": "..."
        }
        
    Example:
        >>> result = predict_oral_disease("oral_image.jpg")
        >>> print(f"Disease: {result['disease']}, Confidence: {result['confidence']}")
    """
    predictor = get_predictor()
    return predictor.predict(image)


# ============================================
# CLI INTERFACE
# ============================================
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Oral Disease Detection Prediction')
    parser.add_argument('image', help='Path to oral image')
    parser.add_argument('--model', help='Path to model file', default=None)
    parser.add_argument('--labels', help='Path to labels file', default=None)
    
    args = parser.parse_args()
    
    # Create predictor
    predictor = OralDiseasePredictor(
        model_path=args.model,
        labels_path=args.labels
    )
    
    # Predict
    result = predictor.predict(args.image)
    
    # Print results
    print("\n" + "="*50)
    print("ORAL DISEASE DETECTION RESULT")
    print("="*50)
    print(f"\n🦷 Disease: {result['disease']}")
    print(f"📊 Confidence: {result['confidence']*100:.1f}%")
    print(f"⚠️  Severity: {result['severity']}")
    print("\nAll predictions:")
    for pred in result['all_predictions'][:5]:
        print(f"  - {pred['disease']}: {pred['confidence']*100:.1f}%")
    print(f"\n⚕️  {result['disclaimer']}")
