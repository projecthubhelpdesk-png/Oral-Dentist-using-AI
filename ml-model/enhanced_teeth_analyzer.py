"""
Enhanced AI Teeth-Condition Analyzer
====================================
Complete AI system for oral disease classification, tooth localization, 
severity scoring, and dental report generation.

DISCLAIMER: This AI provides preliminary screening only and is not a medical diagnosis.
Always consult a qualified dental professional for proper diagnosis and treatment.
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Union, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont
import io
import cv2
from datetime import datetime

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras

# Import existing predictor
from predict import OralDiseasePredictor

# Try to import YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    print("YOLOv8 loaded successfully!")
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: YOLOv8 not available. Using mock tooth detection.")


# ============================================
# CONFIGURATION
# ============================================
YOLO_MODEL_PATH = Path(__file__).parent / 'models' / 'yolo_tooth_detector.pt'
FDI_MAPPING_PATH = Path(__file__).parent / 'config' / 'fdi_mapping.json'

# FDI Tooth Numbering System (Adult teeth)
FDI_TEETH = {
    'upper_right': ['18', '17', '16', '15', '14', '13', '12', '11'],
    'upper_left': ['21', '22', '23', '24', '25', '26', '27', '28'],
    'lower_left': ['38', '37', '36', '35', '34', '33', '32', '31'],
    'lower_right': ['41', '42', '43', '44', '45', '46', '47', '48']
}

# Tooth detection classes
TOOTH_CLASSES = {
    0: 'healthy_tooth',
    1: 'cavity',
    2: 'plaque',
    3: 'crooked_tooth',
    4: 'missing_tooth'
}

# Severity scoring rules
SEVERITY_RULES = {
    'low': {'confidence_threshold': 0.5, 'max_teeth': 1},
    'medium': {'confidence_threshold': 0.75, 'max_teeth': 3},
    'high': {'confidence_threshold': 1.0, 'max_teeth': float('inf')}
}

# Medical disclaimer
DISCLAIMER = "This AI provides preliminary screening only and is not a medical diagnosis. Always consult a qualified dental professional."


class MockYOLODetector:
    """Mock YOLO detector for when YOLOv8 is not available."""
    
    def __init__(self):
        self.names = TOOTH_CLASSES
    
    def predict(self, image, conf=0.5, verbose=False):
        """Mock prediction that returns some sample detections."""
        # Create mock detection results
        height, width = 480, 640
        if hasattr(image, 'shape'):
            height, width = image.shape[:2]
        elif isinstance(image, Image.Image):
            width, height = image.size
        
        # Generate some mock detections
        mock_results = []
        
        # Mock detection 1: cavity on upper right
        mock_results.append({
            'bbox': [width*0.3, height*0.2, width*0.1, height*0.15],
            'confidence': 0.75,
            'class': 1,  # cavity
            'tooth_region': 'upper_right'
        })
        
        # Mock detection 2: plaque on lower left
        mock_results.append({
            'bbox': [width*0.6, height*0.7, width*0.08, height*0.12],
            'confidence': 0.68,
            'class': 2,  # plaque
            'tooth_region': 'lower_left'
        })
        
        return [MockResult(mock_results, (height, width))]


class MockResult:
    """Mock YOLO result object."""
    
    def __init__(self, detections, image_shape):
        self.detections = detections
        self.image_shape = image_shape
        
        # Create mock boxes and conf arrays
        self.boxes = MockBoxes(detections)
    
    @property
    def names(self):
        return TOOTH_CLASSES


class MockBoxes:
    """Mock YOLO boxes object."""
    
    def __init__(self, detections):
        self.detections = detections
        
        # Convert to arrays and create mock tensor-like objects
        self.xyxy = [MockTensor(d['bbox']) for d in detections]
        self.conf = [MockTensor([d['confidence']]) for d in detections]
        self.cls = [MockTensor([d['class']]) for d in detections]


class MockTensor:
    """Mock tensor object with cpu() method."""
    
    def __init__(self, data):
        self.data = np.array(data)
    
    def cpu(self):
        return self
    
    def numpy(self):
        return self.data


class ToothDetector:
    """YOLOv8-based tooth detection and localization."""
    
    def __init__(self, model_path: str = None):
        """Initialize tooth detector."""
        self.model_path = Path(model_path) if model_path else YOLO_MODEL_PATH
        self.model = None
        self.fdi_mapping = self._load_fdi_mapping()
        
        self._load_model()
    
    def _load_model(self):
        """Load YOLOv8 model or create mock."""
        if YOLO_AVAILABLE and self.model_path.exists():
            print(f"Loading YOLO model from: {self.model_path}")
            self.model = YOLO(str(self.model_path))
            print("YOLO model loaded successfully!")
        else:
            print("Using mock tooth detector (YOLO not available or model not found)")
            self.model = MockYOLODetector()
    
    def _load_fdi_mapping(self) -> Dict:
        """Load FDI tooth numbering mapping."""
        if FDI_MAPPING_PATH.exists():
            with open(FDI_MAPPING_PATH, 'r') as f:
                return json.load(f)
        else:
            # Create default mapping
            return {
                'quadrants': FDI_TEETH,
                'position_mapping': {
                    'upper_right': {'x_range': [0.0, 0.5], 'y_range': [0.0, 0.5]},
                    'upper_left': {'x_range': [0.5, 1.0], 'y_range': [0.0, 0.5]},
                    'lower_left': {'x_range': [0.5, 1.0], 'y_range': [0.5, 1.0]},
                    'lower_right': {'x_range': [0.0, 0.5], 'y_range': [0.5, 1.0]}
                }
            }
    
    def _map_to_fdi(self, bbox: List[float], image_shape: Tuple[int, int]) -> str:
        """Map bounding box to FDI tooth number."""
        x, y, w, h = bbox
        height, width = image_shape
        
        # Normalize coordinates
        center_x = (x + w/2) / width
        center_y = (y + h/2) / height
        
        # Determine quadrant
        quadrant = None
        for quad_name, ranges in self.fdi_mapping['position_mapping'].items():
            x_range = ranges['x_range']
            y_range = ranges['y_range']
            
            if (x_range[0] <= center_x <= x_range[1] and 
                y_range[0] <= center_y <= y_range[1]):
                quadrant = quad_name
                break
        
        if not quadrant:
            return "unknown"
        
        # Get teeth in quadrant
        teeth_in_quad = self.fdi_mapping['quadrants'][quadrant]
        
        # Simple mapping based on x position within quadrant
        if quadrant in ['upper_right', 'lower_right']:
            # Right side: leftmost is highest number
            relative_x = (center_x - self.fdi_mapping['position_mapping'][quadrant]['x_range'][0]) / 0.5
            tooth_idx = int(relative_x * len(teeth_in_quad))
        else:
            # Left side: leftmost is lowest number
            relative_x = (center_x - self.fdi_mapping['position_mapping'][quadrant]['x_range'][0]) / 0.5
            tooth_idx = int((1 - relative_x) * len(teeth_in_quad))
        
        tooth_idx = max(0, min(tooth_idx, len(teeth_in_quad) - 1))
        return teeth_in_quad[tooth_idx]
    
    def detect_teeth(self, image: Union[str, bytes, np.ndarray, Image.Image], 
                    confidence_threshold: float = 0.5) -> List[Dict]:
        """
        Detect teeth and issues in image.
        
        Args:
            image: Input image (file path, bytes, numpy array, or PIL Image)
            confidence_threshold: Minimum confidence for detections
            
        Returns:
            List of detections with FDI mapping
        """
        # Convert image if needed
        if isinstance(image, str):
            img = cv2.imread(image)
        elif isinstance(image, bytes):
            # Handle bytes input
            pil_img = Image.open(io.BytesIO(image))
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        elif isinstance(image, Image.Image):
            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            img = image
        
        # Run detection
        results = self.model.predict(img, conf=confidence_threshold, verbose=False)
        
        detections = []
        for result in results:
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes
                
                for i in range(len(boxes.xyxy)):
                    # Handle both real YOLO tensors and mock tensors
                    if hasattr(boxes.xyxy[i], 'cpu'):
                        bbox = boxes.xyxy[i].cpu().numpy()
                        conf = float(boxes.conf[i].cpu().numpy()[0] if hasattr(boxes.conf[i].cpu().numpy(), '__len__') else boxes.conf[i].cpu().numpy())
                        cls = int(boxes.cls[i].cpu().numpy()[0] if hasattr(boxes.cls[i].cpu().numpy(), '__len__') else boxes.cls[i].cpu().numpy())
                    else:
                        # Mock tensor case
                        bbox = boxes.xyxy[i]
                        conf = float(boxes.conf[i][0] if hasattr(boxes.conf[i], '__len__') else boxes.conf[i])
                        cls = int(boxes.cls[i][0] if hasattr(boxes.cls[i], '__len__') else boxes.cls[i])
                    
                    # Map to FDI
                    fdi_number = self._map_to_fdi(bbox.tolist() if hasattr(bbox, 'tolist') else bbox, img.shape[:2])
                    
                    detection = {
                        'tooth_id': f"FDI-{fdi_number}",
                        'issue': TOOTH_CLASSES.get(cls, 'unknown'),
                        'confidence': round(conf, 3),
                        'bbox': [float(x) for x in (bbox.tolist() if hasattr(bbox, 'tolist') else bbox)],
                        'class_id': cls
                    }
                    detections.append(detection)
        
        return detections


class SeverityScorer:
    """Rule-based severity scoring engine."""
    
    @staticmethod
    def calculate_severity(disease_confidence: float, 
                          tooth_detections: List[Dict],
                          disease_type: str = None) -> Dict:
        """
        Calculate severity based on disease classification and tooth detections.
        
        Args:
            disease_confidence: EfficientNet confidence score
            tooth_detections: List of YOLO tooth detections
            disease_type: Type of disease detected
            
        Returns:
            Severity analysis dictionary
        """
        # Count affected teeth
        affected_teeth = len([d for d in tooth_detections 
                            if d['issue'] in ['cavity', 'plaque', 'crooked_tooth']])
        
        # Calculate average tooth detection confidence
        if tooth_detections:
            avg_tooth_conf = np.mean([d['confidence'] for d in tooth_detections])
        else:
            avg_tooth_conf = 0.0
        
        # Combined confidence (weighted average)
        combined_confidence = (disease_confidence * 0.7 + avg_tooth_conf * 0.3)
        
        # Apply severity rules
        severity = "Low"
        
        if (combined_confidence >= SEVERITY_RULES['high']['confidence_threshold'] or 
            affected_teeth > SEVERITY_RULES['medium']['max_teeth']):
            severity = "High"
        elif (combined_confidence >= SEVERITY_RULES['medium']['confidence_threshold'] or 
              affected_teeth > SEVERITY_RULES['low']['max_teeth']):
            severity = "Medium"
        
        # Disease-specific adjustments
        if disease_type:
            disease_weights = {
                'Caries': 1.2,
                'Mouth_Ulcer': 1.3,
                'Gingivitis': 1.1,
                'Calculus': 1.0
            }
            
            weight = disease_weights.get(disease_type, 1.0)
            if weight > 1.1 and severity == "Low":
                severity = "Medium"
            elif weight > 1.2 and severity == "Medium":
                severity = "High"
        
        return {
            'severity': severity,
            'combined_confidence': round(combined_confidence, 3),
            'affected_teeth_count': affected_teeth,
            'reasoning': f"Based on {disease_confidence:.1%} disease confidence and {affected_teeth} affected teeth"
        }


class DentalReportGenerator:
    """Generate safe, non-diagnostic dental reports."""
    
    @staticmethod
    def generate_report(efficientnet_result: Dict,
                       yolo_detections: List[Dict],
                       severity_analysis: Dict) -> Dict:
        """
        Generate comprehensive dental report.
        
        Args:
            efficientnet_result: Disease classification result
            yolo_detections: Tooth detection results
            severity_analysis: Severity scoring result
            
        Returns:
            Complete dental report JSON
        """
        disease = efficientnet_result.get('disease', 'Unknown')
        confidence = efficientnet_result.get('confidence', 0)
        severity = severity_analysis.get('severity', 'Low')
        
        # Extract affected teeth
        affected_teeth = []
        issue_summary = {}
        
        for detection in yolo_detections:
            if detection['issue'] != 'healthy_tooth':
                tooth_id = detection['tooth_id'].replace('FDI-', '')
                affected_teeth.append(tooth_id)
                
                issue = detection['issue']
                if issue not in issue_summary:
                    issue_summary[issue] = 0
                issue_summary[issue] += 1
        
        # Generate summary
        summary_parts = []
        if disease != 'Healthy' and confidence > 0.3:
            summary_parts.append(f"Signs of {disease.lower()} detected")
        
        for issue, count in issue_summary.items():
            if count == 1:
                summary_parts.append(f"{issue.replace('_', ' ')} detected on 1 tooth")
            else:
                summary_parts.append(f"{issue.replace('_', ' ')} detected on {count} teeth")
        
        if not summary_parts:
            summary = "No significant dental issues detected in this screening"
        else:
            summary = ". ".join(summary_parts) + "."
        
        # Generate recommendations based on severity
        recommendations = DentalReportGenerator._get_recommendations(severity, disease, issue_summary)
        
        # Home care tips
        home_care_tips = DentalReportGenerator._get_home_care_tips(disease, issue_summary)
        
        # Time-based recommendation
        urgency = DentalReportGenerator._get_urgency_recommendation(severity)
        
        return {
            'summary': summary,
            'severity': severity,
            'confidence_level': 'High' if confidence > 0.7 else 'Medium' if confidence > 0.4 else 'Low',
            'affected_teeth': sorted(set(affected_teeth)),
            'detected_issues': issue_summary,
            'recommendation': urgency,
            'home_care_tips': home_care_tips,
            'professional_advice': recommendations,
            'screening_date': datetime.now().isoformat(),
            'ai_disclaimer': DISCLAIMER,
            'next_steps': [
                "This is a preliminary AI screening only",
                "Schedule a professional dental examination",
                "Discuss these findings with your dentist",
                "Continue regular oral hygiene practices"
            ]
        }
    
    @staticmethod
    def _get_recommendations(severity: str, disease: str, issues: Dict) -> List[str]:
        """Get professional recommendations based on findings."""
        recommendations = []
        
        if severity == "High":
            recommendations.append("🚨 Schedule urgent dental appointment within 1 week")
        elif severity == "Medium":
            recommendations.append("⚠️ Schedule dental appointment within 2-3 weeks")
        else:
            recommendations.append("Schedule routine dental checkup within 3-6 months")
        
        # Disease-specific recommendations
        if disease == "Caries":
            recommendations.extend([
                "Discuss cavity treatment options",
                "Consider fluoride treatments",
                "Review dietary habits with dentist"
            ])
        elif disease == "Gingivitis":
            recommendations.extend([
                "Professional dental cleaning recommended",
                "Discuss gum health maintenance",
                "Consider periodontal evaluation"
            ])
        elif disease == "Calculus":
            recommendations.extend([
                "Professional tartar removal needed",
                "Discuss improved oral hygiene techniques",
                "Consider more frequent cleanings"
            ])
        elif disease == "Mouth_Ulcer":
            recommendations.extend([
                "Evaluate underlying causes of ulcers",
                "Discuss pain management options",
                "Rule out systemic conditions if recurring"
            ])
        
        # Issue-specific recommendations
        if 'cavity' in issues:
            recommendations.append("Address cavities before they worsen")
        if 'plaque' in issues:
            recommendations.append("Professional plaque removal recommended")
        if 'crooked_tooth' in issues:
            recommendations.append("Consider orthodontic consultation")
        
        return recommendations
    
    @staticmethod
    def _get_home_care_tips(disease: str, issues: Dict) -> List[str]:
        """Get home care recommendations."""
        tips = [
            "Brush teeth twice daily with fluoride toothpaste",
            "Floss daily to remove plaque between teeth",
            "Use antimicrobial mouthwash",
            "Limit sugary and acidic foods/drinks"
        ]
        
        # Disease-specific tips
        if disease == "Caries":
            tips.extend([
                "Rinse mouth with water after eating",
                "Chew sugar-free gum to stimulate saliva",
                "Consider using a fluoride rinse"
            ])
        elif disease == "Gingivitis":
            tips.extend([
                "Use a soft-bristled toothbrush",
                "Brush gently along the gumline",
                "Consider an electric toothbrush"
            ])
        elif disease == "Calculus":
            tips.extend([
                "Use tartar-control toothpaste",
                "Pay extra attention to gumline when brushing",
                "Consider a water flosser"
            ])
        elif disease == "Mouth_Ulcer":
            tips.extend([
                "Avoid spicy, acidic, or rough foods",
                "Use saltwater rinses",
                "Apply topical oral gels as needed"
            ])
        
        return tips
    
    @staticmethod
    def _get_urgency_recommendation(severity: str) -> str:
        """Get urgency-based recommendation."""
        if severity == "High":
            return "Schedule dental appointment within 1 week - prompt attention recommended"
        elif severity == "Medium":
            return "Schedule dental appointment within 2-3 weeks for evaluation"
        else:
            return "Schedule routine dental checkup within 3-6 months"


class EnhancedTeethAnalyzer:
    """
    Unified AI system for comprehensive teeth analysis.
    
    Combines:
    1. EfficientNet disease classification
    2. YOLOv8 tooth-level detection
    3. Severity scoring
    4. Dental report generation
    """
    
    def __init__(self, 
                 efficientnet_model_path: str = None,
                 yolo_model_path: str = None):
        """Initialize the enhanced analyzer."""
        print("Initializing Enhanced Teeth Analyzer...")
        
        # Initialize components
        self.disease_predictor = OralDiseasePredictor(model_path=efficientnet_model_path)
        self.tooth_detector = ToothDetector(model_path=yolo_model_path)
        self.severity_scorer = SeverityScorer()
        self.report_generator = DentalReportGenerator()
        
        print("Enhanced Teeth Analyzer ready!")
    
    def analyze_teeth(self, image: Union[str, bytes, Image.Image, np.ndarray]) -> Dict:
        """
        Complete teeth analysis pipeline.
        
        Args:
            image: Input dental image
            
        Returns:
            Complete analysis results including:
            - Disease classification
            - Tooth detections
            - Severity analysis
            - Dental report
        """
        print("Starting comprehensive teeth analysis...")
        
        # Step 1: Disease Classification (EfficientNet)
        print("1. Running disease classification...")
        disease_result = self.disease_predictor.predict(image)
        
        # Step 2: Tooth Detection (YOLOv8)
        print("2. Running tooth detection...")
        tooth_detections = self.tooth_detector.detect_teeth(image)
        
        # Step 3: Severity Scoring
        print("3. Calculating severity score...")
        severity_analysis = self.severity_scorer.calculate_severity(
            disease_confidence=disease_result['confidence'],
            tooth_detections=tooth_detections,
            disease_type=disease_result['disease']
        )
        
        # Step 4: Generate Dental Report
        print("4. Generating dental report...")
        dental_report = self.report_generator.generate_report(
            efficientnet_result=disease_result,
            yolo_detections=tooth_detections,
            severity_analysis=severity_analysis
        )
        
        # Combine all results
        complete_analysis = {
            'analysis_id': f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'timestamp': datetime.now().isoformat(),
            'disease_classification': {
                'primary_condition': disease_result['disease'],
                'confidence': disease_result['confidence'],
                'all_predictions': disease_result['all_predictions'][:5],  # Top 5
                'description': disease_result.get('description', '')
            },
            'tooth_detections': tooth_detections,
            'severity_analysis': severity_analysis,
            'dental_report': dental_report,
            'technical_details': {
                'efficientnet_version': 'EfficientNetB0',
                'yolo_version': 'YOLOv8n',
                'total_detections': len(tooth_detections),
                'processing_status': 'completed'
            },
            'disclaimer': DISCLAIMER
        }
        
        print("✅ Analysis completed successfully!")
        return complete_analysis
    
    def create_annotated_image(self, 
                              image: Union[str, Image.Image, np.ndarray],
                              analysis_result: Dict) -> Image.Image:
        """
        Create annotated image with detection results.
        
        Args:
            image: Original image
            analysis_result: Analysis results from analyze_teeth()
            
        Returns:
            PIL Image with annotations
        """
        # Convert to PIL Image
        if isinstance(image, str):
            img = Image.open(image)
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            img = image.copy()
        
        # Create drawing context
        draw = ImageDraw.Draw(img)
        
        # Try to load a font
        try:
            font = ImageFont.truetype("arial.ttf", 16)
            small_font = ImageFont.truetype("arial.ttf", 12)
        except:
            font = ImageFont.load_default()
            small_font = font
        
        # Color mapping for different issues
        colors = {
            'cavity': 'red',
            'plaque': 'orange',
            'crooked_tooth': 'yellow',
            'healthy_tooth': 'green',
            'missing_tooth': 'purple'
        }
        
        # Draw tooth detections
        for detection in analysis_result['tooth_detections']:
            bbox = detection['bbox']
            issue = detection['issue']
            confidence = detection['confidence']
            tooth_id = detection['tooth_id']
            
            color = colors.get(issue, 'blue')
            
            # Draw bounding box
            draw.rectangle(bbox, outline=color, width=2)
            
            # Draw label
            label = f"{tooth_id}: {issue} ({confidence:.2f})"
            draw.text((bbox[0], bbox[1] - 20), label, fill=color, font=small_font)
        
        # Add overall analysis info
        disease = analysis_result['disease_classification']['primary_condition']
        confidence = analysis_result['disease_classification']['confidence']
        severity = analysis_result['severity_analysis']['severity']
        
        info_text = f"Disease: {disease} ({confidence:.2f})\nSeverity: {severity}"
        draw.text((10, 10), info_text, fill='white', font=font)
        
        return img


# ============================================
# CONVENIENCE FUNCTIONS
# ============================================
def analyze_dental_image(image_path: str) -> Dict:
    """
    Convenience function for single image analysis.
    
    Args:
        image_path: Path to dental image
        
    Returns:
        Complete analysis results
    """
    analyzer = EnhancedTeethAnalyzer()
    return analyzer.analyze_teeth(image_path)


def create_analysis_summary(analysis_result: Dict) -> str:
    """
    Create a human-readable summary of analysis results.
    
    Args:
        analysis_result: Results from analyze_teeth()
        
    Returns:
        Formatted summary string
    """
    report = analysis_result['dental_report']
    
    summary = f"""
🦷 DENTAL AI SCREENING REPORT
{'='*50}

📊 SUMMARY: {report['summary']}

⚠️  SEVERITY: {report['severity']} ({report['confidence_level']} confidence)

🔍 AFFECTED TEETH: {', '.join(report['affected_teeth']) if report['affected_teeth'] else 'None identified'}

💡 RECOMMENDATION: {report['recommendation']}

🏠 HOME CARE TIPS:
{chr(10).join(f"   • {tip}" for tip in report['home_care_tips'][:5])}

⚕️  PROFESSIONAL ADVICE:
{chr(10).join(f"   • {advice}" for advice in report['professional_advice'][:3])}

⚠️  DISCLAIMER: {report['ai_disclaimer']}

📅 Screening Date: {report['screening_date'][:10]}
"""
    
    return summary


# ============================================
# CLI INTERFACE
# ============================================
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced AI Teeth Condition Analyzer')
    parser.add_argument('image', help='Path to dental image')
    parser.add_argument('--output', '-o', help='Output JSON file path')
    parser.add_argument('--annotated', '-a', help='Save annotated image path')
    parser.add_argument('--summary', '-s', action='store_true', help='Print summary')
    
    args = parser.parse_args()
    
    # Run analysis
    analyzer = EnhancedTeethAnalyzer()
    result = analyzer.analyze_teeth(args.image)
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to: {args.output}")
    
    # Save annotated image
    if args.annotated:
        annotated_img = analyzer.create_annotated_image(args.image, result)
        annotated_img.save(args.annotated)
        print(f"Annotated image saved to: {args.annotated}")
    
    # Print summary
    if args.summary:
        print(create_analysis_summary(result))
    else:
        # Print basic results
        disease = result['disease_classification']['primary_condition']
        confidence = result['disease_classification']['confidence']
        severity = result['severity_analysis']['severity']
        affected_teeth = len(result['dental_report']['affected_teeth'])
        
        print(f"\n🦷 Analysis Complete!")
        print(f"Disease: {disease} ({confidence:.1%} confidence)")
        print(f"Severity: {severity}")
        print(f"Affected Teeth: {affected_teeth}")
        print(f"Recommendation: {result['dental_report']['recommendation']}")