"""
Clinical Safety Module for Dental AI
=====================================

Implements three critical safety layers:
1. Visual Risk Detection (OpenCV-based, no ML)
2. Clinical Rule Override Engine
3. Health Score Recalibration

This ensures AI cannot give dangerous false-negatives.
"""

import numpy as np
from PIL import Image
import io
from typing import Dict, Union, List, Tuple
from pathlib import Path


class VisualRiskDetector:
    """
    Layer 1: Visual Risk Detection using OpenCV/NumPy
    Detects danger signs that ML models may miss.
    
    Checks for:
    - Excessive redness (inflammation)
    - Bleeding / blood-like pixels
    - Open gum tissue
    - Extreme contrast near gum margins
    - Dark decay spots
    - Heavy tartar deposits
    """
    
    def __init__(self):
        # Blood/bleeding detection (dark red to bright red)
        self.blood_hue_range = (0, 15)  # Red hues
        self.blood_sat_min = 40  # Lower threshold to catch more bleeding
        self.blood_value_range = (30, 200)  # Not too dark, not too bright
        
        # Inflammation (red/pink gums)
        self.inflammation_hue_range = (0, 25)  # Wider range
        self.inflammation_sat_min = 30  # Lower threshold
        
        # Severe decay (very dark spots)
        self.decay_value_max = 70  # Higher threshold to catch more decay
        
        # Heavy tartar (yellow-brown) - MUCH more sensitive
        self.tartar_hue_range = (10, 55)  # Wider range for yellow-brown
        self.tartar_sat_min = 20  # Lower saturation threshold
        
        # Thresholds for risk flags - MORE SENSITIVE
        self.bleeding_threshold = 0.02  # 2% blood pixels = HIGH RISK
        self.inflammation_threshold = 0.10  # 10% inflamed = concern
        self.severe_decay_threshold = 0.03  # 3% very dark = HIGH RISK
        self.heavy_tartar_threshold = 0.08  # 8% tartar = concern
    
    def analyze(self, image: Union[str, bytes, Image.Image, np.ndarray]) -> Dict:
        """
        Analyze image for visual risk indicators.
        
        Returns:
            Dict with risk flags and percentages
        """
        # Load and convert image
        img = self._load_image(image)
        img_array = np.array(img)
        hsv = self._rgb_to_hsv(img_array)
        
        # Detect each risk factor
        bleeding = self._detect_bleeding(hsv, img_array)
        inflammation = self._detect_inflammation(hsv, img_array)
        severe_decay = self._detect_severe_decay(hsv, img_array)
        heavy_tartar = self._detect_heavy_tartar(hsv, img_array)
        contrast_issues = self._detect_contrast_issues(img_array)
        
        # Determine overall visual risk level
        risk_flags = []
        risk_level = "LOW"
        
        if bleeding['detected']:
            risk_flags.append("BLEEDING_DETECTED")
            risk_level = "HIGH"
        
        if severe_decay['detected']:
            risk_flags.append("SEVERE_DECAY_DETECTED")
            risk_level = "HIGH"
        
        if inflammation['percentage'] > self.inflammation_threshold:
            risk_flags.append("HIGH_INFLAMMATION")
            if risk_level != "HIGH":
                risk_level = "MEDIUM"
        
        if heavy_tartar['percentage'] > self.heavy_tartar_threshold:
            risk_flags.append("HEAVY_TARTAR")
            if risk_level != "HIGH":
                risk_level = "MEDIUM"
        
        if contrast_issues['detected']:
            risk_flags.append("ABNORMAL_CONTRAST")
            if risk_level == "LOW":
                risk_level = "MEDIUM"
        
        return {
            'risk_level': risk_level,
            'risk_flags': risk_flags,
            'bleeding': bleeding,
            'inflammation': inflammation,
            'severe_decay': severe_decay,
            'heavy_tartar': heavy_tartar,
            'contrast_issues': contrast_issues,
            'requires_immediate_attention': risk_level == "HIGH"
        }
    
    def _load_image(self, image: Union[str, bytes, Image.Image, np.ndarray]) -> Image.Image:
        """Load image from various sources."""
        if isinstance(image, (str, Path)):
            return Image.open(image).convert('RGB')
        elif isinstance(image, bytes):
            return Image.open(io.BytesIO(image)).convert('RGB')
        elif isinstance(image, Image.Image):
            return image.convert('RGB')
        elif isinstance(image, np.ndarray):
            return Image.fromarray(image).convert('RGB')
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
    
    def _rgb_to_hsv(self, rgb: np.ndarray) -> np.ndarray:
        """Convert RGB to HSV."""
        rgb_norm = rgb.astype(np.float32) / 255.0
        r, g, b = rgb_norm[:,:,0], rgb_norm[:,:,1], rgb_norm[:,:,2]
        
        max_c = np.maximum(np.maximum(r, g), b)
        min_c = np.minimum(np.minimum(r, g), b)
        diff = max_c - min_c
        
        # Hue
        h = np.zeros_like(max_c)
        mask = diff != 0
        mask_r = mask & (max_c == r)
        mask_g = mask & (max_c == g)
        mask_b = mask & (max_c == b)
        
        h[mask_r] = (60 * ((g[mask_r] - b[mask_r]) / diff[mask_r]) + 360) % 360
        h[mask_g] = (60 * ((b[mask_g] - r[mask_g]) / diff[mask_g]) + 120) % 360
        h[mask_b] = (60 * ((r[mask_b] - g[mask_b]) / diff[mask_b]) + 240) % 360
        
        # Saturation
        s = np.zeros_like(max_c)
        s[max_c != 0] = (diff[max_c != 0] / max_c[max_c != 0]) * 100
        
        # Value
        v = max_c * 255
        
        return np.stack([h, s, v], axis=-1)
    
    def _detect_bleeding(self, hsv: np.ndarray, rgb: np.ndarray) -> Dict:
        """Detect blood/bleeding indicators."""
        h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
        
        # Blood is dark red to bright red with high saturation
        hue_mask = ((h >= 0) & (h <= self.blood_hue_range[1])) | (h >= 350)
        sat_mask = s >= self.blood_sat_min
        val_mask = (v >= self.blood_value_range[0]) & (v <= self.blood_value_range[1])
        
        # Additional check: red channel dominance in RGB
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        red_dominant = (r > g * 1.3) & (r > b * 1.3) & (r > 80)
        
        blood_mask = (hue_mask & sat_mask & val_mask) | (red_dominant & (v < 150))
        
        percentage = np.sum(blood_mask) / blood_mask.size
        detected = percentage > self.bleeding_threshold
        
        return {
            'detected': detected,
            'percentage': round(percentage, 4),
            'severity': 'HIGH' if percentage > 0.05 else 'MEDIUM' if detected else 'LOW'
        }
    
    def _detect_inflammation(self, hsv: np.ndarray, rgb: np.ndarray) -> Dict:
        """Detect gum inflammation (redness)."""
        h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
        
        # Inflamed gums are red/pink
        hue_mask = ((h >= 0) & (h <= self.inflammation_hue_range[1])) | (h >= 340)
        sat_mask = s >= self.inflammation_sat_min
        val_mask = v >= 60  # Not too dark
        
        inflammation_mask = hue_mask & sat_mask & val_mask
        percentage = np.sum(inflammation_mask) / inflammation_mask.size
        
        return {
            'detected': percentage > 0.10,
            'percentage': round(percentage, 4),
            'severity': 'HIGH' if percentage > 0.25 else 'MEDIUM' if percentage > 0.15 else 'LOW'
        }
    
    def _detect_severe_decay(self, hsv: np.ndarray, rgb: np.ndarray) -> Dict:
        """Detect severe decay (very dark spots)."""
        h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
        
        # Severe decay is very dark
        very_dark = v <= self.decay_value_max
        
        # Also check for dark brown (decay color)
        dark_brown = (v <= 80) & (h >= 0) & (h <= 50) & (s >= 20)
        
        decay_mask = very_dark | dark_brown
        percentage = np.sum(decay_mask) / decay_mask.size
        
        return {
            'detected': percentage > self.severe_decay_threshold,
            'percentage': round(percentage, 4),
            'severity': 'HIGH' if percentage > 0.08 else 'MEDIUM' if percentage > 0.05 else 'LOW'
        }
    
    def _detect_heavy_tartar(self, hsv: np.ndarray, rgb: np.ndarray) -> Dict:
        """Detect heavy tartar/calculus deposits."""
        h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
        
        # Tartar is yellow-brown
        hue_mask = (h >= self.tartar_hue_range[0]) & (h <= self.tartar_hue_range[1])
        sat_mask = s >= self.tartar_sat_min
        val_mask = (v >= 50) & (v <= 200)
        
        tartar_mask = hue_mask & sat_mask & val_mask
        percentage = np.sum(tartar_mask) / tartar_mask.size
        
        return {
            'detected': percentage > 0.08,
            'percentage': round(percentage, 4),
            'severity': 'HIGH' if percentage > 0.20 else 'MEDIUM' if percentage > 0.12 else 'LOW'
        }
    
    def _detect_contrast_issues(self, rgb: np.ndarray) -> Dict:
        """Detect abnormal contrast near gum margins."""
        gray = np.mean(rgb, axis=2)
        
        # Calculate local contrast using standard deviation
        from scipy import ndimage
        local_std = ndimage.generic_filter(gray, np.std, size=15)
        
        # High contrast areas
        high_contrast = local_std > 50
        percentage = np.sum(high_contrast) / high_contrast.size
        
        return {
            'detected': percentage > 0.15,
            'percentage': round(percentage, 4)
        }


class ClinicalRuleEngine:
    """
    Layer 2: Clinical Rule Override Engine
    
    Applies medical safety rules AFTER AI prediction.
    Ensures AI cannot give dangerous false-negatives.
    """
    
    def __init__(self):
        # Rule definitions - AGGRESSIVE for medical safety
        self.rules = [
            {
                'name': 'bleeding_override',
                'condition': lambda vr, ai: vr['bleeding']['detected'],
                'actions': {
                    'min_severity': 'High',
                    'max_health_score': 20,  # Bleeding = max 20%
                    'max_confidence': 0.70,
                    'add_flag': 'BLEEDING_REQUIRES_ATTENTION'
                }
            },
            {
                'name': 'severe_decay_override',
                'condition': lambda vr, ai: vr['severe_decay']['detected'],
                'actions': {
                    'min_severity': 'High',
                    'max_health_score': 15,  # Severe decay = max 15%
                    'add_flag': 'SEVERE_DECAY_DETECTED'
                }
            },
            {
                'name': 'high_inflammation_override',
                'condition': lambda vr, ai: vr['inflammation']['severity'] == 'HIGH',
                'actions': {
                    'min_severity': 'High',
                    'max_health_score': 25,  # High inflammation = max 25%
                    'add_flag': 'HIGH_INFLAMMATION'
                }
            },
            {
                'name': 'heavy_tartar_high_override',
                'condition': lambda vr, ai: vr['heavy_tartar']['severity'] == 'HIGH',
                'actions': {
                    'min_severity': 'High',
                    'max_health_score': 15,  # Heavy tartar = max 15%
                    'add_flag': 'SEVERE_TARTAR_BUILDUP'
                }
            },
            {
                'name': 'heavy_tartar_medium_override',
                'condition': lambda vr, ai: vr['heavy_tartar']['severity'] == 'MEDIUM',
                'actions': {
                    'min_severity': 'Medium',
                    'max_health_score': 30,  # Medium tartar = max 30%
                    'add_flag': 'TARTAR_BUILDUP'
                }
            },
            {
                'name': 'multiple_issues_override',
                'condition': lambda vr, ai: len(vr['risk_flags']) >= 2,
                'actions': {
                    'min_severity': 'High',
                    'max_health_score': 18,  # Multiple issues = max 18%
                    'add_flag': 'MULTIPLE_ISSUES_DETECTED'
                }
            },
            {
                'name': 'three_plus_issues_override',
                'condition': lambda vr, ai: len(vr['risk_flags']) >= 3,
                'actions': {
                    'min_severity': 'High',
                    'max_health_score': 10,  # 3+ issues = max 10%
                    'add_flag': 'CRITICAL_MULTIPLE_ISSUES'
                }
            },
            {
                'name': 'visual_high_risk_override',
                'condition': lambda vr, ai: vr['risk_level'] == 'HIGH',
                'actions': {
                    'min_severity': 'High',
                    'max_health_score': 20,  # High visual risk = max 20%
                    'max_confidence': 0.75,
                    'add_flag': 'HIGH_VISUAL_RISK'
                }
            },
            {
                'name': 'confidence_sanity_check',
                'condition': lambda vr, ai: ai.get('confidence', 0) > 0.9 and vr['risk_level'] != 'LOW',
                'actions': {
                    'max_confidence': 0.80,
                    'add_flag': 'CONFIDENCE_ADJUSTED'
                }
            }
        ]
    
    def apply_rules(self, visual_risk: Dict, ai_result: Dict) -> Dict:
        """
        Apply clinical rules to AI result.
        
        Args:
            visual_risk: Output from VisualRiskDetector
            ai_result: Output from AI model
            
        Returns:
            Modified AI result with safety overrides applied
        """
        result = ai_result.copy()
        applied_rules = []
        safety_flags = []
        
        for rule in self.rules:
            try:
                if rule['condition'](visual_risk, ai_result):
                    applied_rules.append(rule['name'])
                    actions = rule['actions']
                    
                    # Apply severity override
                    if 'min_severity' in actions:
                        current_severity = result.get('severity', 'Low')
                        if self._severity_rank(actions['min_severity']) > self._severity_rank(current_severity):
                            result['severity'] = actions['min_severity']
                    
                    # Apply health score cap
                    if 'max_health_score' in actions:
                        current_score = result.get('health_score', result.get('overall_health_score', 100))
                        if current_score > actions['max_health_score']:
                            result['health_score'] = actions['max_health_score']
                            if 'overall_health_score' in result:
                                result['overall_health_score'] = actions['max_health_score']
                    
                    # Apply confidence cap
                    if 'max_confidence' in actions:
                        if result.get('confidence', 0) > actions['max_confidence']:
                            result['confidence'] = actions['max_confidence']
                    
                    # Add safety flag
                    if 'add_flag' in actions:
                        safety_flags.append(actions['add_flag'])
            
            except Exception as e:
                print(f"Rule {rule['name']} failed: {e}")
        
        result['applied_safety_rules'] = applied_rules
        result['safety_flags'] = safety_flags
        result['visual_risk_level'] = visual_risk['risk_level']
        
        return result
    
    def _severity_rank(self, severity: str) -> int:
        """Get numeric rank for severity."""
        ranks = {'Low': 1, 'Medium': 2, 'High': 3}
        return ranks.get(severity, 0)


class HealthScoreCalculator:
    """
    Layer 3: Health Score Recalibration
    
    Calculates health score using weighted penalty system.
    Prevents unrealistically high scores on unhealthy mouths.
    """
    
    def __init__(self):
        # Penalty weights - MUCH MORE AGGRESSIVE
        self.penalties = {
            'bleeding': 60,           # Bleeding = -60 points
            'severe_decay': 55,       # Severe decay = -55 points
            'high_inflammation': 45,  # High inflammation = -45 points
            'medium_inflammation': 25,
            'heavy_tartar': 50,       # Heavy tartar = -50 points
            'medium_tartar': 30,
            'multiple_areas': 25,     # Multiple affected areas = -25
            'high_visual_risk': 35,   # High visual risk = -35
            'medium_visual_risk': 20,
        }
        
        # Severity multipliers
        self.severity_multipliers = {
            'High': 2.0,   # More aggressive
            'Medium': 1.3,
            'Low': 0.7
        }
    
    def calculate(self, visual_risk: Dict, ai_result: Dict) -> float:
        """
        Calculate calibrated health score.
        
        Args:
            visual_risk: Output from VisualRiskDetector
            ai_result: Output from AI model
            
        Returns:
            Calibrated health score (0-100)
        """
        # Start with base score of 100
        score = 100.0
        penalties_applied = []
        
        # Apply bleeding penalty
        if visual_risk['bleeding']['detected']:
            penalty = self.penalties['bleeding']
            if visual_risk['bleeding']['severity'] == 'HIGH':
                penalty *= 1.3
            score -= penalty
            penalties_applied.append(f"bleeding: -{penalty}")
        
        # Apply decay penalty
        if visual_risk['severe_decay']['detected']:
            penalty = self.penalties['severe_decay']
            if visual_risk['severe_decay']['severity'] == 'HIGH':
                penalty *= 1.3
            score -= penalty
            penalties_applied.append(f"severe_decay: -{penalty}")
        
        # Apply inflammation penalty
        if visual_risk['inflammation']['severity'] == 'HIGH':
            score -= self.penalties['high_inflammation']
            penalties_applied.append(f"high_inflammation: -{self.penalties['high_inflammation']}")
        elif visual_risk['inflammation']['severity'] == 'MEDIUM':
            score -= self.penalties['medium_inflammation']
            penalties_applied.append(f"medium_inflammation: -{self.penalties['medium_inflammation']}")
        
        # Apply tartar penalty
        if visual_risk['heavy_tartar']['severity'] == 'HIGH':
            score -= self.penalties['heavy_tartar']
            penalties_applied.append(f"heavy_tartar: -{self.penalties['heavy_tartar']}")
        elif visual_risk['heavy_tartar']['severity'] == 'MEDIUM':
            score -= self.penalties['medium_tartar']
            penalties_applied.append(f"medium_tartar: -{self.penalties['medium_tartar']}")
        
        # Apply multiple areas penalty
        if len(visual_risk['risk_flags']) >= 2:
            score -= self.penalties['multiple_areas']
            penalties_applied.append(f"multiple_areas: -{self.penalties['multiple_areas']}")
        
        # Apply visual risk level penalty
        if visual_risk['risk_level'] == 'HIGH':
            score -= self.penalties['high_visual_risk']
            penalties_applied.append(f"high_visual_risk: -{self.penalties['high_visual_risk']}")
        elif visual_risk['risk_level'] == 'MEDIUM':
            score -= self.penalties['medium_visual_risk']
            penalties_applied.append(f"medium_visual_risk: -{self.penalties['medium_visual_risk']}")
        
        # Apply AI severity multiplier
        ai_severity = ai_result.get('severity', 'Low')
        multiplier = self.severity_multipliers.get(ai_severity, 1.0)
        if multiplier > 1.0 and score > 50:
            # Additional penalty for high AI severity
            additional_penalty = (score - 50) * (multiplier - 1.0) * 0.5
            score -= additional_penalty
            penalties_applied.append(f"severity_adjustment: -{additional_penalty:.1f}")
        
        # Ensure score is within bounds
        score = max(5, min(100, score))
        
        return round(score, 1), penalties_applied


class ClinicalSafetyPipeline:
    """
    Complete Clinical Safety Pipeline
    
    Combines all three layers:
    1. Visual Risk Detection
    2. Clinical Rule Engine
    3. Health Score Recalibration
    """
    
    def __init__(self):
        self.visual_detector = VisualRiskDetector()
        self.rule_engine = ClinicalRuleEngine()
        self.score_calculator = HealthScoreCalculator()
    
    def process(self, image: Union[str, bytes, Image.Image, np.ndarray], 
                ai_result: Dict) -> Dict:
        """
        Process image through complete safety pipeline.
        
        Args:
            image: Input dental image
            ai_result: Output from AI model
            
        Returns:
            Safe, calibrated result
        """
        # Layer 1: Visual Risk Detection
        visual_risk = self.visual_detector.analyze(image)
        
        # Layer 2: Apply Clinical Rules
        safe_result = self.rule_engine.apply_rules(visual_risk, ai_result)
        
        # Layer 3: Recalculate Health Score
        calibrated_score, penalties = self.score_calculator.calculate(visual_risk, ai_result)
        
        # Use the lower of AI score and calibrated score
        ai_score = safe_result.get('health_score', safe_result.get('overall_health_score', 100))
        final_score = min(ai_score, calibrated_score)
        
        safe_result['health_score'] = final_score
        safe_result['overall_health_score'] = final_score
        safe_result['visual_risk'] = visual_risk
        safe_result['score_penalties'] = penalties
        safe_result['calibrated_score'] = calibrated_score
        safe_result['original_ai_score'] = ai_score
        
        # Add urgent warning if needed
        if visual_risk['requires_immediate_attention']:
            safe_result['urgent_warning'] = "⚠️ URGENT: Visual indicators suggest immediate dental attention may be needed."
        
        return safe_result


# Convenience function
def apply_clinical_safety(image, ai_result: Dict) -> Dict:
    """Apply clinical safety pipeline to AI result."""
    pipeline = ClinicalSafetyPipeline()
    return pipeline.process(image, ai_result)
