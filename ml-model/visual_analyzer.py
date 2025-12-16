"""
Visual Analysis Module for Dental Images
=========================================
Uses color and texture analysis to detect dental conditions
when the ML model predictions are unreliable.

This supplements the ML model with rule-based visual analysis.
"""

import numpy as np
from PIL import Image
import io
from typing import Dict, Union, Tuple
from pathlib import Path


class VisualDentalAnalyzer:
    """
    Analyzes dental images using color and texture features
    to detect tartar, decay, and overall dental health.
    """
    
    def __init__(self):
        # Color ranges for different conditions (in HSV)
        # Tartar/Calculus: Yellow-brown deposits
        self.tartar_hue_range = (15, 45)  # Yellow-brown hues
        self.tartar_sat_min = 30  # Minimum saturation
        
        # Decay/Caries: Dark brown to black spots
        self.decay_value_max = 80  # Dark areas
        self.decay_sat_range = (20, 100)
        
        # Healthy teeth: White/off-white
        self.healthy_sat_max = 30  # Low saturation (white)
        self.healthy_value_min = 180  # Bright
        
        # Gum inflammation: Red/pink areas
        self.inflammation_hue_range = (0, 15)  # Red hues
        self.inflammation_sat_min = 40
    
    def analyze(self, image: Union[str, bytes, Image.Image, np.ndarray]) -> Dict:
        """
        Analyze dental image for visual indicators of conditions.
        
        Returns:
            Dict with detected conditions and confidence scores
        """
        # Load image
        img = self._load_image(image)
        img_array = np.array(img)
        
        # Convert to HSV for color analysis
        hsv = self._rgb_to_hsv(img_array)
        
        # Analyze different conditions
        tartar_score = self._detect_tartar(hsv, img_array)
        decay_score = self._detect_decay(hsv, img_array)
        healthy_score = self._detect_healthy(hsv, img_array)
        inflammation_score = self._detect_inflammation(hsv, img_array)
        
        # Calculate overall health score - EXTREMELY AGGRESSIVE penalties
        # Higher tartar/decay = MUCH lower health
        
        # Base health from healthy tooth detection
        base_health = healthy_score * 70 + 15  # 15-85 range (lower ceiling)
        
        # VERY AGGRESSIVE penalties for detected conditions
        decay_penalty = decay_score * 80  # Decay is most serious
        tartar_penalty = tartar_score * 70  # Tartar is very serious
        inflammation_penalty = inflammation_score * 40
        
        # Combined penalty (can exceed 100)
        total_penalty = decay_penalty + tartar_penalty + inflammation_penalty
        
        # Apply penalty
        overall_health = max(5, base_health - total_penalty)
        
        # EXTRA penalty for multiple conditions
        conditions_detected = sum([
            1 if decay_score > 0.25 else 0,  # Lower threshold
            1 if tartar_score > 0.25 else 0,
            1 if inflammation_score > 0.25 else 0
        ])
        if conditions_detected >= 2:
            overall_health = min(overall_health, 20)  # Cap at 20% for multiple conditions
        if conditions_detected >= 3:
            overall_health = min(overall_health, 12)  # Cap at 12% for all conditions
        
        # Severe single condition cap - MUCH LOWER
        if decay_score > 0.5 or tartar_score > 0.5:
            overall_health = min(overall_health, 18)  # Cap at 18%
        if decay_score > 0.7 or tartar_score > 0.7:
            overall_health = min(overall_health, 12)  # Cap at 12% for severe
        
        # Combined severe conditions
        if (decay_score + tartar_score) > 0.8:
            overall_health = min(overall_health, 10)  # Cap at 10%
        
        # Determine primary condition - prioritize decay over tartar
        conditions = {
            'Caries': decay_score * 1.2,  # Boost decay priority
            'Calculus': tartar_score,
            'Gingivitis': inflammation_score,
            'Healthy': healthy_score if healthy_score > 0.6 and decay_score < 0.2 and tartar_score < 0.2 else 0
        }
        
        primary_condition = max(conditions, key=conditions.get)
        # Get actual confidence (without boost)
        actual_scores = {'Caries': decay_score, 'Calculus': tartar_score, 'Gingivitis': inflammation_score, 'Healthy': healthy_score}
        primary_confidence = actual_scores.get(primary_condition, 0)
        
        # Determine severity based on scores
        severity = self._calculate_severity(tartar_score, decay_score, inflammation_score)
        
        return {
            'primary_condition': primary_condition,
            'confidence': primary_confidence,
            'severity': severity,
            'overall_health_score': overall_health,
            'condition_scores': {
                'tartar_buildup': round(tartar_score, 3),
                'decay_cavities': round(decay_score, 3),
                'gum_inflammation': round(inflammation_score, 3),
                'healthy_teeth': round(healthy_score, 3)
            },
            'visual_indicators': self._get_visual_indicators(tartar_score, decay_score, inflammation_score, healthy_score)
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
        """Convert RGB image to HSV."""
        rgb_normalized = rgb.astype(np.float32) / 255.0
        
        r, g, b = rgb_normalized[:,:,0], rgb_normalized[:,:,1], rgb_normalized[:,:,2]
        
        max_c = np.maximum(np.maximum(r, g), b)
        min_c = np.minimum(np.minimum(r, g), b)
        diff = max_c - min_c
        
        # Hue calculation
        h = np.zeros_like(max_c)
        mask = diff != 0
        
        # Red is max
        mask_r = mask & (max_c == r)
        h[mask_r] = (60 * ((g[mask_r] - b[mask_r]) / diff[mask_r]) + 360) % 360
        
        # Green is max
        mask_g = mask & (max_c == g)
        h[mask_g] = (60 * ((b[mask_g] - r[mask_g]) / diff[mask_g]) + 120) % 360
        
        # Blue is max
        mask_b = mask & (max_c == b)
        h[mask_b] = (60 * ((r[mask_b] - g[mask_b]) / diff[mask_b]) + 240) % 360
        
        # Saturation
        s = np.zeros_like(max_c)
        s[max_c != 0] = (diff[max_c != 0] / max_c[max_c != 0]) * 100
        
        # Value
        v = max_c * 255
        
        return np.stack([h, s, v], axis=-1)
    
    def _detect_tartar(self, hsv: np.ndarray, rgb: np.ndarray) -> float:
        """
        Detect tartar/calculus buildup.
        Tartar appears as yellow-brown deposits, often near gumline.
        """
        h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
        
        # Yellow-brown color detection - WIDER RANGE
        hue_mask = (h >= 10) & (h <= 55)  # Yellow to brown
        sat_mask = s >= 20  # Lower threshold
        val_mask = (v >= 40) & (v <= 220)  # Not too dark or bright
        
        # Also check for brownish tones (higher hue range)
        brown_hue_mask = (h >= 15) & (h <= 60)
        brown_mask = brown_hue_mask & (s >= 15) & (v >= 40) & (v <= 200)
        
        # Dark yellow/orange (severe tartar)
        dark_tartar_mask = (h >= 15) & (h <= 45) & (s >= 30) & (v >= 50) & (v <= 150)
        
        # Combine masks
        tartar_mask = (hue_mask & sat_mask & val_mask) | brown_mask | dark_tartar_mask
        
        # Calculate percentage of image with tartar indicators
        tartar_ratio = np.sum(tartar_mask) / tartar_mask.size
        
        # Scale to 0-1 confidence - MORE SENSITIVE (tartar covering >8% is severe)
        confidence = min(1.0, tartar_ratio / 0.08)
        
        # Boost if significant tartar detected
        if tartar_ratio > 0.15:
            confidence = min(1.0, confidence + 0.3)
        
        return confidence
    
    def _detect_decay(self, hsv: np.ndarray, rgb: np.ndarray) -> float:
        """
        Detect decay/cavities.
        Decay appears as dark brown/black spots on teeth.
        """
        h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
        
        # AGGRESSIVE: Multiple decay indicators
        
        # 1. Very dark spots (black cavities) - strongest indicator
        very_dark_mask = v <= 50  # Increased threshold
        
        # 2. Dark brown areas (decay)
        dark_brown_mask = (v <= 120) & (h >= 0) & (h <= 50) & (s >= 20)
        
        # 3. Dark areas with some saturation (not pure shadows)
        dark_mask = (v <= 90) & (s >= 10)
        
        # 4. Gray-black areas on teeth (cavities often appear grayish)
        gray_decay_mask = (v <= 130) & (s <= 35) & (v >= 15)
        
        # 5. Dark spots near brown (decay with staining)
        stained_decay_mask = (v <= 100) & (h >= 0) & (h <= 60) & (s >= 15)
        
        # Combine all decay indicators
        decay_mask = very_dark_mask | dark_brown_mask | (dark_mask & (h <= 70)) | gray_decay_mask | stained_decay_mask
        
        # Calculate ratio
        decay_ratio = np.sum(decay_mask) / decay_mask.size
        
        # MORE SENSITIVE scaling (decay covering >3% is severe)
        confidence = min(1.0, decay_ratio / 0.03)
        
        # Boost confidence if very dark spots are present
        very_dark_ratio = np.sum(very_dark_mask) / very_dark_mask.size
        if very_dark_ratio > 0.01:  # More than 1% very dark
            confidence = min(1.0, confidence + 0.4)
        
        # Additional boost for significant dark areas
        if decay_ratio > 0.08:
            confidence = min(1.0, confidence + 0.3)
        
        return confidence
    
    def _detect_healthy(self, hsv: np.ndarray, rgb: np.ndarray) -> float:
        """
        Detect healthy white teeth.
        Healthy teeth are white/off-white with low saturation.
        """
        h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
        
        # White/bright areas with low saturation
        white_mask = (s <= self.healthy_sat_max) & (v >= self.healthy_value_min)
        
        # Also include slightly off-white (cream colored healthy teeth)
        offwhite_mask = (s <= 40) & (v >= 160) & (h <= 60)
        
        healthy_mask = white_mask | offwhite_mask
        
        # Calculate ratio of healthy-looking areas
        healthy_ratio = np.sum(healthy_mask) / healthy_mask.size
        
        # Scale (>40% white/healthy areas indicates good health)
        confidence = min(1.0, healthy_ratio / 0.40)
        
        return confidence
    
    def _detect_inflammation(self, hsv: np.ndarray, rgb: np.ndarray) -> float:
        """
        Detect gum inflammation (gingivitis).
        Inflamed gums appear red/dark pink.
        """
        h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
        
        # Red/pink areas (gums)
        red_mask = ((h >= 0) & (h <= self.inflammation_hue_range[1])) | (h >= 340)
        sat_mask = s >= self.inflammation_sat_min
        bright_mask = v >= 80  # Not too dark
        
        inflammation_mask = red_mask & sat_mask & bright_mask
        
        # Calculate ratio
        inflammation_ratio = np.sum(inflammation_mask) / inflammation_mask.size
        
        # Scale (>20% red areas might indicate inflammation)
        # But some red is normal (gums), so threshold is higher
        confidence = min(1.0, max(0, (inflammation_ratio - 0.1) / 0.15))
        
        return confidence
    
    def _calculate_severity(self, tartar: float, decay: float, inflammation: float) -> str:
        """Calculate overall severity based on condition scores."""
        max_score = max(tartar, decay, inflammation)
        # Weight decay and tartar more heavily
        combined = tartar * 0.4 + decay * 0.5 + inflammation * 0.3
        
        # VERY AGGRESSIVE thresholds - any significant detection = High
        if max_score >= 0.35 or combined >= 0.25:
            return 'High'
        elif max_score >= 0.15 or combined >= 0.10:
            return 'Medium'
        else:
            return 'Low'
    
    def _get_visual_indicators(self, tartar: float, decay: float, 
                               inflammation: float, healthy: float) -> list:
        """Get list of visual indicators found."""
        indicators = []
        
        if tartar >= 0.3:
            indicators.append(f"Yellow/brown deposits detected (tartar likelihood: {tartar:.0%})")
        if decay >= 0.3:
            indicators.append(f"Dark spots detected (decay likelihood: {decay:.0%})")
        if inflammation >= 0.3:
            indicators.append(f"Redness detected (inflammation likelihood: {inflammation:.0%})")
        if healthy >= 0.5:
            indicators.append(f"White/healthy tooth surfaces detected ({healthy:.0%})")
        
        if not indicators:
            indicators.append("No significant visual indicators detected")
        
        return indicators


def analyze_dental_image_visual(image: Union[str, bytes, Image.Image, np.ndarray]) -> Dict:
    """Convenience function for visual analysis."""
    analyzer = VisualDentalAnalyzer()
    return analyzer.analyze(image)
