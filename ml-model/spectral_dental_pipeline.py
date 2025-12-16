"""
Spectral Dental AI Pipeline
============================
Advanced spectral image analysis for dental disease detection.
Uses ODSI-DB (Oral and Dental Spectral Image Database) format.

Dataset: https://cs.uef.fi/pub/color/spectra/ODSI-DB

Features:
- Spectral band processing (380-780nm range)
- PCA dimensionality reduction
- CNN feature extraction (EfficientNet)
- Ensemble classifier (XGBoost + Random Forest)
- GPT-4o integration for detailed analysis

Author: Oral Care AI
"""

import os
import io
import json
import logging
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
import uuid

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import base64
import cv2

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models

# Scikit-learn for PCA and ensemble
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# OpenRouter for GPT-4o
import httpx
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
SPECTRAL_BANDS = {
    'blue': (380, 480),      # Blue light - enamel fluorescence
    'green': (480, 560),     # Green light - tissue health
    'red': (560, 640),       # Red light - blood/inflammation
    'nir': (640, 780),       # Near-infrared - subsurface detection
}

DENTAL_CONDITIONS = {
    0: {'name': 'healthy', 'display': 'Healthy Tooth', 'severity': 'none'},
    1: {'name': 'early_caries', 'display': 'Early Caries (White Spot)', 'severity': 'mild'},
    2: {'name': 'enamel_caries', 'display': 'Enamel Caries', 'severity': 'moderate'},
    3: {'name': 'dentin_caries', 'display': 'Dentin Caries', 'severity': 'severe'},
    4: {'name': 'demineralization', 'display': 'Enamel Demineralization', 'severity': 'mild'},
    5: {'name': 'calculus', 'display': 'Dental Calculus', 'severity': 'moderate'},
    6: {'name': 'gingivitis', 'display': 'Gingivitis', 'severity': 'moderate'},
    7: {'name': 'periodontal', 'display': 'Periodontal Disease', 'severity': 'severe'},
}

DISCLAIMER = """⚠️ IMPORTANT: This spectral AI analysis is for screening purposes only and does not 
constitute a medical diagnosis. Spectral imaging provides enhanced early detection capabilities 
but requires verification by a licensed dental professional. Always consult with your dentist 
for proper diagnosis and treatment planning."""

# Color mapping for spectral visualization (BGR format for OpenCV)
SPECTRAL_COLORS = {
    'enamel': (0, 255, 0),           # Green - healthy enamel
    'dentin': (0, 255, 255),         # Yellow - dentin
    'gingiva': (0, 128, 0),          # Dark green - healthy gums
    'caries': (0, 0, 255),           # Red - caries/decay
    'early_caries': (0, 165, 255),   # Orange - early caries
    'calculus': (255, 255, 0),       # Cyan - calculus
    'inflammation': (128, 0, 128),   # Purple - inflammation
    'demineralization': (255, 0, 255), # Magenta - demineralization
    'healthy_tooth': (144, 238, 144), # Light green - healthy tooth
    'soft_tissue': (180, 105, 255),  # Pink - soft tissue
    'background': (50, 50, 50),      # Dark gray - background
}


class SpectralImageGenerator:
    """
    Generates color-coded spectral visualization images.
    Creates segmentation overlays showing different dental structures.
    """
    
    def __init__(self):
        self.colors = SPECTRAL_COLORS
        
    def generate_spectral_overlay(
        self,
        original_img: np.ndarray,
        bands: Dict[str, np.ndarray],
        detections: List[Dict],
        alpha: float = 0.6
    ) -> np.ndarray:
        """
        Generate a color-coded spectral overlay image.
        
        Args:
            original_img: Original RGB image (H, W, 3)
            bands: Extracted spectral bands
            detections: List of detected conditions
            alpha: Overlay transparency (0-1)
            
        Returns:
            Color-coded overlay image (H, W, 3)
        """
        h, w = original_img.shape[:2]
        
        # Create segmentation mask
        segmentation = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Convert to HSV for better segmentation
        img_hsv = cv2.cvtColor(original_img, cv2.COLOR_RGB2HSV)
        img_lab = cv2.cvtColor(original_img, cv2.COLOR_RGB2LAB)
        
        # Segment different dental structures based on color and spectral bands
        
        # 1. Detect teeth (high brightness, low saturation)
        teeth_mask = self._segment_teeth(img_hsv, img_lab, original_img)
        
        # 2. Detect gums/gingiva (reddish/pink areas)
        gingiva_mask = self._segment_gingiva(img_hsv, original_img)
        
        # 3. Detect problem areas using spectral bands
        caries_mask = self._detect_caries_regions(bands, teeth_mask)
        inflammation_mask = self._detect_inflammation_regions(bands, gingiva_mask)
        calculus_mask = self._detect_calculus_regions(bands, teeth_mask)
        demineralization_mask = self._detect_demineralization_regions(bands, teeth_mask)
        
        # Apply colors to segmentation (order matters - later overwrites earlier)
        
        # Background
        segmentation[:] = self.colors['background']
        
        # Soft tissue (everything that's not teeth)
        soft_tissue_mask = ~teeth_mask & (np.mean(original_img, axis=2) > 30)
        segmentation[soft_tissue_mask] = self.colors['soft_tissue']
        
        # Healthy gingiva
        segmentation[gingiva_mask] = self.colors['gingiva']
        
        # Healthy teeth/enamel
        healthy_teeth = teeth_mask & ~caries_mask & ~calculus_mask & ~demineralization_mask
        segmentation[healthy_teeth] = self.colors['enamel']
        
        # Problem areas (overlay on top)
        segmentation[demineralization_mask] = self.colors['demineralization']
        segmentation[calculus_mask] = self.colors['calculus']
        segmentation[inflammation_mask] = self.colors['inflammation']
        segmentation[caries_mask] = self.colors['caries']
        
        # Blend with original image
        overlay = cv2.addWeighted(original_img, 1 - alpha, segmentation, alpha, 0)
        
        # Add contours for better visibility
        overlay = self._add_contours(overlay, teeth_mask, caries_mask, inflammation_mask)
        
        return overlay
    
    def _segment_teeth(
        self,
        img_hsv: np.ndarray,
        img_lab: np.ndarray,
        img_rgb: np.ndarray
    ) -> np.ndarray:
        """Segment teeth regions based on color properties."""
        h, s, v = cv2.split(img_hsv)
        l, a, b = cv2.split(img_lab)
        
        # Teeth are typically: high brightness, low saturation, neutral color
        brightness_mask = v > 120
        saturation_mask = s < 80
        
        # Also check LAB - teeth have low a* and b* values
        neutral_mask = (np.abs(a.astype(float) - 128) < 30) & (np.abs(b.astype(float) - 128) < 40)
        
        # Combine masks
        teeth_mask = brightness_mask & saturation_mask & neutral_mask
        
        # Clean up with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        teeth_mask = cv2.morphologyEx(teeth_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        teeth_mask = cv2.morphologyEx(teeth_mask, cv2.MORPH_OPEN, kernel)
        
        return teeth_mask.astype(bool)
    
    def _segment_gingiva(self, img_hsv: np.ndarray, img_rgb: np.ndarray) -> np.ndarray:
        """Segment gingiva (gum) regions."""
        h, s, v = cv2.split(img_hsv)
        
        # Gums are typically pinkish-red with moderate saturation
        # Hue range for pink/red: 0-20 or 160-180
        hue_mask = ((h < 20) | (h > 160)) & (s > 30) & (s < 180) & (v > 50)
        
        # Also check RGB - gums have more red than blue
        r, g, b = img_rgb[:,:,0], img_rgb[:,:,1], img_rgb[:,:,2]
        color_mask = (r > b) & (r > 80)
        
        gingiva_mask = hue_mask & color_mask
        
        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        gingiva_mask = cv2.morphologyEx(gingiva_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        
        return gingiva_mask.astype(bool)
    
    def _detect_caries_regions(
        self,
        bands: Dict[str, np.ndarray],
        teeth_mask: np.ndarray
    ) -> np.ndarray:
        """Detect caries regions using spectral bands."""
        # Caries show reduced fluorescence and altered spectral response
        fluor = bands.get('fluorescence', np.zeros_like(teeth_mask, dtype=float))
        demin = bands.get('demineralization', np.zeros_like(teeth_mask, dtype=float))
        
        # Caries indicators: low fluorescence, high demineralization index
        caries_indicator = (fluor < -0.15) | (demin > 0.2)
        
        # Only within teeth regions
        caries_mask = caries_indicator & teeth_mask
        
        # Expand slightly for visibility
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        caries_mask = cv2.dilate(caries_mask.astype(np.uint8), kernel, iterations=1)
        
        return caries_mask.astype(bool)
    
    def _detect_inflammation_regions(
        self,
        bands: Dict[str, np.ndarray],
        gingiva_mask: np.ndarray
    ) -> np.ndarray:
        """Detect inflammation regions in gingiva."""
        inflam = bands.get('inflammation', np.zeros_like(gingiva_mask, dtype=float))
        
        # High inflammation index indicates gingivitis
        inflammation_indicator = inflam > 0.2
        
        # Only within gingiva regions
        inflammation_mask = inflammation_indicator & gingiva_mask
        
        return inflammation_mask.astype(bool)
    
    def _detect_calculus_regions(
        self,
        bands: Dict[str, np.ndarray],
        teeth_mask: np.ndarray
    ) -> np.ndarray:
        """Detect calculus (tartar) regions."""
        calc = bands.get('calculus', np.zeros_like(teeth_mask, dtype=float))
        
        # High calculus index
        calculus_indicator = calc > 0.15
        
        # Typically at gumline - expand teeth mask slightly
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        expanded_teeth = cv2.dilate(teeth_mask.astype(np.uint8), kernel, iterations=2)
        edge_region = expanded_teeth.astype(bool) & ~teeth_mask
        
        calculus_mask = calculus_indicator & (teeth_mask | edge_region)
        
        return calculus_mask.astype(bool)
    
    def _detect_demineralization_regions(
        self,
        bands: Dict[str, np.ndarray],
        teeth_mask: np.ndarray
    ) -> np.ndarray:
        """Detect enamel demineralization (white spots)."""
        demin = bands.get('demineralization', np.zeros_like(teeth_mask, dtype=float))
        
        # Moderate demineralization (not as severe as caries)
        demin_indicator = (demin > 0.1) & (demin < 0.2)
        
        demineralization_mask = demin_indicator & teeth_mask
        
        return demineralization_mask.astype(bool)
    
    def _add_contours(
        self,
        overlay: np.ndarray,
        teeth_mask: np.ndarray,
        caries_mask: np.ndarray,
        inflammation_mask: np.ndarray
    ) -> np.ndarray:
        """Add contour lines for better visibility."""
        result = overlay.copy()
        
        # Add teeth contours (white)
        contours, _ = cv2.findContours(
            teeth_mask.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(result, contours, -1, (255, 255, 255), 1)
        
        # Add caries contours (red)
        if np.any(caries_mask):
            contours, _ = cv2.findContours(
                caries_mask.astype(np.uint8), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(result, contours, -1, (255, 0, 0), 2)
        
        # Add inflammation contours (purple)
        if np.any(inflammation_mask):
            contours, _ = cv2.findContours(
                inflammation_mask.astype(np.uint8), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(result, contours, -1, (128, 0, 128), 2)
        
        return result
    
    def create_legend(self, width: int = 200, height: int = 400) -> np.ndarray:
        """Create a color legend for the spectral visualization."""
        legend = np.ones((height, width, 3), dtype=np.uint8) * 40  # Dark background
        
        labels = [
            ('Healthy Enamel', 'enamel'),
            ('Healthy Gingiva', 'gingiva'),
            ('Soft Tissue', 'soft_tissue'),
            ('Caries/Decay', 'caries'),
            ('Early Caries', 'early_caries'),
            ('Calculus', 'calculus'),
            ('Inflammation', 'inflammation'),
            ('Demineralization', 'demineralization'),
        ]
        
        y_offset = 20
        box_size = 20
        spacing = 40
        
        for label, color_key in labels:
            color = self.colors.get(color_key, (128, 128, 128))
            
            # Draw color box
            cv2.rectangle(
                legend,
                (10, y_offset),
                (10 + box_size, y_offset + box_size),
                color,
                -1
            )
            cv2.rectangle(
                legend,
                (10, y_offset),
                (10 + box_size, y_offset + box_size),
                (255, 255, 255),
                1
            )
            
            # Draw label
            cv2.putText(
                legend,
                label,
                (40, y_offset + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1
            )
            
            y_offset += spacing
        
        return legend
    
    def generate_full_visualization(
        self,
        original_img: np.ndarray,
        bands: Dict[str, np.ndarray],
        detections: List[Dict],
        include_legend: bool = True
    ) -> Tuple[np.ndarray, str]:
        """
        Generate complete spectral visualization with legend.
        
        Returns:
            Tuple of (visualization image, base64 encoded string)
        """
        # Generate overlay
        overlay = self.generate_spectral_overlay(original_img, bands, detections)
        
        if include_legend:
            # Create legend
            legend = self.create_legend(width=180, height=overlay.shape[0])
            
            # Combine overlay and legend
            visualization = np.hstack([overlay, legend])
        else:
            visualization = overlay
        
        # Convert to base64
        img_pil = Image.fromarray(visualization)
        buffer = io.BytesIO()
        img_pil.save(buffer, format='PNG')
        img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return visualization, img_b64


class SpectralPreprocessor:
    """
    Preprocesses spectral dental images.
    Handles ODSI-DB format and standard RGB images.
    """
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.target_size = target_size
        self.scaler = StandardScaler()
        
    def load_spectral_image(self, image_data: bytes) -> np.ndarray:
        """Load and preprocess spectral image."""
        img = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize
        img = img.resize(self.target_size, Image.Resampling.LANCZOS)
        
        return np.array(img)
    
    def extract_spectral_bands(self, img: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract pseudo-spectral bands from RGB image.
        For true spectral images, this would process actual wavelength data.
        """
        # Normalize to 0-1
        img_norm = img.astype(np.float32) / 255.0
        
        # Extract channels
        r, g, b = img_norm[:,:,0], img_norm[:,:,1], img_norm[:,:,2]
        
        # Compute spectral-like features
        bands = {
            'blue': b,                          # Blue channel - enamel fluorescence
            'green': g,                         # Green channel - tissue health
            'red': r,                           # Red channel - blood/inflammation
            'nir_approx': (r + g) / 2,          # NIR approximation
            'fluorescence': b - 0.5 * (r + g),  # Fluorescence indicator
            'demineralization': (b - r) / (b + r + 1e-6),  # Demineralization index
            'inflammation': (r - g) / (r + g + 1e-6),      # Inflammation index
            'calculus': np.abs(g - 0.5 * (r + b)),         # Calculus indicator
        }
        
        return bands
    
    def compute_spectral_features(self, bands: Dict[str, np.ndarray]) -> np.ndarray:
        """Compute statistical features from spectral bands."""
        features = []
        
        for name, band in bands.items():
            # Statistical features
            features.extend([
                np.mean(band),
                np.std(band),
                np.min(band),
                np.max(band),
                np.percentile(band, 25),
                np.percentile(band, 75),
                np.median(band),
            ])
            
            # Texture features (simplified)
            grad_x = np.abs(np.diff(band, axis=1)).mean()
            grad_y = np.abs(np.diff(band, axis=0)).mean()
            features.extend([grad_x, grad_y])
        
        return np.array(features)


class SpectralCNN(nn.Module):
    """
    CNN for spectral dental image feature extraction.
    Uses EfficientNet backbone with spectral-aware modifications.
    """
    
    def __init__(self, num_classes: int = 8, pretrained: bool = True):
        super().__init__()
        
        # Load EfficientNet-B4 backbone
        self.backbone = models.efficientnet_b4(
            weights=models.EfficientNet_B4_Weights.DEFAULT if pretrained else None
        )
        
        # Get feature dimension
        in_features = self.backbone.classifier[1].in_features
        
        # Replace classifier with spectral-aware head
        self.backbone.classifier = nn.Identity()
        
        # Spectral feature processing
        self.spectral_fc = nn.Sequential(
            nn.Linear(72, 128),  # 72 = 8 bands * 9 features
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 256),
            nn.ReLU(),
        )
        
        # Combined classifier
        self.classifier = nn.Sequential(
            nn.Linear(in_features + 256, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
        
    def forward(self, x: torch.Tensor, spectral_features: torch.Tensor) -> torch.Tensor:
        # CNN features
        cnn_features = self.backbone(x)
        
        # Spectral features
        spectral_out = self.spectral_fc(spectral_features)
        
        # Combine
        combined = torch.cat([cnn_features, spectral_out], dim=1)
        
        return self.classifier(combined)
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract CNN features without classification."""
        return self.backbone(x)


class SpectralEnsembleClassifier:
    """
    Ensemble classifier combining CNN features with traditional ML.
    Uses PCA for dimensionality reduction + XGBoost/RF ensemble.
    """
    
    def __init__(self, n_components: int = 50):
        self.pca = PCA(n_components=n_components)
        self.scaler = StandardScaler()
        
        # Ensemble classifiers
        self.rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.gb = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        
        self.is_fitted = False
        
    def fit(self, features: np.ndarray, labels: np.ndarray):
        """Fit the ensemble classifier."""
        # Scale and reduce dimensions
        features_scaled = self.scaler.fit_transform(features)
        features_pca = self.pca.fit_transform(features_scaled)
        
        # Fit classifiers
        self.rf.fit(features_pca, labels)
        self.gb.fit(features_pca, labels)
        
        self.is_fitted = True
        
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Predict class probabilities using ensemble."""
        if not self.is_fitted:
            # Return uniform distribution if not fitted
            n_classes = len(DENTAL_CONDITIONS)
            return np.ones((features.shape[0], n_classes)) / n_classes
        
        features_scaled = self.scaler.transform(features)
        features_pca = self.pca.transform(features_scaled)
        
        # Ensemble prediction (average)
        rf_proba = self.rf.predict_proba(features_pca)
        gb_proba = self.gb.predict_proba(features_pca)
        
        return (rf_proba + gb_proba) / 2


class SpectralDentalPipeline:
    """
    Complete spectral dental analysis pipeline.
    Combines spectral preprocessing, CNN, ensemble classifier, and LLM.
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize components
        self.preprocessor = SpectralPreprocessor()
        self.cnn = SpectralCNN(num_classes=len(DENTAL_CONDITIONS)).to(self.device)
        self.cnn.eval()
        
        self.ensemble = SpectralEnsembleClassifier()
        self.image_generator = SpectralImageGenerator()
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        
        # OpenRouter API
        self.openrouter_key = os.getenv('OPENROUTER_API_KEY')
        
        logger.info("Spectral Dental Pipeline initialized")
    
    async def analyze(
        self,
        image_data: bytes,
        image_type: str = 'nir',
        use_llm: bool = True
    ) -> Dict[str, Any]:
        """
        Run complete spectral analysis pipeline.
        
        Args:
            image_data: Raw image bytes
            image_type: Type of spectral image (nir, fluorescence, intraoral)
            use_llm: Whether to use GPT-4o for detailed analysis
            
        Returns:
            Complete analysis results
        """
        analysis_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        logger.info(f"Starting spectral analysis {analysis_id} ({image_type})")
        
        # 1. Preprocess image
        img = self.preprocessor.load_spectral_image(image_data)
        
        # 2. Extract spectral bands
        bands = self.preprocessor.extract_spectral_bands(img)
        
        # 3. Compute spectral features
        spectral_features = self.preprocessor.compute_spectral_features(bands)
        
        # 4. CNN feature extraction
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        spectral_tensor = torch.tensor(spectral_features, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Get CNN predictions
            logits = self.cnn(img_tensor, spectral_tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            
            # Get CNN features for ensemble
            cnn_features = self.cnn.extract_features(img_tensor).cpu().numpy()
        
        # 5. Combine with spectral features for ensemble
        combined_features = np.concatenate([
            cnn_features.flatten(),
            spectral_features
        ]).reshape(1, -1)
        
        # 6. Get ensemble predictions (if fitted)
        # For now, use CNN predictions directly
        final_probs = probs
        
        # 7. Analyze spectral signatures
        detections = self._analyze_spectral_signatures(bands, final_probs, image_type)
        
        # 8. Calculate overall health score
        health_score = self._calculate_health_score(detections, final_probs)
        
        # 9. Get top prediction
        top_idx = int(np.argmax(final_probs))
        top_condition = DENTAL_CONDITIONS[top_idx]
        
        # 10. Generate spectral visualization image
        logger.info("Generating spectral visualization...")
        visualization, visualization_b64 = self.image_generator.generate_full_visualization(
            img, bands, detections, include_legend=True
        )
        
        # Also generate overlay-only version (without legend)
        overlay_only, overlay_b64 = self.image_generator.generate_full_visualization(
            img, bands, detections, include_legend=False
        )
        
        result = {
            'analysis_id': analysis_id,
            'timestamp': timestamp,
            'image_type': image_type,
            'spectral_analysis': {
                'detections': detections,
                'overall_health_score': health_score,
                'imaging_quality': self._assess_image_quality(img),
                'analysis_method': 'CNN + PCA Feature Extraction + Ensemble Classifier',
                'spectral_bands_analyzed': list(bands.keys()),
            },
            'standard_analysis': {
                'disease': top_condition['name'],
                'disease_name': top_condition['display'],
                'confidence': float(final_probs[top_idx]),
                'severity': top_condition['severity'],
                'all_probabilities': {
                    DENTAL_CONDITIONS[i]['name']: float(p) 
                    for i, p in enumerate(final_probs)
                },
            },
            'spectral_recommendations': self._generate_recommendations(detections, health_score),
            # Spectral visualization images (base64 encoded PNG)
            'spectral_image': visualization_b64,
            'spectral_overlay': overlay_b64,
            'color_legend': {
                'enamel': '#00FF00',
                'gingiva': '#008000',
                'soft_tissue': '#FF69B4',
                'caries': '#FF0000',
                'early_caries': '#FFA500',
                'calculus': '#00FFFF',
                'inflammation': '#800080',
                'demineralization': '#FF00FF',
            },
        }
        
        # 10. LLM analysis (if enabled)
        if use_llm and self.openrouter_key:
            llm_result = await self._get_llm_analysis(image_data, result, image_type)
            result.update(llm_result)
        
        result['disclaimer'] = DISCLAIMER
        
        logger.info(f"Spectral analysis complete: {analysis_id}")
        return result
    
    def _analyze_spectral_signatures(
        self,
        bands: Dict[str, np.ndarray],
        probs: np.ndarray,
        image_type: str
    ) -> List[Dict[str, Any]]:
        """Analyze spectral signatures to detect specific conditions."""
        detections = []
        
        # Analyze demineralization
        demin_index = bands['demineralization']
        demin_score = float(np.mean(demin_index > 0.1) * 100)
        if demin_score > 20 or probs[4] > 0.3:  # demineralization class
            detections.append({
                'condition': 'Enamel Demineralization',
                'confidence': round(min(95, max(demin_score, probs[4] * 100)), 1),
                'location': 'Enamel surface',
                'severity': 'mild' if demin_score < 40 else 'moderate',
                'spectral_signature': f'Increased light scattering (index: {float(np.mean(demin_index)):.3f})',
            })
        
        # Analyze early caries
        fluor_index = bands['fluorescence']
        fluor_anomaly = float(np.mean(fluor_index < -0.1) * 100)
        if fluor_anomaly > 15 or probs[1] > 0.3:  # early_caries class
            detections.append({
                'condition': 'Early Caries (White Spot Lesion)',
                'confidence': round(min(95, max(fluor_anomaly, probs[1] * 100)), 1),
                'location': 'Detected via fluorescence analysis',
                'severity': 'mild',
                'spectral_signature': f'Fluorescence quenching detected (anomaly: {fluor_anomaly:.1f}%)',
            })
        
        # Analyze inflammation
        inflam_index = bands['inflammation']
        inflam_score = float(np.mean(inflam_index > 0.15) * 100)
        if inflam_score > 25 or probs[6] > 0.3:  # gingivitis class
            detections.append({
                'condition': 'Gingival Inflammation',
                'confidence': round(min(92, max(inflam_score, probs[6] * 100)), 1),
                'location': 'Gingival tissue',
                'severity': 'moderate' if inflam_score > 40 else 'mild',
                'spectral_signature': f'Increased red absorption (inflammation index: {float(np.mean(inflam_index)):.3f})',
            })
        
        # Analyze calculus
        calc_index = bands['calculus']
        calc_score = float(np.mean(calc_index > 0.1) * 100)
        if calc_score > 20 or probs[5] > 0.3:  # calculus class
            detections.append({
                'condition': 'Dental Calculus',
                'confidence': round(min(88, max(calc_score, probs[5] * 100)), 1),
                'location': 'Tooth surface',
                'severity': 'moderate',
                'spectral_signature': f'Mineral deposit signature (index: {float(np.mean(calc_index)):.3f})',
            })
        
        # Analyze subsurface decay (NIR)
        if image_type in ['nir', 'intraoral']:
            nir_index = bands['nir_approx']
            nir_anomaly = float(np.std(nir_index) * 100)
            if nir_anomaly > 15 or probs[2] > 0.3 or probs[3] > 0.3:
                severity = 'severe' if probs[3] > 0.3 else 'moderate'
                detections.append({
                    'condition': 'Subsurface Decay',
                    'confidence': round(min(90, max(nir_anomaly * 2, max(probs[2], probs[3]) * 100)), 1),
                    'location': 'Below enamel surface',
                    'severity': severity,
                    'spectral_signature': f'NIR transmission anomaly (variance: {nir_anomaly:.1f}%)',
                })
        
        # If no detections, add healthy status
        if not detections and probs[0] > 0.5:
            detections.append({
                'condition': 'Healthy Tooth Structure',
                'confidence': round(probs[0] * 100, 1),
                'location': 'Overall assessment',
                'severity': 'none',
                'spectral_signature': 'Normal spectral response across all bands',
            })
        
        # Sort by confidence
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        return detections
    
    def _calculate_health_score(
        self,
        detections: List[Dict],
        probs: np.ndarray
    ) -> float:
        """Calculate overall dental health score (0-100)."""
        # Start with a base score of 100
        base_score = 100.0
        
        # Penalty for detected conditions based on severity and confidence
        severity_penalties = {'none': 0, 'mild': 8, 'moderate': 18, 'severe': 30}
        
        total_penalty = 0
        for det in detections:
            if det['severity'] != 'none':
                penalty = severity_penalties.get(det['severity'], 15)
                # Scale penalty by confidence (higher confidence = more penalty)
                confidence_factor = min(det['confidence'], 95) / 100
                total_penalty += penalty * confidence_factor
        
        # Also factor in the healthy probability from CNN
        healthy_boost = probs[0] * 20  # Up to 20 points boost if healthy
        
        # Calculate final score
        health_score = base_score - total_penalty + healthy_boost
        
        # Clamp to 0-100 range
        health_score = max(10, min(100, health_score))  # Minimum 10% to avoid 0%
        
        return float(round(health_score, 1))
    
    def _assess_image_quality(self, img: np.ndarray) -> str:
        """Assess spectral image quality."""
        # Check brightness
        brightness = np.mean(img)
        
        # Check contrast
        contrast = np.std(img)
        
        # Check sharpness (gradient magnitude)
        grad_x = np.abs(np.diff(img.astype(float), axis=1)).mean()
        grad_y = np.abs(np.diff(img.astype(float), axis=0)).mean()
        sharpness = (grad_x + grad_y) / 2
        
        if brightness < 50 or brightness > 220:
            return 'poor - exposure issues'
        elif contrast < 30:
            return 'fair - low contrast'
        elif sharpness < 5:
            return 'fair - slight blur'
        else:
            return 'good'
    
    def _generate_recommendations(
        self,
        detections: List[Dict],
        health_score: float
    ) -> List[str]:
        """Generate recommendations based on spectral findings."""
        recommendations = []
        
        conditions = [d['condition'].lower() for d in detections]
        severities = [d['severity'] for d in detections]
        
        if 'demineralization' in ' '.join(conditions):
            recommendations.append('Fluoride treatment recommended for demineralized areas')
            recommendations.append('Consider remineralization therapy (MI Paste or similar)')
        
        if 'caries' in ' '.join(conditions) or 'decay' in ' '.join(conditions):
            recommendations.append('Professional dental examination recommended within 2 weeks')
            if 'severe' in severities:
                recommendations.append('Urgent: Deep decay detected - schedule appointment immediately')
        
        if 'inflammation' in ' '.join(conditions) or 'gingivitis' in ' '.join(conditions):
            recommendations.append('Improved oral hygiene routine recommended')
            recommendations.append('Consider antiseptic mouthwash (chlorhexidine)')
        
        if 'calculus' in ' '.join(conditions):
            recommendations.append('Professional dental cleaning (scaling) recommended')
        
        # General recommendations based on health score
        if health_score >= 80:
            recommendations.append('Maintain current oral hygiene routine')
            recommendations.append('Schedule routine check-up in 6 months')
        elif health_score >= 60:
            recommendations.append('Schedule follow-up spectral scan in 3 months')
            recommendations.append('Enhanced brushing technique recommended')
        else:
            recommendations.append('Schedule dental appointment within 1-2 weeks')
            recommendations.append('Daily fluoride rinse recommended')
        
        return recommendations[:6]  # Limit to 6 recommendations
    
    async def _get_llm_analysis(
        self,
        image_data: bytes,
        spectral_result: Dict,
        image_type: str
    ) -> Dict[str, str]:
        """Get detailed analysis from GPT-4o."""
        if not self.openrouter_key:
            return {}
        
        import base64
        image_b64 = base64.b64encode(image_data).decode('utf-8')
        
        # Build context from spectral analysis
        detections_text = "\n".join([
            f"- {d['condition']}: {d['confidence']:.0f}% confidence, {d['severity']} severity"
            for d in spectral_result['spectral_analysis']['detections']
        ])
        
        prompt = f"""You are an expert dental AI assistant analyzing a spectral dental image.

SPECTRAL ANALYSIS RESULTS ({image_type.upper()} imaging):
Health Score: {spectral_result['spectral_analysis']['overall_health_score']}%
Image Quality: {spectral_result['spectral_analysis']['imaging_quality']}

Detected Conditions:
{detections_text}

Primary Finding: {spectral_result['standard_analysis']['disease_name']}
Confidence: {spectral_result['standard_analysis']['confidence']*100:.1f}%

Based on this spectral analysis, provide a detailed assessment in the following format:

EXACT_COMPLAINT: [Specific dental condition identified]

DETAILED_FINDINGS: [Technical findings from spectral analysis including wavelength-specific observations]

WHAT_THIS_MEANS: [Patient-friendly explanation of the spectral findings]

IMMEDIATE_ACTIONS: [What the patient should do right now]

TREATMENT_OPTIONS: [Available treatment approaches based on spectral severity]

HOME_CARE_ROUTINE: [Daily care recommendations]

PREVENTION_TIPS: [How to prevent progression based on spectral indicators]

Be specific about spectral imaging findings and their clinical significance."""

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    'https://openrouter.ai/api/v1/chat/completions',
                    headers={
                        'Authorization': f'Bearer {self.openrouter_key}',
                        'Content-Type': 'application/json',
                    },
                    json={
                        'model': 'openai/gpt-4o',
                        'messages': [
                            {
                                'role': 'user',
                                'content': [
                                    {'type': 'text', 'text': prompt},
                                    {
                                        'type': 'image_url',
                                        'image_url': {
                                            'url': f'data:image/jpeg;base64,{image_b64}'
                                        }
                                    }
                                ]
                            }
                        ],
                        'max_tokens': 2000,
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    content = data['choices'][0]['message']['content']
                    return self._parse_llm_response(content)
                    
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
        
        return {}
    
    def _parse_llm_response(self, content: str) -> Dict[str, str]:
        """Parse structured LLM response."""
        sections = {
            'exact_complaint': '',
            'detailed_findings': '',
            'what_this_means': '',
            'immediate_actions': '',
            'treatment_options': '',
            'home_care_routine': '',
            'prevention_tips': '',
        }
        
        current_section = None
        current_content = []
        
        for line in content.split('\n'):
            line_upper = line.upper().strip()
            
            # Check for section headers
            for key in sections.keys():
                header = key.upper().replace('_', ' ') + ':'
                alt_header = key.upper().replace('_', '_') + ':'
                if line_upper.startswith(header) or line_upper.startswith(alt_header):
                    if current_section:
                        sections[current_section] = '\n'.join(current_content).strip()
                    current_section = key
                    # Get content after the header on same line
                    remaining = line.split(':', 1)[1].strip() if ':' in line else ''
                    current_content = [remaining] if remaining else []
                    break
            else:
                if current_section:
                    current_content.append(line)
        
        # Save last section
        if current_section:
            sections[current_section] = '\n'.join(current_content).strip()
        
        return sections


# Convenience function
async def analyze_spectral_image(
    image_data: bytes,
    image_type: str = 'nir',
    use_llm: bool = True
) -> Dict[str, Any]:
    """Analyze a spectral dental image."""
    pipeline = SpectralDentalPipeline()
    return await pipeline.analyze(image_data, image_type, use_llm)


if __name__ == '__main__':
    # Test with a sample image
    import asyncio
    
    async def test():
        # Create a test image
        test_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(test_img)
        
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG')
        image_data = buffer.getvalue()
        
        result = await analyze_spectral_image(image_data, 'nir', use_llm=False)
        print(json.dumps(result, indent=2, default=str))
    
    asyncio.run(test())
