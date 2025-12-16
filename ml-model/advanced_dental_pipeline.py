"""
Advanced Dental AI Pipeline
===========================
Ultimate multi-stage AI system for dental disease detection and analysis.

Pipeline Flow:
    Image -> CNN Ensemble -> Disease Probability
                |
            LLM/VLM -> Explanation + Report + Advice
                |
            Final Report with Confidence

CNN Models: EfficientNet-B4, EfficientNet-B3, ResNet50, MobileNetV3
LLM Models: GPT-4o, LLaMA-3, Mistral
VLM Models: LLaVA-1.6, Qwen-VL

DISCLAIMER: This AI provides preliminary screening only.
Always consult a qualified dental professional.
"""

import os
import json
import base64
import asyncio
import numpy as np
from pathlib import Path
from typing import Dict, List, Union, Optional, Tuple
from PIL import Image
from datetime import datetime
import io
import cv2
import logging

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # python-dotenv not installed, use system env vars

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import (
    EfficientNetB4, EfficientNetB3, 
    ResNet50, MobileNetV3Large
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
CONFIG_PATH = Path(__file__).parent / 'config'
MODELS_PATH = Path(__file__).parent / 'models'
LLM_CONFIG_PATH = CONFIG_PATH / 'llm_config.json'
LABELS_PATH = CONFIG_PATH / 'class_labels.json'

DISCLAIMER = "This AI provides preliminary screening only. Always consult a qualified dental professional."


# Disease information for LLM context
DISEASE_INFO = {
    "Calculus": {
        "name": "Dental Calculus (Tartar)",
        "description": "Hardened dental plaque that forms on teeth surfaces",
        "causes": ["Poor oral hygiene", "Irregular brushing", "Diet high in sugars"],
        "symptoms": ["Yellow/brown deposits on teeth", "Bad breath", "Gum irritation"],
        "severity_factors": ["Amount of buildup", "Location", "Gum involvement"]
    },
    "Caries": {
        "name": "Dental Caries (Cavities)",
        "description": "Tooth decay caused by bacterial acid production",
        "causes": ["Bacteria", "Sugary foods", "Poor oral hygiene", "Dry mouth"],
        "symptoms": ["Dark spots on teeth", "Tooth sensitivity", "Pain when eating"],
        "severity_factors": ["Depth of decay", "Number of teeth affected", "Proximity to nerve"]
    },
    "Gingivitis": {
        "name": "Gingivitis (Gum Inflammation)",
        "description": "Early stage of gum disease with inflammation",
        "causes": ["Plaque buildup", "Poor brushing technique", "Hormonal changes"],
        "symptoms": ["Red, swollen gums", "Bleeding when brushing", "Bad breath"],
        "severity_factors": ["Extent of inflammation", "Bleeding frequency", "Gum recession"]
    },
    "Mouth_Ulcer": {
        "name": "Mouth Ulcer (Canker Sore)",
        "description": "Painful sores in the mouth lining",
        "causes": ["Stress", "Injury", "Acidic foods", "Immune response"],
        "symptoms": ["Painful white/yellow sores", "Difficulty eating", "Burning sensation"],
        "severity_factors": ["Size", "Number of ulcers", "Duration", "Recurrence"]
    },
    "Healthy": {
        "name": "Healthy Teeth",
        "description": "No significant dental issues detected",
        "causes": [],
        "symptoms": [],
        "severity_factors": []
    }
}


class CNNEnsemble:
    """
    Ensemble of CNN models for robust disease classification.
    Uses weighted voting from multiple architectures.
    """
    
    def __init__(self, num_classes: int = 4, use_pretrained: bool = True):
        self.num_classes = num_classes
        self.use_pretrained = use_pretrained
        self.models = {}
        self.weights = {
            'efficientnet_b4': 0.4,
            'efficientnet_b3': 0.3,
            'resnet50': 0.2,
            'mobilenetv3': 0.1
        }
        self.input_sizes = {
            'efficientnet_b4': (380, 380),
            'efficientnet_b3': (300, 300),
            'resnet50': (224, 224),
            'mobilenetv3': (224, 224)
        }
        self.loaded_model = None
        self._load_existing_model()
    
    def _load_existing_model(self):
        """Load existing trained model if available."""
        model_path = MODELS_PATH / 'oral_disease_rgb_model.h5'
        if model_path.exists():
            logger.info(f"Loading existing model: {model_path}")
            self.loaded_model = keras.models.load_model(str(model_path))
            logger.info("Existing model loaded successfully")
    
    def _build_model(self, architecture: str) -> keras.Model:
        """Build a single CNN model with specified architecture."""
        input_size = self.input_sizes[architecture]
        
        if architecture == 'efficientnet_b4':
            base = EfficientNetB4(
                weights='imagenet' if self.use_pretrained else None,
                include_top=False,
                input_shape=(*input_size, 3)
            )
        elif architecture == 'efficientnet_b3':
            base = EfficientNetB3(
                weights='imagenet' if self.use_pretrained else None,
                include_top=False,
                input_shape=(*input_size, 3)
            )
        elif architecture == 'resnet50':
            base = ResNet50(
                weights='imagenet' if self.use_pretrained else None,
                include_top=False,
                input_shape=(*input_size, 3)
            )
        elif architecture == 'mobilenetv3':
            base = MobileNetV3Large(
                weights='imagenet' if self.use_pretrained else None,
                include_top=False,
                input_shape=(*input_size, 3)
            )
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        # Freeze base layers for transfer learning
        base.trainable = False
        
        # Build classification head
        model = keras.Sequential([
            base,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def preprocess_image(self, image: Union[str, bytes, Image.Image, np.ndarray], 
                        target_size: Tuple[int, int]) -> np.ndarray:
        """Preprocess image for model input."""
        if isinstance(image, str):
            img = Image.open(image)
        elif isinstance(image, bytes):
            img = Image.open(io.BytesIO(image))
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(image)
        else:
            img = image
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        img_array = np.array(img, dtype=np.float32) / 255.0
        return np.expand_dims(img_array, axis=0)
    
    def predict(self, image: Union[str, bytes, Image.Image, np.ndarray]) -> Dict:
        """
        Run ensemble prediction on image.
        Returns weighted average of all model predictions.
        """
        # Use existing model for now (faster)
        if self.loaded_model:
            img = self.preprocess_image(image, (224, 224))
            predictions = self.loaded_model.predict(img, verbose=0)[0]
            
            return {
                'predictions': predictions.tolist(),
                'model_used': 'existing_efficientnet',
                'ensemble': False
            }
        
        # Full ensemble prediction (when models are trained)
        all_predictions = []
        total_weight = 0
        
        for arch, weight in self.weights.items():
            if arch in self.models:
                input_size = self.input_sizes[arch]
                img = self.preprocess_image(image, input_size)
                pred = self.models[arch].predict(img, verbose=0)[0]
                all_predictions.append(pred * weight)
                total_weight += weight
        
        if not all_predictions:
            raise RuntimeError("No models available for prediction")
        
        # Weighted average
        ensemble_pred = np.sum(all_predictions, axis=0) / total_weight
        
        return {
            'predictions': ensemble_pred.tolist(),
            'model_used': 'ensemble',
            'ensemble': True,
            'models_used': list(self.models.keys())
        }


class LLMProvider:
    """
    LLM integration for intelligent dental analysis and report generation.
    Supports OpenAI GPT-4o, OpenRouter, local LLaMA, and Mistral.
    """
    
    def __init__(self):
        self.config = self._load_config()
        self.openai_client = None
        self.use_openrouter = False
        self._init_providers()
    
    def _load_config(self) -> Dict:
        """Load LLM configuration."""
        if LLM_CONFIG_PATH.exists():
            with open(LLM_CONFIG_PATH, 'r') as f:
                return json.load(f)
        return {}
    
    def _init_providers(self):
        """Initialize available LLM providers."""
        try:
            import openai
            
            # Check for OpenRouter key first (starts with sk-or-)
            api_key = os.environ.get('OPENAI_API_KEY', '')
            
            if api_key.startswith('sk-or-'):
                # OpenRouter API
                self.openai_client = openai.OpenAI(
                    api_key=api_key,
                    base_url="https://openrouter.ai/api/v1"
                )
                self.use_openrouter = True
                logger.info("OpenRouter client initialized (GPT-4o via OpenRouter)")
            elif api_key:
                # Standard OpenAI
                self.openai_client = openai.OpenAI(api_key=api_key)
                logger.info("OpenAI client initialized")
            else:
                logger.warning("No API key found (set OPENAI_API_KEY)")
        except ImportError:
            logger.warning("OpenAI package not installed")
        except Exception as e:
            logger.warning(f"LLM client init failed: {e}")
    
    def _encode_image_base64(self, image: Union[str, bytes, Image.Image]) -> str:
        """Encode image to base64 for vision models."""
        if isinstance(image, str):
            with open(image, 'rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')
        elif isinstance(image, bytes):
            return base64.b64encode(image).decode('utf-8')
        elif isinstance(image, Image.Image):
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG')
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        return ""
    
    async def generate_analysis(self, 
                               cnn_results: Dict,
                               image: Union[str, bytes, Image.Image] = None,
                               use_vision: bool = True) -> Dict:
        """
        Generate intelligent analysis using LLM.
        
        Args:
            cnn_results: Results from CNN ensemble
            image: Original image for vision analysis
            use_vision: Whether to use vision capabilities
            
        Returns:
            LLM-generated analysis with explanation, report, and advice
        """
        # Build context from CNN results
        disease_idx = np.argmax(cnn_results['predictions'])
        confidence = cnn_results['predictions'][disease_idx]
        
        # Load class labels
        with open(LABELS_PATH, 'r') as f:
            labels_data = json.load(f)
        class_labels = labels_data['class_labels']
        disease_name = class_labels[str(disease_idx)]
        
        # Get disease info
        disease_info = DISEASE_INFO.get(disease_name, DISEASE_INFO.get('Healthy'))
        
        # Build all predictions
        all_predictions = []
        for idx, conf in enumerate(cnn_results['predictions']):
            all_predictions.append({
                'disease': class_labels[str(idx)],
                'confidence': float(conf)
            })
        all_predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Try GPT-4o with vision
        if self.openai_client and use_vision and image:
            try:
                return await self._generate_with_gpt4o_vision(
                    disease_name, confidence, all_predictions, disease_info, image
                )
            except Exception as e:
                logger.warning(f"GPT-4o vision failed: {e}")
        
        # Try GPT-4o text-only
        if self.openai_client:
            try:
                return await self._generate_with_gpt4o(
                    disease_name, confidence, all_predictions, disease_info
                )
            except Exception as e:
                logger.warning(f"GPT-4o failed: {e}")
        
        # Fallback to rule-based generation
        return self._generate_fallback(disease_name, confidence, all_predictions, disease_info)
    
    async def _generate_with_gpt4o_vision(self, disease: str, confidence: float,
                                          predictions: List, disease_info: Dict,
                                          image: Union[str, bytes, Image.Image]) -> Dict:
        """Generate analysis using GPT-4o with vision."""
        image_b64 = self._encode_image_base64(image)
        
        prompt = f"""You are an expert dental AI assistant analyzing a dental image.

CNN Analysis Results:
- Primary Detection: {disease} ({confidence*100:.1f}% confidence)
- All Predictions: {json.dumps(predictions[:4], indent=2)}

Disease Information:
{json.dumps(disease_info, indent=2)}

Please analyze the dental image and provide:

1. **EXPLANATION** (2-3 sentences): Explain what the AI detected in simple terms.

2. **DETAILED FINDINGS**: What specific signs are visible in the image?

3. **SEVERITY ASSESSMENT**: Rate as Low/Medium/High with reasoning.

4. **RECOMMENDATIONS**: 3-5 specific actionable recommendations.

5. **HOME CARE TIPS**: 3-4 practical tips the patient can do at home.

6. **WHEN TO SEE A DENTIST**: Urgency guidance.

Be thorough but reassuring. Avoid causing unnecessary alarm while being honest about findings.
Always emphasize this is a preliminary AI screening, not a diagnosis."""

        # Use appropriate model based on provider
        model = "openai/gpt-4o" if self.use_openrouter else "gpt-4o"
        
        extra_headers = {}
        if self.use_openrouter:
            extra_headers = {
                "HTTP-Referer": "https://dental-ai.local",
                "X-Title": "Dental AI Analysis"
            }
        
        response = self.openai_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=2048,
            temperature=0.3,
            extra_headers=extra_headers if self.use_openrouter else None
        )
        
        llm_response = response.choices[0].message.content
        
        return {
            'source': 'gpt-4o-vision',
            'disease': disease,
            'confidence': confidence,
            'all_predictions': predictions,
            'llm_analysis': llm_response,
            'severity': self._extract_severity(llm_response),
            'model_info': disease_info
        }
    
    async def _generate_with_gpt4o(self, disease: str, confidence: float,
                                   predictions: List, disease_info: Dict) -> Dict:
        """Generate analysis using GPT-4o text-only."""
        prompt = f"""You are an expert dental AI assistant.

CNN Analysis Results:
- Primary Detection: {disease} ({confidence*100:.1f}% confidence)
- All Predictions: {json.dumps(predictions[:4], indent=2)}

Disease Information:
{json.dumps(disease_info, indent=2)}

Based on the AI detection results, provide:

1. **EXPLANATION**: What does this detection mean for the patient?

2. **SEVERITY ASSESSMENT**: Rate as Low/Medium/High based on confidence and condition type.

3. **RECOMMENDATIONS**: 3-5 specific actionable recommendations.

4. **HOME CARE TIPS**: 3-4 practical oral hygiene tips.

5. **WHEN TO SEE A DENTIST**: Urgency guidance.

Be informative and supportive. This is a preliminary AI screening."""

        model = "openai/gpt-4o" if self.use_openrouter else "gpt-4o"
        
        extra_headers = {}
        if self.use_openrouter:
            extra_headers = {
                "HTTP-Referer": "https://dental-ai.local",
                "X-Title": "Dental AI Analysis"
            }
        
        response = self.openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
            temperature=0.3,
            extra_headers=extra_headers if self.use_openrouter else None
        )
        
        llm_response = response.choices[0].message.content
        
        return {
            'source': 'gpt-4o',
            'disease': disease,
            'confidence': confidence,
            'all_predictions': predictions,
            'llm_analysis': llm_response,
            'severity': self._extract_severity(llm_response),
            'model_info': disease_info
        }
    
    def _extract_severity(self, text: str) -> str:
        """Extract severity from LLM response."""
        text_lower = text.lower()
        if 'high' in text_lower and 'severity' in text_lower:
            return 'High'
        elif 'medium' in text_lower or 'moderate' in text_lower:
            return 'Medium'
        return 'Low'
    
    def _generate_fallback(self, disease: str, confidence: float,
                          predictions: List, disease_info: Dict) -> Dict:
        """Generate analysis without LLM (rule-based fallback)."""
        # Determine severity
        if confidence >= 0.8:
            severity = 'High'
        elif confidence >= 0.5:
            severity = 'Medium'
        else:
            severity = 'Low'
        
        # Generate explanation
        if disease == 'Healthy' or confidence < 0.3:
            explanation = "The AI analysis suggests your teeth appear generally healthy. Continue maintaining good oral hygiene practices."
        else:
            explanation = f"The AI detected signs of {disease_info['name']} with {confidence*100:.1f}% confidence. {disease_info['description']}."
        
        # Generate recommendations
        recommendations = [
            "Schedule a dental checkup for professional evaluation",
            "Maintain regular brushing twice daily with fluoride toothpaste",
            "Floss daily to remove plaque between teeth",
            "Limit sugary and acidic foods and drinks"
        ]
        
        if disease == 'Calculus':
            recommendations.insert(0, "Professional dental cleaning recommended to remove tartar buildup")
        elif disease == 'Caries':
            recommendations.insert(0, "Dental examination needed to assess cavity depth and treatment options")
        elif disease == 'Gingivitis':
            recommendations.insert(0, "Focus on gentle brushing along the gumline")
        elif disease == 'Mouth_Ulcer':
            recommendations.insert(0, "Avoid spicy/acidic foods; use saltwater rinses")
        
        # Home care tips
        home_care = [
            "Brush teeth for 2 minutes, twice daily",
            "Use a soft-bristled toothbrush",
            "Replace toothbrush every 3 months",
            "Use antimicrobial mouthwash"
        ]
        
        # Urgency
        if severity == 'High':
            urgency = "Schedule a dental appointment within 1-2 weeks"
        elif severity == 'Medium':
            urgency = "Schedule a dental appointment within 2-4 weeks"
        else:
            urgency = "Schedule routine dental checkup within 3-6 months"
        
        analysis_text = f"""## Dental AI Screening Results

### Explanation
{explanation}

### Severity Assessment: {severity}
Based on {confidence*100:.1f}% detection confidence for {disease_info['name']}.

### Recommendations
{chr(10).join(f"- {r}" for r in recommendations)}

### Home Care Tips
{chr(10).join(f"- {t}" for t in home_care)}

### When to See a Dentist
{urgency}

---
*{DISCLAIMER}*"""
        
        return {
            'source': 'rule-based',
            'disease': disease,
            'confidence': confidence,
            'all_predictions': predictions,
            'llm_analysis': analysis_text,
            'severity': severity,
            'model_info': disease_info,
            'recommendations': recommendations,
            'home_care': home_care,
            'urgency': urgency
        }


class VisionLanguageModel:
    """
    Vision-Language Model integration for advanced image understanding.
    Supports LLaVA, Qwen-VL via Ollama or API.
    """
    
    def __init__(self):
        self.config = self._load_config()
        self.ollama_available = self._check_ollama()
    
    def _load_config(self) -> Dict:
        if LLM_CONFIG_PATH.exists():
            with open(LLM_CONFIG_PATH, 'r') as f:
                return json.load(f).get('vlm_providers', {})
        return {}
    
    def _check_ollama(self) -> bool:
        """Check if Ollama is available."""
        try:
            import requests
            response = requests.get('http://localhost:11434/api/tags', timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def _encode_image(self, image: Union[str, bytes, Image.Image]) -> str:
        """Encode image to base64."""
        if isinstance(image, str):
            with open(image, 'rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')
        elif isinstance(image, bytes):
            return base64.b64encode(image).decode('utf-8')
        elif isinstance(image, Image.Image):
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG')
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        return ""
    
    async def analyze_with_llava(self, image: Union[str, bytes, Image.Image],
                                 cnn_context: Dict) -> Optional[Dict]:
        """Analyze image using LLaVA model via Ollama."""
        if not self.ollama_available:
            return None
        
        try:
            import requests
            
            image_b64 = self._encode_image(image)
            
            prompt = f"""Analyze this dental image. The AI has detected: {cnn_context.get('disease', 'Unknown')} 
with {cnn_context.get('confidence', 0)*100:.1f}% confidence.

Please describe:
1. What dental conditions are visible?
2. Are there signs of plaque, tartar, cavities, or gum issues?
3. Overall assessment of oral health
4. Any areas of concern?

Be specific about what you observe in the image."""

            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': 'llava',
                    'prompt': prompt,
                    'images': [image_b64],
                    'stream': False
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    'source': 'llava',
                    'analysis': result.get('response', ''),
                    'success': True
                }
        except Exception as e:
            logger.warning(f"LLaVA analysis failed: {e}")
        
        return None
    
    async def analyze_with_qwen(self, image: Union[str, bytes, Image.Image],
                                cnn_context: Dict) -> Optional[Dict]:
        """Analyze image using Qwen-VL model."""
        if not self.ollama_available:
            return None
        
        try:
            import requests
            
            image_b64 = self._encode_image(image)
            
            prompt = f"""As a dental analysis AI, examine this oral image.
CNN Detection: {cnn_context.get('disease', 'Unknown')} ({cnn_context.get('confidence', 0)*100:.1f}%)

Provide detailed observations about:
- Tooth condition and color
- Gum health indicators
- Presence of plaque or tartar
- Any visible decay or damage
- Overall oral health assessment"""

            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': 'qwen-vl',
                    'prompt': prompt,
                    'images': [image_b64],
                    'stream': False
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    'source': 'qwen-vl',
                    'analysis': result.get('response', ''),
                    'success': True
                }
        except Exception as e:
            logger.warning(f"Qwen-VL analysis failed: {e}")
        
        return None


class AdvancedDentalPipeline:
    """
    Complete advanced dental AI pipeline.
    
    Combines:
    1. CNN Ensemble (EfficientNet-B4, B3, ResNet50, MobileNetV3)
    2. LLM Analysis (GPT-4o, LLaMA-3, Mistral)
    3. VLM Understanding (LLaVA, Qwen-VL)
    4. Comprehensive Report Generation
    """
    
    def __init__(self):
        logger.info("Initializing Advanced Dental Pipeline...")
        
        self.cnn_ensemble = CNNEnsemble()
        self.llm_provider = LLMProvider()
        self.vlm = VisionLanguageModel()
        
        # Load class labels
        with open(LABELS_PATH, 'r') as f:
            labels_data = json.load(f)
        self.class_labels = labels_data['class_labels']
        
        logger.info("Advanced Dental Pipeline ready!")
    
    async def analyze(self, image: Union[str, bytes, Image.Image, np.ndarray],
                     use_llm: bool = True,
                     use_vlm: bool = False) -> Dict:
        """
        Run complete dental analysis pipeline.
        
        Args:
            image: Input dental image
            use_llm: Whether to use LLM for analysis
            use_vlm: Whether to use Vision-Language models
            
        Returns:
            Complete analysis with CNN predictions, LLM analysis, and report
        """
        logger.info("Starting advanced dental analysis...")
        timestamp = datetime.now().isoformat()
        
        # Stage 1: CNN Ensemble Prediction
        logger.info("Stage 1: CNN Ensemble prediction...")
        cnn_results = self.cnn_ensemble.predict(image)
        
        # Get primary prediction
        predictions = cnn_results['predictions']
        disease_idx = np.argmax(predictions)
        confidence = predictions[disease_idx]
        disease = self.class_labels[str(disease_idx)]
        
        # Build all predictions list
        all_predictions = []
        for idx, conf in enumerate(predictions):
            all_predictions.append({
                'disease': self.class_labels[str(idx)],
                'confidence': float(conf)
            })
        all_predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Stage 2: LLM Analysis
        llm_analysis = None
        if use_llm:
            logger.info("Stage 2: LLM analysis...")
            try:
                llm_analysis = await self.llm_provider.generate_analysis(
                    cnn_results, image, use_vision=True
                )
            except Exception as e:
                logger.warning(f"LLM analysis failed: {e}")
        
        # Stage 3: VLM Analysis (optional)
        vlm_analysis = None
        if use_vlm:
            logger.info("Stage 3: VLM analysis...")
            vlm_analysis = await self.vlm.analyze_with_llava(
                image, {'disease': disease, 'confidence': confidence}
            )
        
        # Stage 4: Generate Final Report
        logger.info("Stage 4: Generating final report...")
        final_report = self._generate_final_report(
            disease, confidence, all_predictions,
            llm_analysis, vlm_analysis
        )
        
        # Compile complete results
        result = {
            'analysis_id': f"adv_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'timestamp': timestamp,
            'pipeline_version': '2.0.0-advanced',
            
            # CNN Results
            'cnn_analysis': {
                'primary_disease': disease,
                'confidence': float(confidence),
                'all_predictions': all_predictions,
                'model_info': cnn_results
            },
            
            # LLM Analysis
            'llm_analysis': llm_analysis,
            
            # VLM Analysis
            'vlm_analysis': vlm_analysis,
            
            # Final Report
            'report': final_report,
            
            # Summary
            'summary': {
                'disease': disease,
                'confidence': float(confidence),
                'severity': final_report['severity'],
                'recommendation': final_report['urgency']
            },
            
            'disclaimer': DISCLAIMER
        }
        
        logger.info("Advanced dental analysis complete!")
        return result
    
    def _generate_final_report(self, disease: str, confidence: float,
                               predictions: List, llm_analysis: Optional[Dict],
                               vlm_analysis: Optional[Dict]) -> Dict:
        """Generate comprehensive final report."""
        disease_info = DISEASE_INFO.get(disease, DISEASE_INFO['Healthy'])
        
        # Determine severity
        if llm_analysis and 'severity' in llm_analysis:
            severity = llm_analysis['severity']
        elif confidence >= 0.8:
            severity = 'High'
        elif confidence >= 0.5:
            severity = 'Medium'
        else:
            severity = 'Low'
        
        # Get recommendations
        if llm_analysis and 'recommendations' in llm_analysis:
            recommendations = llm_analysis['recommendations']
        else:
            recommendations = self._get_default_recommendations(disease, severity)
        
        # Get home care tips
        if llm_analysis and 'home_care' in llm_analysis:
            home_care = llm_analysis['home_care']
        else:
            home_care = self._get_default_home_care(disease)
        
        # Urgency
        if severity == 'High':
            urgency = "Schedule dental appointment within 1-2 weeks"
        elif severity == 'Medium':
            urgency = "Schedule dental appointment within 2-4 weeks"
        else:
            urgency = "Schedule routine checkup within 3-6 months"
        
        # Build report
        report = {
            'disease_detected': disease,
            'disease_name': disease_info['name'],
            'confidence': float(confidence),
            'confidence_level': 'High' if confidence > 0.7 else 'Medium' if confidence > 0.4 else 'Low',
            'severity': severity,
            
            'description': disease_info['description'],
            'causes': disease_info['causes'],
            'symptoms': disease_info['symptoms'],
            
            'recommendations': recommendations,
            'home_care_tips': home_care,
            'urgency': urgency,
            
            'all_predictions': predictions[:5],
            
            'llm_explanation': llm_analysis.get('llm_analysis', '') if llm_analysis else '',
            'vlm_observations': vlm_analysis.get('analysis', '') if vlm_analysis else '',
            
            'next_steps': [
                "This is a preliminary AI screening only",
                "Schedule a professional dental examination",
                "Discuss these findings with your dentist",
                "Continue regular oral hygiene practices"
            ],
            
            'generated_at': datetime.now().isoformat(),
            'disclaimer': DISCLAIMER
        }
        
        return report
    
    def _get_default_recommendations(self, disease: str, severity: str) -> List[str]:
        """Get default recommendations based on disease."""
        base = [
            "Schedule a dental checkup for professional evaluation",
            "Maintain regular brushing twice daily",
            "Floss daily to remove plaque between teeth"
        ]
        
        disease_specific = {
            'Calculus': ["Professional cleaning to remove tartar buildup"],
            'Caries': ["Dental examination to assess cavity treatment options"],
            'Gingivitis': ["Focus on gentle brushing along the gumline", "Consider antiseptic mouthwash"],
            'Mouth_Ulcer': ["Avoid spicy/acidic foods", "Use saltwater rinses for relief"],
            'Healthy': ["Continue current oral hygiene routine"]
        }
        
        return disease_specific.get(disease, []) + base
    
    def _get_default_home_care(self, disease: str) -> List[str]:
        """Get default home care tips."""
        return [
            "Brush teeth for 2 minutes, twice daily",
            "Use fluoride toothpaste",
            "Floss daily",
            "Use antimicrobial mouthwash",
            "Limit sugary foods and drinks",
            "Stay hydrated"
        ]
    
    def analyze_sync(self, image: Union[str, bytes, Image.Image, np.ndarray],
                    use_llm: bool = True, use_vlm: bool = False) -> Dict:
        """Synchronous wrapper for analyze method."""
        return asyncio.run(self.analyze(image, use_llm, use_vlm))


class DentalReportFormatter:
    """Format dental reports for different outputs."""
    
    @staticmethod
    def to_text(report: Dict) -> str:
        """Generate human-readable text report."""
        r = report['report']
        
        text = f"""
{'='*60}
🦷 ADVANCED DENTAL AI SCREENING REPORT
{'='*60}

📋 ANALYSIS ID: {report['analysis_id']}
📅 DATE: {report['timestamp'][:10]}

{'─'*60}
📊 PRIMARY FINDINGS
{'─'*60}

Detected Condition: {r['disease_name']}
Confidence: {r['confidence']*100:.1f}% ({r['confidence_level']})
Severity: {r['severity']}

Description:
{r['description']}

{'─'*60}
🔍 ALL PREDICTIONS
{'─'*60}
"""
        for pred in r['all_predictions']:
            bar = '█' * int(pred['confidence'] * 20)
            text += f"  {pred['disease']:15} {bar} {pred['confidence']*100:.1f}%\n"
        
        text += f"""
{'─'*60}
💡 RECOMMENDATIONS
{'─'*60}
"""
        for i, rec in enumerate(r['recommendations'], 1):
            text += f"  {i}. {rec}\n"
        
        text += f"""
{'─'*60}
🏠 HOME CARE TIPS
{'─'*60}
"""
        for tip in r['home_care_tips']:
            text += f"  • {tip}\n"
        
        text += f"""
{'─'*60}
⏰ WHEN TO SEE A DENTIST
{'─'*60}
  {r['urgency']}

{'─'*60}
📝 AI ANALYSIS
{'─'*60}
{r['llm_explanation'][:1000] if r['llm_explanation'] else 'No LLM analysis available'}

{'='*60}
⚠️  DISCLAIMER
{'='*60}
{r['disclaimer']}

{'='*60}
"""
        return text
    
    @staticmethod
    def to_html(report: Dict) -> str:
        """Generate HTML report."""
        r = report['report']
        
        severity_colors = {
            'Low': '#28a745',
            'Medium': '#ffc107', 
            'High': '#dc3545'
        }
        severity_color = severity_colors.get(r['severity'], '#6c757d')
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Dental AI Screening Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; background: #f5f5f5; }}
        .report {{ background: white; border-radius: 12px; padding: 30px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .header {{ text-align: center; border-bottom: 2px solid #e0e0e0; padding-bottom: 20px; margin-bottom: 20px; }}
        .header h1 {{ color: #2c3e50; margin: 0; }}
        .header .subtitle {{ color: #7f8c8d; font-size: 14px; }}
        .section {{ margin: 25px 0; }}
        .section-title {{ color: #34495e; font-size: 18px; font-weight: 600; margin-bottom: 15px; display: flex; align-items: center; gap: 8px; }}
        .finding-card {{ background: #f8f9fa; border-radius: 8px; padding: 20px; margin: 15px 0; }}
        .disease-name {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
        .confidence {{ font-size: 16px; color: #7f8c8d; }}
        .severity {{ display: inline-block; padding: 5px 15px; border-radius: 20px; color: white; font-weight: bold; background: {severity_color}; }}
        .prediction-bar {{ background: #e9ecef; border-radius: 4px; height: 24px; margin: 8px 0; position: relative; }}
        .prediction-fill {{ background: #3498db; height: 100%; border-radius: 4px; }}
        .prediction-label {{ position: absolute; left: 10px; top: 2px; font-size: 12px; }}
        .recommendation {{ padding: 10px 15px; background: #e8f4fd; border-left: 4px solid #3498db; margin: 10px 0; border-radius: 0 8px 8px 0; }}
        .tip {{ padding: 8px 15px; background: #e8f8e8; border-left: 4px solid #28a745; margin: 8px 0; border-radius: 0 8px 8px 0; }}
        .urgency {{ background: #fff3cd; border: 1px solid #ffc107; border-radius: 8px; padding: 15px; text-align: center; }}
        .disclaimer {{ background: #f8d7da; border: 1px solid #f5c6cb; border-radius: 8px; padding: 15px; font-size: 12px; color: #721c24; margin-top: 30px; }}
        .llm-analysis {{ background: #f0f0f0; border-radius: 8px; padding: 20px; white-space: pre-wrap; font-size: 14px; line-height: 1.6; }}
    </style>
</head>
<body>
    <div class="report">
        <div class="header">
            <h1>🦷 Dental AI Screening Report</h1>
            <div class="subtitle">Analysis ID: {report['analysis_id']} | Date: {report['timestamp'][:10]}</div>
        </div>
        
        <div class="section">
            <div class="section-title">📊 Primary Findings</div>
            <div class="finding-card">
                <div class="disease-name">{r['disease_name']}</div>
                <div class="confidence">Confidence: {r['confidence']*100:.1f}% ({r['confidence_level']})</div>
                <div style="margin-top: 10px;">Severity: <span class="severity">{r['severity']}</span></div>
                <p style="margin-top: 15px; color: #555;">{r['description']}</p>
            </div>
        </div>
        
        <div class="section">
            <div class="section-title">🔍 All Predictions</div>
"""
        
        for pred in r['all_predictions']:
            width = pred['confidence'] * 100
            html += f"""
            <div class="prediction-bar">
                <div class="prediction-fill" style="width: {width}%;"></div>
                <span class="prediction-label">{pred['disease']}: {pred['confidence']*100:.1f}%</span>
            </div>
"""
        
        html += """
        </div>
        
        <div class="section">
            <div class="section-title">💡 Recommendations</div>
"""
        for rec in r['recommendations']:
            html += f'            <div class="recommendation">{rec}</div>\n'
        
        html += """
        </div>
        
        <div class="section">
            <div class="section-title">🏠 Home Care Tips</div>
"""
        for tip in r['home_care_tips']:
            html += f'            <div class="tip">{tip}</div>\n'
        
        html += f"""
        </div>
        
        <div class="section">
            <div class="section-title">⏰ When to See a Dentist</div>
            <div class="urgency">{r['urgency']}</div>
        </div>
"""
        
        if r['llm_explanation']:
            html += f"""
        <div class="section">
            <div class="section-title">📝 AI Analysis</div>
            <div class="llm-analysis">{r['llm_explanation'][:2000]}</div>
        </div>
"""
        
        html += f"""
        <div class="disclaimer">
            <strong>⚠️ Disclaimer:</strong> {r['disclaimer']}
        </div>
    </div>
</body>
</html>
"""
        return html
    
    @staticmethod
    def to_json(report: Dict) -> str:
        """Generate JSON report."""
        return json.dumps(report, indent=2, default=str)


# Convenience functions
def analyze_dental_image_advanced(image_path: str, use_llm: bool = True) -> Dict:
    """Quick analysis function."""
    pipeline = AdvancedDentalPipeline()
    return pipeline.analyze_sync(image_path, use_llm=use_llm)


def generate_report(analysis_result: Dict, format: str = 'text') -> str:
    """Generate formatted report from analysis."""
    formatter = DentalReportFormatter()
    if format == 'html':
        return formatter.to_html(analysis_result)
    elif format == 'json':
        return formatter.to_json(analysis_result)
    return formatter.to_text(analysis_result)


# CLI Interface
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Advanced Dental AI Pipeline')
    parser.add_argument('image', help='Path to dental image')
    parser.add_argument('--no-llm', action='store_true', help='Disable LLM analysis')
    parser.add_argument('--vlm', action='store_true', help='Enable VLM analysis')
    parser.add_argument('--format', choices=['text', 'html', 'json'], default='text', help='Output format')
    parser.add_argument('--output', '-o', help='Output file path')
    
    args = parser.parse_args()
    
    # Run analysis
    pipeline = AdvancedDentalPipeline()
    result = pipeline.analyze_sync(
        args.image,
        use_llm=not args.no_llm,
        use_vlm=args.vlm
    )
    
    # Generate report
    report = generate_report(result, args.format)
    
    # Output
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Report saved to: {args.output}")
    else:
        print(report)
