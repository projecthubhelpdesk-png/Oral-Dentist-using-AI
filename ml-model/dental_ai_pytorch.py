"""
Dental AI Pipeline (PyTorch Version)
====================================
Works with Python 3.14+

Uses PyTorch + timm for EfficientNet models
Uses OpenRouter/GPT-4o for LLM analysis
"""

import os
import json
import base64
import asyncio
from pathlib import Path
from typing import Dict, List, Union, Optional
from PIL import Image
from datetime import datetime
import io
import logging

# Load .env
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
CONFIG_PATH = Path(__file__).parent / 'config'
MODELS_PATH = Path(__file__).parent / 'models'
LABELS_PATH = CONFIG_PATH / 'class_labels.json'

DISCLAIMER = "This AI provides preliminary screening only. Always consult a qualified dental professional."

# Disease info for reports
DISEASE_INFO = {
    "Calculus": {
        "name": "Dental Calculus (Tartar)",
        "description": "Hardened dental plaque on teeth surfaces",
        "recommendations": ["Professional dental cleaning", "Improve brushing technique", "Use tartar-control toothpaste"]
    },
    "Caries": {
        "name": "Dental Caries (Cavities)",
        "description": "Tooth decay caused by bacterial acid",
        "recommendations": ["Dental examination for treatment", "Fluoride treatments", "Reduce sugary foods"]
    },
    "Gingivitis": {
        "name": "Gingivitis (Gum Inflammation)",
        "description": "Early stage gum disease with inflammation",
        "recommendations": ["Gentle brushing along gumline", "Daily flossing", "Antiseptic mouthwash"]
    },
    "Mouth_Ulcer": {
        "name": "Mouth Ulcer",
        "description": "Painful sores in the mouth",
        "recommendations": ["Avoid spicy/acidic foods", "Saltwater rinses", "Topical oral gels"]
    }
}


class DentalClassifier:
    """
    PyTorch-based dental disease classifier using EfficientNet.
    """
    
    def __init__(self):
        self.model = None
        self.class_labels = self._load_labels()
        self.device = None
        self._init_model()
    
    def _load_labels(self) -> Dict:
        """Load class labels."""
        if LABELS_PATH.exists():
            with open(LABELS_PATH, 'r') as f:
                data = json.load(f)
            return data.get('class_labels', {
                "0": "Calculus", "1": "Caries", "2": "Gingivitis", "3": "Mouth_Ulcer"
            })
        return {"0": "Calculus", "1": "Caries", "2": "Gingivitis", "3": "Mouth_Ulcer"}
    
    def _init_model(self):
        """Initialize PyTorch model."""
        try:
            import torch
            import timm
            
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"Using device: {self.device}")
            
            # Load pretrained EfficientNet-B4
            self.model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=4)
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Load custom weights if available
            weights_path = MODELS_PATH / 'dental_efficientnet_b4.pth'
            if weights_path.exists():
                self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
                logger.info("Loaded custom dental model weights")
            else:
                logger.info("Using pretrained ImageNet weights (fine-tuning recommended)")
            
        except ImportError as e:
            logger.warning(f"PyTorch/timm not installed: {e}")
            self.model = None
    
    def preprocess(self, image: Union[str, bytes, Image.Image]) -> 'torch.Tensor':
        """Preprocess image for model."""
        import torch
        from torchvision import transforms
        
        # Load image
        if isinstance(image, str):
            img = Image.open(image)
        elif isinstance(image, bytes):
            img = Image.open(io.BytesIO(image))
        else:
            img = image
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # EfficientNet-B4 preprocessing
        transform = transforms.Compose([
            transforms.Resize((380, 380)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        tensor = transform(img).unsqueeze(0)
        return tensor.to(self.device)
    
    def predict(self, image: Union[str, bytes, Image.Image]) -> Dict:
        """Run prediction on image."""
        if self.model is None:
            # Fallback: return mock prediction for testing
            return self._mock_predict()
        
        import torch
        import torch.nn.functional as F
        
        # Preprocess
        tensor = self.preprocess(image)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(tensor)
            probs = F.softmax(outputs, dim=1)[0]
        
        # Get predictions
        probs_np = probs.cpu().numpy()
        top_idx = int(np.argmax(probs_np))
        confidence = float(probs_np[top_idx])
        disease = self.class_labels.get(str(top_idx), "Unknown")
        
        # All predictions
        all_preds = []
        for idx, conf in enumerate(probs_np):
            all_preds.append({
                'disease': self.class_labels.get(str(idx), f"Class_{idx}"),
                'confidence': float(conf)
            })
        all_preds.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Severity
        if confidence >= 0.8:
            severity = 'High'
        elif confidence >= 0.5:
            severity = 'Medium'
        else:
            severity = 'Low'
        
        return {
            'disease': disease,
            'confidence': confidence,
            'severity': severity,
            'all_predictions': all_preds,
            'model': 'EfficientNet-B4 (PyTorch)'
        }
    
    def _mock_predict(self) -> Dict:
        """Mock prediction when model not available."""
        return {
            'disease': 'Calculus',
            'confidence': 0.75,
            'severity': 'Medium',
            'all_predictions': [
                {'disease': 'Calculus', 'confidence': 0.75},
                {'disease': 'Gingivitis', 'confidence': 0.15},
                {'disease': 'Caries', 'confidence': 0.07},
                {'disease': 'Mouth_Ulcer', 'confidence': 0.03}
            ],
            'model': 'Mock (install torch & timm for real predictions)'
        }


class LLMAnalyzer:
    """LLM-based analysis using OpenRouter/GPT-4o."""
    
    def __init__(self):
        self.client = None
        self.use_openrouter = False
        self._init_client()
    
    def _init_client(self):
        """Initialize OpenAI/OpenRouter client."""
        try:
            import openai
            
            api_key = os.environ.get('OPENAI_API_KEY', '')
            
            if api_key.startswith('sk-or-'):
                # OpenRouter
                self.client = openai.OpenAI(
                    api_key=api_key,
                    base_url="https://openrouter.ai/api/v1"
                )
                self.use_openrouter = True
                logger.info("OpenRouter client initialized")
            elif api_key:
                self.client = openai.OpenAI(api_key=api_key)
                logger.info("OpenAI client initialized")
            else:
                logger.warning("No API key found")
        except ImportError:
            logger.warning("openai package not installed")
    
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
    
    async def analyze(self, cnn_result: Dict, image: Union[str, bytes, Image.Image] = None) -> Dict:
        """Generate LLM analysis."""
        disease = cnn_result['disease']
        confidence = cnn_result['confidence']
        severity = cnn_result['severity']
        disease_info = DISEASE_INFO.get(disease, {})
        
        # Try GPT-4o with vision
        if self.client and image:
            try:
                return await self._analyze_with_vision(disease, confidence, severity, disease_info, image)
            except Exception as e:
                logger.warning(f"LLM analysis failed: {e}")
        
        # Fallback to rule-based
        return self._fallback_analysis(disease, confidence, severity, disease_info)
    
    async def _analyze_with_vision(self, disease: str, confidence: float, 
                                   severity: str, disease_info: Dict,
                                   image: Union[str, bytes, Image.Image]) -> Dict:
        """Analyze with GPT-4o vision."""
        image_b64 = self._encode_image(image)
        
        prompt = f"""You are an expert dental AI assistant. Analyze this dental image carefully and provide a detailed assessment.

AI Detection Results:
- Primary Detection: {disease} ({confidence*100:.1f}% confidence)
- Severity Level: {severity}

IMPORTANT: Examine the image thoroughly and provide:

## 1. EXACT COMPLAINT / DIAGNOSIS
Describe exactly what you see in the image:
- What specific dental condition is visible?
- Where exactly is the problem located (which teeth, gum area)?
- What are the visible signs/symptoms?
- How severe does it appear?

## 2. DETAILED FINDINGS
- Color changes (yellowing, dark spots, white patches)
- Gum condition (redness, swelling, recession)
- Plaque/tartar buildup locations
- Any visible decay, cavities, or damage
- Alignment issues if any

## 3. WHAT THIS MEANS FOR YOU
Explain in simple terms:
- What is happening to your teeth/gums
- Why this condition occurs
- What will happen if left untreated

## 4. IMMEDIATE ACTIONS TO TAKE
Step-by-step guidance:
- What to do RIGHT NOW
- What to avoid doing
- Emergency signs to watch for

## 5. TREATMENT OPTIONS
- Professional treatments available
- Estimated urgency (immediate/within days/within weeks)
- What to expect at the dentist

## 6. HOME CARE ROUTINE
Specific daily routine to follow:
- Morning routine
- After meals
- Night routine
- Products to use

## 7. PREVENTION TIPS
How to prevent this from getting worse or recurring.

Be specific, actionable, and helpful. This is a preliminary AI screening - always recommend professional consultation."""

        model = "openai/gpt-4o" if self.use_openrouter else "gpt-4o"
        
        response = self.client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}", "detail": "high"}}
                ]
            }],
            max_tokens=2500,
            temperature=0.3
        )
        
        llm_text = response.choices[0].message.content
        
        # Parse the response into structured sections
        parsed = self._parse_gpt_response(llm_text)
        
        return {
            'source': 'gpt-4o-vision',
            'analysis': llm_text,
            'parsed': parsed,
            'disease': disease,
            'confidence': confidence,
            'severity': severity
        }
    
    def _parse_gpt_response(self, text: str) -> Dict:
        """Parse GPT response into structured sections."""
        sections = {
            'exact_complaint': '',
            'detailed_findings': '',
            'what_this_means': '',
            'immediate_actions': '',
            'treatment_options': '',
            'home_care_routine': '',
            'prevention_tips': ''
        }
        
        # Simple parsing by section headers
        current_section = None
        current_content = []
        
        for line in text.split('\n'):
            line_lower = line.lower().strip()
            
            if 'exact complaint' in line_lower or 'diagnosis' in line_lower:
                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = 'exact_complaint'
                current_content = []
            elif 'detailed findings' in line_lower:
                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = 'detailed_findings'
                current_content = []
            elif 'what this means' in line_lower:
                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = 'what_this_means'
                current_content = []
            elif 'immediate action' in line_lower:
                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = 'immediate_actions'
                current_content = []
            elif 'treatment option' in line_lower:
                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = 'treatment_options'
                current_content = []
            elif 'home care' in line_lower:
                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = 'home_care_routine'
                current_content = []
            elif 'prevention' in line_lower:
                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = 'prevention_tips'
                current_content = []
            elif current_section:
                current_content.append(line)
        
        # Save last section
        if current_section and current_content:
            sections[current_section] = '\n'.join(current_content).strip()
        
        return sections
    
    def _fallback_analysis(self, disease: str, confidence: float, 
                          severity: str, disease_info: Dict) -> Dict:
        """Rule-based fallback analysis."""
        name = disease_info.get('name', disease)
        desc = disease_info.get('description', '')
        recs = disease_info.get('recommendations', ['Consult a dentist'])
        
        if severity == 'High':
            urgency = "Schedule dental appointment within 1-2 weeks"
        elif severity == 'Medium':
            urgency = "Schedule dental appointment within 2-4 weeks"
        else:
            urgency = "Schedule routine checkup within 3-6 months"
        
        analysis = f"""## Dental AI Screening Results

### Detection
The AI detected signs of **{name}** with {confidence*100:.1f}% confidence.

{desc}

### Severity: {severity}

### Recommendations
{chr(10).join(f"- {r}" for r in recs)}

### Home Care Tips
- Brush teeth twice daily with fluoride toothpaste
- Floss daily
- Use antimicrobial mouthwash
- Limit sugary foods and drinks

### When to See a Dentist
{urgency}

---
*{DISCLAIMER}*"""
        
        return {
            'source': 'rule-based',
            'analysis': analysis,
            'disease': disease,
            'confidence': confidence,
            'severity': severity,
            'recommendations': recs,
            'urgency': urgency
        }


class DentalAIPipeline:
    """
    Complete Dental AI Pipeline (PyTorch version).
    Works with Python 3.14+
    """
    
    def __init__(self):
        logger.info("Initializing Dental AI Pipeline (PyTorch)...")
        self.classifier = DentalClassifier()
        self.llm = LLMAnalyzer()
        logger.info("Pipeline ready!")
    
    async def analyze(self, image: Union[str, bytes, Image.Image], 
                     use_llm: bool = True) -> Dict:
        """
        Run complete dental analysis.
        
        Args:
            image: Input dental image
            use_llm: Whether to use LLM for analysis
            
        Returns:
            Complete analysis results
        """
        timestamp = datetime.now().isoformat()
        
        # Step 1: CNN Classification
        logger.info("Running CNN classification...")
        cnn_result = self.classifier.predict(image)
        
        # Step 2: LLM Analysis (optional)
        llm_result = None
        if use_llm:
            logger.info("Running LLM analysis...")
            llm_result = await self.llm.analyze(cnn_result, image)
        
        # Step 3: Build report
        disease = cnn_result['disease']
        confidence = cnn_result['confidence']
        severity = cnn_result['severity']
        disease_info = DISEASE_INFO.get(disease, {})
        
        # Urgency
        if severity == 'High':
            urgency = "Schedule dental appointment within 1-2 weeks"
        elif severity == 'Medium':
            urgency = "Schedule dental appointment within 2-4 weeks"
        else:
            urgency = "Schedule routine checkup within 3-6 months"
        
        # Extract parsed sections from LLM if available
        parsed = {}
        if llm_result and 'parsed' in llm_result:
            parsed = llm_result['parsed']
        
        result = {
            'analysis_id': f"dental_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'timestamp': timestamp,
            
            'summary': {
                'disease': disease,
                'disease_name': disease_info.get('name', disease),
                'confidence': confidence,
                'severity': severity,
                'recommendation': urgency
            },
            
            'cnn_analysis': cnn_result,
            'llm_analysis': llm_result,
            
            # Detailed guidance sections
            'exact_complaint': parsed.get('exact_complaint', ''),
            'detailed_findings': parsed.get('detailed_findings', ''),
            'what_this_means': parsed.get('what_this_means', ''),
            'immediate_actions': parsed.get('immediate_actions', ''),
            'treatment_options': parsed.get('treatment_options', ''),
            'home_care_routine': parsed.get('home_care_routine', ''),
            'prevention_tips': parsed.get('prevention_tips', ''),
            
            'report': {
                'disease': disease,
                'disease_name': disease_info.get('name', disease),
                'description': disease_info.get('description', ''),
                'confidence': confidence,
                'severity': severity,
                'recommendations': disease_info.get('recommendations', []),
                'urgency': urgency,
                'llm_explanation': llm_result.get('analysis', '') if llm_result else '',
                'exact_complaint': parsed.get('exact_complaint', ''),
                'immediate_actions': parsed.get('immediate_actions', ''),
                'home_care_routine': parsed.get('home_care_routine', '')
            },
            
            'disclaimer': DISCLAIMER
        }
        
        logger.info(f"Analysis complete: {disease} ({confidence*100:.1f}%)")
        return result
    
    def analyze_sync(self, image: Union[str, bytes, Image.Image], 
                    use_llm: bool = True) -> Dict:
        """Synchronous wrapper."""
        return asyncio.run(self.analyze(image, use_llm))


def format_report(result: Dict) -> str:
    """Format analysis result as text report."""
    r = result['report']
    s = result['summary']
    
    report = f"""
{'='*60}
🦷 DENTAL AI SCREENING REPORT
{'='*60}

📋 Analysis ID: {result['analysis_id']}
📅 Date: {result['timestamp'][:10]}

{'─'*60}
🔍 EXACT COMPLAINT / DIAGNOSIS
{'─'*60}

Detected Condition: {s['disease_name']}
Confidence: {s['confidence']*100:.1f}%
Severity: {s['severity']}

"""
    
    # Add exact complaint from GPT analysis
    if result.get('exact_complaint'):
        report += f"{result['exact_complaint']}\n"
    else:
        report += f"{r['description']}\n"
    
    # Detailed findings
    if result.get('detailed_findings'):
        report += f"""
{'─'*60}
🔬 DETAILED FINDINGS
{'─'*60}
{result['detailed_findings']}
"""
    
    # What this means
    if result.get('what_this_means'):
        report += f"""
{'─'*60}
📖 WHAT THIS MEANS FOR YOU
{'─'*60}
{result['what_this_means']}
"""
    
    # Immediate actions
    report += f"""
{'─'*60}
⚡ IMMEDIATE ACTIONS TO TAKE
{'─'*60}
"""
    if result.get('immediate_actions'):
        report += f"{result['immediate_actions']}\n"
    else:
        for rec in r['recommendations']:
            report += f"  • {rec}\n"
    
    # Treatment options
    if result.get('treatment_options'):
        report += f"""
{'─'*60}
🏥 TREATMENT OPTIONS
{'─'*60}
{result['treatment_options']}
"""
    
    # Home care routine
    report += f"""
{'─'*60}
🏠 HOME CARE ROUTINE
{'─'*60}
"""
    if result.get('home_care_routine'):
        report += f"{result['home_care_routine']}\n"
    else:
        report += """  Morning: Brush for 2 minutes with fluoride toothpaste
  After meals: Rinse with water or mouthwash
  Night: Brush, floss, use antiseptic mouthwash
"""
    
    # Prevention tips
    if result.get('prevention_tips'):
        report += f"""
{'─'*60}
🛡️ PREVENTION TIPS
{'─'*60}
{result['prevention_tips']}
"""
    
    # When to see dentist
    report += f"""
{'─'*60}
⏰ WHEN TO SEE A DENTIST
{'─'*60}
  {r['urgency']}
"""
    
    # Full AI analysis if available
    if r.get('llm_explanation') and not result.get('exact_complaint'):
        report += f"""
{'─'*60}
📝 FULL AI ANALYSIS
{'─'*60}
{r['llm_explanation']}
"""
    
    report += f"""
{'='*60}
⚠️  DISCLAIMER
{'='*60}
{result['disclaimer']}
"""
    return report


# CLI
if __name__ == '__main__':
    import sys
    
    print("="*60)
    print("Dental AI Pipeline (PyTorch) - Test")
    print("="*60)
    
    # Get image path
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Try to find a test image
        test_paths = [
            Path(__file__).parent / 'test_image.jpg',
            Path(__file__).parent / 'data' / 'Calculus' / 'Calculus_0001.jpg'
        ]
        image_path = None
        for p in test_paths:
            if p.exists():
                image_path = str(p)
                break
    
    if not image_path or not Path(image_path).exists():
        print("Usage: python dental_ai_pytorch.py <image_path>")
        print("No test image found.")
        sys.exit(1)
    
    print(f"\nTest image: {image_path}")
    
    # Initialize pipeline
    print("\n1. Initializing pipeline...")
    pipeline = DentalAIPipeline()
    
    # Run analysis
    print("\n2. Running analysis...")
    use_llm = bool(os.environ.get('OPENAI_API_KEY'))
    result = pipeline.analyze_sync(image_path, use_llm=use_llm)
    
    # Print report
    print("\n3. Results:")
    print(format_report(result))
