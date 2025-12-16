# Advanced Dental AI Pipeline

Ultimate multi-stage AI system for dental disease detection with CNN ensemble and LLM integration.

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    ADVANCED DENTAL AI PIPELINE                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐    ┌─────────────────────────────────────────┐   │
│  │  Image   │───▶│           CNN ENSEMBLE                   │   │
│  └──────────┘    │  ┌─────────────┐  ┌─────────────┐       │   │
│                  │  │EfficientNet │  │EfficientNet │       │   │
│                  │  │    B4 (40%) │  │    B3 (30%) │       │   │
│                  │  └─────────────┘  └─────────────┘       │   │
│                  │  ┌─────────────┐  ┌─────────────┐       │   │
│                  │  │  ResNet50   │  │ MobileNetV3 │       │   │
│                  │  │     (20%)   │  │     (10%)   │       │   │
│                  │  └─────────────┘  └─────────────┘       │   │
│                  └──────────────────────┬──────────────────┘   │
│                                         │                       │
│                                         ▼                       │
│                  ┌─────────────────────────────────────────┐   │
│                  │         Disease Probability              │   │
│                  │   Calculus | Caries | Gingivitis | Ulcer │   │
│                  └──────────────────────┬──────────────────┘   │
│                                         │                       │
│                                         ▼                       │
│                  ┌─────────────────────────────────────────┐   │
│                  │           LLM / VLM ANALYSIS             │   │
│                  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  │   │
│                  │  │ GPT-4o  │  │ LLaVA   │  │ Qwen-VL │  │   │
│                  │  │ (Vision)│  │  1.6    │  │         │  │   │
│                  │  └─────────┘  └─────────┘  └─────────┘  │   │
│                  └──────────────────────┬──────────────────┘   │
│                                         │                       │
│                                         ▼                       │
│                  ┌─────────────────────────────────────────┐   │
│                  │         COMPREHENSIVE REPORT             │   │
│                  │  • Explanation  • Recommendations        │   │
│                  │  • Severity     • Home Care Tips         │   │
│                  │  • Urgency      • Next Steps             │   │
│                  └─────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Features

### CNN Ensemble
- **EfficientNet-B4** (40% weight) - Best accuracy for medical imaging
- **EfficientNet-B3** (30% weight) - Good balance of speed/accuracy
- **ResNet50** (20% weight) - Robust feature extraction
- **MobileNetV3** (10% weight) - Fast inference, mobile-ready

### LLM Integration
- **GPT-4o** with vision - Intelligent image analysis and explanations
- **LLaMA-3-Instruct** - Local medical explanations (via Ollama)
- **Mistral-Instruct** - Lightweight local reasoning (via Ollama)

### VLM Support
- **LLaVA-1.6** - Advanced image + text reasoning
- **Qwen-VL** - Vision understanding

## Installation

```bash
# Install dependencies
pip install -r requirements_advanced.txt

# Set OpenAI API key (for GPT-4o)
export OPENAI_API_KEY=your_api_key

# Optional: Install Ollama for local LLM/VLM
# https://ollama.ai
```

## Usage

### Python API

```python
from advanced_dental_pipeline import AdvancedDentalPipeline

# Initialize pipeline
pipeline = AdvancedDentalPipeline()

# Run analysis
result = await pipeline.analyze(
    "dental_image.jpg",
    use_llm=True,   # Enable GPT-4o analysis
    use_vlm=False   # Enable LLaVA (requires Ollama)
)

# Get results
print(f"Disease: {result['summary']['disease']}")
print(f"Confidence: {result['summary']['confidence']}")
print(f"Severity: {result['summary']['severity']}")
print(f"Report: {result['report']['llm_explanation']}")
```

### REST API

```bash
# Start the API server
uvicorn advanced_api:app --host 0.0.0.0 --port 8001

# Full analysis with LLM
curl -X POST "http://localhost:8001/analyze?use_llm=true" \
  -F "file=@dental_image.jpg"

# Quick CNN-only analysis
curl -X POST "http://localhost:8001/analyze/quick" \
  -F "file=@dental_image.jpg"

# Get HTML report
curl -X POST "http://localhost:8001/analyze/report?format=html" \
  -F "file=@dental_image.jpg" > report.html
```

### Command Line

```bash
# Basic analysis
python advanced_dental_pipeline.py dental_image.jpg

# With LLM analysis
python advanced_dental_pipeline.py dental_image.jpg --format html --output report.html

# Quick test
python test_advanced_pipeline.py dental_image.jpg
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/analyze` | POST | Full advanced analysis with LLM |
| `/analyze/quick` | POST | Fast CNN-only analysis |
| `/analyze/report` | POST | Get formatted report (text/html/json) |
| `/health` | GET | Health check |
| `/models/info` | GET | Model information |

## Training EfficientNet-B4

```bash
# Train the best model
python train_efficientnet_b4.py \
  --data-dir datasets/oral-diseases/train \
  --epochs 30 \
  --fine-tune-epochs 20
```

## Configuration

Edit `config/llm_config.json` to configure:
- LLM providers (OpenAI, local LLaMA, Mistral)
- VLM providers (LLaVA, Qwen-VL)
- CNN ensemble weights
- Dental prompts

## Output Example

```json
{
  "analysis_id": "adv_20241214_120000",
  "summary": {
    "disease": "Calculus",
    "confidence": 0.87,
    "severity": "Medium",
    "recommendation": "Schedule dental appointment within 2-4 weeks"
  },
  "cnn_analysis": {
    "primary_disease": "Calculus",
    "all_predictions": [
      {"disease": "Calculus", "confidence": 0.87},
      {"disease": "Gingivitis", "confidence": 0.08},
      {"disease": "Caries", "confidence": 0.03},
      {"disease": "Mouth_Ulcer", "confidence": 0.02}
    ]
  },
  "llm_analysis": {
    "source": "gpt-4o-vision",
    "llm_analysis": "The AI analysis detected signs of dental calculus..."
  },
  "report": {
    "disease_name": "Dental Calculus (Tartar)",
    "severity": "Medium",
    "recommendations": [...],
    "home_care_tips": [...]
  }
}
```

## Disclaimer

⚠️ This AI provides preliminary screening only and is not a medical diagnosis. Always consult a qualified dental professional for proper diagnosis and treatment.
