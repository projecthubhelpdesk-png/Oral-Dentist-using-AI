# Oral Disease Detection ML Model

Medical-grade AI screening model for oral disease detection using RGB images.

## Folder Structure
```
ml-model/
├── train.py              # Complete training pipeline
├── inference_api.py      # FastAPI inference server
├── predict.py            # Standalone prediction function
├── requirements.txt      # Python dependencies
├── models/               # Saved models
│   └── oral_disease_rgb_model.h5
├── config/
│   └── class_labels.json
└── notebooks/
    └── exploration.ipynb
```

## Setup
```bash
cd ml-model
pip install -r requirements.txt
```

## Training
```bash
python train.py
```

## Run Inference API
```bash
uvicorn inference_api:app --host 0.0.0.0 --port 8000
```

## API Usage
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@oral_image.jpg"
```

## Disclaimer
⚠️ This AI provides preliminary screening only and is not a medical diagnosis.
Always consult a qualified dental professional for proper diagnosis and treatment.
