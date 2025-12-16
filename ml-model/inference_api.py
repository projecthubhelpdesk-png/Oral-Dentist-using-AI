"""
Oral Disease Detection - FastAPI Inference Server
=================================================
REST API for oral disease prediction.

DISCLAIMER: This AI provides preliminary screening only and is not a medical diagnosis.
Always consult a qualified dental professional for proper diagnosis and treatment.

Usage:
    uvicorn inference_api:app --host 0.0.0.0 --port 8000
    
    # Or with auto-reload for development:
    uvicorn inference_api:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import io
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List

# Suppress TensorFlow warnings before importing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from predict import OralDiseasePredictor, DISCLAIMER
from enhanced_teeth_analyzer import EnhancedTeethAnalyzer, create_analysis_summary
from teeth_condition_analyzer import TeethConditionAnalyzer
from roboflow_analyzer import RoboflowDentalAnalyzer

# Advanced Pipeline (LLM + CNN Ensemble)
try:
    from advanced_dental_pipeline import AdvancedDentalPipeline, DentalReportFormatter
    ADVANCED_PIPELINE_AVAILABLE = True
except ImportError:
    ADVANCED_PIPELINE_AVAILABLE = False
    print("Warning: Advanced pipeline not available")

# ============================================
# LOGGING SETUP
# ============================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================
# PYDANTIC MODELS
# ============================================
class PredictionResult(BaseModel):
    """Single prediction result."""
    disease: str = Field(..., description="Detected disease name")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score (0-1)")

class PredictionResponse(BaseModel):
    """API response for prediction endpoint."""
    success: bool = Field(True, description="Whether prediction was successful")
    disease: str = Field(..., description="Primary detected disease")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score (0-1)")
    severity: str = Field(..., description="Severity level: Low, Medium, or High")
    description: str = Field(..., description="Description of the detected condition")
    recommendations: List[str] = Field(..., description="Recommended actions")
    all_predictions: List[PredictionResult] = Field(..., description="All class predictions")
    disclaimer: str = Field(DISCLAIMER, description="Medical disclaimer")
    timestamp: str = Field(..., description="Prediction timestamp")

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field("healthy", description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    version: str = Field("1.0.0", description="API version")
    disclaimer: str = Field(DISCLAIMER, description="Medical disclaimer")

class ErrorResponse(BaseModel):
    """Error response."""
    success: bool = Field(False)
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")

class ToothDetection(BaseModel):
    """Single tooth detection result."""
    tooth_id: str = Field(..., description="FDI tooth number (e.g., 'FDI-11')")
    issue: str = Field(..., description="Detected issue type")
    confidence: float = Field(..., ge=0, le=1, description="Detection confidence")
    bbox: List[float] = Field(..., description="Bounding box [x, y, w, h]")

class EnhancedAnalysisResponse(BaseModel):
    """Enhanced analysis response with tooth-level detection."""
    success: bool = Field(True, description="Whether analysis was successful")
    analysis_id: str = Field(..., description="Unique analysis identifier")
    timestamp: str = Field(..., description="Analysis timestamp")
    
    # Disease classification
    disease: str = Field(..., description="Primary detected disease")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score (0-1)")
    severity: str = Field(..., description="Severity level: Low, Medium, or High")
    description: str = Field(..., description="Description of the detected condition")
    all_predictions: List[PredictionResult] = Field(..., description="All class predictions")
    
    # Enhanced features
    tooth_detections: List[ToothDetection] = Field(..., description="Individual tooth detections")
    affected_teeth: List[str] = Field(..., description="List of affected tooth numbers")
    dental_report: dict = Field(..., description="Complete dental report")
    
    # Recommendations
    recommendations: List[str] = Field(..., description="Recommended actions")
    home_care_tips: List[str] = Field(..., description="Home care recommendations")
    
    disclaimer: str = Field(DISCLAIMER, description="Medical disclaimer")

# ============================================
# FASTAPI APP
# ============================================
app = FastAPI(
    title="Enhanced Oral Disease Detection API",
    description="""
    🦷 **AI-powered comprehensive dental analysis from dental images.**
    
    ## Features
    - **Disease Classification**: EfficientNet-based oral disease detection
    - **Tooth-Level Detection**: Individual tooth issue identification
    - **Severity Assessment**: Rule-based severity scoring
    - **Dental Reports**: Comprehensive, safe dental screening reports
    - **Multi-format Support**: JPEG, PNG, WebP image formats
    
    ## Analysis Pipeline
    1. **EfficientNet Classification** → Disease detection with confidence
    2. **Tooth Detection** → Individual tooth issue localization  
    3. **Severity Assessment** → Rule-based severity scoring
    4. **Report Generation** → Safe, non-diagnostic dental reports
    
    ## Disclaimer
    ⚠️ This AI provides preliminary screening only and is not a medical diagnosis.
    Always consult a qualified dental professional for proper diagnosis and treatment.
    """,
    version="2.0.0-enhanced",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for web app integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# GLOBAL PREDICTORS
# ============================================
predictor: Optional[OralDiseasePredictor] = None
enhanced_analyzer: Optional[EnhancedTeethAnalyzer] = None
condition_analyzer: Optional[TeethConditionAnalyzer] = None
roboflow_analyzer: Optional[RoboflowDentalAnalyzer] = None
advanced_pipeline = None

@app.on_event("startup")
async def load_model():
    """Load models on startup."""
    global predictor, enhanced_analyzer, condition_analyzer, roboflow_analyzer, advanced_pipeline
    try:
        logger.info("Loading oral disease detection model...")
        predictor = OralDiseasePredictor()
        logger.info("Model loaded successfully!")
        
        # Load enhanced analyzer
        logger.info("Loading enhanced teeth analyzer...")
        enhanced_analyzer = EnhancedTeethAnalyzer()
        logger.info("Enhanced analyzer loaded successfully!")
        
        # Load teeth condition analyzer
        logger.info("Loading teeth condition analyzer...")
        condition_analyzer = TeethConditionAnalyzer()
        logger.info("Teeth condition analyzer loaded successfully!")
        
        # Load Roboflow analyzer
        logger.info("Loading Roboflow dental analyzer...")
        roboflow_analyzer = RoboflowDentalAnalyzer()
        logger.info("Roboflow analyzer loaded successfully!")
        
        # Load Advanced Pipeline (LLM + CNN Ensemble)
        if ADVANCED_PIPELINE_AVAILABLE:
            logger.info("Loading advanced dental pipeline (LLM + CNN Ensemble)...")
            advanced_pipeline = AdvancedDentalPipeline()
            logger.info("Advanced pipeline loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        predictor = None
        enhanced_analyzer = None
        condition_analyzer = None
        roboflow_analyzer = None
        advanced_pipeline = None

# ============================================
# API ENDPOINTS
# ============================================

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - API information."""
    return HealthResponse(
        status="healthy",
        model_loaded=predictor is not None and enhanced_analyzer is not None,
        version="2.0.0-enhanced",
        disclaimer=DISCLAIMER
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if predictor else "degraded",
        model_loaded=predictor is not None,
        version="1.0.0",
        disclaimer=DISCLAIMER
    )

@app.post("/predict", response_model=PredictionResponse, responses={
    400: {"model": ErrorResponse, "description": "Invalid input"},
    500: {"model": ErrorResponse, "description": "Server error"},
    503: {"model": ErrorResponse, "description": "Model not loaded"}
})
async def predict(
    file: UploadFile = File(..., description="Dental image file (JPEG, PNG, WebP)")
):
    """
    Predict oral disease from uploaded image.
    
    - **file**: Dental image file (JPEG, PNG, or WebP format)
    
    Returns prediction with disease name, confidence score, and severity level.
    
    ## Example Response
    ```json
    {
        "success": true,
        "disease": "Gingivitis",
        "confidence": 0.87,
        "severity": "High",
        "all_predictions": [
            {"disease": "Gingivitis", "confidence": 0.87},
            {"disease": "Healthy", "confidence": 0.08},
            ...
        ],
        "disclaimer": "This AI provides preliminary screening only...",
        "timestamp": "2024-12-13T10:30:00"
    }
    ```
    """
    # Check if model is loaded
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please try again later."
        )
    
    # Validate file type - be more lenient and check file extension if content_type is None
    allowed_types = ['image/jpeg', 'image/png', 'image/webp', 'image/jpg']
    allowed_extensions = ['.jpg', '.jpeg', '.png', '.webp']
    
    # Check content type first, then fall back to file extension
    valid_type = False
    if file.content_type and file.content_type in allowed_types:
        valid_type = True
    elif file.filename:
        # Check file extension if content_type is None or not recognized
        import os
        _, ext = os.path.splitext(file.filename.lower())
        if ext in allowed_extensions:
            valid_type = True
    
    if not valid_type:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type} (filename: {file.filename}). Allowed: JPEG, PNG, WebP"
        )
    
    # Validate file size (max 10MB)
    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="File too large. Maximum size is 10MB."
        )
    
    try:
        # Run prediction
        logger.info(f"Processing image: {file.filename} ({len(contents)} bytes)")
        result = predictor.predict_from_bytes(contents)
        
        # Format response
        return PredictionResponse(
            success=True,
            disease=result['disease'],
            confidence=result['confidence'],
            severity=result['severity'],
            description=result.get('description', f"{result['disease']} detected"),
            recommendations=result.get('recommendations', ['Consult a dental professional']),
            all_predictions=[
                PredictionResult(disease=p['disease'], confidence=p['confidence'])
                for p in result['all_predictions']
            ],
            disclaimer=DISCLAIMER,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/predict/batch", responses={
    400: {"model": ErrorResponse},
    500: {"model": ErrorResponse},
    503: {"model": ErrorResponse}
})
async def predict_batch(
    files: List[UploadFile] = File(..., description="Multiple dental images")
):
    """
    Predict oral disease from multiple images.
    
    - **files**: List of dental image files
    
    Returns list of predictions for each image.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 images per batch")
    
    results = []
    for file in files:
        try:
            contents = await file.read()
            result = predictor.predict_from_bytes(contents)
            results.append({
                "filename": file.filename,
                "success": True,
                "disease": result['disease'],
                "confidence": result['confidence'],
                "severity": result['severity']
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
    
    return {
        "success": True,
        "count": len(results),
        "results": results,
        "disclaimer": DISCLAIMER,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/analyze-teeth", response_model=EnhancedAnalysisResponse, responses={
    400: {"model": ErrorResponse, "description": "Invalid input"},
    500: {"model": ErrorResponse, "description": "Server error"},
    503: {"model": ErrorResponse, "description": "Models not loaded"}
})
async def analyze_teeth_enhanced(
    file: UploadFile = File(..., description="Dental image file (JPEG, PNG, WebP)")
):
    """
    🦷 **Complete Enhanced Teeth Analysis**
    
    Performs comprehensive dental analysis including:
    - Disease classification (EfficientNet)
    - Individual tooth detection
    - Severity assessment
    - Dental report generation
    
    **Input**: Dental image file (JPEG, PNG, WebP)
    
    **Output**: Complete analysis with:
    - Primary condition detection
    - Individual tooth issues
    - Severity scoring
    - Professional recommendations
    - Home care tips
    - Safe, non-diagnostic report
    """
    # Check if enhanced analyzer is loaded
    if enhanced_analyzer is None:
        raise HTTPException(
            status_code=503,
            detail="Enhanced analyzer not loaded. Please try again later."
        )
    
    # Validate file type
    allowed_types = ['image/jpeg', 'image/png', 'image/webp', 'image/jpg']
    allowed_extensions = ['.jpg', '.jpeg', '.png', '.webp']
    
    valid_type = False
    if file.content_type and file.content_type in allowed_types:
        valid_type = True
    elif file.filename:
        import os
        _, ext = os.path.splitext(file.filename.lower())
        if ext in allowed_extensions:
            valid_type = True
    
    if not valid_type:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type} (filename: {file.filename}). Allowed: JPEG, PNG, WebP"
        )
    
    # Validate file size (max 10MB)
    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="File too large. Maximum size is 10MB."
        )
    
    try:
        # Run enhanced analysis
        logger.info(f"Processing enhanced dental analysis: {file.filename} ({len(contents)} bytes)")
        result = enhanced_analyzer.analyze_teeth(contents)
        
        # Extract key components
        disease_classification = result['disease_classification']
        tooth_detections = result['tooth_detections']
        dental_report = result['dental_report']
        
        # Format response
        return EnhancedAnalysisResponse(
            success=True,
            analysis_id=result['analysis_id'],
            timestamp=result['timestamp'],
            
            # Disease classification
            disease=disease_classification['primary_condition'],
            confidence=disease_classification['confidence'],
            severity=result['severity_analysis']['severity'],
            description=disease_classification.get('description', f"{disease_classification['primary_condition']} detected"),
            all_predictions=[
                PredictionResult(disease=p['disease'], confidence=p['confidence'])
                for p in disease_classification['all_predictions'][:5]
            ],
            
            # Enhanced features
            tooth_detections=[
                ToothDetection(**detection) for detection in tooth_detections
            ],
            affected_teeth=dental_report['affected_teeth'],
            dental_report=dental_report,
            
            # Recommendations
            recommendations=dental_report['professional_advice'],
            home_care_tips=dental_report['home_care_tips'],
            
            disclaimer=DISCLAIMER
        )
        
    except Exception as e:
        logger.error(f"Enhanced analysis error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Enhanced analysis failed: {str(e)}"
        )

@app.post("/analyze-teeth/summary")
async def analyze_teeth_summary(
    file: UploadFile = File(..., description="Dental image file")
):
    """
    🦷 **Quick Enhanced Analysis Summary**
    
    Returns a human-readable summary of enhanced dental analysis.
    Useful for simple integrations or testing.
    """
    if enhanced_analyzer is None:
        raise HTTPException(status_code=503, detail="Enhanced analyzer not loaded")
    
    # Validate file
    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large")
    
    try:
        # Run enhanced analysis
        result = enhanced_analyzer.analyze_teeth(contents)
        
        # Create summary
        summary = create_analysis_summary(result)
        
        return {
            "success": True,
            "analysis_id": result['analysis_id'],
            "summary": summary,
            "key_findings": {
                "primary_condition": result['disease_classification']['primary_condition'],
                "confidence": result['disease_classification']['confidence'],
                "severity": result['severity_analysis']['severity'],
                "affected_teeth": len(result['dental_report']['affected_teeth']),
                "tooth_detections": len(result['tooth_detections']),
                "recommendation": result['dental_report']['recommendation']
            },
            "timestamp": result['timestamp']
        }
        
    except Exception as e:
        logger.error(f"Enhanced summary analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/info")
async def get_model_info():
    """Get information about loaded models."""
    models_info = {
        "basic_predictor": {
            "loaded": predictor is not None,
            "classes": list(predictor.class_labels.values()) if predictor else [],
            "count": predictor.num_classes if predictor else 0
        },
        "enhanced_analyzer": {
            "loaded": enhanced_analyzer is not None,
            "features": [
                "Disease Classification (EfficientNet)",
                "Tooth Detection",
                "Severity Scoring", 
                "Dental Report Generation"
            ] if enhanced_analyzer else []
        }
    }
    
    if enhanced_analyzer:
        models_info["enhanced_analyzer"].update({
            "efficientnet": {
                "architecture": "EfficientNetB0",
                "input_size": [224, 224, 3],
                "classes": enhanced_analyzer.disease_predictor.num_classes,
                "class_labels": enhanced_analyzer.disease_predictor.class_labels
            },
            "tooth_detector": {
                "detection_classes": ["healthy_tooth", "cavity", "plaque", "crooked_tooth", "missing_tooth"],
                "fdi_mapping": "Enabled"
            }
        })
    
    return models_info

@app.post("/analyze-condition")
async def analyze_teeth_condition(
    file: UploadFile = File(..., description="Dental image file (JPEG, PNG, WebP)")
):
    """
    🦷 **EfficientNet Teeth Condition Analysis**
    
    Comprehensive teeth condition analysis including:
    - Overall dental health assessment
    - Teeth whiteness/discoloration
    - Decay/cavity risk
    - Gum health indicators
    - Plaque/tartar level
    - Teeth alignment
    
    Returns detailed condition breakdown with personalized recommendations.
    """
    if condition_analyzer is None:
        raise HTTPException(status_code=503, detail="Condition analyzer not loaded")
    
    # Validate file type
    allowed_types = ['image/jpeg', 'image/png', 'image/webp', 'image/jpg']
    allowed_extensions = ['.jpg', '.jpeg', '.png', '.webp']
    
    valid_type = False
    if file.content_type and file.content_type in allowed_types:
        valid_type = True
    elif file.filename:
        import os
        _, ext = os.path.splitext(file.filename.lower())
        if ext in allowed_extensions:
            valid_type = True
    
    if not valid_type:
        raise HTTPException(status_code=400, detail=f"Invalid file type. Allowed: JPEG, PNG, WebP")
    
    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Maximum 10MB.")
    
    try:
        logger.info(f"Analyzing teeth condition: {file.filename}")
        result = condition_analyzer.analyze(contents)
        
        return {
            "success": True,
            **result
        }
    except Exception as e:
        logger.error(f"Condition analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-complete")
async def analyze_complete(
    file: UploadFile = File(..., description="Dental image file")
):
    """
    🦷 **Complete Dental Analysis**
    
    Combines all analysis types:
    1. Disease classification (EfficientNet)
    2. Tooth-level detection (YOLOv8)
    3. Teeth condition assessment (EfficientNet multi-output)
    4. Severity scoring
    5. Comprehensive dental report
    
    Returns the most comprehensive analysis available.
    """
    if enhanced_analyzer is None or condition_analyzer is None:
        raise HTTPException(status_code=503, detail="Analyzers not loaded")
    
    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large")
    
    try:
        logger.info(f"Running complete dental analysis: {file.filename}")
        
        # Run both analyses
        enhanced_result = enhanced_analyzer.analyze_teeth(contents)
        condition_result = condition_analyzer.analyze(contents)
        
        # Combine results
        combined = {
            "success": True,
            "analysis_id": f"complete_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            
            # Disease analysis
            "disease_analysis": {
                "primary_condition": enhanced_result['disease_classification']['primary_condition'],
                "confidence": enhanced_result['disease_classification']['confidence'],
                "severity": enhanced_result['severity_analysis']['severity'],
                "all_predictions": enhanced_result['disease_classification']['all_predictions'][:5]
            },
            
            # Tooth detections
            "tooth_detections": enhanced_result['tooth_detections'],
            "affected_teeth": enhanced_result['dental_report']['affected_teeth'],
            
            # Condition analysis
            "condition_analysis": {
                "overall_score": condition_result['overall_score'],
                "condition_breakdown": condition_result['condition_breakdown'],
                "risk_factors": condition_result['risk_factors'],
                "positive_aspects": condition_result['positive_aspects']
            },
            
            # Combined recommendations
            "recommendations": {
                "professional": enhanced_result['dental_report']['professional_advice'],
                "home_care": enhanced_result['dental_report']['home_care_tips'],
                "condition_based": condition_result['recommendations'][:5]
            },
            
            # Summary
            "summary": {
                "disease_summary": enhanced_result['dental_report']['summary'],
                "condition_summary": condition_result['summary'],
                "follow_up": condition_result['follow_up']
            },
            
            "disclaimer": DISCLAIMER
        }
        
        return combined
        
    except Exception as e:
        logger.error(f"Complete analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-roboflow")
async def analyze_with_roboflow_api(
    file: UploadFile = File(..., description="Dental image file")
):
    """
    🦷 **Roboflow Dental Analysis**
    
    Uses Roboflow's trained model for detecting:
    - Gums
    - Teeth
    - Plaques/Tartar
    - Other dental conditions
    
    This provides more accurate plaque/tartar detection.
    """
    if roboflow_analyzer is None:
        raise HTTPException(status_code=503, detail="Roboflow analyzer not loaded")
    
    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large")
    
    try:
        logger.info(f"Running Roboflow analysis: {file.filename}")
        result = roboflow_analyzer.analyze(contents)
        
        return {
            "success": result.get('success', False),
            "source": "roboflow",
            "detections": result.get('detections', []),
            "counts": result.get('counts', {}),
            "health_assessment": result.get('health_assessment', {}),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Roboflow analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/classes")
async def get_classes():
    """Get list of detectable disease classes."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "classes": list(predictor.class_labels.values()),
        "count": predictor.num_classes
    }


# ============================================
# ADVANCED PIPELINE ENDPOINTS (LLM + CNN)
# ============================================

@app.post("/analyze-advanced")
async def analyze_with_advanced_pipeline(
    file: UploadFile = File(..., description="Dental image file"),
    use_llm: bool = Query(True, description="Enable LLM analysis (GPT-4o)"),
    use_vlm: bool = Query(False, description="Enable VLM analysis (requires Ollama)")
):
    """
    🦷 **ADVANCED AI DENTAL ANALYSIS**
    
    Ultimate multi-stage pipeline combining:
    
    **Stage 1 - CNN Ensemble:**
    - EfficientNet-B4 (40% weight)
    - EfficientNet-B3 (30% weight)
    - ResNet50 (20% weight)
    - MobileNetV3 (10% weight)
    
    **Stage 2 - LLM Analysis:**
    - GPT-4o with vision for intelligent explanations
    - Contextual recommendations
    - Patient-friendly reports
    
    **Stage 3 - VLM Analysis (optional):**
    - LLaVA-1.6 for detailed image understanding
    - Qwen-VL for vision comprehension
    
    Returns comprehensive analysis with AI-generated explanations.
    """
    if not ADVANCED_PIPELINE_AVAILABLE or advanced_pipeline is None:
        raise HTTPException(
            status_code=503, 
            detail="Advanced pipeline not available. Check if advanced_dental_pipeline.py exists."
        )
    
    # Validate file
    allowed_extensions = ['.jpg', '.jpeg', '.png', '.webp']
    ext = Path(file.filename).suffix.lower() if file.filename else ''
    if ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Invalid file type. Allowed: {allowed_extensions}")
    
    contents = await file.read()
    if len(contents) > 15 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Maximum 15MB.")
    
    try:
        logger.info(f"Running advanced pipeline analysis: {file.filename}")
        result = await advanced_pipeline.analyze(contents, use_llm=use_llm, use_vlm=use_vlm)
        
        return {
            "success": True,
            **result
        }
    except Exception as e:
        logger.error(f"Advanced pipeline error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze-advanced/report")
async def get_advanced_report(
    file: UploadFile = File(..., description="Dental image file"),
    format: str = Query("text", enum=["text", "html", "json"]),
    use_llm: bool = Query(True, description="Enable LLM analysis")
):
    """
    📄 **Get Advanced Analysis Report**
    
    Returns formatted report from advanced pipeline:
    - `text`: Human-readable text report
    - `html`: Styled HTML report (can be displayed in browser)
    - `json`: Raw JSON data
    """
    if not ADVANCED_PIPELINE_AVAILABLE or advanced_pipeline is None:
        raise HTTPException(status_code=503, detail="Advanced pipeline not available")
    
    contents = await file.read()
    if len(contents) > 15 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large")
    
    try:
        logger.info(f"Generating advanced report: {file.filename} (format: {format})")
        result = await advanced_pipeline.analyze(contents, use_llm=use_llm, use_vlm=False)
        
        formatter = DentalReportFormatter()
        
        if format == "html":
            from fastapi.responses import HTMLResponse
            html = formatter.to_html(result)
            return HTMLResponse(content=html)
        elif format == "json":
            return result
        else:
            text = formatter.to_text(result)
            return {
                "success": True,
                "format": "text",
                "report": text,
                "summary": result['summary']
            }
    except Exception as e:
        logger.error(f"Advanced report error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze-advanced/quick")
async def quick_advanced_analysis(
    file: UploadFile = File(..., description="Dental image file")
):
    """
    ⚡ **Quick Advanced Analysis (CNN Only)**
    
    Fast analysis using CNN ensemble without LLM.
    Returns in ~1-2 seconds.
    """
    if not ADVANCED_PIPELINE_AVAILABLE or advanced_pipeline is None:
        raise HTTPException(status_code=503, detail="Advanced pipeline not available")
    
    contents = await file.read()
    
    try:
        result = await advanced_pipeline.analyze(contents, use_llm=False, use_vlm=False)
        
        return {
            "success": True,
            "analysis_id": result['analysis_id'],
            "disease": result['summary']['disease'],
            "confidence": result['summary']['confidence'],
            "severity": result['summary']['severity'],
            "recommendation": result['summary']['recommendation'],
            "all_predictions": result['cnn_analysis']['all_predictions'][:5],
            "disclaimer": DISCLAIMER
        }
    except Exception as e:
        logger.error(f"Quick advanced analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# ERROR HANDLERS
# ============================================
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "detail": str(exc)
        }
    )

# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    uvicorn.run(
        "inference_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
