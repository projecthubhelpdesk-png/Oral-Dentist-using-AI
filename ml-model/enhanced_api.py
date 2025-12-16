"""
Enhanced Teeth Analyzer - FastAPI Integration
=============================================
Complete API for oral disease classification, tooth localization, 
severity scoring, and dental report generation.

DISCLAIMER: This AI provides preliminary screening only and is not a medical diagnosis.
Always consult a qualified dental professional for proper diagnosis and treatment.
"""

import os
import io
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List
import base64

# Suppress TensorFlow warnings before importing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
import uvicorn
from PIL import Image
import numpy as np

from enhanced_teeth_analyzer import EnhancedTeethAnalyzer, create_analysis_summary
from spectral_dental_pipeline import SpectralDentalPipeline

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
class ToothDetection(BaseModel):
    """Single tooth detection result."""
    tooth_id: str = Field(..., description="FDI tooth number (e.g., 'FDI-11')")
    issue: str = Field(..., description="Detected issue type")
    confidence: float = Field(..., ge=0, le=1, description="Detection confidence")
    bbox: List[float] = Field(..., description="Bounding box [x, y, w, h]")

class DiseaseClassification(BaseModel):
    """Disease classification result."""
    primary_condition: str = Field(..., description="Primary detected condition")
    confidence: float = Field(..., ge=0, le=1, description="Classification confidence")
    all_predictions: List[dict] = Field(..., description="All class predictions")
    description: str = Field(..., description="Condition description")

class SeverityAnalysis(BaseModel):
    """Severity analysis result."""
    severity: str = Field(..., description="Severity level: Low, Medium, or High")
    combined_confidence: float = Field(..., description="Combined confidence score")
    affected_teeth_count: int = Field(..., description="Number of affected teeth")
    reasoning: str = Field(..., description="Severity reasoning")

class DentalReport(BaseModel):
    """Complete dental report."""
    summary: str = Field(..., description="Analysis summary")
    severity: str = Field(..., description="Overall severity")
    confidence_level: str = Field(..., description="Confidence level")
    affected_teeth: List[str] = Field(..., description="List of affected tooth numbers")
    detected_issues: dict = Field(..., description="Summary of detected issues")
    recommendation: str = Field(..., description="Urgency recommendation")
    home_care_tips: List[str] = Field(..., description="Home care recommendations")
    professional_advice: List[str] = Field(..., description="Professional recommendations")
    screening_date: str = Field(..., description="Analysis timestamp")
    ai_disclaimer: str = Field(..., description="Medical disclaimer")
    next_steps: List[str] = Field(..., description="Recommended next steps")

class CompleteAnalysisResponse(BaseModel):
    """Complete analysis response."""
    analysis_id: str = Field(..., description="Unique analysis identifier")
    timestamp: str = Field(..., description="Analysis timestamp")
    disease_classification: DiseaseClassification
    tooth_detections: List[ToothDetection]
    severity_analysis: SeverityAnalysis
    dental_report: DentalReport
    technical_details: dict = Field(..., description="Technical processing details")
    disclaimer: str = Field(..., description="Medical disclaimer")

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field("healthy", description="Service status")
    models_loaded: dict = Field(..., description="Model loading status")
    version: str = Field("2.0.0", description="API version")
    features: List[str] = Field(..., description="Available features")
    disclaimer: str = Field(..., description="Medical disclaimer")

class ErrorResponse(BaseModel):
    """Error response."""
    success: bool = Field(False)
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: str = Field(..., description="Error timestamp")

# ============================================
# FASTAPI APP
# ============================================
app = FastAPI(
    title="Enhanced Teeth Analyzer API",
    description="""
    🦷 **Complete AI-Powered Dental Analysis System**
    
    ## Features
    - **Disease Classification**: EfficientNet-based oral disease detection
    - **Tooth Localization**: YOLOv8-powered individual tooth analysis
    - **Severity Scoring**: Rule-based severity assessment
    - **Dental Reports**: Comprehensive, safe dental screening reports
    - **Multi-format Support**: JPEG, PNG, WebP image formats
    
    ## Analysis Pipeline
    1. **EfficientNet Classification** → Disease detection with confidence
    2. **YOLOv8 Tooth Detection** → Individual tooth issue localization
    3. **Severity Assessment** → Rule-based severity scoring
    4. **Report Generation** → Safe, non-diagnostic dental reports
    
    ## Medical Safety
    ⚠️ **IMPORTANT**: This AI provides preliminary screening only and is not a medical diagnosis.
    Always consult a qualified dental professional for proper diagnosis and treatment.
    
    ## Supported Conditions
    - Caries (Cavities)
    - Gingivitis (Gum inflammation)
    - Calculus (Tartar buildup)
    - Mouth Ulcers
    - Tooth-level issues (plaque, cavities, alignment)
    """,
    version="2.0.0",
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
# GLOBAL ANALYZERS
# ============================================
analyzer: Optional[EnhancedTeethAnalyzer] = None
spectral_pipeline: Optional[SpectralDentalPipeline] = None

@app.on_event("startup")
async def load_models():
    """Load models on startup."""
    global analyzer, spectral_pipeline
    try:
        logger.info("Loading Enhanced Teeth Analyzer...")
        analyzer = EnhancedTeethAnalyzer()
        logger.info("Enhanced Teeth Analyzer loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load analyzer: {e}")
        analyzer = None
    
    try:
        logger.info("Loading Spectral Dental Pipeline...")
        spectral_pipeline = SpectralDentalPipeline()
        logger.info("Spectral Dental Pipeline loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load spectral pipeline: {e}")
        spectral_pipeline = None

# ============================================
# API ENDPOINTS
# ============================================

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - API information."""
    models_status = {
        "efficientnet": analyzer is not None and analyzer.disease_predictor is not None,
        "yolo": analyzer is not None and analyzer.tooth_detector is not None,
        "analyzer": analyzer is not None
    }
    
    return HealthResponse(
        status="healthy" if analyzer else "degraded",
        models_loaded=models_status,
        version="2.0.0",
        features=[
            "Disease Classification (EfficientNet)",
            "Tooth Detection (YOLOv8)",
            "Severity Scoring",
            "Dental Report Generation",
            "Image Annotation",
            "Batch Processing"
        ],
        disclaimer="This AI provides preliminary screening only and is not a medical diagnosis."
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check endpoint."""
    models_status = {
        "efficientnet": analyzer is not None and analyzer.disease_predictor is not None,
        "yolo": analyzer is not None and analyzer.tooth_detector is not None,
        "analyzer": analyzer is not None
    }
    
    status = "healthy"
    if not analyzer:
        status = "unhealthy"
    elif not all(models_status.values()):
        status = "degraded"
    
    return HealthResponse(
        status=status,
        models_loaded=models_status,
        version="2.0.0",
        features=[
            "Disease Classification (EfficientNet)",
            "Tooth Detection (YOLOv8)", 
            "Severity Scoring",
            "Dental Report Generation",
            "Image Annotation",
            "Batch Processing"
        ],
        disclaimer="This AI provides preliminary screening only and is not a medical diagnosis."
    )

@app.post("/analyze-teeth", response_model=CompleteAnalysisResponse, responses={
    400: {"model": ErrorResponse, "description": "Invalid input"},
    500: {"model": ErrorResponse, "description": "Server error"},
    503: {"model": ErrorResponse, "description": "Models not loaded"}
})
async def analyze_teeth(
    file: UploadFile = File(..., description="Dental image file (JPEG, PNG, WebP)"),
    include_annotations: bool = Query(False, description="Include annotated image in response")
):
    """
    🦷 **Complete Teeth Analysis**
    
    Performs comprehensive dental analysis including:
    - Disease classification (EfficientNet)
    - Individual tooth detection (YOLOv8)
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
    
    **Example Response**:
    ```json
    {
        "analysis_id": "analysis_20241214_143022",
        "disease_classification": {
            "primary_condition": "Gingivitis",
            "confidence": 0.87
        },
        "tooth_detections": [
            {
                "tooth_id": "FDI-11",
                "issue": "plaque",
                "confidence": 0.75
            }
        ],
        "severity_analysis": {
            "severity": "Medium",
            "affected_teeth_count": 2
        },
        "dental_report": {
            "summary": "Signs of gingivitis detected...",
            "recommendation": "Schedule dental appointment within 2-3 weeks"
        }
    }
    ```
    """
    # Check if analyzer is loaded
    if analyzer is None:
        raise HTTPException(
            status_code=503,
            detail="Enhanced Teeth Analyzer not loaded. Please try again later."
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
        # Run complete analysis
        logger.info(f"Processing dental image: {file.filename} ({len(contents)} bytes)")
        result = analyzer.analyze_teeth(contents)
        
        # Convert to response model
        response = CompleteAnalysisResponse(
            analysis_id=result['analysis_id'],
            timestamp=result['timestamp'],
            disease_classification=DiseaseClassification(**result['disease_classification']),
            tooth_detections=[ToothDetection(**detection) for detection in result['tooth_detections']],
            severity_analysis=SeverityAnalysis(**result['severity_analysis']),
            dental_report=DentalReport(**result['dental_report']),
            technical_details=result['technical_details'],
            disclaimer=result['disclaimer']
        )
        
        # Add annotated image if requested
        if include_annotations:
            try:
                annotated_img = analyzer.create_annotated_image(contents, result)
                
                # Convert to base64
                img_buffer = io.BytesIO()
                annotated_img.save(img_buffer, format='PNG')
                img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
                
                # Add to technical details
                response.technical_details['annotated_image'] = f"data:image/png;base64,{img_base64}"
            except Exception as e:
                logger.warning(f"Failed to create annotated image: {e}")
        
        return response
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

@app.post("/analyze-teeth/summary")
async def analyze_teeth_summary(
    file: UploadFile = File(..., description="Dental image file")
):
    """
    🦷 **Quick Analysis Summary**
    
    Returns a human-readable summary of dental analysis.
    Useful for simple integrations or testing.
    """
    if analyzer is None:
        raise HTTPException(status_code=503, detail="Analyzer not loaded")
    
    # Validate file
    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large")
    
    try:
        # Run analysis
        result = analyzer.analyze_teeth(contents)
        
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
                "recommendation": result['dental_report']['recommendation']
            },
            "timestamp": result['timestamp']
        }
        
    except Exception as e:
        logger.error(f"Summary analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-teeth/batch")
async def analyze_teeth_batch(
    files: List[UploadFile] = File(..., description="Multiple dental images")
):
    """
    🦷 **Batch Analysis**
    
    Analyze multiple dental images in a single request.
    Maximum 5 images per batch.
    """
    if analyzer is None:
        raise HTTPException(status_code=503, detail="Analyzer not loaded")
    
    if len(files) > 5:
        raise HTTPException(status_code=400, detail="Maximum 5 images per batch")
    
    results = []
    for i, file in enumerate(files):
        try:
            contents = await file.read()
            if len(contents) > 10 * 1024 * 1024:
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": "File too large"
                })
                continue
            
            # Run analysis
            result = analyzer.analyze_teeth(contents)
            
            # Simplified result for batch
            results.append({
                "filename": file.filename,
                "success": True,
                "analysis_id": result['analysis_id'],
                "primary_condition": result['disease_classification']['primary_condition'],
                "confidence": result['disease_classification']['confidence'],
                "severity": result['severity_analysis']['severity'],
                "affected_teeth": len(result['dental_report']['affected_teeth']),
                "recommendation": result['dental_report']['recommendation']
            })
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
    
    return {
        "success": True,
        "batch_id": f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "total_images": len(files),
        "successful_analyses": len([r for r in results if r.get('success')]),
        "results": results,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/analyze/spectral")
async def analyze_spectral(
    file: UploadFile = File(..., description="Spectral dental image (JPEG, PNG, WebP, TIFF)"),
    image_type: str = Query("nir", description="Image type: nir, fluorescence, intraoral"),
    use_llm: bool = Query(True, description="Enable LLM analysis for detailed findings")
):
    """
    🔬 **Spectral Dental Analysis**
    
    Advanced spectral imaging analysis for early detection of:
    - Enamel demineralization
    - Early caries (white spot lesions)
    - Subsurface decay
    - Gingival inflammation
    - Dental calculus
    
    Supports NIR, fluorescence, and intraoral camera images.
    """
    if spectral_pipeline is None:
        raise HTTPException(status_code=503, detail="Spectral pipeline not loaded")
    
    # Validate image type
    valid_types = ['nir', 'fluorescence', 'intraoral']
    if image_type not in valid_types:
        raise HTTPException(status_code=400, detail=f"Invalid image_type. Use: {valid_types}")
    
    # Validate file type
    allowed_ext = ['.jpg', '.jpeg', '.png', '.webp', '.tiff', '.tif']
    ext = Path(file.filename).suffix.lower() if file.filename else ''
    if ext not in allowed_ext:
        raise HTTPException(status_code=400, detail=f"Invalid file type. Allowed: {allowed_ext}")
    
    contents = await file.read()
    if len(contents) > 25 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Max 25MB.")
    
    try:
        logger.info(f"Spectral analysis: {file.filename} ({image_type})")
        result = await spectral_pipeline.analyze(contents, image_type=image_type, use_llm=use_llm)
        result['success'] = True
        return result
    except Exception as e:
        logger.error(f"Spectral analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/info")
async def get_model_info():
    """Get information about loaded models."""
    if analyzer is None:
        raise HTTPException(status_code=503, detail="Analyzer not loaded")
    
    return {
        "efficientnet": {
            "architecture": "EfficientNetB0",
            "input_size": [224, 224, 3],
            "classes": analyzer.disease_predictor.num_classes,
            "class_labels": analyzer.disease_predictor.class_labels
        },
        "yolo": {
            "architecture": "YOLOv8n",
            "detection_classes": ["healthy_tooth", "cavity", "plaque", "crooked_tooth", "missing_tooth"],
            "fdi_mapping": "Enabled"
        },
        "features": {
            "disease_classification": True,
            "tooth_detection": True,
            "severity_scoring": True,
            "report_generation": True,
            "image_annotation": True
        }
    }

# ============================================
# ERROR HANDLERS
# ============================================
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            success=False,
            error=exc.detail,
            detail=f"HTTP {exc.status_code}",
            timestamp=datetime.now().isoformat()
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            success=False,
            error="Internal server error",
            detail=str(exc),
            timestamp=datetime.now().isoformat()
        ).dict()
    )

# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    uvicorn.run(
        "enhanced_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )