"""
Advanced Dental AI API
======================
FastAPI server for the advanced dental AI pipeline.

Features:
- CNN Ensemble (EfficientNet-B4, B3, ResNet50, MobileNetV3)
- LLM Integration (GPT-4o, LLaMA-3, Mistral)
- VLM Support (LLaVA, Qwen-VL)
- Comprehensive Report Generation

Usage:
    uvicorn advanced_api:app --host 0.0.0.0 --port 8001 --reload
"""

import os
import io
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field
import uvicorn

from advanced_dental_pipeline import (
    AdvancedDentalPipeline, 
    DentalReportFormatter,
    DISCLAIMER
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Pydantic Models
class PredictionItem(BaseModel):
    disease: str
    confidence: float

class CNNAnalysis(BaseModel):
    primary_disease: str
    confidence: float
    all_predictions: List[PredictionItem]

class ReportSummary(BaseModel):
    disease: str
    confidence: float
    severity: str
    recommendation: str

class AdvancedAnalysisResponse(BaseModel):
    success: bool = True
    analysis_id: str
    timestamp: str
    pipeline_version: str
    
    cnn_analysis: dict
    llm_analysis: Optional[dict] = None
    vlm_analysis: Optional[dict] = None
    
    report: dict
    summary: ReportSummary
    
    disclaimer: str = DISCLAIMER

class QuickAnalysisResponse(BaseModel):
    success: bool = True
    disease: str
    confidence: float
    severity: str
    recommendation: str
    disclaimer: str = DISCLAIMER


# FastAPI App
app = FastAPI(
    title="Advanced Dental AI API",
    description="""
    🦷 **Ultimate Dental AI Analysis Pipeline**
    
    ## Features
    - **CNN Ensemble**: EfficientNet-B4, B3, ResNet50, MobileNetV3
    - **LLM Analysis**: GPT-4o with vision for intelligent explanations
    - **VLM Support**: LLaVA, Qwen-VL for advanced image understanding
    - **Comprehensive Reports**: Text, HTML, JSON formats
    
    ## Pipeline Flow
    ```
    Image → CNN Ensemble → Disease Probability
                ↓
            LLM/VLM → Explanation + Report + Advice
                ↓
            Final Comprehensive Report
    ```
    
    ## Disclaimer
    ⚠️ This AI provides preliminary screening only.
    Always consult a qualified dental professional.
    """,
    version="2.0.0-advanced",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
pipeline: Optional[AdvancedDentalPipeline] = None


@app.on_event("startup")
async def startup():
    """Initialize pipeline on startup."""
    global pipeline
    try:
        logger.info("Initializing Advanced Dental Pipeline...")
        pipeline = AdvancedDentalPipeline()
        logger.info("Pipeline ready!")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        pipeline = None


@app.get("/")
async def root():
    """API information."""
    return {
        "name": "Advanced Dental AI API",
        "version": "2.0.0-advanced",
        "status": "ready" if pipeline else "initializing",
        "features": [
            "CNN Ensemble (EfficientNet-B4, B3, ResNet50, MobileNetV3)",
            "LLM Analysis (GPT-4o)",
            "VLM Support (LLaVA, Qwen-VL)",
            "Multi-format Reports"
        ],
        "endpoints": {
            "/analyze": "Full advanced analysis",
            "/analyze/quick": "Quick CNN-only analysis",
            "/analyze/report": "Get formatted report",
            "/health": "Health check"
        },
        "disclaimer": DISCLAIMER
    }


@app.get("/health")
async def health():
    """Health check."""
    return {
        "status": "healthy" if pipeline else "degraded",
        "pipeline_loaded": pipeline is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/analyze", response_model=AdvancedAnalysisResponse)
async def analyze_advanced(
    file: UploadFile = File(..., description="Dental image (JPEG, PNG, WebP)"),
    use_llm: bool = Query(True, description="Enable LLM analysis"),
    use_vlm: bool = Query(False, description="Enable VLM analysis (requires Ollama)")
):
    """
    🦷 **Full Advanced Dental Analysis**
    
    Runs the complete pipeline:
    1. CNN Ensemble prediction
    2. LLM analysis (GPT-4o with vision)
    3. VLM analysis (optional, requires local Ollama)
    4. Comprehensive report generation
    
    Returns detailed analysis with explanations and recommendations.
    """
    if not pipeline:
        raise HTTPException(503, "Pipeline not initialized")
    
    # Validate file
    allowed_ext = ['.jpg', '.jpeg', '.png', '.webp']
    ext = Path(file.filename).suffix.lower() if file.filename else ''
    if ext not in allowed_ext:
        raise HTTPException(400, f"Invalid file type. Allowed: {allowed_ext}")
    
    contents = await file.read()
    if len(contents) > 15 * 1024 * 1024:
        raise HTTPException(400, "File too large. Max 15MB.")
    
    try:
        logger.info(f"Analyzing: {file.filename} ({len(contents)} bytes)")
        result = await pipeline.analyze(contents, use_llm=use_llm, use_vlm=use_vlm)
        
        return AdvancedAnalysisResponse(
            success=True,
            analysis_id=result['analysis_id'],
            timestamp=result['timestamp'],
            pipeline_version=result['pipeline_version'],
            cnn_analysis=result['cnn_analysis'],
            llm_analysis=result['llm_analysis'],
            vlm_analysis=result['vlm_analysis'],
            report=result['report'],
            summary=ReportSummary(**result['summary']),
            disclaimer=DISCLAIMER
        )
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(500, f"Analysis failed: {str(e)}")


@app.post("/analyze/quick", response_model=QuickAnalysisResponse)
async def analyze_quick(
    file: UploadFile = File(..., description="Dental image")
):
    """
    ⚡ **Quick CNN-Only Analysis**
    
    Fast analysis using only the CNN ensemble.
    No LLM/VLM processing - returns in ~1 second.
    """
    if not pipeline:
        raise HTTPException(503, "Pipeline not initialized")
    
    contents = await file.read()
    if len(contents) > 15 * 1024 * 1024:
        raise HTTPException(400, "File too large")
    
    try:
        result = await pipeline.analyze(contents, use_llm=False, use_vlm=False)
        summary = result['summary']
        
        return QuickAnalysisResponse(
            success=True,
            disease=summary['disease'],
            confidence=summary['confidence'],
            severity=summary['severity'],
            recommendation=summary['recommendation'],
            disclaimer=DISCLAIMER
        )
    except Exception as e:
        logger.error(f"Quick analysis failed: {e}")
        raise HTTPException(500, str(e))


@app.post("/analyze/report")
async def get_report(
    file: UploadFile = File(...),
    format: str = Query("text", enum=["text", "html", "json"]),
    use_llm: bool = Query(True)
):
    """
    📄 **Get Formatted Report**
    
    Returns analysis report in specified format:
    - `text`: Human-readable text report
    - `html`: Styled HTML report
    - `json`: Raw JSON data
    """
    if not pipeline:
        raise HTTPException(503, "Pipeline not initialized")
    
    contents = await file.read()
    
    try:
        result = await pipeline.analyze(contents, use_llm=use_llm, use_vlm=False)
        formatter = DentalReportFormatter()
        
        if format == "html":
            html = formatter.to_html(result)
            return HTMLResponse(content=html)
        elif format == "json":
            return JSONResponse(content=result)
        else:
            text = formatter.to_text(result)
            return {"report": text, "format": "text"}
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        raise HTTPException(500, str(e))


@app.get("/models/info")
async def models_info():
    """Get information about loaded models."""
    return {
        "cnn_ensemble": {
            "models": ["EfficientNet-B4", "EfficientNet-B3", "ResNet50", "MobileNetV3"],
            "weights": {
                "efficientnet_b4": 0.4,
                "efficientnet_b3": 0.3,
                "resnet50": 0.2,
                "mobilenetv3": 0.1
            },
            "status": "loaded" if pipeline else "not loaded"
        },
        "llm": {
            "provider": "OpenAI GPT-4o",
            "vision_enabled": True,
            "status": "available" if os.environ.get('OPENAI_API_KEY') else "no API key"
        },
        "vlm": {
            "models": ["LLaVA-1.6", "Qwen-VL"],
            "backend": "Ollama",
            "status": "check /health for availability"
        },
        "classes": ["Calculus", "Caries", "Gingivitis", "Mouth_Ulcer"]
    }


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "error": exc.detail}
    )


if __name__ == "__main__":
    uvicorn.run("advanced_api:app", host="0.0.0.0", port=8001, reload=True)
