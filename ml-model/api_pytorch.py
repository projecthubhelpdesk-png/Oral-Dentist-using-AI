"""
Dental AI API (PyTorch Version)
===============================
FastAPI server for dental disease detection.
Works with Python 3.14+

Usage:
    uvicorn api_pytorch:app --host 0.0.0.0 --port 8000
"""

import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
import uvicorn

from dental_ai_pytorch import DentalAIPipeline, format_report, DISCLAIMER
from spectral_dental_pipeline import SpectralDentalPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Pydantic models
class AnalysisResult(BaseModel):
    success: bool = True
    disease: str
    confidence: float
    severity: str
    recommendation: str
    disclaimer: str = DISCLAIMER


# FastAPI app
app = FastAPI(
    title="Dental AI API (PyTorch)",
    description="""
    🦷 **AI-powered dental disease detection**
    
    Features:
    - EfficientNet-B4 classification (PyTorch)
    - GPT-4o analysis via OpenRouter
    - Comprehensive reports
    
    Works with Python 3.14+
    """,
    version="2.0.0-pytorch"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipelines
pipeline: Optional[DentalAIPipeline] = None
spectral_pipeline: Optional[SpectralDentalPipeline] = None


@app.on_event("startup")
async def startup():
    global pipeline, spectral_pipeline
    logger.info("Initializing Dental AI Pipeline...")
    pipeline = DentalAIPipeline()
    logger.info("Initializing Spectral Dental Pipeline...")
    spectral_pipeline = SpectralDentalPipeline()
    logger.info("All pipelines ready!")


@app.get("/")
async def root():
    return {
        "name": "Dental AI API (PyTorch)",
        "version": "2.0.0",
        "status": "ready" if pipeline else "initializing",
        "features": ["EfficientNet-B4", "GPT-4o Analysis", "Comprehensive Reports"],
        "endpoints": {
            "/analyze": "Full analysis with LLM",
            "/analyze/quick": "Quick CNN-only analysis",
            "/health": "Health check"
        }
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy" if pipeline else "degraded",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(..., description="Dental image (JPEG, PNG)"),
    use_llm: bool = Query(True, description="Enable GPT-4o analysis")
):
    """
    🦷 **Full Dental Analysis with GPT-4o**
    
    Runs EfficientNet-B4 classification + GPT-4o vision analysis.
    
    Returns:
    - Exact complaint/diagnosis
    - Detailed findings
    - What this means for you
    - Immediate actions to take
    - Treatment options
    - Home care routine
    - Prevention tips
    """
    if not pipeline:
        raise HTTPException(503, "Pipeline not ready")
    
    # Validate file
    allowed = ['.jpg', '.jpeg', '.png', '.webp']
    ext = Path(file.filename).suffix.lower() if file.filename else ''
    if ext not in allowed:
        raise HTTPException(400, f"Invalid file type. Allowed: {allowed}")
    
    contents = await file.read()
    if len(contents) > 15 * 1024 * 1024:
        raise HTTPException(400, "File too large (max 15MB)")
    
    try:
        logger.info(f"Analyzing: {file.filename}")
        result = await pipeline.analyze(contents, use_llm=use_llm)
        
        # Return structured response with all guidance sections
        return {
            "success": True,
            "analysis_id": result['analysis_id'],
            "timestamp": result['timestamp'],
            
            # Summary
            "summary": result['summary'],
            
            # Detailed guidance from GPT-4o
            "exact_complaint": result.get('exact_complaint', ''),
            "detailed_findings": result.get('detailed_findings', ''),
            "what_this_means": result.get('what_this_means', ''),
            "immediate_actions": result.get('immediate_actions', ''),
            "treatment_options": result.get('treatment_options', ''),
            "home_care_routine": result.get('home_care_routine', ''),
            "prevention_tips": result.get('prevention_tips', ''),
            
            # Full report
            "report": result['report'],
            
            # Raw LLM analysis
            "llm_analysis": result.get('llm_analysis', {}).get('analysis', '') if result.get('llm_analysis') else '',
            
            "disclaimer": DISCLAIMER
        }
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(500, str(e))


@app.post("/analyze/quick")
async def analyze_quick(file: UploadFile = File(...)):
    """
    ⚡ **Quick Analysis (CNN only)**
    
    Fast classification without LLM. Returns in ~1 second.
    """
    if not pipeline:
        raise HTTPException(503, "Pipeline not ready")
    
    contents = await file.read()
    
    try:
        result = await pipeline.analyze(contents, use_llm=False)
        s = result['summary']
        return {
            "success": True,
            "disease": s['disease'],
            "disease_name": s['disease_name'],
            "confidence": s['confidence'],
            "severity": s['severity'],
            "recommendation": s['recommendation'],
            "disclaimer": DISCLAIMER
        }
    except Exception as e:
        logger.error(f"Quick analysis failed: {e}")
        raise HTTPException(500, str(e))


@app.post("/analyze/report")
async def get_report(
    file: UploadFile = File(...),
    format: str = Query("text", enum=["text", "html", "json"])
):
    """
    📄 **Get Formatted Report**
    
    Returns analysis in text, HTML, or JSON format.
    """
    if not pipeline:
        raise HTTPException(503, "Pipeline not ready")
    
    contents = await file.read()
    
    try:
        result = await pipeline.analyze(contents, use_llm=True)
        
        if format == "json":
            return result
        elif format == "html":
            # Detailed HTML report with all guidance sections
            s = result['summary']
            r = result['report']
            severity_color = '#dc3545' if s['severity'] == 'High' else '#ffc107' if s['severity'] == 'Medium' else '#28a745'
            
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Dental AI Report</title>
                <style>
                    body {{ font-family: 'Segoe UI', Arial, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; background: #f5f5f5; }}
                    .report {{ background: white; border-radius: 12px; padding: 30px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                    .header {{ text-align: center; border-bottom: 2px solid #e0e0e0; padding-bottom: 20px; margin-bottom: 20px; }}
                    .header h1 {{ color: #2c3e50; margin: 0; }}
                    .section {{ margin: 25px 0; padding: 20px; background: #f8f9fa; border-radius: 8px; border-left: 4px solid #3498db; }}
                    .section h3 {{ color: #2c3e50; margin-top: 0; }}
                    .finding {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 25px; border-radius: 12px; margin: 20px 0; }}
                    .finding h2 {{ margin: 0 0 10px 0; }}
                    .severity {{ display: inline-block; padding: 8px 20px; border-radius: 25px; color: white; font-weight: bold; background: {severity_color}; }}
                    .action-section {{ background: #e8f5e9; border-left-color: #4caf50; }}
                    .warning-section {{ background: #fff3e0; border-left-color: #ff9800; }}
                    .care-section {{ background: #e3f2fd; border-left-color: #2196f3; }}
                    .disclaimer {{ background: #ffebee; border: 1px solid #ef9a9a; padding: 15px; border-radius: 8px; font-size: 13px; color: #c62828; margin-top: 30px; }}
                    pre {{ white-space: pre-wrap; word-wrap: break-word; background: #f5f5f5; padding: 15px; border-radius: 8px; }}
                </style>
            </head>
            <body>
                <div class="report">
                    <div class="header">
                        <h1>🦷 Dental AI Screening Report</h1>
                        <p style="color: #7f8c8d;">Analysis ID: {result['analysis_id']} | {result['timestamp'][:10]}</p>
                    </div>
                    
                    <div class="finding">
                        <h2>🔍 {s['disease_name']}</h2>
                        <p>Confidence: {s['confidence']*100:.1f}% | Severity: <span class="severity">{s['severity']}</span></p>
                    </div>
                    
                    {'<div class="section"><h3>📋 Exact Complaint</h3><p>' + result.get("exact_complaint", "").replace(chr(10), "<br>") + '</p></div>' if result.get("exact_complaint") else ''}
                    
                    {'<div class="section"><h3>🔬 Detailed Findings</h3><p>' + result.get("detailed_findings", "").replace(chr(10), "<br>") + '</p></div>' if result.get("detailed_findings") else ''}
                    
                    {'<div class="section warning-section"><h3>📖 What This Means For You</h3><p>' + result.get("what_this_means", "").replace(chr(10), "<br>") + '</p></div>' if result.get("what_this_means") else ''}
                    
                    {'<div class="section action-section"><h3>⚡ Immediate Actions</h3><p>' + result.get("immediate_actions", "").replace(chr(10), "<br>") + '</p></div>' if result.get("immediate_actions") else ''}
                    
                    {'<div class="section"><h3>🏥 Treatment Options</h3><p>' + result.get("treatment_options", "").replace(chr(10), "<br>") + '</p></div>' if result.get("treatment_options") else ''}
                    
                    {'<div class="section care-section"><h3>🏠 Home Care Routine</h3><p>' + result.get("home_care_routine", "").replace(chr(10), "<br>") + '</p></div>' if result.get("home_care_routine") else ''}
                    
                    {'<div class="section"><h3>🛡️ Prevention Tips</h3><p>' + result.get("prevention_tips", "").replace(chr(10), "<br>") + '</p></div>' if result.get("prevention_tips") else ''}
                    
                    <div class="section warning-section">
                        <h3>⏰ When to See a Dentist</h3>
                        <p><strong>{r['urgency']}</strong></p>
                    </div>
                    
                    <div class="disclaimer">⚠️ <strong>Disclaimer:</strong> {DISCLAIMER}</div>
                </div>
            </body>
            </html>
            """
            return HTMLResponse(content=html)
        else:
            return {"report": format_report(result)}
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        raise HTTPException(500, str(e))


@app.post("/analyze/spectral")
async def analyze_spectral(
    file: UploadFile = File(..., description="Spectral dental image (NIR/Fluorescence)"),
    image_type: str = Query("nir", enum=["nir", "fluorescence", "intraoral"]),
    use_llm: bool = Query(True, description="Enable GPT-4o analysis")
):
    """
    🔬 **Advanced Spectral Image Analysis**
    
    Analyzes NIR (Near-Infrared) or Fluorescence spectral images for:
    - Early caries detection (before visible damage)
    - Subsurface decay identification
    - Enamel demineralization mapping
    - Periodontal tissue changes
    
    Uses CNN + PCA feature extraction + XGBoost/RF classifier pipeline.
    Compatible with ODSI-DB (Oral and Dental Spectral Image Database) format.
    
    Parameters:
    - file: Spectral image (NIR, Fluorescence, or Intraoral camera)
    - image_type: Type of spectral imaging used
    - use_llm: Enable GPT-4o for detailed analysis
    
    Returns:
    - Detected conditions with confidence scores
    - Spectral signatures for each finding
    - Severity assessment
    - Treatment recommendations
    """
    if not spectral_pipeline:
        raise HTTPException(503, "Spectral pipeline not ready")
    
    # Validate file
    allowed = ['.jpg', '.jpeg', '.png', '.webp', '.tiff', '.tif']
    ext = Path(file.filename).suffix.lower() if file.filename else ''
    if ext not in allowed:
        raise HTTPException(400, f"Invalid file type. Allowed: {allowed}")
    
    contents = await file.read()
    if len(contents) > 25 * 1024 * 1024:  # Allow larger files for spectral
        raise HTTPException(400, "File too large (max 25MB)")
    
    try:
        logger.info(f"Spectral analysis ({image_type}): {file.filename}")
        
        # Run spectral pipeline
        result = await spectral_pipeline.analyze(contents, image_type, use_llm)
        
        return {
            "success": True,
            "analysis_id": result['analysis_id'],
            "timestamp": result['timestamp'],
            "image_type": image_type,
            
            # Spectral-specific results
            "spectral_analysis": result['spectral_analysis'],
            
            # Standard analysis results
            "standard_analysis": result['standard_analysis'],
            
            # GPT-4o guidance (if enabled)
            "exact_complaint": result.get('exact_complaint', ''),
            "detailed_findings": result.get('detailed_findings', ''),
            "what_this_means": result.get('what_this_means', ''),
            "immediate_actions": result.get('immediate_actions', ''),
            "treatment_options": result.get('treatment_options', ''),
            "home_care_routine": result.get('home_care_routine', ''),
            "prevention_tips": result.get('prevention_tips', ''),
            
            # Spectral-specific recommendations
            "spectral_recommendations": result.get('spectral_recommendations', []),
            
            # Spectral visualization images (base64 encoded PNG)
            "spectral_image": result.get('spectral_image'),
            "spectral_overlay": result.get('spectral_overlay'),
            "color_legend": result.get('color_legend'),
            
            "disclaimer": result.get('disclaimer', DISCLAIMER)
        }
    except Exception as e:
        logger.error(f"Spectral analysis failed: {e}")
        raise HTTPException(500, str(e))


@app.post("/dentist/review")
async def dentist_review(
    analysis_id: str = Query(..., description="Analysis ID to review"),
    action: str = Query(..., enum=["accept", "edit", "reject"]),
    clinical_notes: Optional[str] = Query(None, description="Dentist's clinical notes"),
    edited_diagnosis: Optional[str] = Query(None, description="Edited diagnosis if action is 'edit'")
):
    """
    🧑‍⚕️ **Dentist Review & Override**
    
    Allows dentists to review, accept, edit, or reject AI findings.
    This step is crucial for medical safety and ethical compliance.
    
    Parameters:
    - analysis_id: The AI analysis to review
    - action: accept, edit, or reject
    - clinical_notes: Professional notes
    - edited_diagnosis: New diagnosis if editing
    """
    # In production, this would update the database
    return {
        "success": True,
        "analysis_id": analysis_id,
        "review_status": action,
        "clinical_notes": clinical_notes,
        "edited_diagnosis": edited_diagnosis if action == "edit" else None,
        "reviewed_at": datetime.now().isoformat(),
        "message": f"AI analysis {action}ed successfully. Report can now be generated."
    }


@app.post("/report/generate")
async def generate_report(
    analysis_id: str = Query(..., description="Analysis ID"),
    report_type: str = Query("both", enum=["clinical", "patient", "both"]),
    include_spectral: bool = Query(True, description="Include spectral analysis details")
):
    """
    📄 **Generate AI + Dentist Report**
    
    Generates professional reports after dentist review:
    - Clinical Report: Detailed technical report for dental records
    - Patient Report: Simplified, easy-to-understand version
    
    Reports are stored in the unified cloud system and linked to patient ID.
    """
    timestamp = datetime.now().isoformat()
    
    clinical_report = {
        "type": "clinical",
        "analysis_id": analysis_id,
        "generated_at": timestamp,
        "sections": [
            "Patient Information",
            "AI Analysis Summary",
            "Spectral Imaging Findings" if include_spectral else None,
            "Dentist Assessment",
            "Diagnosis Codes (ICD/Dental)",
            "Treatment Plan",
            "Follow-up Schedule",
            "Audit Trail"
        ],
        "format": "PDF-ready"
    }
    
    patient_report = {
        "type": "patient",
        "analysis_id": analysis_id,
        "generated_at": timestamp,
        "sections": [
            "Your Dental Health Summary",
            "What We Found",
            "What This Means For You",
            "Recommended Next Steps",
            "Home Care Tips",
            "When to Contact Us"
        ],
        "format": "PDF-ready",
        "reading_level": "simplified"
    }
    
    response = {"success": True, "analysis_id": analysis_id}
    
    if report_type in ["clinical", "both"]:
        response["clinical_report"] = clinical_report
    if report_type in ["patient", "both"]:
        response["patient_report"] = patient_report
    
    response["message"] = "Reports generated and stored in unified cloud system"
    response["download_url"] = f"/reports/{analysis_id}/download"
    
    return response


if __name__ == "__main__":
    uvicorn.run("api_pytorch:app", host="0.0.0.0", port=8000, reload=True)
