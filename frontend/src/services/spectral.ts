import { api } from './api';

export interface SpectralDetection {
  condition: string;
  confidence: number;
  location: string;
  severity: 'mild' | 'moderate' | 'severe';
  spectral_signature?: string;
}

export interface SpectralAnalysisResult {
  success: boolean;
  analysisId: string;
  imageType: 'nir' | 'fluorescence' | 'intraoral';
  spectralAnalysis: {
    detections: SpectralDetection[];
    overall_health_score: number;
    imaging_quality: string;
    analysis_method: string;
  };
  standardAnalysis: {
    disease: string;
    disease_name: string;
    confidence: number;
    severity: string;
    recommendation: string;
  };
  guidance: {
    exactComplaint: string;
    detailedFindings: string;
    whatThisMeans: string;
    immediateActions: string;
    treatmentOptions: string;
    homeCareRoutine: string;
    preventionTips: string;
  };
  spectralRecommendations: string[];
  disclaimer: string;
}

export interface SpectralReviewRequest {
  action: 'accept' | 'edit' | 'reject';
  clinicalNotes?: string;
  editedDiagnosis?: string;
}

export interface SpectralReportRequest {
  reportType: 'clinical' | 'patient' | 'both';
}

export interface SpectralHistoryItem {
  id: string;
  patientId: string | null;
  patientName: string | null;
  imageType: 'nir' | 'fluorescence' | 'intraoral';
  healthScore: number;
  status: 'pending_review' | 'approved' | 'edited' | 'rejected';
  reviewAction: string | null;
  reportId: string | null;
  createdAt: string;
  reviewedAt: string | null;
}

/**
 * Run spectral AI analysis on an image
 */
export async function analyzeSpectralImage(
  file: File,
  imageType: 'nir' | 'fluorescence' | 'intraoral' = 'nir',
  patientId?: string
): Promise<SpectralAnalysisResult> {
  const formData = new FormData();
  formData.append('image', file);
  formData.append('imageType', imageType);
  if (patientId) {
    formData.append('patientId', patientId);
  }

  const response = await api.post('/spectral/analyze', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    timeout: 120000, // 2 minutes for spectral analysis
  });

  return response.data;
}

/**
 * Submit dentist review for spectral analysis
 */
export async function reviewSpectralAnalysis(
  analysisId: string,
  review: SpectralReviewRequest
): Promise<{ success: boolean; status: string; message: string }> {
  const response = await api.post(`/spectral/${analysisId}/review`, review);
  return response.data;
}

/**
 * Generate report after dentist review
 */
export async function generateSpectralReport(
  analysisId: string,
  reportType: 'clinical' | 'patient' | 'both' = 'both'
): Promise<any> {
  const response = await api.post(`/spectral/${analysisId}/report`, { reportType });
  return response.data;
}

/**
 * Get spectral analysis history for dentist
 */
export async function getSpectralHistory(
  options: { limit?: number; offset?: number; status?: string } = {}
): Promise<{ data: SpectralHistoryItem[] }> {
  const params = new URLSearchParams();
  if (options.limit) params.append('limit', options.limit.toString());
  if (options.offset) params.append('offset', options.offset.toString());
  if (options.status) params.append('status', options.status);

  const response = await api.get(`/spectral/history?${params.toString()}`);
  return response.data;
}
