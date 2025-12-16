// User types
export interface User {
  id: string;
  email: string;
  firstName?: string;
  lastName?: string;
  role: 'user' | 'dentist' | 'admin';
  phoneLastFour?: string;
  profileImageUrl?: string;
  emailVerified: boolean;
  createdAt: string;
  lastLoginAt?: string;
}

export interface DentistProfile {
  id: string;
  userId: string;
  email?: string;
  name?: string;
  licenseState: string;
  licenseVerified: boolean;
  specialty: string;
  clinicName?: string;
  yearsExperience: number;
  bio?: string;
  acceptingPatients: boolean;
  consultationFeeCents: number;
  averageRating: number;
  totalReviews: number;
}

// Scan types
export type ScanType = 'basic_rgb' | 'advanced_spectral';
export type ScanStatus = 'uploaded' | 'processing' | 'analyzed' | 'failed' | 'archived';

// GPT-4o detailed guidance sections
export interface GPT4oGuidance {
  exact_complaint: string;
  detailed_findings: string;
  what_this_means: string;
  immediate_actions: string;
  treatment_options: string;
  home_care_routine: string;
  prevention_tips: string;
  llm_analysis: string;
}

export interface ScanAnalysis {
  id: string;
  overallScore: number;
  confidenceScore: number;
  findings: Finding[];
  riskAreas: RiskArea[];
  recommendations: string[];
  modelVersion: string;
  analysisDate: string;
  // Enhanced features
  enhanced_features?: EnhancedAnalysis;
  // GPT-4o detailed guidance
  gpt4o_guidance?: GPT4oGuidance;
}

export interface Scan {
  id: string;
  userId: string;
  scanType: ScanType;
  imageUrl?: string;
  thumbnailUrl?: string;
  status: ScanStatus;
  captureDevice?: string;
  uploadedAt: string;
  processedAt?: string;
  analysis?: ScanAnalysis;
  patientEmail?: string;
  patientName?: string;
}

// Analysis types
export type Severity = 'none' | 'minimal' | 'mild' | 'moderate' | 'severe';

export interface Finding {
  type: string;
  severity: Severity;
  location: string;
  confidence: number;
}

export interface RiskArea {
  x: number;
  y: number;
  width: number;
  height: number;
  label: string;
  tooth_id?: string;
}

// Enhanced analysis types (tooth-level detection)
export interface ToothDetection {
  tooth_id: string;
  issue: string;
  confidence: number;
  bbox: number[];
}

export interface EnhancedAnalysis {
  tooth_detections: ToothDetection[];
  affected_teeth: string[];
  home_care_tips: string[];
  analysis_id?: string;
  dental_report?: DentalReport;
}

export interface DentalReport {
  summary: string;
  severity: string;
  confidence_level: string;
  affected_teeth: string[];
  detected_issues: Record<string, number>;
  recommendation: string;
  home_care_tips: string[];
  professional_advice: string[];
  screening_date: string;
  ai_disclaimer: string;
  next_steps: string[];
}

export interface AnalysisResult {
  id: string;
  scanId: string;
  modelType: ScanType;
  modelVersion: string;
  overallScore: number;
  confidenceScore: number;
  findings: Finding[];
  riskAreas: RiskArea[];
  recommendations: string[];
  createdAt: string;
  // Enhanced features
  enhanced_features?: EnhancedAnalysis;
  // GPT-4o detailed guidance
  gpt4o_guidance?: GPT4oGuidance;
}

// Review types
export type UrgencyLevel = 'routine' | 'soon' | 'urgent' | 'emergency';

export interface DentistReview {
  id: string;
  scanId: string;
  dentistId: string;
  agreesWithAi?: boolean;
  professionalAssessment?: string;
  diagnosisCodes?: string[];
  treatmentRecommendations?: string;
  urgencyLevel: UrgencyLevel;
  followUpDays?: number;
  reviewedAt?: string;
}

// Connection types
export type ConnectionStatus = 'pending' | 'active' | 'declined' | 'terminated';

export interface PatientDentistConnection {
  id: string;
  patientId: string;
  dentistId: string;
  status: ConnectionStatus;
  initiatedBy: 'patient' | 'dentist';
  shareScanHistory: boolean;
  connectedAt?: string;
}

export interface Connection {
  id: string;
  patientId?: string;
  dentistId?: string;
  status: ConnectionStatus;
  initiatedBy: 'patient' | 'dentist';
  shareScanHistory: boolean;
  connectedAt?: string;
  createdAt: string;
  otherUser?: {
    id: string;
    email: string;
    name?: string;
    role: 'user' | 'dentist';
    specialty?: string;
    clinicName?: string;
  };
}

// Extended dentist profile with user info
export interface DentistWithUser extends DentistProfile {
  email: string;
  profileImageUrl?: string;
}

// Auth types
export interface AuthTokens {
  accessToken: string;
  refreshToken: string;
  expiresIn: number;
  tokenType: string;
}

export interface LoginResponse extends AuthTokens {
  user: User;
}

// API types
export interface PaginatedResponse<T> {
  data: T[];
  total: number;
  limit: number;
  offset: number;
}

export interface ApiError {
  error: string;
  message: string;
  details?: Record<string, string>;
}
