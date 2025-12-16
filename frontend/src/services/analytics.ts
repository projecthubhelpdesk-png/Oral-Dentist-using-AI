import { api } from './api';

export interface DentistAnalytics {
  totalPatients: number;
  totalScans: number;
  analyzedScans: number;
  pendingScans: number;
  criticalCases: number;
  attentionCases: number;
  normalCases: number;
  pendingRequests: number;
  consultations: number;
  reviewsCount: number;
  newPatientsMonth: number;
  scansMonth: number;
  avgHealthScore: number;
  procedures: number;
  earnings: number;
  successRate: number;
}

export interface UserAnalytics {
  totalScans: number;
  analyzedScans: number;
  avgHealthScore: number;
  latestHealthScore: number;
  connectedDentists: number;
}

export async function getDentistAnalytics(): Promise<DentistAnalytics> {
  const response = await api.get<{ success: boolean; analytics: DentistAnalytics }>('/analytics/dentist');
  return response.data.analytics;
}

export async function getUserAnalytics(): Promise<UserAnalytics> {
  const response = await api.get<{ success: boolean; analytics: UserAnalytics }>('/analytics/user');
  return response.data.analytics;
}
