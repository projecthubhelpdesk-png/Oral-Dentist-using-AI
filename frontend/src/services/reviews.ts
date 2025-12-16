import { api } from './api';
import type { DentistReview } from '@/types';

export async function getReviews(scanId: string): Promise<DentistReview[]> {
  const response = await api.get(`/scans/${scanId}/reviews`);
  return response.data;
}

export async function createReview(
  scanId: string,
  data: {
    agreesWithAi?: boolean;
    professionalAssessment?: string;
    diagnosisCodes?: string[];
    treatmentRecommendations?: string;
    urgencyLevel?: 'routine' | 'soon' | 'urgent' | 'emergency';
    followUpDays?: number;
  }
): Promise<{ id: string }> {
  const response = await api.post(`/scans/${scanId}/reviews`, data);
  return response.data;
}
