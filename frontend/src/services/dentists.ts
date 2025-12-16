import { api } from './api';
import type { DentistProfile } from '@/types';

interface DentistsResponse {
  data: DentistProfile[];
  total: number;
  limit: number;
  offset: number;
}

interface DentistFilters {
  specialty?: string;
  acceptingPatients?: boolean;
  limit?: number;
  offset?: number;
}

export async function getDentists(filters: DentistFilters = {}): Promise<DentistsResponse> {
  const params = new URLSearchParams();
  if (filters.specialty) params.append('specialty', filters.specialty);
  if (filters.acceptingPatients !== undefined) params.append('acceptingPatients', String(filters.acceptingPatients));
  if (filters.limit) params.append('limit', String(filters.limit));
  if (filters.offset) params.append('offset', String(filters.offset));

  const response = await api.get(`/dentists?${params.toString()}`);
  return response.data;
}

export async function getDentist(id: string): Promise<DentistProfile> {
  const response = await api.get(`/dentists/${id}`);
  return response.data;
}

export async function getMyDentistProfile(): Promise<DentistProfile> {
  const response = await api.get('/dentists/me');
  return response.data;
}

export async function updateMyDentistProfile(data: Partial<DentistProfile>): Promise<DentistProfile> {
  const response = await api.patch('/dentists/me', data);
  return response.data;
}
