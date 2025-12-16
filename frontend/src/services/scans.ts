import { api } from './api';
import type { Scan, AnalysisResult, PaginatedResponse } from '@/types';

export interface ScanFilters {
  status?: string;
  scanType?: string;
  limit?: number;
  offset?: number;
}

export const scanService = {
  async getScans(filters: ScanFilters = {}): Promise<PaginatedResponse<Scan>> {
    const params = new URLSearchParams();
    if (filters.status) params.append('status', filters.status);
    if (filters.scanType) params.append('scanType', filters.scanType);
    if (filters.limit) params.append('limit', filters.limit.toString());
    if (filters.offset) params.append('offset', filters.offset.toString());
    
    const response = await api.get<PaginatedResponse<Scan>>(`/scans?${params}`);
    return response.data;
  },
  
  async getScan(id: string): Promise<Scan> {
    const response = await api.get<Scan>(`/scans/${id}`);
    return response.data;
  },
  
  async uploadScan(
    file: File,
    scanType: 'basic_rgb' | 'advanced_spectral',
    captureDevice?: string
  ): Promise<Scan> {
    const formData = new FormData();
    formData.append('image', file);
    formData.append('scanType', scanType);
    if (captureDevice) {
      formData.append('captureDevice', captureDevice);
    }
    
    const response = await api.post<Scan>('/scans', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },
  
  async archiveScan(id: string): Promise<void> {
    await api.delete(`/scans/${id}`);
  },
  
  async triggerAnalysis(id: string): Promise<void> {
    await api.post(`/scans/${id}/analyze`);
  },
  
  async getAnalysis(scanId: string): Promise<AnalysisResult> {
    const response = await api.get<AnalysisResult>(`/scans/${scanId}/analysis`);
    return response.data;
  },
};

// Export individual functions for convenience
export const getScans = scanService.getScans;
export const getScan = scanService.getScan;
export const uploadScan = scanService.uploadScan;
export const archiveScan = scanService.archiveScan;
export const triggerAnalysis = scanService.triggerAnalysis;
export const getAnalysis = scanService.getAnalysis;
