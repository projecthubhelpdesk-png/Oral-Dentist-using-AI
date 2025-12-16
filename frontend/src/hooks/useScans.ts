import { useState, useEffect, useCallback } from 'react';
import { scanService, ScanFilters } from '@/services/scans';
import type { Scan, AnalysisResult } from '@/types';

export function useScans(initialFilters: ScanFilters = {}) {
  const [scans, setScans] = useState<Scan[]>([]);
  const [total, setTotal] = useState(0);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filters, setFilters] = useState(initialFilters);

  const fetchScans = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await scanService.getScans(filters);
      setScans(response.data);
      setTotal(response.total);
    } catch (err) {
      setError('Failed to load scans');
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  }, [filters]);

  useEffect(() => {
    fetchScans();
  }, [fetchScans]);

  const uploadScan = async (
    file: File,
    scanType: 'basic_rgb' | 'advanced_spectral',
    captureDevice?: string
  ) => {
    const newScan = await scanService.uploadScan(file, scanType, captureDevice);
    setScans((prev) => [newScan, ...prev]);
    return newScan;
  };

  const archiveScan = async (id: string) => {
    await scanService.archiveScan(id);
    setScans((prev) => prev.filter((s) => s.id !== id));
  };

  const triggerAnalysis = async (id: string) => {
    // Set to processing first
    setScans((prev) =>
      prev.map((s) => (s.id === id ? { ...s, status: 'processing' as const } : s))
    );
    
    try {
      await scanService.triggerAnalysis(id);
      // Update to analyzed after completion
      setScans((prev) =>
        prev.map((s) => (s.id === id ? { ...s, status: 'analyzed' as const } : s))
      );
    } catch (err) {
      // Revert to uploaded on failure
      setScans((prev) =>
        prev.map((s) => (s.id === id ? { ...s, status: 'failed' as const } : s))
      );
      throw err;
    }
  };

  return {
    scans,
    total,
    isLoading,
    error,
    filters,
    setFilters,
    refresh: fetchScans,
    uploadScan,
    archiveScan,
    triggerAnalysis,
  };
}

export function useScanAnalysis(scanId: string | null) {
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!scanId) {
      setAnalysis(null);
      return;
    }

    const fetchAnalysis = async () => {
      setIsLoading(true);
      setError(null);
      try {
        const result = await scanService.getAnalysis(scanId);
        setAnalysis(result);
      } catch (err) {
        setError('Failed to load analysis');
        console.error(err);
      } finally {
        setIsLoading(false);
      }
    };

    fetchAnalysis();
  }, [scanId]);

  return { analysis, isLoading, error };
}
