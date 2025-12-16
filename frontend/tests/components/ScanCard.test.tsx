import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { ScanCard } from '@/components/scans/ScanCard';
import type { Scan } from '@/types';

const mockScan: Scan = {
  id: 'scan-123',
  userId: 'user-456',
  scanType: 'basic_rgb',
  status: 'analyzed',
  captureDevice: 'iPhone 14 Pro',
  uploadedAt: '2024-12-01T10:30:00Z',
  processedAt: '2024-12-01T10:31:15Z',
};

describe('ScanCard', () => {
  it('renders scan information', () => {
    render(<ScanCard scan={mockScan} onView={vi.fn()} />);
    
    expect(screen.getByText('iPhone 14 Pro')).toBeInTheDocument();
    expect(screen.getByText('Basic RGB')).toBeInTheDocument();
    expect(screen.getByText('Analyzed')).toBeInTheDocument();
  });

  it('calls onView when View button is clicked', () => {
    const onView = vi.fn();
    render(<ScanCard scan={mockScan} onView={onView} />);
    
    fireEvent.click(screen.getByRole('button', { name: /view/i }));
    expect(onView).toHaveBeenCalledWith(mockScan);
  });

  it('shows Analyze button for uploaded scans', () => {
    const uploadedScan = { ...mockScan, status: 'uploaded' as const };
    const onAnalyze = vi.fn();
    
    render(<ScanCard scan={uploadedScan} onView={vi.fn()} onAnalyze={onAnalyze} />);
    
    expect(screen.getByRole('button', { name: /analyze/i })).toBeInTheDocument();
  });

  it('shows Results button for analyzed scans', () => {
    render(<ScanCard scan={mockScan} onView={vi.fn()} />);
    
    expect(screen.getByRole('button', { name: /results/i })).toBeInTheDocument();
  });

  it('displays correct badge for spectral scans', () => {
    const spectralScan = { ...mockScan, scanType: 'advanced_spectral' as const };
    render(<ScanCard scan={spectralScan} onView={vi.fn()} />);
    
    expect(screen.getByText('Spectral')).toBeInTheDocument();
  });

  it('shows processing status badge', () => {
    const processingScan = { ...mockScan, status: 'processing' as const };
    render(<ScanCard scan={processingScan} onView={vi.fn()} />);
    
    expect(screen.getByText('Processing')).toBeInTheDocument();
  });
});
