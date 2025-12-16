import { useState } from 'react';
import { Link } from 'react-router-dom';
import { useScans } from '@/hooks/useScans';
import { Layout } from '@/components/layout/Layout';
import { Card } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import { ScanCard } from '@/components/scans/ScanCard';
import { ScanUploader } from '@/components/scans/ScanUploader';
import type { Scan, ScanType, ScanStatus } from '@/types';

export function ScansPage() {
  const [statusFilter, setStatusFilter] = useState<ScanStatus | ''>('');
  const { scans, isLoading, uploadScan, triggerAnalysis } = useScans({ 
    limit: 50,
    status: statusFilter || undefined 
  });
  const [showUploader, setShowUploader] = useState(false);
  const [isUploading, setIsUploading] = useState(false);

  const handleUpload = async (file: File, scanType: ScanType, captureDevice?: string) => {
    setIsUploading(true);
    try {
      const scan = await uploadScan(file, scanType, captureDevice);
      await triggerAnalysis(scan.id);
      setShowUploader(false);
    } finally {
      setIsUploading(false);
    }
  };

  const handleViewScan = (scan: Scan) => {
    window.location.href = `/scans/${scan.id}`;
  };

  const statusOptions: { value: ScanStatus | ''; label: string }[] = [
    { value: '', label: 'All Scans' },
    { value: 'analyzed', label: 'Analyzed' },
    { value: 'processing', label: 'Processing' },
    { value: 'uploaded', label: 'Uploaded' },
    { value: 'failed', label: 'Failed' },
  ];

  return (
    <Layout>
      <div className="space-y-6">
        {/* Back to Dashboard */}
        <Link to="/dashboard" className="inline-flex items-center gap-1 text-primary-600 hover:text-primary-700 text-sm font-medium">
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
          </svg>
          Back to Dashboard
        </Link>

        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">My Scans</h1>
            <p className="text-gray-600">View and manage all your dental scans</p>
          </div>
          <Button onClick={() => setShowUploader(!showUploader)}>
            {showUploader ? 'Cancel' : 'Upload New Scan'}
          </Button>
        </div>

        {showUploader && (
          <Card className="p-6">
            <h2 className="text-lg font-semibold mb-4">Upload New Scan</h2>
            <ScanUploader onUpload={handleUpload} isUploading={isUploading} />
          </Card>
        )}

        {/* Filters */}
        <div className="flex gap-2 flex-wrap">
          {statusOptions.map((option) => (
            <button
              key={option.value}
              onClick={() => setStatusFilter(option.value)}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                statusFilter === option.value
                  ? 'bg-primary-600 text-white'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              {option.label}
            </button>
          ))}
        </div>

        {/* Scans Grid */}
        {isLoading ? (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
            {[...Array(8)].map((_, i) => (
              <div key={i} className="bg-gray-100 rounded-xl h-64 animate-pulse" />
            ))}
          </div>
        ) : scans.length > 0 ? (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
            {scans.map((scan) => (
              <ScanCard
                key={scan.id}
                scan={scan}
                onView={handleViewScan}
                onAnalyze={() => triggerAnalysis(scan.id)}
              />
            ))}
          </div>
        ) : (
          <Card className="text-center py-12">
            <p className="text-gray-500 mb-4">
              {statusFilter ? `No ${statusFilter} scans found.` : 'No scans yet. Upload your first dental scan!'}
            </p>
            <Button onClick={() => setShowUploader(true)}>Upload Scan</Button>
          </Card>
        )}
      </div>
    </Layout>
  );
}
