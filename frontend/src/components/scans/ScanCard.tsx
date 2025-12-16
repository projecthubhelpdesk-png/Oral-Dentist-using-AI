import { useState, useEffect } from 'react';
import { format } from 'date-fns';
import { Card } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { getTokens } from '@/services/auth';
import type { Scan, ScanStatus } from '@/types';

interface ScanCardProps {
  scan: Scan;
  onView: (scan: Scan) => void;
  onAnalyze?: (scan: Scan) => void;
  onArchive?: (scan: Scan) => void;
}

const statusConfig: Record<ScanStatus, { label: string; variant: 'default' | 'success' | 'warning' | 'danger' | 'info' }> = {
  uploaded: { label: 'Uploaded', variant: 'default' },
  processing: { label: 'Processing', variant: 'info' },
  analyzed: { label: 'Analyzed', variant: 'success' },
  failed: { label: 'Failed', variant: 'danger' },
  archived: { label: 'Archived', variant: 'default' },
};

export function ScanCard({ scan, onView, onAnalyze, onArchive }: ScanCardProps) {
  const status = statusConfig[scan.status];
  const isBasicRgb = scan.scanType === 'basic_rgb';
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [imageLoaded, setImageLoaded] = useState(false);
  const [imageError, setImageError] = useState(false);

  useEffect(() => {
    // Build image URL with auth token for authenticated image requests
    // Always get fresh token from localStorage
    const accessToken = localStorage.getItem('accessToken');
    const baseUrl = import.meta.env.VITE_API_URL || 'http://localhost/oral-care-ai/backend-php/api';
    
    if (scan.id && accessToken) {
      setImageUrl(`${baseUrl}/scans/${scan.id}/image?token=${encodeURIComponent(accessToken)}`);
      setImageLoaded(false);
      setImageError(false);
    } else {
      // No token available - image will show placeholder
      setImageUrl(null);
      setImageError(true);
    }
  }, [scan.id]);

  return (
    <Card padding="none" className="overflow-hidden hover:shadow-md transition-shadow">
      {/* Thumbnail - Blurred for privacy */}
      <div className="aspect-video bg-gray-100 relative overflow-hidden group cursor-pointer" onClick={() => onView(scan)}>
        {imageUrl && (
          <img
            src={imageUrl}
            alt="Scan thumbnail"
            className={`w-full h-full object-cover absolute inset-0 transition-all duration-300 blur-lg ${imageLoaded ? 'opacity-100' : 'opacity-0'}`}
            onLoad={() => setImageLoaded(true)}
            onError={() => setImageError(true)}
          />
        )}
        {/* Placeholder shown when image not loaded */}
        {(!imageLoaded || imageError) && (
          <div className="w-full h-full flex items-center justify-center absolute inset-0">
            <svg className="w-12 h-12 text-gray-300" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1.5}
                d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
              />
            </svg>
          </div>
        )}
        
        {/* Overlay with "Click to view" message */}
        {imageLoaded && !imageError && (
          <div className="absolute inset-0 bg-black/30 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
            <div className="text-white text-center">
              <svg className="w-8 h-8 mx-auto mb-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
              </svg>
              <span className="text-xs font-medium">Click to view</span>
            </div>
          </div>
        )}
        
        {/* Privacy indicator */}
        {imageLoaded && !imageError && (
          <div className="absolute bottom-2 left-2 bg-black/50 text-white text-xs px-2 py-1 rounded-full flex items-center gap-1">
            <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
            </svg>
            Protected
          </div>
        )}
        
        {/* Status badge overlay */}
        <div className="absolute top-2 right-2 flex gap-1 z-10">
          {/* Warning icon for low health scores */}
          {scan.analysis && scan.analysis.overallScore < 50 && (
            <div className="bg-red-500 text-white rounded-full p-1.5 shadow-lg animate-pulse" title="Urgent: Low health score - Dentist notified">
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
              </svg>
            </div>
          )}
          <Badge variant={status.variant}>{status.label}</Badge>
        </div>
      </div>

      {/* Content */}
      <div className="p-4">
        <div className="flex items-center justify-between mb-2">
          <Badge variant={isBasicRgb ? 'info' : 'warning'} size="sm">
            {isBasicRgb ? 'Basic RGB' : 'Spectral'}
          </Badge>
          <span className="text-xs text-gray-500">
            {scan.uploadedAt ? format(new Date(scan.uploadedAt), 'MMM d, yyyy') : 'Unknown date'}
          </span>
        </div>

        {/* Analysis Score */}
        {scan.analysis && (
          <div className="mb-3">
            <div className="flex items-center justify-between mb-1">
              <span className="text-xs text-gray-500">Health Score</span>
              <span className={`text-sm font-semibold ${
                scan.analysis.overallScore >= 70 ? 'text-green-600' :
                scan.analysis.overallScore >= 50 ? 'text-yellow-600' :
                scan.analysis.overallScore >= 30 ? 'text-orange-600' : 'text-red-600'
              }`}>
                {scan.analysis.overallScore.toFixed(0)}%
              </span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-1.5">
              <div 
                className={`h-1.5 rounded-full ${
                  scan.analysis.overallScore >= 70 ? 'bg-green-500' :
                  scan.analysis.overallScore >= 50 ? 'bg-yellow-500' :
                  scan.analysis.overallScore >= 30 ? 'bg-orange-500' : 'bg-red-500'
                }`}
                style={{ width: `${scan.analysis.overallScore}%` }}
              />
            </div>
            {/* Top finding with severity indicator */}
            {scan.analysis.findings?.[0] && (
              <p className={`text-xs mt-1 truncate capitalize ${
                scan.analysis.findings[0].severity === 'severe' ? 'text-red-600 font-medium' :
                scan.analysis.findings[0].severity === 'moderate' ? 'text-orange-600' :
                'text-gray-600'
              }`}>
                {scan.analysis.findings[0].type.replace(/_/g, ' ')} ({scan.analysis.findings[0].severity})
              </p>
            )}
            {/* Show additional findings if multiple conditions detected */}
            {scan.analysis.findings && scan.analysis.findings.length > 1 && (
              <p className="text-xs text-orange-600 mt-0.5">
                +{scan.analysis.findings.length - 1} more condition{scan.analysis.findings.length > 2 ? 's' : ''} detected
              </p>
            )}
            {/* Enhanced: Affected teeth count */}
            {scan.analysis.riskAreas && scan.analysis.riskAreas.length > 0 && (
              <p className="text-xs text-blue-600 mt-1">
                🦷 {scan.analysis.riskAreas.length} area{scan.analysis.riskAreas.length > 1 ? 's' : ''} detected
              </p>
            )}
          </div>
        )}

        {scan.captureDevice && !scan.analysis && (
          <p className="text-sm text-gray-600 mb-3 truncate">{scan.captureDevice}</p>
        )}

        {/* Actions */}
        <div className="flex gap-2">
          {/* Chat with Doctor button */}
          <Button 
            size="sm" 
            variant="secondary" 
            onClick={() => window.location.href = `/scans/${scan.id}?chat=true`} 
            className="flex-1 flex items-center justify-center gap-1"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
            </svg>
            Chat
          </Button>
          
          {scan.status === 'uploaded' && onAnalyze && (
            <Button size="sm" onClick={() => onAnalyze(scan)} className="flex-1">
              Analyze
            </Button>
          )}
          
          {scan.status === 'analyzed' && (
            <Button size="sm" onClick={() => onView(scan)} className="flex-1">
              Results
            </Button>
          )}
        </div>
      </div>
    </Card>
  );
}
