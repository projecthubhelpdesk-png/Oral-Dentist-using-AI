import { useEffect, useState } from 'react';
import { useParams, Link, useSearchParams } from 'react-router-dom';
import { useAuth } from '@/context/AuthContext';
import { Layout } from '@/components/layout/Layout';
import { Card, CardHeader, CardTitle } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import { HealthScoreGauge } from '@/components/analysis/HealthScoreGauge';
import { AnalysisReport } from '@/components/analysis/AnalysisReport';
import { DownloadReport } from '@/components/analysis/DownloadReport';
import { ScanChat } from '@/components/chat/ScanChat';
import { ScanImage } from '@/components/scans/ScanImage';
import { getScan, getAnalysis } from '@/services/scans';
import { getReviews } from '@/services/reviews';
import type { Scan, AnalysisResult, DentistReview } from '@/types';

// Generate a user-friendly report ID from the analysis ID
function generateReportId(analysisId: string, scanDate: string): string {
  const date = new Date(scanDate);
  const dateStr = date.toISOString().slice(0, 10).replace(/-/g, '');
  const shortId = analysisId.substring(0, 8).toUpperCase();
  return `RPT-${dateStr}-${shortId}`;
}

export function ScanDetailPage() {
  const { id } = useParams<{ id: string }>();
  const [searchParams] = useSearchParams();
  const openChat = searchParams.get('chat') === 'true';
  const { user, isLoading: authLoading } = useAuth();
  const [scan, setScan] = useState<Scan | null>(null);
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [reviews, setReviews] = useState<DentistReview[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    async function loadData() {
      if (!id || authLoading) return;
      
      try {
        setIsLoading(true);
        const scanData = await getScan(id);
        setScan(scanData);

        // Use analysis from scan if available, otherwise fetch separately
        if (scanData.analysis) {
          setAnalysis({
            id: scanData.analysis.id,
            scanId: id,
            modelType: scanData.scanType,
            modelVersion: scanData.analysis.modelVersion,
            overallScore: scanData.analysis.overallScore,
            confidenceScore: scanData.analysis.confidenceScore,
            findings: scanData.analysis.findings,
            riskAreas: scanData.analysis.riskAreas,
            recommendations: scanData.analysis.recommendations,
            createdAt: scanData.analysis.analysisDate,
            enhanced_features: scanData.analysis.enhanced_features,
            gpt4o_guidance: scanData.analysis.gpt4o_guidance,
          });
        } else if (scanData.status === 'analyzed') {
          const [analysisData, reviewsData] = await Promise.all([
            getAnalysis(id).catch(() => null),
            getReviews(id).catch(() => []),
          ]);
          setAnalysis(analysisData);
          setReviews(reviewsData);
        }
      } catch (err) {
        setError('Failed to load scan');
      } finally {
        setIsLoading(false);
      }
    }

    loadData();
  }, [id, authLoading]);

  if (isLoading) {
    return (
      <Layout>
        <div className="animate-pulse space-y-6">
          <div className="h-8 bg-gray-200 rounded w-1/3" />
          <div className="h-64 bg-gray-200 rounded" />
        </div>
      </Layout>
    );
  }

  const isDentist = user?.role === 'dentist';
  const backLink = isDentist ? '/dashboard' : '/scans';
  const backText = isDentist ? '← Back to Dashboard' : '← Back to Scans';

  if (error || !scan) {
    return (
      <Layout>
        <Card className="text-center py-12">
          <p className="text-red-600 mb-4">{error || 'Scan not found'}</p>
          <Link to={backLink}>
            <Button variant="secondary">{backText.replace('← ', '')}</Button>
          </Link>
        </Card>
      </Layout>
    );
  }


  const statusColors: Record<string, 'default' | 'success' | 'warning' | 'error'> = {
    uploaded: 'default',
    processing: 'warning',
    analyzed: 'success',
    failed: 'error',
    archived: 'default',
  };

  return (
    <Layout>
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <Link to={backLink} className="text-primary-600 hover:underline text-sm mb-2 inline-block">
              {backText}
            </Link>
            <h1 className="text-2xl font-bold text-gray-900">Scan Details</h1>
            <p className="text-gray-600">
              {new Date(scan.uploadedAt).toLocaleDateString('en-US', {
                year: 'numeric',
                month: 'long',
                day: 'numeric',
                hour: '2-digit',
                minute: '2-digit',
              })}
            </p>
          </div>
          <div className="flex items-center gap-3">
            {analysis && <DownloadReport scan={scan} analysis={analysis} />}
            <Badge variant={statusColors[scan.status]}>{scan.status}</Badge>
          </div>
        </div>

        {/* Report ID Banner */}
        {analysis && (
          <div className="bg-gradient-to-r from-primary-50 to-blue-50 border border-primary-200 rounded-lg p-4">
            <div className="flex items-center justify-between flex-wrap gap-4">
              <div className="flex items-center gap-3">
                <div className="bg-primary-100 rounded-full p-2">
                  <svg className="w-5 h-5 text-primary-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                </div>
                <div>
                  <p className="text-xs text-gray-500 uppercase tracking-wide">Report ID</p>
                  <p className="text-lg font-mono font-bold text-primary-700">
                    {generateReportId(analysis.id, scan.uploadedAt)}
                  </p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => {
                    navigator.clipboard.writeText(generateReportId(analysis.id, scan.uploadedAt));
                    setCopied(true);
                    setTimeout(() => setCopied(false), 2000);
                  }}
                  className="flex items-center gap-1 px-3 py-1.5 text-sm bg-white border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
                >
                  {copied ? (
                    <>
                      <svg className="w-4 h-4 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                      </svg>
                      Copied!
                    </>
                  ) : (
                    <>
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                      </svg>
                      Copy ID
                    </>
                  )}
                </button>
                <span className="text-xs text-gray-500">Share this ID with your dentist</span>
              </div>
            </div>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Scan Info & Image */}
          <div className="lg:col-span-1 space-y-6">
            {/* Scan Image */}
            <Card>
              <CardHeader>
                <CardTitle>Scanned Image</CardTitle>
              </CardHeader>
              <div className="p-2">
                {scan.imageUrl ? (
                  <ScanImage imageUrl={scan.imageUrl} alt="Dental Scan" className="w-full" />
                ) : (
                  <div className="flex flex-col items-center justify-center py-12 text-gray-400">
                    <svg className="w-16 h-16 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                    </svg>
                    <p className="text-sm">Image not available</p>
                  </div>
                )}
              </div>
            </Card>

            {/* Scan Info */}
            <Card>
              <CardHeader>
                <CardTitle>Scan Information</CardTitle>
              </CardHeader>
              <div className="space-y-3 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-500">Type</span>
                  <span className="font-medium">{scan.scanType === 'basic_rgb' ? 'Basic RGB' : 'Advanced Spectral'}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-500">Device</span>
                  <span className="font-medium">{scan.captureDevice || 'Unknown'}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-500">Uploaded</span>
                  <span className="font-medium">{new Date(scan.uploadedAt).toLocaleDateString()}</span>
                </div>
                {scan.processedAt && (
                  <div className="flex justify-between">
                    <span className="text-gray-500">Processed</span>
                    <span className="font-medium">{new Date(scan.processedAt).toLocaleDateString()}</span>
                  </div>
                )}
              </div>

              {/* Health Score */}
              {analysis && (
                <div className="mt-6 pt-6 border-t flex flex-col items-center">
                  <HealthScoreGauge score={analysis.overallScore} size="lg" />
                  <p className="text-sm text-gray-500 mt-2">
                    Confidence: {(analysis.confidenceScore * 100).toFixed(1)}%
                  </p>
                </div>
              )}
            </Card>
          </div>

          {/* Analysis Results */}
          <div className="lg:col-span-2 space-y-6">
            {analysis ? (
              <AnalysisReport analysis={analysis} />
            ) : scan.status === 'processing' ? (
              <Card className="text-center py-12">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600 mx-auto mb-4" />
                <p className="text-gray-600">Analysis in progress...</p>
              </Card>
            ) : scan.status === 'uploaded' ? (
              <Card className="text-center py-12">
                <p className="text-gray-600 mb-4">This scan hasn't been analyzed yet.</p>
                <Button>Start Analysis</Button>
              </Card>
            ) : null}

            {/* Dentist Reviews */}
            {reviews.length > 0 && (
              <Card>
                <CardHeader>
                  <CardTitle>Professional Reviews</CardTitle>
                </CardHeader>
                <div className="space-y-4">
                  {reviews.map((review) => (
                    <div key={review.id} className="border-b last:border-0 pb-4 last:pb-0">
                      <div className="flex items-center justify-between mb-2">
                        <span className="font-medium">Dr. Review</span>
                        <Badge variant={review.urgencyLevel === 'routine' ? 'success' : review.urgencyLevel === 'urgent' ? 'error' : 'warning'}>
                          {review.urgencyLevel}
                        </Badge>
                      </div>
                      {review.professionalAssessment && (
                        <p className="text-gray-700 text-sm mb-2">{review.professionalAssessment}</p>
                      )}
                      {review.treatmentRecommendations && (
                        <div className="bg-blue-50 p-3 rounded-lg text-sm">
                          <span className="font-medium text-blue-800">Recommendations: </span>
                          <span className="text-blue-700">{review.treatmentRecommendations}</span>
                        </div>
                      )}
                      {review.followUpDays && (
                        <p className="text-sm text-gray-500 mt-2">
                          Follow-up recommended in {review.followUpDays} days
                        </p>
                      )}
                    </div>
                  ))}
                </div>
              </Card>
            )}

          </div>
        </div>
      </div>

      {/* Floating Chat - Bottom Right Corner */}
      {scan.status === 'analyzed' && (
        <ScanChat scanId={scan.id} scanOwnerId={scan.userId} defaultOpen={openChat} />
      )}
    </Layout>
  );
}
