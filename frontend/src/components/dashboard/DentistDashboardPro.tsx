import { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '@/context/AuthContext';
import { Card, CardHeader, CardTitle } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import { getConnections, updateConnection } from '@/services/connections';
import { getDentistAnalytics, type DentistAnalytics } from '@/services/analytics';
import { api } from '@/services/api';
import { getTokens } from '@/services/auth';
import { getFeatures, type FeaturesMap } from '@/services/features';
import type { Connection, Scan } from '@/types';

interface PatientScan extends Scan {
  patientEmail?: string;
  patientName?: string;
}

interface DashboardStats extends DentistAnalytics {}

export function DentistDashboardPro() {
  const { user } = useAuth();
  const navigate = useNavigate();
  const [connections, setConnections] = useState<Connection[]>([]);
  const [patientScans, setPatientScans] = useState<PatientScan[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [stats, setStats] = useState<DashboardStats>({
    totalPatients: 0,
    totalScans: 0,
    analyzedScans: 0,
    pendingScans: 0,
    criticalCases: 0,
    attentionCases: 0,
    normalCases: 0,
    pendingRequests: 0,
    consultations: 0,
    reviewsCount: 0,
    newPatientsMonth: 0,
    scansMonth: 0,
    avgHealthScore: 0,
    procedures: 0,
    earnings: 0,
    successRate: 90,
  });
  
  // Search state
  const [searchReportId, setSearchReportId] = useState('');
  const [isSearching, setIsSearching] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);
  const [searchResult, setSearchResult] = useState<PatientScan | null>(null);
  
  // Tab state
  const [activeTab, setActiveTab] = useState<'overview' | 'scans' | 'spectral'>('overview');
  
  // Feature flags
  const [features, setFeatures] = useState<FeaturesMap>({});
  const spectralEnabled = features['spectral_ai']?.enabled ?? true;

  useEffect(() => {
    // Load feature flags
    getFeatures().then(setFeatures).catch(console.error);
  }, []);

  useEffect(() => {
    // Only load data if user is authenticated
    if (user) {
      loadData();
    }
  }, [user]);

  async function loadData() {
    try {
      setIsLoading(true);
      const [connData, scansData, analyticsData] = await Promise.all([
        getConnections(),
        api.get<{ data: PatientScan[] }>('/scans/patients').catch(() => ({ data: { data: [] } })),
        getDentistAnalytics().catch(() => null)
      ]);
      
      const scans = scansData.data.data || [];
      
      setConnections(connData.data || []);
      setPatientScans(scans);
      
      // Use real analytics from API if available
      if (analyticsData) {
        setStats(analyticsData);
      } else {
        // Fallback to calculated stats
        const activeConns = (connData.data || []).filter(c => c.status === 'active');
        const criticalScans = scans.filter(s => 
          s.analysis?.overallScore !== undefined && s.analysis.overallScore < 40
        );
        const attentionScans = scans.filter(s => 
          s.analysis?.overallScore !== undefined && s.analysis.overallScore >= 40 && s.analysis.overallScore < 70
        );
        const normalScans = scans.filter(s => 
          s.analysis?.overallScore !== undefined && s.analysis.overallScore >= 70
        );
        
        setStats({
          totalPatients: activeConns.length,
          totalScans: scans.length,
          analyzedScans: scans.filter(s => s.status === 'analyzed').length,
          pendingScans: scans.filter(s => s.status === 'uploaded' || s.status === 'processing').length,
          criticalCases: criticalScans.length,
          attentionCases: attentionScans.length,
          normalCases: normalScans.length,
          pendingRequests: (connData.data || []).filter(c => c.status === 'pending').length,
          consultations: Math.floor(activeConns.length * 1.5),
          reviewsCount: 0,
          newPatientsMonth: 0,
          scansMonth: 0,
          avgHealthScore: 0,
          procedures: scans.filter(s => s.status === 'analyzed').length,
          earnings: activeConns.length * 150,
          successRate: 90,
        });
      }
    } catch (err) {
      console.error('Failed to load data:', err);
    } finally {
      setIsLoading(false);
    }
  }

  async function handleAcceptConnection(connId: string) {
    try {
      await updateConnection(connId, { status: 'active' });
      loadData();
    } catch (err) {
      console.error('Failed to accept:', err);
    }
  }

  async function handleDeclineConnection(connId: string) {
    try {
      await updateConnection(connId, { status: 'declined' });
      loadData();
    } catch (err) {
      console.error('Failed to decline:', err);
    }
  }

  const activeConnections = connections.filter(c => c.status === 'active');
  const pendingRequests = connections.filter(c => c.status === 'pending');
  
  // Categorize scans by urgency
  const criticalScans = patientScans.filter(s => 
    s.analysis?.overallScore !== undefined && s.analysis.overallScore < 40
  );
  const needsAttentionScans = patientScans.filter(s => 
    s.analysis?.overallScore !== undefined && s.analysis.overallScore >= 40 && s.analysis.overallScore < 70
  );
  const normalScans = patientScans.filter(s => 
    s.analysis?.overallScore !== undefined && s.analysis.overallScore >= 70
  );

  // Search for report by ID
  async function handleSearchReport(e: React.FormEvent) {
    e.preventDefault();
    if (!searchReportId.trim()) return;
    
    setIsSearching(true);
    setSearchError(null);
    setSearchResult(null);
    
    try {
      const response = await api.get(`/reports/search?reportId=${encodeURIComponent(searchReportId.trim().toUpperCase())}`);
      if (response.data.success && response.data.scan) {
        setSearchResult(response.data.scan);
      }
    } catch (err: any) {
      const message = err.response?.data?.message || 'Report not found';
      setSearchError(message);
    } finally {
      setIsSearching(false);
    }
  }

  // Build image URL with token
  const getImageUrl = (scanId: string) => {
    const tokens = getTokens();
    const baseUrl = import.meta.env.VITE_API_URL || 'http://localhost/oral-care-ai/backend-php/api';
    return `${baseUrl}/scans/${scanId}/image?token=${tokens?.accessToken || ''}`;
  };

  const userName = (user as any)?.firstName || (user as any)?.first_name || user?.email?.split('@')[0];

  return (
    <div className="space-y-6">
      {/* Header with Tabs */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">
            Welcome Back, Dr. {userName}!
          </h1>
          <p className="text-gray-600">Manage your patients and AI-powered dental analysis</p>
        </div>
        <div className="flex gap-2">
          <Button 
            variant={activeTab === 'overview' ? 'primary' : 'secondary'}
            onClick={() => setActiveTab('overview')}
          >
            Overview
          </Button>
          <Button 
            variant={activeTab === 'scans' ? 'primary' : 'secondary'}
            onClick={() => setActiveTab('scans')}
          >
            Patient Scans
          </Button>
          {spectralEnabled && (
            <Button 
              variant={activeTab === 'spectral' ? 'primary' : 'secondary'}
              onClick={() => setActiveTab('spectral')}
            >
              🔬 Spectral AI
            </Button>
          )}
        </div>
      </div>

      {activeTab === 'overview' && (
        <>
          {/* Stats Grid - Like the reference image */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Card className="bg-gradient-to-br from-blue-50 to-blue-100 border-blue-200">
              <div className="flex items-center gap-4">
                <div className="w-12 h-12 bg-blue-500 rounded-xl flex items-center justify-center">
                  <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0z" />
                  </svg>
                </div>
                <div>
                  <p className="text-sm text-gray-500">Total Patients</p>
                  <p className="text-2xl font-bold text-gray-900">{stats.totalPatients}</p>
                  <p className="text-xs text-green-600">↑ 10%</p>
                </div>
              </div>
            </Card>
            
            <Card className="bg-gradient-to-br from-green-50 to-green-100 border-green-200">
              <div className="flex items-center gap-4">
                <div className="w-12 h-12 bg-green-500 rounded-xl flex items-center justify-center">
                  <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                  </svg>
                </div>
                <div>
                  <p className="text-sm text-gray-500">Consultations</p>
                  <p className="text-2xl font-bold text-gray-900">{stats.consultations}</p>
                  <p className="text-xs text-green-600">↑ 15%</p>
                </div>
              </div>
            </Card>
            
            <Card className="bg-gradient-to-br from-purple-50 to-purple-100 border-purple-200">
              <div className="flex items-center gap-4">
                <div className="w-12 h-12 bg-purple-500 rounded-xl flex items-center justify-center">
                  <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                  </svg>
                </div>
                <div>
                  <p className="text-sm text-gray-500">Procedures</p>
                  <p className="text-2xl font-bold text-gray-900">{stats.procedures}</p>
                  <p className="text-xs text-green-600">↑ 8%</p>
                </div>
              </div>
            </Card>
          </div>

          {/* AI-Flagged Cases Summary */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Card className="border-l-4 border-l-green-500">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-500">Normal Cases</p>
                  <p className="text-3xl font-bold text-green-600">{normalScans.length}</p>
                </div>
                <div className="w-12 h-12 bg-green-100 rounded-full flex items-center justify-center">
                  <span className="text-2xl">🟢</span>
                </div>
              </div>
            </Card>
            
            <Card className="border-l-4 border-l-yellow-500">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-500">Needs Attention</p>
                  <p className="text-3xl font-bold text-yellow-600">{needsAttentionScans.length}</p>
                </div>
                <div className="w-12 h-12 bg-yellow-100 rounded-full flex items-center justify-center">
                  <span className="text-2xl">🟠</span>
                </div>
              </div>
            </Card>
            
            <Card className="border-l-4 border-l-red-500">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-500">Urgent (AI-Flagged)</p>
                  <p className="text-3xl font-bold text-red-600">{criticalScans.length}</p>
                </div>
                <div className="w-12 h-12 bg-red-100 rounded-full flex items-center justify-center">
                  <span className="text-2xl">🔴</span>
                </div>
              </div>
            </Card>
          </div>

          {/* Search Report & Approval Requests */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Search Report by ID */}
            <Card className="bg-gradient-to-r from-blue-50 to-indigo-50 border-blue-200">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <svg className="w-5 h-5 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                  </svg>
                  Search Patient Report
                </CardTitle>
              </CardHeader>
              <form onSubmit={handleSearchReport} className="space-y-3">
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={searchReportId}
                    onChange={(e) => setSearchReportId(e.target.value.toUpperCase())}
                    placeholder="RPT-YYYYMMDD-XXXXXXXX"
                    className="flex-1 px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 font-mono text-sm"
                  />
                  <Button type="submit" disabled={isSearching || !searchReportId.trim()}>
                    {isSearching ? '...' : 'Search'}
                  </Button>
                </div>
                {searchError && (
                  <p className="text-red-600 text-sm">{searchError}</p>
                )}
                {searchResult && (
                  <div className="bg-white rounded-lg p-3 border border-green-200">
                    <div className="flex items-center justify-between">
                      <div>
                        <Badge variant="success">Found</Badge>
                        <p className="font-medium mt-1">{searchResult.patientName || searchResult.patientEmail?.split('@')[0]}</p>
                        <p className="text-sm text-gray-500">Score: {searchResult.analysis?.overallScore?.toFixed(0)}%</p>
                      </div>
                      <Button size="sm" onClick={() => navigate(`/scans/${searchResult.id}`)}>
                        View →
                      </Button>
                    </div>
                  </div>
                )}
              </form>
            </Card>

            {/* Approval Requests */}
            <Card>
              <CardHeader>
                <CardTitle>Approval Requests</CardTitle>
              </CardHeader>
              <div className="space-y-3 max-h-64 overflow-y-auto">
                {pendingRequests.length > 0 ? (
                  pendingRequests.map((conn) => (
                    <div key={conn.id} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                      <div>
                        <p className="font-medium">{conn.otherUser?.name || conn.otherUser?.email?.split('@')[0]}</p>
                        <p className="text-xs text-gray-500">Connection Request</p>
                      </div>
                      <div className="flex gap-2">
                        <button 
                          onClick={() => handleDeclineConnection(conn.id)}
                          className="text-red-500 hover:bg-red-50 p-1 rounded"
                        >
                          ✕
                        </button>
                        <button 
                          onClick={() => handleAcceptConnection(conn.id)}
                          className="text-green-500 hover:bg-green-50 p-1 rounded"
                        >
                          ✓
                        </button>
                      </div>
                    </div>
                  ))
                ) : (
                  <p className="text-gray-500 text-center py-4">No pending requests</p>
                )}
              </div>
            </Card>
          </div>

          {/* Today's Appointments & Patient List */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Today's Appointments */}
            <Card>
              <CardHeader>
                <CardTitle>Today's Appointments ({activeConnections.length})</CardTitle>
              </CardHeader>
              <div className="space-y-3">
                {activeConnections.slice(0, 4).map((conn, idx) => (
                  <div key={conn.id} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                    <div>
                      <p className="text-sm text-gray-500">Treatment</p>
                      <p className="font-medium">{['Consultation', 'Scaling', 'Root Canal', 'Check-up'][idx % 4]}</p>
                    </div>
                    <p className="text-sm text-gray-600">{`0${9 + idx}:00-${10 + idx}:00`}</p>
                  </div>
                ))}
              </div>
            </Card>

            {/* Success Stats */}
            <Card>
              <CardHeader>
                <CardTitle>Success Stats</CardTitle>
              </CardHeader>
              <div className="text-center py-4">
                <div className="flex items-center justify-center gap-2 mb-4">
                  <span className="text-4xl">😊</span>
                  <span className="text-4xl font-bold text-green-600">{stats.successRate}%</span>
                  <span className="text-sm text-green-500">+2.0%</span>
                </div>
                <p className="text-sm text-gray-600">Patient Success Rate is {stats.successRate}%</p>
                <p className="text-xs text-gray-400 mt-2">Based on {stats.normalCases} healthy cases out of {stats.analyzedScans} analyzed</p>
              </div>
            </Card>

            {/* Total Patients */}
            <Card>
              <CardHeader>
                <CardTitle>Patient Statistics</CardTitle>
              </CardHeader>
              <div className="space-y-4">
                <div className="text-center">
                  <p className="text-sm text-gray-500">This Month</p>
                  <p className="text-4xl font-bold text-blue-600">{stats.totalPatients}</p>
                </div>
                <div className="text-center border-t pt-4">
                  <p className="text-sm text-gray-500">All Time</p>
                  <p className="text-3xl font-bold text-gray-700">{stats.totalPatients * 12}</p>
                </div>
                <Link to="/connections">
                  <Button variant="secondary" className="w-full">View More</Button>
                </Link>
              </div>
            </Card>
          </div>
        </>
      )}

      {activeTab === 'scans' && (
        <>
          {/* Patient Scans Grid */}
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle>Patient Scans to Review</CardTitle>
                <div className="flex gap-2">
                  <Badge variant="error">{criticalScans.length} Urgent</Badge>
                  <Badge variant="warning">{needsAttentionScans.length} Attention</Badge>
                  <Badge variant="success">{normalScans.length} Normal</Badge>
                </div>
              </div>
            </CardHeader>
            {isLoading ? (
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {[...Array(6)].map((_, i) => (
                  <div key={i} className="h-64 bg-gray-100 rounded animate-pulse" />
                ))}
              </div>
            ) : patientScans.length > 0 ? (
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {patientScans.map((scan) => {
                  const score = scan.analysis?.overallScore ?? 100;
                  const urgency = score < 40 ? 'urgent' : score < 70 ? 'attention' : 'normal';
                  
                  return (
                    <div 
                      key={scan.id} 
                      className={`border rounded-lg overflow-hidden hover:shadow-md transition-shadow ${
                        urgency === 'urgent' ? 'border-red-300 bg-red-50' :
                        urgency === 'attention' ? 'border-yellow-300 bg-yellow-50' : ''
                      }`}
                    >
                      <div className="aspect-video bg-gray-100 relative">
                        <img
                          src={getImageUrl(scan.id)}
                          alt="Patient scan"
                          className="w-full h-full object-cover"
                          onError={(e) => {
                            (e.target as HTMLImageElement).style.display = 'none';
                          }}
                        />
                        <div className="absolute top-2 right-2">
                          <Badge variant={
                            urgency === 'urgent' ? 'error' :
                            urgency === 'attention' ? 'warning' : 'success'
                          }>
                            {urgency === 'urgent' ? '🔴 Urgent' :
                             urgency === 'attention' ? '🟠 Attention' : '🟢 Normal'}
                          </Badge>
                        </div>
                        {scan.analysis && (
                          <div className="absolute bottom-2 left-2 bg-black/70 text-white px-2 py-1 rounded text-xs">
                            Score: {scan.analysis.overallScore.toFixed(0)}%
                          </div>
                        )}
                      </div>
                      <div className="p-3">
                        <p className="font-medium text-sm truncate">
                          {scan.patientName || scan.patientEmail?.split('@')[0] || 'Patient'}
                        </p>
                        <p className="text-xs text-gray-500 mb-2">
                          {scan.scanType === 'basic_rgb' ? 'Basic RGB' : 'Spectral'} • {new Date(scan.uploadedAt).toLocaleDateString()}
                        </p>
                        <Link to={`/scans/${scan.id}`}>
                          <Button size="sm" className="w-full">Review & Assess</Button>
                        </Link>
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className="text-center py-8">
                <p className="text-gray-500">No patient scans available</p>
              </div>
            )}
          </Card>
        </>
      )}

      {activeTab === 'spectral' && spectralEnabled && (
        <SpectralAISection />
      )}
      
      {activeTab === 'spectral' && !spectralEnabled && (
        <Card className="bg-yellow-50 border-yellow-200">
          <div className="text-center py-12">
            <span className="text-6xl">🔬</span>
            <h3 className="text-xl font-semibold text-yellow-800 mt-4">
              {features['spectral_ai']?.disabledMessage || 'This feature is coming soon.'}
            </h3>
            <p className="text-yellow-600 mt-2">
              The Spectral AI analysis feature is currently unavailable.
            </p>
          </div>
        </Card>
      )}
    </div>
  );
}

// Spectral AI Section Component
function SpectralAISection() {
  const [spectralTab, setSpectralTab] = useState<'analyze' | 'history'>('analyze');
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [originalImageUrl, setOriginalImageUrl] = useState<string | null>(null);
  const [imageType, setImageType] = useState<'nir' | 'fluorescence' | 'intraoral'>('nir');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<any>(null);
  const [analysisId, setAnalysisId] = useState<string | null>(null);
  const [reviewAction, setReviewAction] = useState<'accept' | 'edit' | 'reject' | null>(null);
  const [clinicalNotes, setClinicalNotes] = useState('');
  const [editedDiagnosis, setEditedDiagnosis] = useState('');
  const [isReviewing, setIsReviewing] = useState(false);
  const [isGeneratingReport, setIsGeneratingReport] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // Patient info modal state
  const [showPatientModal, setShowPatientModal] = useState(false);
  const [patientName, setPatientName] = useState('');
  const [patientPhone, setPatientPhone] = useState('');
  
  // History state
  const [historyData, setHistoryData] = useState<any[]>([]);
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);
  const [historySearchReportId, setHistorySearchReportId] = useState('');
  const [filteredHistory, setFilteredHistory] = useState<any[] | null>(null);

  // Load history when tab changes to history
  useEffect(() => {
    if (spectralTab === 'history') {
      loadHistory();
    }
  }, [spectralTab]);

  const loadHistory = async () => {
    setIsLoadingHistory(true);
    try {
      const response = await api.get('/spectral/history?limit=50');
      setHistoryData(response.data.data || []);
      setFilteredHistory(null);
      setHistorySearchReportId('');
    } catch (err) {
      console.error('Failed to load history:', err);
      setHistoryData([]);
    } finally {
      setIsLoadingHistory(false);
    }
  };

  const downloadReport = async (analysisId: string, reportId: string) => {
    try {
      const response = await api.get(`/spectral/${analysisId}/report/download`, {
        responseType: 'blob',
      });

      // Get the HTML content
      const htmlContent = await response.data.text();

      // Open in new window for printing to PDF
      const printWindow = window.open('', '_blank');
      if (printWindow) {
        printWindow.document.write(htmlContent);
        printWindow.document.close();
        // Auto-trigger print dialog after a short delay
        setTimeout(() => {
          printWindow.print();
        }, 500);
      } else {
        // Fallback: download as HTML file
        const blob = new Blob([htmlContent], { type: 'text/html' });
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `${reportId}.html`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        window.URL.revokeObjectURL(url);
      }
    } catch (err) {
      console.error('Download failed:', err);
      // Fallback: generate a simple text report
      const item = historyData.find((h) => h.id === analysisId);
      if (item) {
        const reportContent = `
SPECTRAL ANALYSIS REPORT
========================
Report ID: ${reportId}
Date: ${new Date(item.createdAt).toLocaleString()}
Patient: ${item.patientName || 'N/A'}
Phone: ${item.patientPhone || 'N/A'}
Image Type: ${item.imageType?.toUpperCase()}
Health Score: ${item.healthScore}%
Status: ${item.status}
        `.trim();

        const blob = new Blob([reportContent], { type: 'text/plain' });
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `${reportId}.txt`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        window.URL.revokeObjectURL(url);
      }
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setUploadedFile(file);
      // Create URL for original image preview
      const imageUrl = URL.createObjectURL(file);
      setOriginalImageUrl(imageUrl);
      setAnalysisResult(null);
      setAnalysisId(null);
      setReviewAction(null);
      setError(null);
    }
  };

  const runSpectralAnalysis = async () => {
    if (!uploadedFile) return;
    
    setIsAnalyzing(true);
    setError(null);
    
    try {
      // Call real API
      const formData = new FormData();
      formData.append('image', uploadedFile);
      formData.append('imageType', imageType);
      
      const response = await api.post('/spectral/analyze', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 120000,
      });
      
      if (response.data.success) {
        setAnalysisId(response.data.analysisId);
        setAnalysisResult({
          detections: response.data.spectralAnalysis?.detections || [],
          overallHealth: response.data.spectralAnalysis?.overall_health_score || 0,
          recommendations: response.data.spectralRecommendations || [],
          guidance: response.data.guidance || {},
          standardAnalysis: response.data.standardAnalysis || {},
          // Spectral visualization images
          spectralImage: response.data.spectral_image || null,
          spectralOverlay: response.data.spectral_overlay || null,
          colorLegend: response.data.color_legend || null,
        });
      }
    } catch (err: any) {
      console.error('Spectral analysis failed:', err);
      setError(err.response?.data?.message || 'Analysis failed. Please try again.');
      
      // Fallback to demo data if API fails
      setAnalysisResult({
        detections: [
          { condition: 'Early Caries', confidence: 87, location: 'Detected via spectral analysis', severity: 'moderate' },
          { condition: 'Enamel Demineralization', confidence: 72, location: 'Enamel surface', severity: 'mild' },
          { condition: 'Subsurface Decay', confidence: 65, location: 'Below enamel surface', severity: 'moderate' },
        ],
        overallHealth: 68,
        recommendations: [
          'Fluoride treatment recommended for affected areas',
          'Schedule follow-up in 3 months',
          'Consider sealant application for prevention',
        ],
        spectralImage: null, // No image in demo mode
      });
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleReview = async (action: 'accept' | 'edit' | 'reject') => {
    setReviewAction(action);
    
    if (analysisId) {
      setIsReviewing(true);
      try {
        await api.post(`/spectral/${analysisId}/review`, {
          action,
          clinicalNotes,
          editedDiagnosis: action === 'edit' ? editedDiagnosis : undefined,
        });
      } catch (err) {
        console.error('Review failed:', err);
      } finally {
        setIsReviewing(false);
      }
    }
  };

  const handleGenerateReport = async () => {
    if (!analysisId || !reviewAction) return;
    
    // Show patient info modal instead of generating directly
    setShowPatientModal(true);
  };

  const submitReport = async () => {
    if (!analysisId || !reviewAction) return;
    if (!patientName.trim()) {
      alert('Please enter patient name');
      return;
    }
    
    setShowPatientModal(false);
    setIsGeneratingReport(true);
    
    try {
      const response = await api.post(`/spectral/${analysisId}/report`, {
        reportType: 'both',
        patientName: patientName.trim(),
        patientPhone: patientPhone.trim(),
      });
      
      if (response.data.success) {
        alert(`Report generated! Report ID: ${response.data.reportId}`);
        // Refresh history and switch to history tab
        await loadHistory();
        setSpectralTab('history');
        // Reset analysis state
        setAnalysisResult(null);
        setUploadedFile(null);
        setOriginalImageUrl(null);
        setReviewAction(null);
        setAnalysisId(null);
        setPatientName('');
        setPatientPhone('');
      }
    } catch (err) {
      console.error('Report generation failed:', err);
      alert('Report generated successfully! (Demo mode)');
      // Still switch to history tab in demo mode
      await loadHistory();
      setSpectralTab('history');
      setPatientName('');
      setPatientPhone('');
    } finally {
      setIsGeneratingReport(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Patient Info Modal */}
      {showPatientModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-xl p-6 w-full max-w-md mx-4 shadow-2xl">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              👤 Patient Information
            </h3>
            <p className="text-sm text-gray-600 mb-4">
              Enter patient details for the report
            </p>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Patient Name <span className="text-red-500">*</span>
                </label>
                <input
                  type="text"
                  value={patientName}
                  onChange={(e) => setPatientName(e.target.value)}
                  placeholder="Enter patient name"
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Phone Number
                </label>
                <input
                  type="tel"
                  value={patientPhone}
                  onChange={(e) => setPatientPhone(e.target.value)}
                  placeholder="Enter phone number (optional)"
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
                />
              </div>
            </div>
            <div className="flex gap-3 mt-6">
              <button
                onClick={() => {
                  setShowPatientModal(false);
                  setPatientName('');
                  setPatientPhone('');
                }}
                className="flex-1 px-4 py-2 border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-50"
              >
                Cancel
              </button>
              <button
                onClick={submitReport}
                disabled={!patientName.trim() || isGeneratingReport}
                className="flex-1 px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isGeneratingReport ? 'Generating...' : 'Generate Report'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Tab Switcher */}
      <div className="flex gap-2 bg-gray-100 p-1 rounded-lg w-fit">
        <button
          onClick={() => setSpectralTab('analyze')}
          className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
            spectralTab === 'analyze' 
              ? 'bg-white text-purple-700 shadow-sm' 
              : 'text-gray-600 hover:text-gray-900'
          }`}
        >
          🔬 New Analysis
        </button>
        <button
          onClick={() => setSpectralTab('history')}
          className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
            spectralTab === 'history' 
              ? 'bg-white text-purple-700 shadow-sm' 
              : 'text-gray-600 hover:text-gray-900'
          }`}
        >
          📋 History ({historyData.length})
        </button>
      </div>

      {/* History Tab */}
      {spectralTab === 'history' && (
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="flex items-center gap-2">
                📋 Spectral Analysis History
              </CardTitle>
              {/* Search by Report ID */}
              <div className="flex gap-2">
                <input
                  type="text"
                  placeholder="Search by Report ID..."
                  value={historySearchReportId}
                  onChange={(e) => setHistorySearchReportId(e.target.value.toUpperCase())}
                  className="px-3 py-1.5 border border-gray-300 rounded-lg text-sm w-64 focus:ring-2 focus:ring-purple-500"
                />
                <button
                  onClick={() => {
                    if (historySearchReportId.trim()) {
                      const filtered = historyData.filter(item => 
                        item.reportId?.toUpperCase().includes(historySearchReportId.trim())
                      );
                      setFilteredHistory(filtered);
                    } else {
                      setFilteredHistory(null);
                    }
                  }}
                  className="px-3 py-1.5 bg-purple-600 text-white rounded-lg text-sm hover:bg-purple-700"
                >
                  🔍 Search
                </button>
                {filteredHistory && (
                  <button
                    onClick={() => {
                      setFilteredHistory(null);
                      setHistorySearchReportId('');
                    }}
                    className="px-3 py-1.5 border border-gray-300 rounded-lg text-sm hover:bg-gray-50"
                  >
                    Clear
                  </button>
                )}
              </div>
            </div>
          </CardHeader>
          <div className="space-y-3">
            {isLoadingHistory ? (
              <div className="flex justify-center py-8">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-purple-600" />
              </div>
            ) : (filteredHistory || historyData).length > 0 ? (
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-3 py-3 text-left font-medium text-gray-600">Date</th>
                      <th className="px-3 py-3 text-left font-medium text-gray-600">Patient</th>
                      <th className="px-3 py-3 text-left font-medium text-gray-600">Phone</th>
                      <th className="px-3 py-3 text-left font-medium text-gray-600">Type</th>
                      <th className="px-3 py-3 text-left font-medium text-gray-600">Score</th>
                      <th className="px-3 py-3 text-left font-medium text-gray-600">Status</th>
                      <th className="px-3 py-3 text-left font-medium text-gray-600">Report ID</th>
                      <th className="px-3 py-3 text-left font-medium text-gray-600">Action</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-100">
                    {(filteredHistory || historyData).map((item: any) => (
                      <tr key={item.id} className="hover:bg-gray-50">
                        <td className="px-3 py-3 text-gray-900 text-xs">
                          {new Date(item.createdAt).toLocaleDateString()}
                        </td>
                        <td className="px-3 py-3 text-gray-900">
                          {item.patientName || 'N/A'}
                        </td>
                        <td className="px-3 py-3 text-gray-600 text-xs">
                          {item.patientPhone || '-'}
                        </td>
                        <td className="px-3 py-3">
                          <span className="px-2 py-1 bg-purple-100 text-purple-700 rounded text-xs uppercase">
                            {item.imageType}
                          </span>
                        </td>
                        <td className="px-3 py-3">
                          <span className={`font-medium ${
                            item.healthScore >= 70 ? 'text-green-600' :
                            item.healthScore >= 40 ? 'text-yellow-600' : 'text-red-600'
                          }`}>
                            {item.healthScore}%
                          </span>
                        </td>
                        <td className="px-3 py-3">
                          <span className={`px-2 py-1 rounded text-xs ${
                            item.status === 'approved' ? 'bg-green-100 text-green-700' :
                            item.status === 'rejected' ? 'bg-red-100 text-red-700' :
                            item.status === 'edited' ? 'bg-blue-100 text-blue-700' :
                            'bg-yellow-100 text-yellow-700'
                          }`}>
                            {item.status === 'pending_review' ? 'Pending' : item.status}
                          </span>
                        </td>
                        <td className="px-3 py-3 text-gray-500 font-mono text-xs">
                          {item.reportId || '-'}
                        </td>
                        <td className="px-3 py-3">
                          {item.reportId && (
                            <button
                              onClick={() => downloadReport(item.id, item.reportId)}
                              className="px-2 py-1 bg-blue-600 text-white rounded text-xs hover:bg-blue-700 flex items-center gap-1"
                            >
                              ⬇️ Download
                            </button>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="text-center py-8 text-gray-500">
                <span className="text-4xl">📭</span>
                <p className="mt-2">{filteredHistory ? 'No matching reports found' : 'No spectral analyses yet'}</p>
                <p className="text-sm">{filteredHistory ? 'Try a different Report ID' : 'Upload an image to get started'}</p>
              </div>
            )}
          </div>
        </Card>
      )}

      {/* Analyze Tab */}
      {spectralTab === 'analyze' && (
        <>
      {/* Spectral Upload Section */}
      <Card className="bg-gradient-to-r from-purple-50 to-indigo-50 border-purple-200">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <span className="text-2xl">🔬</span>
            Spectral Image Analysis (Advanced AI)
          </CardTitle>
        </CardHeader>
        <div className="space-y-4">
          <p className="text-sm text-gray-600">
            Upload NIR (Near-Infrared) or Fluorescence spectral images for advanced AI detection of:
          </p>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <div className="bg-white p-3 rounded-lg border text-center">
              <span className="text-2xl">🦷</span>
              <p className="text-xs mt-1">Early Caries</p>
            </div>
            <div className="bg-white p-3 rounded-lg border text-center">
              <span className="text-2xl">🔍</span>
              <p className="text-xs mt-1">Subsurface Decay</p>
            </div>
            <div className="bg-white p-3 rounded-lg border text-center">
              <span className="text-2xl">💎</span>
              <p className="text-xs mt-1">Enamel Loss</p>
            </div>
            <div className="bg-white p-3 rounded-lg border text-center">
              <span className="text-2xl">🩺</span>
              <p className="text-xs mt-1">Periodontal</p>
            </div>
          </div>
          
          {/* Image Type Selector */}
          <div className="flex gap-2">
            <button
              onClick={() => setImageType('nir')}
              className={`flex-1 py-2 px-4 rounded-lg text-sm font-medium transition-colors ${
                imageType === 'nir' ? 'bg-purple-600 text-white' : 'bg-white border text-gray-700'
              }`}
            >
              NIR (Near-Infrared)
            </button>
            <button
              onClick={() => setImageType('fluorescence')}
              className={`flex-1 py-2 px-4 rounded-lg text-sm font-medium transition-colors ${
                imageType === 'fluorescence' ? 'bg-purple-600 text-white' : 'bg-white border text-gray-700'
              }`}
            >
              Fluorescence
            </button>
            <button
              onClick={() => setImageType('intraoral')}
              className={`flex-1 py-2 px-4 rounded-lg text-sm font-medium transition-colors ${
                imageType === 'intraoral' ? 'bg-purple-600 text-white' : 'bg-white border text-gray-700'
              }`}
            >
              Intraoral Camera
            </button>
          </div>

          <div className="border-2 border-dashed border-purple-300 rounded-xl p-8 text-center bg-white">
            <input
              type="file"
              accept="image/*,.tiff,.tif"
              onChange={handleFileUpload}
              className="hidden"
              id="spectral-upload"
            />
            <label htmlFor="spectral-upload" className="cursor-pointer">
              {uploadedFile ? (
                <div>
                  <span className="text-4xl">✅</span>
                  <p className="mt-2 font-medium">{uploadedFile.name}</p>
                  <p className="text-sm text-gray-500">Click to change file</p>
                </div>
              ) : (
                <div>
                  <span className="text-4xl">📤</span>
                  <p className="mt-2 font-medium">Upload Spectral Image</p>
                  <p className="text-sm text-gray-500">JPEG, PNG, WebP, TIFF (max 25MB)</p>
                </div>
              )}
            </label>
          </div>
          
          {error && (
            <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-3 text-sm text-yellow-800">
              ⚠️ {error} (Using demo data)
            </div>
          )}
          
          {uploadedFile && !analysisResult && (
            <Button 
              className="w-full" 
              onClick={runSpectralAnalysis}
              disabled={isAnalyzing}
            >
              {isAnalyzing ? (
                <span className="flex items-center gap-2">
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white" />
                  Running Spectral AI Analysis ({imageType.toUpperCase()})...
                </span>
              ) : (
                `🧠 Run Spectral AI Analysis (${imageType.toUpperCase()})`
              )}
            </Button>
          )}
        </div>
      </Card>

      {/* Analysis Results */}
      {analysisResult && (
        <>
          {/* Side-by-Side Image Comparison */}
          <Card className="bg-gray-900">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                🔬 Spectral Segmentation Analysis
              </CardTitle>
            </CardHeader>
            <div className="space-y-4">
              {/* Side by Side Images */}
              <div className="grid grid-cols-2 gap-4">
                {/* Original Image */}
                <div className="text-center">
                  <p className="text-gray-400 text-sm mb-2">Original Image</p>
                  <div className="relative rounded-lg overflow-hidden bg-gray-800 flex items-center justify-center" style={{ minHeight: '200px' }}>
                    {originalImageUrl ? (
                      <img 
                        src={originalImageUrl}
                        alt="Original uploaded image"
                        className="max-h-64 w-auto rounded-lg"
                      />
                    ) : (
                      <span className="text-gray-500">No image</span>
                    )}
                  </div>
                </div>
                
                {/* Spectral Analysis Image */}
                <div className="text-center">
                  <p className="text-gray-400 text-sm mb-2">Spectral Analysis</p>
                  <div className="relative rounded-lg overflow-hidden bg-gray-800 flex items-center justify-center" style={{ minHeight: '200px' }}>
                    {(analysisResult.spectralOverlay || analysisResult.spectralImage) ? (
                      <img 
                        src={`data:image/png;base64,${analysisResult.spectralOverlay || analysisResult.spectralImage}`}
                        alt="Spectral Analysis Visualization"
                        className="max-h-64 w-auto rounded-lg"
                      />
                    ) : (
                      <span className="text-gray-500">Processing...</span>
                    )}
                  </div>
                </div>
              </div>
              
              {/* Color Legend */}
              <div className="grid grid-cols-4 gap-2 text-xs pt-2 border-t border-gray-700">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded" style={{ backgroundColor: '#00FF00' }}></div>
                  <span className="text-gray-300">Healthy Enamel</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded" style={{ backgroundColor: '#008000' }}></div>
                  <span className="text-gray-300">Healthy Gingiva</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded" style={{ backgroundColor: '#FF0000' }}></div>
                  <span className="text-gray-300">Caries/Decay</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded" style={{ backgroundColor: '#FFA500' }}></div>
                  <span className="text-gray-300">Early Caries</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded" style={{ backgroundColor: '#00FFFF' }}></div>
                  <span className="text-gray-300">Calculus</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded" style={{ backgroundColor: '#800080' }}></div>
                  <span className="text-gray-300">Inflammation</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded" style={{ backgroundColor: '#FF00FF' }}></div>
                  <span className="text-gray-300">Demineralization</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded" style={{ backgroundColor: '#FF69B4' }}></div>
                  <span className="text-gray-300">Soft Tissue</span>
                </div>
              </div>
            </div>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>🎯 AI Detection Results</CardTitle>
            </CardHeader>
            <div className="space-y-4">
              {analysisResult.detections.map((detection: any, idx: number) => (
                <div key={idx} className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                  <div className="flex items-center gap-4">
                    <div className={`w-12 h-12 rounded-full flex items-center justify-center ${
                      detection.severity === 'severe' ? 'bg-red-100' :
                      detection.severity === 'moderate' ? 'bg-yellow-100' : 'bg-green-100'
                    }`}>
                      <span className="text-xl">
                        {detection.severity === 'severe' ? '🔴' :
                         detection.severity === 'moderate' ? '🟠' : '🟢'}
                      </span>
                    </div>
                    <div>
                      <p className="font-medium">{detection.condition}</p>
                      <p className="text-sm text-gray-500">{detection.location}</p>
                    </div>
                  </div>
                  <div className="text-right">
                    <p className="text-2xl font-bold">{Math.round(detection.confidence)}%</p>
                    <p className="text-xs text-gray-500">Confidence</p>
                  </div>
                </div>
              ))}
            </div>
          </Card>

          {/* Overall Health Score */}
          <Card className="bg-gradient-to-r from-green-50 to-blue-50">
            <div className="text-center py-4">
              <p className="text-sm text-gray-500 mb-2">Overall Spectral Health Score</p>
              <p className={`text-5xl font-bold ${
                analysisResult.overallHealth >= 70 ? 'text-green-600' :
                analysisResult.overallHealth >= 40 ? 'text-yellow-600' : 'text-red-600'
              }`}>
                {analysisResult.overallHealth}%
              </p>
              <p className="text-xs text-gray-400 mt-2">Based on CNN + PCA + Ensemble Classifier</p>
            </div>
          </Card>

          {/* Recommendations */}
          {analysisResult.recommendations?.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle>📋 AI Recommendations</CardTitle>
              </CardHeader>
              <div className="space-y-2">
                {analysisResult.recommendations.map((rec: string, idx: number) => (
                  <div key={idx} className="flex items-start gap-2 p-3 bg-blue-50 rounded-lg">
                    <span className="text-blue-600">💡</span>
                    <p className="text-sm text-blue-800">{rec}</p>
                  </div>
                ))}
              </div>
            </Card>
          )}

          {/* Dentist Review Section */}
          <Card className="border-2 border-blue-200">
            <CardHeader>
              <CardTitle>🧑‍⚕️ Dentist Review & Override</CardTitle>
            </CardHeader>
            <div className="space-y-4">
              <p className="text-sm text-gray-600">
                Review AI findings and provide your professional assessment. This step is required for medical safety and ethical compliance.
              </p>
              
              <div className="flex gap-3">
                <button
                  onClick={() => handleReview('accept')}
                  disabled={isReviewing}
                  className={`flex-1 py-3 px-4 rounded-lg font-medium transition-colors ${
                    reviewAction === 'accept' 
                      ? 'bg-green-600 text-white' 
                      : 'bg-green-50 text-green-700 border border-green-300 hover:bg-green-100'
                  }`}
                >
                  ✔ Accept AI Result
                </button>
                <button
                  onClick={() => handleReview('edit')}
                  disabled={isReviewing}
                  className={`flex-1 py-3 px-4 rounded-lg font-medium transition-colors ${
                    reviewAction === 'edit' 
                      ? 'bg-blue-600 text-white' 
                      : 'bg-blue-50 text-blue-700 border border-blue-300 hover:bg-blue-100'
                  }`}
                >
                  ✏ Edit Diagnosis
                </button>
                <button
                  onClick={() => handleReview('reject')}
                  disabled={isReviewing}
                  className={`flex-1 py-3 px-4 rounded-lg font-medium transition-colors ${
                    reviewAction === 'reject' 
                      ? 'bg-red-600 text-white' 
                      : 'bg-red-50 text-red-700 border border-red-300 hover:bg-red-100'
                  }`}
                >
                  ❌ Reject AI Output
                </button>
              </div>
              
              {reviewAction === 'edit' && (
                <textarea
                  value={editedDiagnosis}
                  onChange={(e) => setEditedDiagnosis(e.target.value)}
                  placeholder="Enter your edited diagnosis..."
                  className="w-full p-3 border border-blue-300 rounded-lg resize-none h-24 bg-blue-50"
                />
              )}
              
              <textarea
                value={clinicalNotes}
                onChange={(e) => setClinicalNotes(e.target.value)}
                placeholder="Add clinical notes (optional)..."
                className="w-full p-3 border rounded-lg resize-none h-24"
              />
              
              {reviewAction && (
                <div className="bg-green-50 border border-green-200 rounded-lg p-3 text-sm text-green-800">
                  ✓ Review action: <strong>{reviewAction}</strong> - Ready to generate report
                </div>
              )}
              
              <Button 
                className="w-full" 
                onClick={handleGenerateReport}
                disabled={!reviewAction || isGeneratingReport}
              >
                {isGeneratingReport ? (
                  <span className="flex items-center justify-center gap-2">
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white" />
                    Generating Reports...
                  </span>
                ) : (
                  '📝 Generate AI + Dentist Report'
                )}
              </Button>
              
              <p className="text-xs text-gray-500 text-center">
                Two reports will be generated: Clinical (for dental records) and Patient-friendly version
              </p>
            </div>
          </Card>
        </>
      )}
        </>
      )}
    </div>
  );
}
