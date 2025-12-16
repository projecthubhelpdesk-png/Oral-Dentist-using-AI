import { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '@/context/AuthContext';
import { Card, CardHeader, CardTitle } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import { getConnections, updateConnection } from '@/services/connections';
import { api } from '@/services/api';
import { getTokens } from '@/services/auth';
import type { Connection, Scan } from '@/types';

interface PatientScan extends Scan {
  patientEmail?: string;
  patientName?: string;
}

export function DentistDashboard() {
  const { user } = useAuth();
  const navigate = useNavigate();
  const [connections, setConnections] = useState<Connection[]>([]);
  const [patientScans, setPatientScans] = useState<PatientScan[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  
  // Search state
  const [searchReportId, setSearchReportId] = useState('');
  const [isSearching, setIsSearching] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);
  const [searchResult, setSearchResult] = useState<PatientScan | null>(null);

  useEffect(() => {
    loadData();
  }, []);

  async function loadData() {
    try {
      setIsLoading(true);
      const [connData, scansData] = await Promise.all([
        getConnections(),
        api.get<{ data: PatientScan[] }>('/scans/patients').catch(() => ({ data: { data: [] } }))
      ]);
      setConnections(connData.data || []);
      setPatientScans(scansData.data.data || []);
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
  const scansToReview = patientScans.filter(s => s.status === 'analyzed');

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

  function handleViewReport() {
    if (searchResult) {
      navigate(`/scans/${searchResult.id}`);
    }
  }

  function clearSearch() {
    setSearchReportId('');
    setSearchResult(null);
    setSearchError(null);
  }

  // Build image URL with token
  const getImageUrl = (scanId: string) => {
    const tokens = getTokens();
    const baseUrl = import.meta.env.VITE_API_URL || 'http://localhost/oral-care-ai/backend-php/api';
    return `${baseUrl}/scans/${scanId}/image?token=${tokens?.accessToken || ''}`;
  };

  return (
    <div className="space-y-6">
      {/* Welcome Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">
            Welcome, Dr. {(user as any)?.firstName || (user as any)?.first_name || user?.email?.split('@')[0]}!
          </h1>
          <p className="text-gray-600">Review patient scans and provide professional assessments</p>
        </div>
      </div>

      {/* Search Report by ID */}
      <Card className="bg-gradient-to-r from-blue-50 to-indigo-50 border-blue-200">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <svg className="w-5 h-5 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
            Search Patient Report by ID
          </CardTitle>
        </CardHeader>
        <form onSubmit={handleSearchReport} className="space-y-4">
          <div className="flex gap-3">
            <div className="flex-1">
              <input
                type="text"
                value={searchReportId}
                onChange={(e) => setSearchReportId(e.target.value.toUpperCase())}
                placeholder="Enter Report ID (e.g., RPT-20251214-626C3768)"
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent font-mono text-sm"
              />
            </div>
            <Button type="submit" disabled={isSearching || !searchReportId.trim()}>
              {isSearching ? (
                <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white" />
              ) : (
                'Search'
              )}
            </Button>
            {(searchResult || searchError) && (
              <Button type="button" variant="secondary" onClick={clearSearch}>
                Clear
              </Button>
            )}
          </div>
          
          {searchError && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-center gap-3">
              <svg className="w-5 h-5 text-red-500 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <p className="text-red-700 text-sm">{searchError}</p>
            </div>
          )}
          
          {searchResult && (
            <div className="bg-white border border-green-200 rounded-lg p-4 shadow-sm">
              <div className="flex items-start justify-between gap-4">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-2">
                    <Badge variant="success">Report Found</Badge>
                    <span className="font-mono text-sm text-gray-600">{searchReportId}</span>
                  </div>
                  <p className="font-medium text-gray-900">{searchResult.patientName || searchResult.patientEmail?.split('@')[0] || 'Patient'}</p>
                  <div className="flex items-center gap-4 mt-2 text-sm text-gray-600">
                    <span>📅 {new Date(searchResult.uploadedAt).toLocaleDateString()}</span>
                    <span>📊 Health Score: <strong className={
                      searchResult.analysis?.overallScore >= 70 ? 'text-green-600' :
                      searchResult.analysis?.overallScore >= 40 ? 'text-yellow-600' : 'text-red-600'
                    }>{searchResult.analysis?.overallScore?.toFixed(0)}%</strong></span>
                    <span>🔬 {searchResult.scanType === 'basic_rgb' ? 'Basic RGB' : 'Spectral'}</span>
                  </div>
                  {searchResult.analysis?.findings && searchResult.analysis.findings.length > 0 && (
                    <div className="mt-3 flex flex-wrap gap-2">
                      {searchResult.analysis.findings.slice(0, 3).map((f: any, idx: number) => (
                        <Badge key={idx} variant={
                          f.severity === 'severe' ? 'error' :
                          f.severity === 'moderate' ? 'warning' : 'info'
                        } size="sm">
                          {f.type.replace(/_/g, ' ')}
                        </Badge>
                      ))}
                    </div>
                  )}
                </div>
                <Button onClick={handleViewReport}>
                  View Full Report →
                </Button>
              </div>
            </div>
          )}
        </form>
      </Card>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <Card className="text-center py-6">
          <p className="text-3xl font-bold text-primary-600">{activeConnections.length}</p>
          <p className="text-sm text-gray-500">Active Patients</p>
        </Card>
        
        <Card className="text-center py-6">
          <p className="text-3xl font-bold text-yellow-600">{pendingRequests.length}</p>
          <p className="text-sm text-gray-500">Pending Requests</p>
        </Card>
        
        <Card className="text-center py-6">
          <p className="text-3xl font-bold text-green-600">{patientScans.length}</p>
          <p className="text-sm text-gray-500">Patient Scans</p>
        </Card>
        
        <Card className="text-center py-6">
          <p className="text-3xl font-bold text-blue-600">{scansToReview.length}</p>
          <p className="text-sm text-gray-500">To Review</p>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Pending Connection Requests */}
        <Card>
          <CardHeader>
            <CardTitle>Pending Connection Requests</CardTitle>
          </CardHeader>
          {isLoading ? (
            <div className="space-y-3">
              {[...Array(3)].map((_, i) => (
                <div key={i} className="h-16 bg-gray-100 rounded animate-pulse" />
              ))}
            </div>
          ) : pendingRequests.length > 0 ? (
            <div className="space-y-3">
              {pendingRequests.map((conn) => (
                <div key={conn.id} className="flex items-center justify-between p-3 bg-yellow-50 rounded-lg">
                  <div>
                    <p className="font-medium">{conn.otherUser?.name || conn.otherUser?.email?.split('@')[0] || 'Patient'}</p>
                    <p className="text-sm text-gray-500">Wants to connect</p>
                  </div>
                  <div className="flex gap-2">
                    <Button size="sm" variant="secondary" onClick={() => handleDeclineConnection(conn.id)}>
                      Decline
                    </Button>
                    <Button size="sm" onClick={() => handleAcceptConnection(conn.id)}>
                      Accept
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-gray-500 text-center py-4">No pending requests</p>
          )}
          <div className="mt-4">
            <Link to="/connections">
              <Button variant="secondary" className="w-full">View All Connections</Button>
            </Link>
          </div>
        </Card>

        {/* Active Patients */}
        <Card>
          <CardHeader>
            <CardTitle>Your Patients</CardTitle>
          </CardHeader>
          {isLoading ? (
            <div className="space-y-3">
              {[...Array(3)].map((_, i) => (
                <div key={i} className="h-16 bg-gray-100 rounded animate-pulse" />
              ))}
            </div>
          ) : activeConnections.length > 0 ? (
            <div className="space-y-3">
              {activeConnections.slice(0, 5).map((conn) => (
                <div key={conn.id} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                  <div>
                    <p className="font-medium">{conn.otherUser?.name || conn.otherUser?.email?.split('@')[0] || 'Patient'}</p>
                    <p className="text-sm text-gray-500">
                      Connected {conn.connectedAt ? new Date(conn.connectedAt).toLocaleDateString() : 'recently'}
                    </p>
                  </div>
                  <Badge variant="success">Active</Badge>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-gray-500 text-center py-4">No connected patients yet</p>
          )}
          <div className="mt-4">
            <Link to="/connections">
              <Button variant="secondary" className="w-full">Manage Patients</Button>
            </Link>
          </div>
        </Card>
      </div>

      {/* Patient Scans */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle>Patient Scans to Review</CardTitle>
            <Badge variant="info">{patientScans.length} total</Badge>
          </div>
        </CardHeader>
        {isLoading ? (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {[...Array(3)].map((_, i) => (
              <div key={i} className="h-64 bg-gray-100 rounded animate-pulse" />
            ))}
          </div>
        ) : patientScans.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {patientScans.slice(0, 6).map((scan) => (
              <div key={scan.id} className="border rounded-lg overflow-hidden hover:shadow-md transition-shadow">
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
                    <Badge variant={scan.status === 'analyzed' ? 'success' : 'warning'}>
                      {scan.status}
                    </Badge>
                  </div>
                  {scan.analysis && (
                    <div className="absolute bottom-2 left-2 bg-black/70 text-white px-2 py-1 rounded text-xs">
                      Score: {scan.analysis.overallScore.toFixed(0)}%
                    </div>
                  )}
                </div>
                <div className="p-3">
                  <p className="font-medium text-sm truncate">{scan.patientName || scan.patientEmail?.split('@')[0] || 'Patient'}</p>
                  <p className="text-xs text-gray-500 mb-2">
                    {scan.scanType === 'basic_rgb' ? 'Basic RGB' : 'Spectral'} • {new Date(scan.uploadedAt).toLocaleDateString()}
                  </p>
                  {/* Show AI findings summary */}
                  {scan.analysis?.findings?.length > 0 && (
                    <div className="mb-2 space-y-1">
                      {scan.analysis.findings.slice(0, 2).map((finding, idx) => (
                        <div key={idx} className="flex items-center justify-between text-xs">
                          <span className="capitalize truncate">{finding.type.replace(/_/g, ' ')}</span>
                          <Badge variant={
                            finding.severity === 'severe' ? 'error' :
                            finding.severity === 'moderate' ? 'warning' : 'info'
                          } className="text-xs px-1">
                            {finding.severity}
                          </Badge>
                        </div>
                      ))}
                    </div>
                  )}
                  <Link to={`/scans/${scan.id}`}>
                    <Button size="sm" className="w-full">Review & Assess</Button>
                  </Link>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center py-8">
            <svg className="w-16 h-16 text-gray-300 mx-auto mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            <p className="text-gray-500 mb-2">No patient scans available</p>
            <p className="text-sm text-gray-400">When patients share scans with you, they'll appear here</p>
          </div>
        )}
      </Card>

      {/* Quick Actions */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Link to="/connections">
          <Card className="p-4 hover:shadow-md transition-shadow cursor-pointer">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center">
                <svg className="w-5 h-5 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
                </svg>
              </div>
              <div>
                <p className="font-medium">Manage Patients</p>
                <p className="text-sm text-gray-500">View and manage connections</p>
              </div>
            </div>
          </Card>
        </Link>

        <Link to="/profile">
          <Card className="p-4 hover:shadow-md transition-shadow cursor-pointer">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-green-100 rounded-lg flex items-center justify-center">
                <svg className="w-5 h-5 text-green-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                </svg>
              </div>
              <div>
                <p className="font-medium">My Profile</p>
                <p className="text-sm text-gray-500">Update your practice info</p>
              </div>
            </div>
          </Card>
        </Link>

        <Card className="p-4 hover:shadow-md transition-shadow cursor-pointer">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-purple-100 rounded-lg flex items-center justify-center">
              <svg className="w-5 h-5 text-purple-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
            </div>
            <div>
              <p className="font-medium">Analytics</p>
              <p className="text-sm text-gray-500">View your statistics</p>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
}
