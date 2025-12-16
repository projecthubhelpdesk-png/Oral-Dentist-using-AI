import { useState } from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '@/context/AuthContext';
import { useTheme } from '@/context/ThemeContext';
import { useScans } from '@/hooks/useScans';
import { Card, CardHeader, CardTitle } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import { ScanCard } from '@/components/scans/ScanCard';
import { ScanUploader } from '@/components/scans/ScanUploader';
import { HealthScoreGauge } from '@/components/analysis/HealthScoreGauge';
import type { Scan, ScanType } from '@/types';

const ToothIcon = ({ className }: { className?: string }) => (
  <svg className={className} viewBox="0 0 24 24" fill="currentColor">
    <path d="M12 2C8.5 2 6 4.5 6 7c0 1.5.5 2.5 1 3.5.5 1 1 2 1 3.5 0 2-1 4-1 6 0 1.5 1 2 2 2s2-.5 2-2v-4h2v4c0 1.5 1 2 2 2s2-.5 2-2c0-2-1-4-1-6 0-1.5.5-2.5 1-3.5.5-1 1-2 1-3.5 0-2.5-2.5-5-6-5z"/>
  </svg>
);

export function UserDashboard() {
  const { user } = useAuth();
  const { theme } = useTheme();
  const { scans, isLoading, uploadScan, triggerAnalysis, refresh } = useScans({ limit: 4 });
  const [showUploader, setShowUploader] = useState(false);
  const [isUploading, setIsUploading] = useState(false);

  const handleUpload = async (file: File, scanType: ScanType, captureDevice?: string) => {
    setIsUploading(true);
    try {
      const scan = await uploadScan(file, scanType, captureDevice);
      await triggerAnalysis(scan.id);
      setShowUploader(false);
      refresh();
    } catch (err) {
      console.error('Upload failed:', err);
    } finally {
      setIsUploading(false);
    }
  };

  const handleViewScan = (scan: Scan) => {
    window.location.href = `/scans/${scan.id}`;
  };

  const handleAnalyzeScan = async (scan: Scan) => {
    await triggerAnalysis(scan.id);
    refresh();
  };

  // Calculate stats from scans
  const analyzedScans = scans.filter(s => s.status === 'analyzed');
  const latestAnalyzedScan = analyzedScans[0];
  const latestScore = latestAnalyzedScan?.analysis?.overallScore ?? 0;

  return (
    <div className="min-h-screen relative" style={{ margin: '-2rem calc(-50vw + 50%)', width: '100vw' }}>
      {/* Hero Header with Themed Gradient - Full Width Edge to Edge */}
      <div className={`relative bg-gradient-to-br ${theme.gradientFrom} ${theme.gradientVia} ${theme.gradientTo} px-4 sm:px-6 lg:px-8 pt-12 pb-40 overflow-hidden`}>
        {/* Dental Pattern Overlay */}
        <div className="absolute inset-0 bg-dental-pattern opacity-15"></div>
        
        {/* Floating Dental Icons in Header */}
        <div className="absolute inset-0 overflow-hidden">
          <ToothIcon className="absolute top-6 left-[10%] w-10 h-10 text-white/[0.08]" />
          <ToothIcon className="absolute top-8 left-[30%] w-14 h-14 text-white/[0.06]" />
          <ToothIcon className="absolute top-4 right-[25%] w-12 h-12 text-white/[0.07]" />
          <ToothIcon className="absolute top-10 right-[8%] w-10 h-10 text-white/[0.08]" />
          <ToothIcon className="absolute top-1/3 left-[5%] w-16 h-16 text-white/[0.05]" />
          <ToothIcon className="absolute top-1/3 right-[15%] w-12 h-12 text-white/[0.06]" />
          <ToothIcon className="absolute bottom-[35%] left-[20%] w-11 h-11 text-white/[0.07]" />
          <ToothIcon className="absolute bottom-[40%] right-[30%] w-14 h-14 text-white/[0.05]" />
          <div className="absolute top-1/4 left-[15%] w-20 h-20 rounded-full bg-white/[0.03]"></div>
          <div className="absolute top-[40%] right-[20%] w-28 h-28 rounded-full bg-white/[0.02]"></div>
        </div>

        <div className="relative z-10 max-w-7xl mx-auto flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div className={theme.textOnGradient}>
            <h1 className="text-3xl font-bold">
              Welcome back{user?.firstName ? `, ${user.firstName}` : user?.email ? `, ${user.email.split('@')[0]}` : ''}!
            </h1>
            <p className={`${theme.textMuted} mt-1`}>Track your oral health with AI-powered analysis</p>
          </div>
          <button 
            onClick={() => setShowUploader(!showUploader)}
            className="bg-gray-900 text-white hover:bg-gray-800 shadow-lg font-semibold px-4 py-2 rounded-lg transition-colors"
          >
            {showUploader ? 'Cancel' : '📷 Upload New Scan'}
          </button>
        </div>
        
        {/* Curved Bottom Edge */}
        <div className="absolute bottom-0 left-0 right-0">
          <svg viewBox="0 0 1440 80" fill="none" xmlns="http://www.w3.org/2000/svg" className="w-full h-auto" preserveAspectRatio="none">
            <path d="M0 80L60 73.3C120 66.7 240 53.3 360 46.7C480 40 600 40 720 43.3C840 46.7 960 53.3 1080 56.7C1200 60 1320 60 1380 60L1440 60V80H1380C1320 80 1200 80 1080 80C960 80 840 80 720 80C600 80 480 80 360 80C240 80 120 80 60 80H0Z" fill="#F9FAFB"/>
          </svg>
        </div>
      </div>

      {/* Main Content Area with Dental Pattern Background */}
      <div className="relative bg-gray-50">
        {/* Dental Icons Pattern for Content Area */}
        <div className={`absolute inset-0 overflow-hidden pointer-events-none text-${theme.accent}`}>
          {/* Row 1 */}
          <ToothIcon className="absolute top-[5%] left-[3%] w-8 h-8 opacity-[0.04]" />
          <ToothIcon className="absolute top-[8%] left-[18%] w-10 h-10 opacity-[0.03]" />
          <ToothIcon className="absolute top-[3%] left-[35%] w-7 h-7 opacity-[0.05]" />
          <ToothIcon className="absolute top-[6%] left-[55%] w-9 h-9 opacity-[0.03]" />
          <ToothIcon className="absolute top-[4%] left-[72%] w-8 h-8 opacity-[0.04]" />
          <ToothIcon className="absolute top-[7%] left-[88%] w-10 h-10 opacity-[0.03]" />
          
          {/* Row 2 */}
          <ToothIcon className="absolute top-[18%] left-[8%] w-9 h-9 opacity-[0.03]" />
          <ToothIcon className="absolute top-[22%] left-[28%] w-11 h-11 opacity-[0.04]" />
          <ToothIcon className="absolute top-[16%] left-[45%] w-8 h-8 opacity-[0.03]" />
          <ToothIcon className="absolute top-[20%] left-[65%] w-10 h-10 opacity-[0.05]" />
          <ToothIcon className="absolute top-[17%] left-[82%] w-7 h-7 opacity-[0.04]" />
          
          {/* Row 3 */}
          <ToothIcon className="absolute top-[32%] left-[2%] w-10 h-10 opacity-[0.04]" />
          <ToothIcon className="absolute top-[35%] left-[22%] w-8 h-8 opacity-[0.03]" />
          <ToothIcon className="absolute top-[30%] left-[40%] w-9 h-9 opacity-[0.05]" />
          <ToothIcon className="absolute top-[33%] left-[58%] w-11 h-11 opacity-[0.03]" />
          <ToothIcon className="absolute top-[36%] left-[75%] w-8 h-8 opacity-[0.04]" />
          <ToothIcon className="absolute top-[31%] left-[92%] w-9 h-9 opacity-[0.03]" />
          
          {/* Row 4 */}
          <ToothIcon className="absolute top-[48%] left-[6%] w-8 h-8 opacity-[0.03]" />
          <ToothIcon className="absolute top-[45%] left-[25%] w-10 h-10 opacity-[0.04]" />
          <ToothIcon className="absolute top-[50%] left-[48%] w-7 h-7 opacity-[0.05]" />
          <ToothIcon className="absolute top-[46%] left-[68%] w-9 h-9 opacity-[0.03]" />
          <ToothIcon className="absolute top-[52%] left-[85%] w-10 h-10 opacity-[0.04]" />
          
          {/* Row 5 */}
          <ToothIcon className="absolute top-[62%] left-[4%] w-9 h-9 opacity-[0.04]" />
          <ToothIcon className="absolute top-[65%] left-[20%] w-8 h-8 opacity-[0.03]" />
          <ToothIcon className="absolute top-[60%] left-[38%] w-10 h-10 opacity-[0.05]" />
          <ToothIcon className="absolute top-[68%] left-[55%] w-8 h-8 opacity-[0.03]" />
          <ToothIcon className="absolute top-[63%] left-[78%] w-11 h-11 opacity-[0.04]" />
          <ToothIcon className="absolute top-[66%] left-[95%] w-7 h-7 opacity-[0.03]" />
          
          {/* Row 6 */}
          <ToothIcon className="absolute top-[78%] left-[10%] w-10 h-10 opacity-[0.03]" />
          <ToothIcon className="absolute top-[82%] left-[30%] w-9 h-9 opacity-[0.04]" />
          <ToothIcon className="absolute top-[76%] left-[50%] w-8 h-8 opacity-[0.05]" />
          <ToothIcon className="absolute top-[80%] left-[70%] w-10 h-10 opacity-[0.03]" />
          <ToothIcon className="absolute top-[84%] left-[90%] w-8 h-8 opacity-[0.04]" />
          
          {/* Row 7 - Bottom */}
          <ToothIcon className="absolute top-[92%] left-[5%] w-8 h-8 opacity-[0.04]" />
          <ToothIcon className="absolute top-[95%] left-[25%] w-9 h-9 opacity-[0.03]" />
          <ToothIcon className="absolute top-[90%] left-[45%] w-10 h-10 opacity-[0.04]" />
          <ToothIcon className="absolute top-[93%] left-[65%] w-7 h-7 opacity-[0.05]" />
          <ToothIcon className="absolute top-[96%] left-[85%] w-9 h-9 opacity-[0.03]" />
        </div>
        
        {/* Content */}
        <div className="relative z-10 -mt-16 space-y-6 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto pb-8">
        {/* Upload Section */}
        {showUploader && (
          <Card className={`${theme.cardBorder} shadow-lg`}>
            <CardHeader className={`bg-gradient-to-r ${theme.cardHeaderBg}`}>
              <CardTitle className={`text-${theme.accentDark}`}>Upload Dental Scan</CardTitle>
            </CardHeader>
            <ScanUploader onUpload={handleUpload} isUploading={isUploading} />
          </Card>
        )}

        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Health Score */}
          <Card className={`flex flex-col items-center justify-center py-8 ${theme.cardBorder} shadow-lg bg-gradient-to-br from-white to-${theme.accentLight}`}>
            <HealthScoreGauge score={latestScore} size="md" />
            <p className="mt-2 text-sm text-gray-500">
              {latestScore > 0 ? 'Based on your latest scan' : 'Upload a scan to see your score'}
            </p>
          </Card>

          {/* Quick Stats */}
          <Card className={`${theme.cardBorder} shadow-lg`}>
            <CardHeader className={`bg-gradient-to-r ${theme.cardHeaderBg}`}>
              <CardTitle className={`text-${theme.accentDark} flex items-center gap-2`}>
                <ToothIcon className={`w-5 h-5 text-${theme.accent}`} />
                Your Scans
              </CardTitle>
            </CardHeader>
            <div className="space-y-4 p-4">
              <div className="flex justify-between items-center p-2 rounded-lg bg-gray-50">
                <span className="text-gray-600">Total Scans</span>
                <span className={`font-semibold text-${theme.accentDark}`}>{scans.length}</span>
              </div>
              <div className="flex justify-between items-center p-2 rounded-lg bg-green-50">
                <span className="text-gray-600">Analyzed</span>
                <span className="font-semibold text-green-600">{analyzedScans.length}</span>
              </div>
              <div className="flex justify-between items-center p-2 rounded-lg bg-yellow-50">
                <span className="text-gray-600">Processing</span>
                <span className="font-semibold text-yellow-600">
                  {scans.filter(s => s.status === 'processing').length}
                </span>
              </div>
            </div>
          </Card>

          {/* Latest Findings */}
          <Card className={`${theme.cardBorder} shadow-lg`}>
            <CardHeader className={`bg-gradient-to-r ${theme.cardHeaderBg}`}>
              <CardTitle className={`text-${theme.accentDark} flex items-center gap-2`}>
                <span className="text-lg">🔍</span>
                Latest Findings
              </CardTitle>
            </CardHeader>
            <div className="space-y-3 p-4">
              {latestAnalyzedScan?.analysis?.findings?.length ? (
                latestAnalyzedScan.analysis.findings.slice(0, 3).map((finding, idx) => (
                  <div key={idx} className="flex items-center justify-between p-2 bg-gray-50 rounded-lg">
                    <span className="text-sm capitalize">{finding.type.replace(/_/g, ' ')}</span>
                    <Badge variant={
                      finding.severity === 'severe' ? 'error' :
                      finding.severity === 'moderate' ? 'warning' : 'info'
                    }>
                      {finding.severity}
                    </Badge>
                  </div>
                ))
              ) : (
                <p className="text-sm text-gray-500">No findings yet. Upload a scan to get analysis.</p>
              )}
            </div>
          </Card>
        </div>

        {/* Recommendations Section */}
        {latestAnalyzedScan?.analysis?.recommendations?.length > 0 && (
          <Card className={`${theme.cardBorder} shadow-lg`}>
            <CardHeader className={`bg-gradient-to-r ${theme.cardHeaderBg}`}>
              <CardTitle className={`text-${theme.accentDark} flex items-center gap-2`}>
                <span className="text-lg">💡</span>
                AI Recommendations
              </CardTitle>
            </CardHeader>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 p-4">
              {latestAnalyzedScan.analysis.recommendations.map((rec, idx) => (
                <div key={idx} className={`flex items-start gap-2 p-3 bg-${theme.accentLight} rounded-lg border border-${theme.accent}/20`}>
                  <span className={`text-${theme.accent} mt-0.5`}>✓</span>
                  <p className={`text-sm text-${theme.accentDark}`}>{rec}</p>
                </div>
              ))}
            </div>
          </Card>
        )}

        {/* Quick Actions */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <Link to="/dentists">
            <Card className={`p-4 hover:shadow-lg transition-all cursor-pointer ${theme.cardBorder} hover:border-${theme.accent} group`}>
              <div className="flex items-center gap-3">
                <div className={`w-12 h-12 bg-gradient-to-br from-${theme.accentLight} to-${theme.accent}/20 rounded-xl flex items-center justify-center group-hover:scale-110 transition-transform`}>
                  <ToothIcon className={`w-6 h-6 text-${theme.accent}`} />
                </div>
                <div>
                  <p className="font-medium text-gray-900">Find a Dentist</p>
                  <p className="text-sm text-gray-500">Connect with professionals</p>
                </div>
              </div>
            </Card>
          </Link>
          <Link to="/connections">
            <Card className={`p-4 hover:shadow-lg transition-all cursor-pointer ${theme.cardBorder} hover:border-${theme.accent} group`}>
              <div className="flex items-center gap-3">
                <div className="w-12 h-12 bg-gradient-to-br from-green-100 to-green-200 rounded-xl flex items-center justify-center group-hover:scale-110 transition-transform">
                  <span className="text-2xl">👥</span>
                </div>
                <div>
                  <p className="font-medium text-gray-900">My Connections</p>
                  <p className="text-sm text-gray-500">Manage dentist connections</p>
                </div>
              </div>
            </Card>
          </Link>
          <Card className={`p-4 hover:shadow-lg transition-all cursor-pointer ${theme.cardBorder} hover:border-${theme.accent} group`} onClick={() => setShowUploader(true)}>
            <div className="flex items-center gap-3">
              <div className="w-12 h-12 bg-gradient-to-br from-purple-100 to-purple-200 rounded-xl flex items-center justify-center group-hover:scale-110 transition-transform">
                <span className="text-2xl">📷</span>
              </div>
              <div>
                <p className="font-medium text-gray-900">New Scan</p>
                <p className="text-sm text-gray-500">Upload dental image</p>
              </div>
            </div>
          </Card>
        </div>

        {/* Recent Scans */}
        <div>
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-gray-900 flex items-center gap-2">
              <span className={`text-${theme.accent}`}>📋</span>
              Recent Scans
            </h2>
            <Link to="/scans" className={`text-${theme.accent} hover:text-${theme.accentHover} text-sm font-medium flex items-center gap-1`}>
              View all 
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
              </svg>
            </Link>
          </div>

          {isLoading ? (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              {[...Array(4)].map((_, i) => (
                <div key={i} className={`bg-gradient-to-br from-${theme.accentLight} to-gray-100 rounded-xl h-64 animate-pulse`} />
              ))}
            </div>
          ) : scans.length > 0 ? (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              {scans.map((scan) => (
                <ScanCard
                  key={scan.id}
                  scan={scan}
                  onView={handleViewScan}
                  onAnalyze={handleAnalyzeScan}
                />
              ))}
            </div>
          ) : (
            <Card className={`text-center py-12 ${theme.cardBorder} bg-gradient-to-br from-white to-${theme.accentLight}`}>
              <div className={`w-16 h-16 mx-auto mb-4 bg-${theme.accentLight} rounded-full flex items-center justify-center`}>
                <ToothIcon className={`w-8 h-8 text-${theme.accent}`} />
              </div>
              <p className="text-gray-500 mb-4">No scans yet. Upload your first dental scan!</p>
              <button 
                onClick={() => setShowUploader(true)} 
                className={`${theme.buttonBg} ${theme.buttonHover} ${theme.buttonText} px-4 py-2 rounded-lg font-medium transition-colors`}
              >
                Upload Scan
              </button>
            </Card>
          )}
        </div>
        </div>
      </div>
    </div>
  );
}
