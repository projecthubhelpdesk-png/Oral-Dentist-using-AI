import { useState } from 'react';
import { Button } from '@/components/ui/Button';
import type { AnalysisResult, Scan } from '@/types';

interface DownloadReportProps {
  scan: Scan;
  analysis: AnalysisResult;
}

export function DownloadReport({ scan, analysis }: DownloadReportProps) {
  const [isGenerating, setIsGenerating] = useState(false);

  const getSeverityColor = (severity: string) => {
    switch (severity?.toLowerCase()) {
      case 'severe':
      case 'high':
        return '#dc2626';
      case 'moderate':
      case 'medium':
        return '#f59e0b';
      case 'mild':
      case 'low':
        return '#3b82f6';
      default:
        return '#22c55e';
    }
  };

  const getScoreColor = (score: number) => {
    if (score >= 70) return '#22c55e';
    if (score >= 40) return '#f59e0b';
    return '#dc2626';
  };

  // Convert image URL to base64
  const getImageAsBase64 = async (imageUrl: string): Promise<string> => {
    try {
      const token = localStorage.getItem('accessToken');
      
      // Build full URL - use the API base URL
      const apiBaseUrl = import.meta.env.VITE_API_URL || 'http://localhost/oral-care-ai/backend-php/api';
      
      let fullUrl = imageUrl;
      // If it's a relative URL starting with /oral-care-ai, use localhost
      if (imageUrl.startsWith('/oral-care-ai')) {
        fullUrl = `http://localhost${imageUrl}`;
      } else if (imageUrl.startsWith('/')) {
        // If it starts with /, prepend the API base (without /api)
        const baseWithoutApi = apiBaseUrl.replace('/api', '');
        fullUrl = `${baseWithoutApi}${imageUrl}`;
      }
      
      // Add token
      fullUrl = fullUrl.includes('?') 
        ? `${fullUrl}&token=${token}` 
        : `${fullUrl}?token=${token}`;
      
      console.log('Fetching image from:', fullUrl);
      
      const response = await fetch(fullUrl, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      
      if (!response.ok) {
        console.error('Image fetch failed:', response.status, response.statusText);
        return '';
      }
      
      const blob = await response.blob();
      console.log('Image blob size:', blob.size, 'type:', blob.type);
      
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onloadend = () => {
          const result = reader.result as string;
          console.log('Base64 result length:', result?.length);
          resolve(result);
        };
        reader.onerror = (e) => {
          console.error('FileReader error:', e);
          reject(e);
        };
        reader.readAsDataURL(blob);
      });
    } catch (error) {
      console.error('Failed to load image:', error);
      return '';
    }
  };

  const downloadReport = async () => {
    setIsGenerating(true);
    
    try {
      // First, convert the image to base64
      const imageBase64 = scan.imageUrl ? await getImageAsBase64(scan.imageUrl) : '';
      
      // Create a new window for the report
      const reportWindow = window.open('', '_blank');
      if (!reportWindow) {
        alert('Please allow popups to download the report');
        setIsGenerating(false);
        return;
      }

      const gpt4o = analysis.gpt4o_guidance;
      const scoreColor = getScoreColor(analysis.overallScore);
      
      // Generate Report ID
      const date = new Date(scan.uploadedAt);
      const dateStr = date.toISOString().slice(0, 10).replace(/-/g, '');
      const shortId = analysis.id.substring(0, 8).toUpperCase();
      const reportId = `RPT-${dateStr}-${shortId}`;
      
      // Build the HTML report
      const reportHTML = `
<!DOCTYPE html>
<html>
<head>
  <title>Dental AI Report - ${reportId}</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { 
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
      background: #f5f5f5; 
      padding: 20px;
      color: #333;
    }
    .report { 
      max-width: 800px; 
      margin: 0 auto; 
      background: white; 
      border-radius: 12px; 
      box-shadow: 0 4px 20px rgba(0,0,0,0.1);
      overflow: hidden;
    }
    .header { 
      background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%); 
      color: white; 
      padding: 30px; 
      text-align: center;
    }
    .header h1 { font-size: 28px; margin-bottom: 5px; }
    .header p { opacity: 0.9; font-size: 14px; }
    .content { padding: 30px; }
    .scan-info { 
      display: flex; 
      gap: 30px; 
      margin-bottom: 30px;
      flex-wrap: wrap;
    }
    .scan-image { 
      flex: 1; 
      min-width: 250px;
    }
    .scan-image img { 
      width: 100%; 
      border-radius: 12px; 
      border: 3px solid #e5e7eb;
    }
    .score-card { 
      flex: 1; 
      min-width: 200px;
      text-align: center;
      padding: 20px;
      background: #f8fafc;
      border-radius: 12px;
    }
    .score-circle {
      width: 120px;
      height: 120px;
      border-radius: 50%;
      border: 8px solid ${scoreColor};
      display: flex;
      align-items: center;
      justify-content: center;
      margin: 0 auto 15px;
      background: white;
    }
    .score-value { font-size: 36px; font-weight: bold; color: ${scoreColor}; }
    .score-label { color: #6b7280; font-size: 14px; }
    .confidence { margin-top: 10px; color: #6b7280; font-size: 13px; }
    .section { 
      margin-bottom: 25px; 
      padding: 20px; 
      background: #f8fafc; 
      border-radius: 12px;
      border-left: 4px solid #3b82f6;
    }
    .section.warning { border-left-color: #f59e0b; background: #fffbeb; }
    .section.danger { border-left-color: #dc2626; background: #fef2f2; }
    .section.success { border-left-color: #22c55e; background: #f0fdf4; }
    .section h3 { 
      color: #1f2937; 
      margin-bottom: 12px; 
      font-size: 16px;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .section p, .section li { 
      color: #4b5563; 
      line-height: 1.7; 
      font-size: 14px;
    }
    .section ul { padding-left: 20px; }
    .section li { margin-bottom: 6px; }
    .findings-grid { display: flex; flex-wrap: wrap; gap: 10px; }
    .finding-badge {
      padding: 8px 16px;
      border-radius: 20px;
      font-size: 13px;
      font-weight: 500;
    }
    .meta-info {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 15px;
      margin-bottom: 25px;
    }
    .meta-item {
      padding: 12px;
      background: #f8fafc;
      border-radius: 8px;
    }
    .meta-item label { font-size: 12px; color: #6b7280; display: block; }
    .meta-item span { font-weight: 600; color: #1f2937; }
    .disclaimer { 
      background: #fef3c7; 
      border: 1px solid #fcd34d; 
      padding: 15px; 
      border-radius: 8px; 
      font-size: 12px; 
      color: #92400e;
      margin-top: 20px;
    }
    .footer { 
      text-align: center; 
      padding: 20px; 
      background: #f8fafc; 
      color: #6b7280; 
      font-size: 12px;
    }
    .print-btn {
      position: fixed;
      top: 20px;
      right: 20px;
      background: #3b82f6;
      color: white;
      border: none;
      padding: 12px 24px;
      border-radius: 8px;
      cursor: pointer;
      font-size: 14px;
      font-weight: 500;
      box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
    }
    .print-btn:hover { background: #2563eb; }
    @media print {
      .print-btn { display: none; }
      body { padding: 0; background: white; }
      .report { box-shadow: none; }
    }
  </style>
</head>
<body>
  <button class="print-btn" onclick="window.print()">🖨️ Print / Save as PDF</button>
  
  <div class="report">
    <div class="header">
      <h1>🦷 Dental AI Screening Report</h1>
      <p style="font-size: 20px; font-weight: bold; margin: 10px 0; letter-spacing: 1px;">${reportId}</p>
      <p>Generated on ${new Date().toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric', hour: '2-digit', minute: '2-digit' })}</p>
    </div>
    
    <div class="content">
      <!-- Report ID Banner -->
      <div style="background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%); border: 2px solid #3b82f6; border-radius: 12px; padding: 20px; margin-bottom: 25px; text-align: center;">
        <p style="font-size: 12px; color: #6b7280; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 5px;">Report ID</p>
        <p style="font-size: 28px; font-weight: bold; color: #1d4ed8; font-family: monospace; letter-spacing: 2px;">${reportId}</p>
        <p style="font-size: 11px; color: #6b7280; margin-top: 8px;">Share this ID with your dentist for reference</p>
      </div>
      
      <div class="meta-info">
        <div class="meta-item">
          <label>Report ID</label>
          <span style="font-family: monospace;">${reportId}</span>
        </div>
        <div class="meta-item">
          <label>Scan Date</label>
          <span>${new Date(scan.uploadedAt).toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' })}</span>
        </div>
        <div class="meta-item">
          <label>Scan Type</label>
          <span>${scan.scanType === 'basic_rgb' ? 'Basic RGB' : 'Advanced Spectral'}</span>
        </div>
        <div class="meta-item">
          <label>Model Version</label>
          <span>${analysis.modelVersion}</span>
        </div>
        <div class="meta-item">
          <label>Analysis ID</label>
          <span>${analysis.id.substring(0, 8)}...</span>
        </div>
      </div>

      <div class="scan-info">
        <div class="scan-image">
          ${imageBase64 ? `<img src="${imageBase64}" alt="Dental Scan" />` : '<p style="color: #6b7280; text-align: center; padding: 40px;">Image not available</p>'}
        </div>
        <div class="score-card">
          <div class="score-circle">
            <span class="score-value">${Math.round(analysis.overallScore)}</span>
          </div>
          <div class="score-label">Health Score</div>
          <div class="confidence">Confidence: ${(analysis.confidenceScore * 100).toFixed(1)}%</div>
        </div>
      </div>

      ${gpt4o?.exact_complaint ? `
      <div class="section danger">
        <h3>🔍 Diagnosis</h3>
        <p>${gpt4o.exact_complaint.replace(/\n/g, '<br>')}</p>
      </div>
      ` : ''}

      ${analysis.findings && analysis.findings.length > 0 ? `
      <div class="section">
        <h3>📋 Findings</h3>
        <div class="findings-grid">
          ${analysis.findings.map(f => `
            <span class="finding-badge" style="background: ${getSeverityColor(f.severity)}20; color: ${getSeverityColor(f.severity)}; border: 1px solid ${getSeverityColor(f.severity)}40;">
              ${f.type.replace(/_/g, ' ')} - ${f.severity}
            </span>
          `).join('')}
        </div>
      </div>
      ` : ''}

      ${gpt4o?.detailed_findings ? `
      <div class="section">
        <h3>🔬 Detailed Findings</h3>
        <p>${gpt4o.detailed_findings.replace(/\n/g, '<br>')}</p>
      </div>
      ` : ''}

      ${gpt4o?.what_this_means ? `
      <div class="section warning">
        <h3>📖 What This Means</h3>
        <p>${gpt4o.what_this_means.replace(/\n/g, '<br>')}</p>
      </div>
      ` : ''}

      ${gpt4o?.immediate_actions ? `
      <div class="section danger">
        <h3>⚡ Immediate Actions</h3>
        <p>${gpt4o.immediate_actions.replace(/\n/g, '<br>')}</p>
      </div>
      ` : ''}

      ${gpt4o?.treatment_options ? `
      <div class="section">
        <h3>🏥 Treatment Options</h3>
        <p>${gpt4o.treatment_options.replace(/\n/g, '<br>')}</p>
      </div>
      ` : ''}

      ${gpt4o?.home_care_routine ? `
      <div class="section success">
        <h3>🏠 Home Care Routine</h3>
        <p>${gpt4o.home_care_routine.replace(/\n/g, '<br>')}</p>
      </div>
      ` : ''}

      ${gpt4o?.prevention_tips ? `
      <div class="section success">
        <h3>🛡️ Prevention Tips</h3>
        <p>${gpt4o.prevention_tips.replace(/\n/g, '<br>')}</p>
      </div>
      ` : ''}

      ${analysis.recommendations && analysis.recommendations.length > 0 ? `
      <div class="section">
        <h3>💡 Recommendations</h3>
        <ul>
          ${analysis.recommendations.map(r => `<li>${r}</li>`).join('')}
        </ul>
      </div>
      ` : ''}

      <div class="disclaimer">
        <strong>⚠️ Disclaimer:</strong> This AI analysis is for informational purposes only and does not constitute medical advice. 
        Please consult with a licensed dental professional for diagnosis and treatment recommendations.
      </div>
    </div>
    
    <div class="footer">
      <p>Oral Care AI - Powered by EfficientNet-B4 & GPT-4o</p>
      <p>Report generated automatically. For professional use only.</p>
    </div>
  </div>
</body>
</html>
      `;

      reportWindow.document.write(reportHTML);
      reportWindow.document.close();
      
    } catch (error) {
      console.error('Failed to generate report:', error);
      alert('Failed to generate report. Please try again.');
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <Button
      onClick={downloadReport}
      disabled={isGenerating}
      variant="secondary"
      className="flex items-center gap-2"
    >
      {isGenerating ? (
        <>
          <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-primary-600" />
          Generating...
        </>
      ) : (
        <>
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
          Download Report
        </>
      )}
    </Button>
  );
}
