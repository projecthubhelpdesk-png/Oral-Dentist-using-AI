import { Card, CardHeader, CardTitle } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { HealthScoreGauge } from './HealthScoreGauge';
import type { AnalysisResult, Finding, Severity, ToothDetection, GPT4oGuidance } from '@/types';

interface AnalysisReportProps {
  analysis: AnalysisResult;
}

const severityConfig: Record<string, { label: string; variant: 'success' | 'warning' | 'danger' | 'info' | 'default' }> = {
  none: { label: 'None', variant: 'success' },
  minimal: { label: 'Minimal', variant: 'success' },
  mild: { label: 'Mild', variant: 'info' },
  moderate: { label: 'Moderate', variant: 'warning' },
  severe: { label: 'Severe', variant: 'danger' },
  // Handle any other values that might come from backend
  low: { label: 'Low', variant: 'info' },
  medium: { label: 'Medium', variant: 'warning' },
  high: { label: 'High', variant: 'danger' },
};

const defaultSeverity = { label: 'Unknown', variant: 'default' as const };

function FindingCard({ finding }: { finding: Finding }) {
  // Safely get severity config with fallback
  const severityKey = finding.severity?.toLowerCase() || 'none';
  const severity = severityConfig[severityKey] || defaultSeverity;
  
  return (
    <div className="flex items-start justify-between p-3 bg-gray-50 rounded-lg">
      <div>
        <p className="font-medium text-gray-900 capitalize">
          {finding.type?.replace(/_/g, ' ') || 'Unknown'}
        </p>
        <p className="text-sm text-gray-500 capitalize">
          Location: {finding.location?.replace(/_/g, ' ') || 'Not specified'}
        </p>
      </div>
      <div className="text-right">
        <Badge variant={severity.variant}>{severity.label}</Badge>
        <p className="text-xs text-gray-500 mt-1">
          {Math.round((finding.confidence || 0) * 100)}% confidence
        </p>
      </div>
    </div>
  );
}

function ToothDetectionCard({ detection }: { detection: ToothDetection }) {
  const issueColors: Record<string, string> = {
    cavity: 'bg-red-100 text-red-800 border-red-200',
    plaque: 'bg-orange-100 text-orange-800 border-orange-200',
    crooked_tooth: 'bg-yellow-100 text-yellow-800 border-yellow-200',
    healthy_tooth: 'bg-green-100 text-green-800 border-green-200',
    missing_tooth: 'bg-purple-100 text-purple-800 border-purple-200',
  };
  
  const colorClass = issueColors[detection.issue] || 'bg-gray-100 text-gray-800 border-gray-200';
  
  return (
    <div className={`flex items-center justify-between p-3 rounded-lg border ${colorClass}`}>
      <div className="flex items-center gap-3">
        <span className="text-2xl">🦷</span>
        <div>
          <p className="font-medium capitalize">
            {detection.tooth_id}
          </p>
          <p className="text-sm capitalize">
            {detection.issue.replace(/_/g, ' ')}
          </p>
        </div>
      </div>
      <div className="text-right">
        <p className="font-semibold">
          {Math.round(detection.confidence * 100)}%
        </p>
        <p className="text-xs opacity-75">confidence</p>
      </div>
    </div>
  );
}

// GPT-4o Guidance Section Component
function GuidanceSection({ 
  title, 
  icon, 
  content, 
  variant = 'default' 
}: { 
  title: string; 
  icon: string; 
  content: string; 
  variant?: 'default' | 'warning' | 'success' | 'danger' | 'info';
}) {
  if (!content || content.trim() === '') return null;
  
  const variantStyles: Record<string, string> = {
    default: 'bg-gray-50 border-gray-200',
    warning: 'bg-amber-50 border-amber-200',
    success: 'bg-green-50 border-green-200',
    danger: 'bg-red-50 border-red-200',
    info: 'bg-blue-50 border-blue-200',
  };
  
  // Convert markdown-like content to formatted text
  const formatContent = (text: string) => {
    return text.split('\n').map((line, idx) => {
      const trimmed = line.trim();
      if (!trimmed) return null;
      
      // Handle bullet points
      if (trimmed.startsWith('-') || trimmed.startsWith('•') || trimmed.startsWith('*')) {
        return (
          <li key={idx} className="ml-4 text-gray-700">
            {trimmed.replace(/^[-•*]\s*/, '')}
          </li>
        );
      }
      
      // Handle numbered items
      if (/^\d+\./.test(trimmed)) {
        return (
          <li key={idx} className="ml-4 text-gray-700 list-decimal">
            {trimmed.replace(/^\d+\.\s*/, '')}
          </li>
        );
      }
      
      // Handle headers (##)
      if (trimmed.startsWith('##')) {
        return (
          <p key={idx} className="font-semibold text-gray-800 mt-2">
            {trimmed.replace(/^#+\s*/, '')}
          </p>
        );
      }
      
      return <p key={idx} className="text-gray-700">{trimmed}</p>;
    });
  };
  
  return (
    <div className={`p-4 rounded-lg border ${variantStyles[variant]}`}>
      <h4 className="font-semibold text-gray-900 mb-3 flex items-center gap-2">
        <span>{icon}</span>
        {title}
      </h4>
      <div className="space-y-1 text-sm">
        {formatContent(content)}
      </div>
    </div>
  );
}

export function AnalysisReport({ analysis }: AnalysisReportProps) {
  const findings = analysis.findings || [];
  const hasFindings = findings.length > 0;
  const significantFindings = findings.filter(
    (f) => f.severity !== 'none' && f.severity !== 'minimal'
  );
  const recommendations = analysis.recommendations || [];
  const gpt4oGuidance = analysis.gpt4o_guidance;
  const hasGPT4oAnalysis = gpt4oGuidance && (
    gpt4oGuidance.exact_complaint || 
    gpt4oGuidance.detailed_findings || 
    gpt4oGuidance.immediate_actions
  );

  return (
    <div className="space-y-6">
      {/* Score Overview */}
      <Card>
        <div className="flex flex-col md:flex-row items-center gap-6">
          <HealthScoreGauge score={analysis.overallScore} size="lg" />
          
          <div className="flex-1 text-center md:text-left">
            <h2 className="text-xl font-semibold text-gray-900 mb-2">
              AI Analysis Complete
              {hasGPT4oAnalysis && (
                <Badge variant="info" size="sm" className="ml-2">GPT-4o Enhanced</Badge>
              )}
            </h2>
            <p className="text-gray-600 mb-4">
              Analyzed using {analysis.modelType === 'basic_rgb' ? 'Basic RGB' : 'Advanced Spectral'} model
              (v{analysis.modelVersion})
            </p>
            <div className="flex flex-wrap gap-4 justify-center md:justify-start">
              <div>
                <p className="text-sm text-gray-500">Confidence</p>
                <p className="font-semibold">{Math.round((analysis.confidenceScore || 0) * 100)}%</p>
              </div>
              <div>
                <p className="text-sm text-gray-500">Findings</p>
                <p className="font-semibold">{findings.length}</p>
              </div>
              <div>
                <p className="text-sm text-gray-500">Attention Areas</p>
                <p className="font-semibold">{significantFindings.length}</p>
              </div>
            </div>
          </div>
        </div>
      </Card>

      {/* GPT-4o Exact Complaint / Diagnosis */}
      {gpt4oGuidance?.exact_complaint && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              🔍 Exact Complaint / Diagnosis
              <Badge variant="info" size="sm">AI Analysis</Badge>
            </CardTitle>
          </CardHeader>
          <GuidanceSection
            title=""
            icon=""
            content={gpt4oGuidance.exact_complaint}
            variant="danger"
          />
        </Card>
      )}

      {/* GPT-4o Detailed Findings */}
      {gpt4oGuidance?.detailed_findings && (
        <Card>
          <CardHeader>
            <CardTitle>🔬 Detailed Findings</CardTitle>
          </CardHeader>
          <GuidanceSection
            title=""
            icon=""
            content={gpt4oGuidance.detailed_findings}
            variant="info"
          />
        </Card>
      )}

      {/* GPT-4o What This Means */}
      {gpt4oGuidance?.what_this_means && (
        <Card>
          <CardHeader>
            <CardTitle>📖 What This Means For You</CardTitle>
          </CardHeader>
          <GuidanceSection
            title=""
            icon=""
            content={gpt4oGuidance.what_this_means}
            variant="warning"
          />
        </Card>
      )}

      {/* GPT-4o Immediate Actions */}
      {gpt4oGuidance?.immediate_actions && (
        <Card>
          <CardHeader>
            <CardTitle>⚡ Immediate Actions To Take</CardTitle>
          </CardHeader>
          <GuidanceSection
            title=""
            icon=""
            content={gpt4oGuidance.immediate_actions}
            variant="danger"
          />
        </Card>
      )}

      {/* GPT-4o Treatment Options */}
      {gpt4oGuidance?.treatment_options && (
        <Card>
          <CardHeader>
            <CardTitle>🏥 Treatment Options</CardTitle>
          </CardHeader>
          <GuidanceSection
            title=""
            icon=""
            content={gpt4oGuidance.treatment_options}
            variant="info"
          />
        </Card>
      )}

      {/* GPT-4o Home Care Routine */}
      {gpt4oGuidance?.home_care_routine && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              🏠 Home Care Routine
              <Badge variant="success" size="sm">Personalized</Badge>
            </CardTitle>
          </CardHeader>
          <GuidanceSection
            title=""
            icon=""
            content={gpt4oGuidance.home_care_routine}
            variant="success"
          />
        </Card>
      )}

      {/* GPT-4o Prevention Tips */}
      {gpt4oGuidance?.prevention_tips && (
        <Card>
          <CardHeader>
            <CardTitle>🛡️ Prevention Tips</CardTitle>
          </CardHeader>
          <GuidanceSection
            title=""
            icon=""
            content={gpt4oGuidance.prevention_tips}
            variant="success"
          />
        </Card>
      )}

      {/* Findings */}
      <Card>
        <CardHeader>
          <CardTitle>Findings</CardTitle>
        </CardHeader>
        
        {hasFindings ? (
          <div className="space-y-3">
            {findings.map((finding, index) => (
              <FindingCard key={index} finding={finding} />
            ))}
          </div>
        ) : (
          <p className="text-gray-500 text-center py-4">
            No significant findings detected. Great oral health!
          </p>
        )}
      </Card>

      {/* Enhanced: Tooth-Level Detections */}
      {analysis.enhanced_features?.tooth_detections && analysis.enhanced_features.tooth_detections.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              🦷 Tooth-Level Analysis
              <Badge variant="info" size="sm">Enhanced</Badge>
            </CardTitle>
          </CardHeader>
          
          <div className="space-y-3">
            {analysis.enhanced_features.tooth_detections.map((detection, index) => (
              <ToothDetectionCard key={index} detection={detection} />
            ))}
          </div>
          
          {analysis.enhanced_features.affected_teeth && analysis.enhanced_features.affected_teeth.length > 0 && (
            <div className="mt-4 pt-4 border-t">
              <p className="text-sm text-gray-600">
                <span className="font-medium">Affected teeth (FDI): </span>
                {analysis.enhanced_features.affected_teeth.join(', ')}
              </p>
            </div>
          )}
        </Card>
      )}

      {/* Recommendations */}
      {recommendations.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Recommendations</CardTitle>
          </CardHeader>
          
          <ul className="space-y-2">
            {recommendations.map((rec, index) => (
              <li key={index} className="flex items-start gap-3">
                <span className="flex-shrink-0 w-6 h-6 bg-primary-100 text-primary-600 rounded-full flex items-center justify-center text-sm font-medium">
                  {index + 1}
                </span>
                <span className="text-gray-700">{rec}</span>
              </li>
            ))}
          </ul>
        </Card>
      )}

      {/* Enhanced: Home Care Tips */}
      {analysis.enhanced_features?.home_care_tips && analysis.enhanced_features.home_care_tips.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              🏠 Home Care Tips
              <Badge variant="success" size="sm">Personalized</Badge>
            </CardTitle>
          </CardHeader>
          
          <ul className="space-y-2">
            {analysis.enhanced_features.home_care_tips.map((tip, index) => (
              <li key={index} className="flex items-start gap-3">
                <span className="flex-shrink-0 text-green-500">✓</span>
                <span className="text-gray-700">{tip}</span>
              </li>
            ))}
          </ul>
        </Card>
      )}

      {/* Disclaimer */}
      <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
        <p className="text-sm text-yellow-800">
          <strong>Disclaimer:</strong> This AI analysis is for informational purposes only and does not
          constitute medical advice. Please consult with a licensed dental professional for diagnosis
          and treatment recommendations.
        </p>
      </div>
    </div>
  );
}
