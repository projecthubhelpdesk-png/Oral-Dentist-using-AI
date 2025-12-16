import { clsx } from 'clsx';

interface HealthScoreGaugeProps {
  score: number;
  size?: 'sm' | 'md' | 'lg';
  showLabel?: boolean;
}

function getScoreColor(score: number): string {
  if (score >= 70) return 'text-green-500';
  if (score >= 50) return 'text-yellow-500';
  if (score >= 30) return 'text-orange-500';
  return 'text-red-500';
}

function getScoreLabel(score: number): string {
  if (score >= 70) return 'Good';
  if (score >= 50) return 'Fair';
  if (score >= 30) return '⚠️ Needs Attention';
  return '🚨 Urgent Care Needed';
}

function getScoreBgColor(score: number): string {
  if (score >= 70) return 'stroke-green-500';
  if (score >= 50) return 'stroke-yellow-500';
  if (score >= 30) return 'stroke-orange-500';
  return 'stroke-red-500';
}

export function HealthScoreGauge({ score, size = 'md', showLabel = true }: HealthScoreGaugeProps) {
  // Handle undefined/NaN scores
  const safeScore = typeof score === 'number' && !isNaN(score) ? score : 0;
  
  const sizes = {
    sm: { width: 80, strokeWidth: 6, fontSize: 'text-lg' },
    md: { width: 120, strokeWidth: 8, fontSize: 'text-2xl' },
    lg: { width: 160, strokeWidth: 10, fontSize: 'text-4xl' },
  };

  const { width, strokeWidth, fontSize } = sizes[size];
  const radius = (width - strokeWidth) / 2;
  const circumference = radius * 2 * Math.PI;
  const offset = circumference - (safeScore / 100) * circumference;

  return (
    <div className="flex flex-col items-center">
      <div className="relative" style={{ width, height: width }}>
        <svg className="transform -rotate-90" width={width} height={width}>
          {/* Background circle */}
          <circle
            cx={width / 2}
            cy={width / 2}
            r={radius}
            fill="none"
            stroke="currentColor"
            strokeWidth={strokeWidth}
            className="text-gray-200"
          />
          {/* Progress circle */}
          <circle
            cx={width / 2}
            cy={width / 2}
            r={radius}
            fill="none"
            strokeWidth={strokeWidth}
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            className={clsx('transition-all duration-1000 ease-out', getScoreBgColor(score))}
          />
        </svg>
        {/* Score text */}
        <div className="absolute inset-0 flex items-center justify-center">
          <span className={clsx('font-bold', fontSize, getScoreColor(safeScore))}>
            {Math.round(safeScore)}
          </span>
        </div>
      </div>
      
      {showLabel && (
        <div className="mt-2 text-center">
          <p className={clsx('font-medium', getScoreColor(safeScore))}>{getScoreLabel(safeScore)}</p>
          <p className="text-sm text-gray-500">Oral Health Score</p>
        </div>
      )}
    </div>
  );
}
