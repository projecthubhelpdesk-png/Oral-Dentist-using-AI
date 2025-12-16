import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { HealthScoreGauge } from '@/components/analysis/HealthScoreGauge';

describe('HealthScoreGauge', () => {
  it('renders the score', () => {
    render(<HealthScoreGauge score={85} />);
    expect(screen.getByText('85')).toBeInTheDocument();
  });

  it('shows "Good" label for scores >= 70', () => {
    render(<HealthScoreGauge score={75} />);
    expect(screen.getByText('Good')).toBeInTheDocument();
  });

  it('shows "Fair" label for scores >= 50', () => {
    render(<HealthScoreGauge score={60} />);
    expect(screen.getByText('Fair')).toBeInTheDocument();
  });

  it('shows "⚠️ Needs Attention" label for scores >= 30', () => {
    render(<HealthScoreGauge score={40} />);
    expect(screen.getByText('⚠️ Needs Attention')).toBeInTheDocument();
  });

  it('shows "🚨 Urgent Care Needed" label for scores < 30', () => {
    render(<HealthScoreGauge score={20} />);
    expect(screen.getByText('🚨 Urgent Care Needed')).toBeInTheDocument();
  });

  it('hides label when showLabel is false', () => {
    render(<HealthScoreGauge score={75} showLabel={false} />);
    expect(screen.queryByText('Good')).not.toBeInTheDocument();
  });

  it('rounds decimal scores', () => {
    render(<HealthScoreGauge score={78.6} />);
    expect(screen.getByText('79')).toBeInTheDocument();
  });
});
