import { useState, FormEvent } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '@/context/AuthContext';
import { useTheme } from '@/context/ThemeContext';
import { Input } from '@/components/ui/Input';
import { Button } from '@/components/ui/Button';

const ToothIcon = ({ className }: { className?: string }) => (
  <svg className={className} viewBox="0 0 24 24" fill="currentColor">
    <path d="M12 2C8.5 2 6 4.5 6 7c0 1.5.5 2.5 1 3.5.5 1 1 2 1 3.5 0 2-1 4-1 6 0 1.5 1 2 2 2s2-.5 2-2v-4h2v4c0 1.5 1 2 2 2s2-.5 2-2c0-2-1-4-1-6 0-1.5.5-2.5 1-3.5.5-1 1-2 1-3.5 0-2.5-2.5-5-6-5z"/>
  </svg>
);

export function LoginPage() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  
  const { login } = useAuth();
  const { theme } = useTheme();
  const navigate = useNavigate();

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError('');
    setIsLoading(true);

    try {
      await login({ email, password });
      navigate('/dashboard');
    } catch (err: any) {
      // Check for feature disabled error
      const apiMessage = err?.response?.data?.message;
      const errorMessage = apiMessage || (err instanceof Error ? err.message : 'Invalid email or password');
      setError(errorMessage);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex">
      {/* Left Side - Themed Gradient Background */}
      <div className={`hidden lg:flex lg:w-1/2 bg-gradient-to-br ${theme.gradientFrom} ${theme.gradientVia} ${theme.gradientTo} relative overflow-hidden`}>
        {/* Pattern Overlay */}
        <div className="absolute inset-0 bg-dental-pattern opacity-20"></div>
        
        {/* Floating Dental Icons */}
        <div className="absolute inset-0">
          <ToothIcon className="absolute top-20 left-20 w-16 h-16 text-white/20" />
          <ToothIcon className="absolute top-40 right-32 w-12 h-12 text-white/15" />
          <ToothIcon className="absolute bottom-32 left-40 w-20 h-20 text-white/10" />
          <ToothIcon className="absolute bottom-20 right-20 w-14 h-14 text-white/20" />
          <div className="absolute top-1/3 left-1/3 w-32 h-32 rounded-full bg-white/5"></div>
          <div className="absolute bottom-1/4 right-1/4 w-48 h-48 rounded-full bg-white/5"></div>
        </div>

        {/* Content */}
        <div className={`relative z-10 flex flex-col justify-center px-12 ${theme.textOnGradient}`}>
          <div className="mb-8">
            <ToothIcon className="w-16 h-16 text-white" />
          </div>
          <h1 className="text-4xl font-bold mb-4">Welcome to Oral Care AI</h1>
          <p className={`text-xl ${theme.textMuted} mb-8`}>
            Advanced AI-powered dental analysis for better oral health outcomes.
          </p>
          <div className="space-y-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-full bg-white/20 flex items-center justify-center">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
              </div>
              <span className="text-lg">AI-Powered Scan Analysis</span>
            </div>
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-full bg-white/20 flex items-center justify-center">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
              </div>
              <span className="text-lg">Connect with Dental Professionals</span>
            </div>
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-full bg-white/20 flex items-center justify-center">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
              </div>
              <span className="text-lg">Track Your Oral Health Journey</span>
            </div>
          </div>
        </div>
      </div>

      {/* Right Side - Login Form */}
      <div className={`flex-1 flex items-center justify-center p-8 bg-gradient-to-br from-${theme.accentLight} to-white`}>
        <div className="w-full max-w-md">
          {/* Mobile Logo */}
          <div className="text-center mb-8 lg:hidden">
            <Link to="/" className="inline-flex items-center gap-2">
              <ToothIcon className={`w-10 h-10 text-${theme.accent}`} />
              <span className="text-2xl font-bold text-gray-900">Oral Care AI</span>
            </Link>
          </div>

          <div className={`bg-white rounded-2xl shadow-xl p-8 border ${theme.cardBorder}`}>
            <h1 className="text-2xl font-bold text-gray-900 mb-2">Welcome back</h1>
            <p className="text-gray-500 mb-6">Sign in to continue to your account</p>

            {error && (
              <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg">
                <p className="text-sm text-red-600">{error}</p>
              </div>
            )}

            <form onSubmit={handleSubmit} className="space-y-5">
              <Input
                label="Email"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="you@example.com"
                required
                autoComplete="email"
              />

              <Input
                label="Password"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••"
                required
                autoComplete="current-password"
              />

              <div className="flex items-center justify-between">
                <label className="flex items-center">
                  <input type="checkbox" className={`rounded border-gray-300 text-${theme.accent} focus:ring-${theme.accent}`} />
                  <span className="ml-2 text-sm text-gray-600">Remember me</span>
                </label>
                <a href="/forgot-password" className={`text-sm text-${theme.accent} hover:text-${theme.accentHover} font-medium`}>
                  Forgot password?
                </a>
              </div>

              <button
                type="submit"
                disabled={isLoading}
                className={`w-full ${theme.buttonBg} ${theme.buttonHover} ${theme.buttonText} py-2.5 px-4 rounded-lg font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 disabled:opacity-50`}
              >
                {isLoading ? 'Signing in...' : 'Sign in'}
              </button>
            </form>

            <div className="mt-6 text-center">
              <p className="text-sm text-gray-600">
                Don't have an account?{' '}
                <Link to="/register" className={`text-${theme.accent} hover:text-${theme.accentHover} font-medium`}>
                  Sign up for free
                </Link>
              </p>
            </div>
          </div>

          {/* Demo credentials */}
          <div className={`mt-6 p-4 bg-${theme.accentLight} rounded-xl border border-${theme.accent}/20`}>
            <p className={`text-sm text-${theme.accentDark} font-medium mb-2`}>Demo Credentials:</p>
            <p className={`text-xs text-${theme.accentDark}/80`}>User: john.doe@example.com / password123</p>
            <p className={`text-xs text-${theme.accentDark}/80`}>Dentist: dr.sarah.chen@dental.com / password123</p>
          </div>
        </div>
      </div>
    </div>
  );
}
