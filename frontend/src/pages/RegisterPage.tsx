import { useState, FormEvent } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '@/context/AuthContext';
import { useTheme } from '@/context/ThemeContext';
import { Input } from '@/components/ui/Input';

const ToothIcon = ({ className }: { className?: string }) => (
  <svg className={className} viewBox="0 0 24 24" fill="currentColor">
    <path d="M12 2C8.5 2 6 4.5 6 7c0 1.5.5 2.5 1 3.5.5 1 1 2 1 3.5 0 2-1 4-1 6 0 1.5 1 2 2 2s2-.5 2-2v-4h2v4c0 1.5 1 2 2 2s2-.5 2-2c0-2-1-4-1-6 0-1.5.5-2.5 1-3.5.5-1 1-2 1-3.5 0-2.5-2.5-5-6-5z"/>
  </svg>
);

export function RegisterPage() {
  const [formData, setFormData] = useState({
    email: '',
    password: '',
    confirmPassword: '',
    role: 'user' as 'user' | 'dentist',
  });
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [isLoading, setIsLoading] = useState(false);
  
  const { register } = useAuth();
  const { theme } = useTheme();
  const navigate = useNavigate();

  const validate = () => {
    const newErrors: Record<string, string> = {};
    
    if (!formData.email || !/\S+@\S+\.\S+/.test(formData.email)) {
      newErrors.email = 'Valid email is required';
    }
    
    if (!formData.password || formData.password.length < 8) {
      newErrors.password = 'Password must be at least 8 characters';
    }
    
    if (formData.password !== formData.confirmPassword) {
      newErrors.confirmPassword = 'Passwords do not match';
    }
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    
    if (!validate()) return;
    
    setIsLoading(true);

    try {
      await register({
        email: formData.email,
        password: formData.password,
        role: formData.role,
      });
      navigate('/dashboard');
    } catch (err: any) {
      // Check for feature disabled error
      const apiMessage = err?.response?.data?.message;
      const errorMessage = apiMessage || (err instanceof Error ? err.message : 'Registration failed. Please try again.');
      setErrors({ form: errorMessage });
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
          <ToothIcon className="absolute top-16 left-16 w-14 h-14 text-white/20" />
          <ToothIcon className="absolute top-32 right-24 w-10 h-10 text-white/15" />
          <ToothIcon className="absolute bottom-40 left-32 w-18 h-18 text-white/10" />
          <ToothIcon className="absolute bottom-16 right-16 w-12 h-12 text-white/20" />
          <div className="absolute top-1/4 left-1/4 w-32 h-32 rounded-full bg-white/5"></div>
          <div className="absolute bottom-1/4 right-1/4 w-48 h-48 rounded-full bg-white/5"></div>
        </div>

        {/* Content */}
        <div className={`relative z-10 flex flex-col justify-center px-12 ${theme.textOnGradient}`}>
          <div className="mb-8">
            <ToothIcon className="w-16 h-16 text-white" />
          </div>
          <h1 className="text-4xl font-bold mb-4">Join Oral Care AI</h1>
          <p className={`text-xl ${theme.textMuted} mb-8`}>
            Start your journey to better oral health with AI-powered insights.
          </p>
          <div className="space-y-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-full bg-white/20 flex items-center justify-center">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                </svg>
              </div>
              <span className="text-lg">Upload dental scans easily</span>
            </div>
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-full bg-white/20 flex items-center justify-center">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
              </div>
              <span className="text-lg">Get instant AI analysis</span>
            </div>
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-full bg-white/20 flex items-center justify-center">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
                </svg>
              </div>
              <span className="text-lg">Connect with dentists</span>
            </div>
          </div>
        </div>
      </div>

      {/* Right Side - Register Form */}
      <div className={`flex-1 flex items-center justify-center p-8 bg-gradient-to-br from-${theme.accentLight} to-white`}>
        <div className="w-full max-w-md">
          {/* Mobile Logo */}
          <div className="text-center mb-6 lg:hidden">
            <Link to="/" className="inline-flex items-center gap-2">
              <ToothIcon className={`w-10 h-10 text-${theme.accent}`} />
              <span className="text-2xl font-bold text-gray-900">Oral Care AI</span>
            </Link>
          </div>

          <div className={`bg-white rounded-2xl shadow-xl p-8 border ${theme.cardBorder}`}>
            <h1 className="text-2xl font-bold text-gray-900 mb-2">Create your account</h1>
            <p className="text-gray-500 mb-6">Start your oral health journey today</p>

            {errors.form && (
              <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg">
                <p className="text-sm text-red-600">{errors.form}</p>
              </div>
            )}

            <form onSubmit={handleSubmit} className="space-y-4">
              {/* Role Selection */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">I am a...</label>
                <div className="grid grid-cols-2 gap-3">
                  <button
                    type="button"
                    onClick={() => setFormData({ ...formData, role: 'user' })}
                    className={`p-4 border-2 rounded-xl text-center transition-all ${
                      formData.role === 'user'
                        ? `border-${theme.accent} bg-${theme.accentLight} text-${theme.accentDark} shadow-sm`
                        : `border-gray-200 hover:border-${theme.accent}/50 hover:bg-${theme.accentLight}/50`
                    }`}
                  >
                    <div className={`w-10 h-10 mx-auto mb-2 rounded-full bg-${theme.accentLight} flex items-center justify-center`}>
                      <svg className={`w-5 h-5 text-${theme.accent}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                      </svg>
                    </div>
                    <span className="block font-medium">Patient</span>
                    <span className="text-xs text-gray-500">Track my oral health</span>
                  </button>
                  <button
                    type="button"
                    onClick={() => setFormData({ ...formData, role: 'dentist' })}
                    className={`p-4 border-2 rounded-xl text-center transition-all ${
                      formData.role === 'dentist'
                        ? `border-${theme.accent} bg-${theme.accentLight} text-${theme.accentDark} shadow-sm`
                        : `border-gray-200 hover:border-${theme.accent}/50 hover:bg-${theme.accentLight}/50`
                    }`}
                  >
                    <div className={`w-10 h-10 mx-auto mb-2 rounded-full bg-${theme.accentLight} flex items-center justify-center`}>
                      <ToothIcon className={`w-5 h-5 text-${theme.accent}`} />
                    </div>
                    <span className="block font-medium">Dentist</span>
                    <span className="text-xs text-gray-500">Review patient scans</span>
                  </button>
                </div>
              </div>

              <Input
                label="Email"
                type="email"
                value={formData.email}
                onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                placeholder="you@example.com"
                error={errors.email}
                required
                autoComplete="email"
              />

              <Input
                label="Password"
                type="password"
                value={formData.password}
                onChange={(e) => setFormData({ ...formData, password: e.target.value })}
                placeholder="••••••••"
                error={errors.password}
                helperText="At least 8 characters"
                required
                autoComplete="new-password"
              />

              <Input
                label="Confirm Password"
                type="password"
                value={formData.confirmPassword}
                onChange={(e) => setFormData({ ...formData, confirmPassword: e.target.value })}
                placeholder="••••••••"
                error={errors.confirmPassword}
                required
                autoComplete="new-password"
              />

              <div className="flex items-start">
                <input
                  type="checkbox"
                  required
                  className={`mt-1 rounded border-gray-300 text-${theme.accent} focus:ring-${theme.accent}`}
                />
                <span className="ml-2 text-sm text-gray-600">
                  I agree to the{' '}
                  <a href="/terms" className={`text-${theme.accent} hover:underline`}>Terms of Service</a>
                  {' '}and{' '}
                  <a href="/privacy" className={`text-${theme.accent} hover:underline`}>Privacy Policy</a>
                </span>
              </div>

              <button
                type="submit"
                disabled={isLoading}
                className={`w-full ${theme.buttonBg} ${theme.buttonHover} ${theme.buttonText} py-2.5 px-4 rounded-lg font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 disabled:opacity-50`}
              >
                {isLoading ? 'Creating account...' : 'Create Account'}
              </button>
            </form>

            <div className="mt-6 text-center">
              <p className="text-sm text-gray-600">
                Already have an account?{' '}
                <Link to="/login" className={`text-${theme.accent} hover:text-${theme.accentHover} font-medium`}>
                  Sign in
                </Link>
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
