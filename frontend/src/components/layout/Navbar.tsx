import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '@/context/AuthContext';
import { Button } from '@/components/ui/Button';

export function Navbar() {
  const { user, isAuthenticated, logout } = useAuth();
  const navigate = useNavigate();

  const handleLogout = async () => {
    await logout();
    navigate('/login');
  };

  return (
    <nav className="bg-white border-b border-gray-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          {/* Logo */}
          <div className="flex items-center">
            <Link to="/" className="flex items-center gap-2">
              <svg className="w-8 h-8 text-primary-600" viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 2C8.5 2 6 4.5 6 7c0 1.5.5 2.5 1 3.5.5 1 1 2 1 3.5 0 2-1 4-1 6 0 1.5 1 2 2 2s2-.5 2-2v-4h2v4c0 1.5 1 2 2 2s2-.5 2-2c0-2-1-4-1-6 0-1.5.5-2.5 1-3.5.5-1 1-2 1-3.5 0-2.5-2.5-5-6-5z"/>
              </svg>
              <span className="text-xl font-bold text-gray-900">Oral Care AI</span>
            </Link>
          </div>

          {/* Navigation Links */}
          {isAuthenticated && (
            <div className="hidden md:flex items-center gap-6">
              <Link to="/dashboard" className="text-gray-600 hover:text-gray-900">
                Dashboard
              </Link>
              {user?.role === 'user' && (
                <Link to="/scans" className="text-gray-600 hover:text-gray-900">
                  My Scans
                </Link>
              )}
              {user?.role === 'dentist' && (
                <Link to="/connections" className="text-gray-600 hover:text-gray-900">
                  My Patients
                </Link>
              )}
              {user?.role === 'user' && (
                <>
                  <Link to="/connections" className="text-gray-600 hover:text-gray-900">
                    Connections
                  </Link>
                  <Link to="/dentists" className="text-gray-600 hover:text-gray-900">
                    Find Dentist
                  </Link>
                </>
              )}
              <Link to="/profile" className="text-gray-600 hover:text-gray-900">
                Profile
              </Link>
            </div>
          )}

          {/* User Menu */}
          <div className="flex items-center gap-4">
            {isAuthenticated ? (
              <>
                <div className="hidden sm:flex items-center gap-2">
                  <div className="w-8 h-8 bg-primary-100 rounded-full flex items-center justify-center">
                    <span className="text-primary-600 font-medium text-sm">
                      {user?.email?.charAt(0).toUpperCase()}
                    </span>
                  </div>
                  <span className="text-sm text-gray-700">{user?.email}</span>
                </div>
                <Button variant="ghost" size="sm" onClick={handleLogout}>
                  Logout
                </Button>
              </>
            ) : (
              <>
                <Link to="/login">
                  <Button variant="ghost" size="sm">Login</Button>
                </Link>
                <Link to="/register">
                  <Button size="sm">Sign Up</Button>
                </Link>
              </>
            )}
          </div>
        </div>
      </div>
    </nav>
  );
}
