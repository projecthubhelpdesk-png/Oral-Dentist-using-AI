import { createContext, useContext, useState, useEffect, useRef, ReactNode } from 'react';
import { authService, LoginData, RegisterData } from '@/services/auth';
import { getStoredRefreshToken, getStoredAccessToken, clearTokens } from '@/services/api';
import type { User } from '@/types';

interface AuthContextType {
  user: User | null;
  isLoading: boolean;
  isAuthenticated: boolean;
  login: (data: LoginData) => Promise<void>;
  register: (data: RegisterData) => Promise<void>;
  logout: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

// Decode JWT token to get user info (without verification - just for display)
function decodeToken(token: string): { sub: string; email: string; role: string; exp: number } | null {
  try {
    const parts = token.split('.');
    if (parts.length !== 3) return null;
    const payload = JSON.parse(atob(parts[1]));
    return payload;
  } catch {
    return null;
  }
}

// Check if token is expired (with 30 second buffer)
function isTokenExpired(token: string): boolean {
  const decoded = decodeToken(token);
  if (!decoded || !decoded.exp) return true;
  // Add 30 second buffer to account for clock skew
  return decoded.exp * 1000 < Date.now() + 30000;
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(() => {
    // Initialize user from token immediately to prevent flash of login page
    const accessToken = getStoredAccessToken();
    const refreshToken = getStoredRefreshToken();
    
    if (accessToken && refreshToken) {
      const decoded = decodeToken(accessToken);
      if (decoded) {
        return {
          id: decoded.sub,
          email: decoded.email,
          role: decoded.role as 'user' | 'dentist',
        } as User;
      }
    }
    return null;
  });
  const [isLoading, setIsLoading] = useState(true);
  const initRef = useRef(false);

  useEffect(() => {
    // Prevent double initialization in React StrictMode
    if (initRef.current) return;
    initRef.current = true;

    // Check for existing session and fetch full user data
    const initAuth = async () => {
      const accessToken = getStoredAccessToken();
      const refreshToken = getStoredRefreshToken();
      
      if (accessToken && refreshToken) {
        // User is already set from initial state, now try to get full user data
        try {
          const currentUser = await authService.getCurrentUser();
          setUser(currentUser);
        } catch (error: any) {
          // API call failed, but we still have token data
          // Check if we still have valid tokens after potential refresh
          const newAccessToken = getStoredAccessToken();
          const newRefreshToken = getStoredRefreshToken();
          
          if (newAccessToken && newRefreshToken) {
            // Tokens still exist (possibly refreshed), decode and use
            const decoded = decodeToken(newAccessToken);
            if (decoded) {
              setUser({
                id: decoded.sub,
                email: decoded.email,
                role: decoded.role as 'user' | 'dentist',
              } as User);
            }
          }
          console.warn('Could not fetch user from API, using token data:', error?.message);
        }
      }
      setIsLoading(false);
    };

    initAuth();
  }, []);

  const login = async (data: LoginData) => {
    const response = await authService.login(data);
    setUser(response.user);
  };

  const register = async (data: RegisterData) => {
    const response = await authService.register(data);
    setUser(response.user);
  };

  const logout = async () => {
    await authService.logout();
    setUser(null);
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        isLoading,
        isAuthenticated: !!user,
        login,
        register,
        logout,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}
