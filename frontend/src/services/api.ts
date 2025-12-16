import axios, { AxiosError, InternalAxiosRequestConfig } from 'axios';
import type { ApiError } from '@/types';

const API_URL = import.meta.env.VITE_API_URL || '/api';

export const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Token management
let accessToken: string | null = null;
let refreshToken: string | null = null;

export const setTokens = (access: string, refresh: string) => {
  accessToken = access;
  refreshToken = refresh;
  localStorage.setItem('accessToken', access);
  localStorage.setItem('refreshToken', refresh);
};

export const clearTokens = () => {
  accessToken = null;
  refreshToken = null;
  localStorage.removeItem('accessToken');
  localStorage.removeItem('refreshToken');
};

export const getStoredRefreshToken = () => {
  return localStorage.getItem('refreshToken');
};

export const getStoredAccessToken = () => {
  return localStorage.getItem('accessToken');
};

export const getTokens = () => {
  const access = accessToken || localStorage.getItem('accessToken');
  const refresh = refreshToken || localStorage.getItem('refreshToken');
  if (access && refresh) {
    return { accessToken: access, refreshToken: refresh };
  }
  return null;
};

// Request interceptor - add auth header
api.interceptors.request.use((config: InternalAxiosRequestConfig) => {
  // Always check localStorage for the latest token
  const token = accessToken || localStorage.getItem('accessToken');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Track if we're currently refreshing to prevent infinite loops
let isRefreshing = false;
let refreshSubscribers: ((token: string) => void)[] = [];

const subscribeTokenRefresh = (cb: (token: string) => void) => {
  refreshSubscribers.push(cb);
};

const onRefreshed = (token: string) => {
  refreshSubscribers.forEach(cb => cb(token));
  refreshSubscribers = [];
};

// Response interceptor - handle token refresh
api.interceptors.response.use(
  (response) => response,
  async (error: AxiosError<ApiError>) => {
    const originalRequest = error.config as InternalAxiosRequestConfig & { _retry?: boolean };
    
    // If 401 and we have a refresh token, try to refresh (but not if already retried)
    if (error.response?.status === 401 && originalRequest && !originalRequest._retry) {
      
      // Skip refresh for auth endpoints to avoid loops
      if (originalRequest.url?.includes('/auth/')) {
        return Promise.reject(error);
      }
      
      // Get refresh token from memory or localStorage
      const storedRefresh = refreshToken || localStorage.getItem('refreshToken');
      if (!storedRefresh) {
        return Promise.reject(error);
      }
      
      // If already refreshing, wait for it
      if (isRefreshing) {
        return new Promise((resolve, reject) => {
          subscribeTokenRefresh((token: string) => {
            if (token) {
              originalRequest.headers.Authorization = `Bearer ${token}`;
              resolve(api(originalRequest));
            } else {
              reject(error);
            }
          });
        });
      }
      
      originalRequest._retry = true;
      isRefreshing = true;
      
      try {
        const response = await axios.post(`${API_URL}/auth/refresh`, {
          refreshToken: storedRefresh,
        });
        
        const { accessToken: newAccess, refreshToken: newRefresh } = response.data;
        setTokens(newAccess, newRefresh);
        isRefreshing = false;
        onRefreshed(newAccess);
        
        // Retry original request with new token
        originalRequest.headers.Authorization = `Bearer ${newAccess}`;
        return api(originalRequest);
      } catch (refreshError: any) {
        // Refresh failed
        isRefreshing = false;
        
        // Notify waiting requests that refresh failed
        refreshSubscribers.forEach(cb => cb(''));
        refreshSubscribers = [];
        
        // Only clear tokens if refresh token is invalid/expired (not network errors)
        if (refreshError?.response?.status === 401) {
          // Refresh token is invalid - clear tokens and let user re-login
          clearTokens();
        }
        
        return Promise.reject(error);
      }
    }
    
    return Promise.reject(error);
  }
);

// Initialize from stored tokens on module load
function initializeTokens() {
  const storedAccess = getStoredAccessToken();
  const storedRefresh = getStoredRefreshToken();
  if (storedAccess) {
    accessToken = storedAccess;
  }
  if (storedRefresh) {
    refreshToken = storedRefresh;
  }
}

initializeTokens();
