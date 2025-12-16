import { api, setTokens, clearTokens, getTokens } from './api';

export { getTokens };
import type { User, LoginResponse } from '@/types';

export interface RegisterData {
  email: string;
  password: string;
  phone?: string;
  role?: 'user' | 'dentist';
  firstName?: string;
  lastName?: string;
}

export interface LoginData {
  email: string;
  password: string;
}

export const authService = {
  async register(data: RegisterData): Promise<LoginResponse> {
    const response = await api.post<LoginResponse>('/auth/register', {
      email: data.email,
      password: data.password,
      phone: data.phone,
      role: data.role || 'user',
      first_name: data.firstName,
      last_name: data.lastName,
    });
    
    setTokens(response.data.accessToken, response.data.refreshToken);
    return response.data;
  },
  
  async login(data: LoginData): Promise<LoginResponse> {
    const response = await api.post<LoginResponse>('/auth/login', data);
    setTokens(response.data.accessToken, response.data.refreshToken);
    return response.data;
  },
  
  async logout(): Promise<void> {
    try {
      await api.post('/auth/logout');
    } finally {
      clearTokens();
    }
  },
  
  async getCurrentUser(): Promise<User> {
    const response = await api.get<User>('/users/me');
    return response.data;
  },
  
  async updateProfile(data: Partial<User>): Promise<User> {
    const response = await api.patch<User>('/users/me', data);
    return response.data;
  },
};
