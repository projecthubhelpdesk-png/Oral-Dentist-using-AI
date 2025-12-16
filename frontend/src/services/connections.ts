import { api } from './api';
import type { Connection } from '@/types';

interface ConnectionsResponse {
  data: Connection[];
}

export async function getConnections(status?: string): Promise<ConnectionsResponse> {
  const params = status ? `?status=${status}` : '';
  const response = await api.get(`/connections${params}`);
  return response.data;
}

export async function createConnection(targetUserId: string, shareScanHistory = false): Promise<{ id: string }> {
  const response = await api.post('/connections', { targetUserId, shareScanHistory });
  return response.data;
}

export async function updateConnection(
  connectionId: string,
  data: { status?: 'active' | 'declined' | 'terminated'; shareScanHistory?: boolean }
): Promise<void> {
  await api.patch(`/connections/${connectionId}`, data);
}
