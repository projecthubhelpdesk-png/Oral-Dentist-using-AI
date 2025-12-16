import { api } from './api';

export interface ChatMessage {
  id: string;
  scanId: string;
  senderId: string;
  senderEmail: string;
  senderName: string;
  senderRole: 'user' | 'dentist' | 'admin';
  message: string;
  messageType: 'text' | 'image' | 'system';
  isRead: boolean;
  createdAt: string;
}

export async function getChatMessages(scanId: string, limit = 50): Promise<ChatMessage[]> {
  const response = await api.get(`/scans/${scanId}/chat?limit=${limit}`);
  return response.data.messages;
}

export async function sendChatMessage(scanId: string, message: string): Promise<ChatMessage> {
  const response = await api.post(`/scans/${scanId}/chat`, { message });
  return response.data;
}

export async function getUnreadCount(scanId: string): Promise<number> {
  const response = await api.get(`/scans/${scanId}/chat/unread`);
  return response.data.unreadCount;
}
