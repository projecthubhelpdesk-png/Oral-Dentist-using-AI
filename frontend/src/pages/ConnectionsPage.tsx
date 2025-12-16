import { useEffect, useState } from 'react';
import { useAuth } from '@/context/AuthContext';
import { Layout } from '@/components/layout/Layout';
import { Card, CardHeader, CardTitle } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import { getConnections, updateConnection } from '@/services/connections';
import type { Connection, ConnectionStatus } from '@/types';

export function ConnectionsPage() {
  const { user } = useAuth();
  const [connections, setConnections] = useState<Connection[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [statusFilter, setStatusFilter] = useState<ConnectionStatus | ''>('');

  useEffect(() => {
    loadConnections();
  }, [statusFilter]);

  async function loadConnections() {
    try {
      setIsLoading(true);
      const response = await getConnections(statusFilter || undefined);
      setConnections(response.data);
    } catch (err) {
      console.error('Failed to load connections:', err);
    } finally {
      setIsLoading(false);
    }
  }

  async function handleUpdateStatus(connectionId: string, status: 'active' | 'declined' | 'terminated') {
    try {
      await updateConnection(connectionId, { status });
      loadConnections();
    } catch (err) {
      console.error('Failed to update connection:', err);
    }
  }

  const statusColors: Record<ConnectionStatus, 'default' | 'success' | 'warning' | 'error'> = {
    pending: 'warning',
    active: 'success',
    declined: 'error',
    terminated: 'default',
  };

  const pendingConnections = connections.filter(c => c.status === 'pending');
  const activeConnections = connections.filter(c => c.status === 'active');
  const otherConnections = connections.filter(c => !['pending', 'active'].includes(c.status));


  return (
    <Layout>
      <div className="space-y-6">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">My Connections</h1>
          <p className="text-gray-600">
            {user?.role === 'dentist' 
              ? 'Manage your patient connections' 
              : 'Manage your dentist connections'}
          </p>
        </div>

        {/* Pending Requests */}
        {pendingConnections.length > 0 && (
          <div>
            <h2 className="text-lg font-semibold text-gray-900 mb-4">Pending Requests</h2>
            <div className="space-y-3">
              {pendingConnections.map((conn) => (
                <Card key={conn.id} className="flex items-center justify-between p-4">
                  <div>
                    <p className="font-medium">{conn.otherUser?.name || conn.otherUser?.email?.split('@')[0]}</p>
                    <p className="text-sm text-gray-500">
                      {conn.otherUser?.role === 'dentist' && conn.otherUser?.specialty}
                      {conn.initiatedBy === 'patient' ? ' • Requested by patient' : ' • Requested by dentist'}
                    </p>
                  </div>
                  <div className="flex gap-2">
                    <Button 
                      variant="secondary" 
                      onClick={() => handleUpdateStatus(conn.id, 'declined')}
                    >
                      Decline
                    </Button>
                    <Button onClick={() => handleUpdateStatus(conn.id, 'active')}>
                      Accept
                    </Button>
                  </div>
                </Card>
              ))}
            </div>
          </div>
        )}

        {/* Active Connections */}
        <div>
          <h2 className="text-lg font-semibold text-gray-900 mb-4">
            Active Connections ({activeConnections.length})
          </h2>
          {activeConnections.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {activeConnections.map((conn) => (
                <Card key={conn.id}>
                  <CardHeader>
                    <div className="flex items-start justify-between">
                      <div>
                        <CardTitle className="text-base">{conn.otherUser?.name || conn.otherUser?.email?.split('@')[0]}</CardTitle>
                        <p className="text-sm text-gray-500 capitalize">{conn.otherUser?.role}</p>
                      </div>
                      <Badge variant="success">Active</Badge>
                    </div>
                  </CardHeader>
                  <div className="space-y-2 text-sm">
                    {conn.otherUser.specialty && (
                      <p className="text-gray-600">{conn.otherUser.specialty}</p>
                    )}
                    {conn.otherUser.clinicName && (
                      <p className="text-gray-600">{conn.otherUser.clinicName}</p>
                    )}
                    <p className="text-gray-500">
                      Connected {conn.connectedAt ? new Date(conn.connectedAt).toLocaleDateString() : 'recently'}
                    </p>
                    {conn.shareScanHistory && (
                      <p className="text-green-600 text-xs">✓ Scan history shared</p>
                    )}
                  </div>
                  <div className="mt-4 pt-4 border-t">
                    <Button 
                      variant="secondary" 
                      className="w-full text-red-600 hover:bg-red-50"
                      onClick={() => handleUpdateStatus(conn.id, 'terminated')}
                    >
                      End Connection
                    </Button>
                  </div>
                </Card>
              ))}
            </div>
          ) : (
            <Card className="text-center py-8">
              <p className="text-gray-500">No active connections yet.</p>
            </Card>
          )}
        </div>

        {/* Past Connections */}
        {otherConnections.length > 0 && (
          <div>
            <h2 className="text-lg font-semibold text-gray-900 mb-4">Past Connections</h2>
            <div className="space-y-2">
              {otherConnections.map((conn) => (
                <Card key={conn.id} className="flex items-center justify-between p-4 opacity-60">
                  <div>
                    <p className="font-medium">{conn.otherUser?.name || conn.otherUser?.email?.split('@')[0]}</p>
                    <p className="text-sm text-gray-500">{conn.otherUser?.role}</p>
                  </div>
                  <Badge variant={statusColors[conn.status]}>{conn.status}</Badge>
                </Card>
              ))}
            </div>
          </div>
        )}
      </div>
    </Layout>
  );
}
