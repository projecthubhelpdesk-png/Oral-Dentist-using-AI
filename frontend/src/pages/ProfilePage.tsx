import { useState } from 'react';
import { useAuth } from '@/context/AuthContext';
import { Layout } from '@/components/layout/Layout';
import { Card, CardHeader, CardTitle } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { Badge } from '@/components/ui/Badge';

export function ProfilePage() {
  const { user, logout } = useAuth();
  const [isEditing, setIsEditing] = useState(false);

  return (
    <Layout>
      <div className="max-w-2xl mx-auto space-y-6">
        <h1 className="text-2xl font-bold text-gray-900">Profile Settings</h1>

        {/* Account Info */}
        <Card>
          <CardHeader>
            <CardTitle>Account Information</CardTitle>
          </CardHeader>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Email</label>
              <div className="flex items-center gap-2">
                <Input value={user?.email || ''} disabled className="flex-1" />
                {user?.emailVerified ? (
                  <Badge variant="success">Verified</Badge>
                ) : (
                  <Badge variant="warning">Unverified</Badge>
                )}
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Role</label>
              <Input value={user?.role || ''} disabled className="capitalize" />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Member Since</label>
              <Input 
                value={user?.createdAt ? new Date(user.createdAt).toLocaleDateString() : ''} 
                disabled 
              />
            </div>

            {user?.phoneLastFour && (
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Phone</label>
                <Input value={`***-***-${user.phoneLastFour}`} disabled />
              </div>
            )}
          </div>
        </Card>

        {/* Security */}
        <Card>
          <CardHeader>
            <CardTitle>Security</CardTitle>
          </CardHeader>
          <div className="space-y-4">
            <Button variant="secondary" className="w-full">
              Change Password
            </Button>
            <Button variant="secondary" className="w-full">
              Enable Two-Factor Authentication
            </Button>
          </div>
        </Card>

        {/* Danger Zone */}
        <Card className="border-red-200">
          <CardHeader>
            <CardTitle className="text-red-600">Danger Zone</CardTitle>
          </CardHeader>
          <div className="space-y-4">
            <Button 
              variant="secondary" 
              className="w-full text-red-600 hover:bg-red-50"
              onClick={logout}
            >
              Sign Out
            </Button>
            <Button 
              variant="secondary" 
              className="w-full text-red-600 hover:bg-red-50"
            >
              Delete Account
            </Button>
          </div>
        </Card>
      </div>
    </Layout>
  );
}
