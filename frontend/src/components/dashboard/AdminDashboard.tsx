import { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import {
  getAdminFeatures,
  updateFeature,
  type FeatureFlagAdmin,
} from '@/services/features';

export function AdminDashboard() {
  const [features, setFeatures] = useState<FeatureFlagAdmin[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [updating, setUpdating] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  useEffect(() => {
    loadFeatures();
  }, []);

  const loadFeatures = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await getAdminFeatures();
      setFeatures(data);
    } catch (err) {
      console.error('Failed to load features:', err);
      setError('Failed to load feature flags');
    } finally {
      setIsLoading(false);
    }
  };

  const handleToggle = async (featureKey: string, currentEnabled: boolean) => {
    setUpdating(featureKey);
    setError(null);
    setSuccessMessage(null);

    try {
      await updateFeature(featureKey, { isEnabled: !currentEnabled });
      // Update local state
      setFeatures((prev) =>
        prev.map((f) =>
          f.featureKey === featureKey ? { ...f, isEnabled: !currentEnabled } : f
        )
      );
      setSuccessMessage(
        `${featureKey} has been ${!currentEnabled ? 'enabled' : 'disabled'}`
      );
      setTimeout(() => setSuccessMessage(null), 3000);
    } catch (err) {
      console.error('Failed to update feature:', err);
      setError('Failed to update feature');
    } finally {
      setUpdating(null);
    }
  };

  const handleMessageUpdate = async (
    featureKey: string,
    newMessage: string
  ) => {
    setUpdating(featureKey);
    try {
      await updateFeature(featureKey, { disabledMessage: newMessage });
      setFeatures((prev) =>
        prev.map((f) =>
          f.featureKey === featureKey
            ? { ...f, disabledMessage: newMessage }
            : f
        )
      );
      setSuccessMessage('Message updated successfully');
      setTimeout(() => setSuccessMessage(null), 3000);
    } catch (err) {
      setError('Failed to update message');
    } finally {
      setUpdating(null);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Admin Dashboard</h1>
        <p className="text-gray-600">
          Manage system features and configurations
        </p>
      </div>

      {/* Stats Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="bg-gradient-to-br from-blue-50 to-blue-100 border-blue-200">
          <div className="text-center py-2">
            <p className="text-3xl font-bold text-blue-600">
              {features.length}
            </p>
            <p className="text-sm text-gray-600">Total Features</p>
          </div>
        </Card>
        <Card className="bg-gradient-to-br from-green-50 to-green-100 border-green-200">
          <div className="text-center py-2">
            <p className="text-3xl font-bold text-green-600">
              {features.filter((f) => f.isEnabled).length}
            </p>
            <p className="text-sm text-gray-600">Enabled</p>
          </div>
        </Card>
        <Card className="bg-gradient-to-br from-red-50 to-red-100 border-red-200">
          <div className="text-center py-2">
            <p className="text-3xl font-bold text-red-600">
              {features.filter((f) => !f.isEnabled).length}
            </p>
            <p className="text-sm text-gray-600">Disabled</p>
          </div>
        </Card>
        <Card className="bg-gradient-to-br from-purple-50 to-purple-100 border-purple-200">
          <div className="text-center py-2">
            <p className="text-3xl font-bold text-purple-600">🛡️</p>
            <p className="text-sm text-gray-600">Admin Mode</p>
          </div>
        </Card>
      </div>

      {/* Alerts */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-red-700">
          ❌ {error}
        </div>
      )}
      {successMessage && (
        <div className="bg-green-50 border border-green-200 rounded-lg p-4 text-green-700">
          ✅ {successMessage}
        </div>
      )}

      {/* Feature Management */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2">
              ⚙️ Feature Management
            </CardTitle>
            <Button variant="secondary" size="sm" onClick={loadFeatures}>
              🔄 Refresh
            </Button>
          </div>
        </CardHeader>

        {isLoading ? (
          <div className="flex justify-center py-8">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600" />
          </div>
        ) : (
          <div className="space-y-4">
            {features.map((feature) => (
              <FeatureCard
                key={feature.featureKey}
                feature={feature}
                isUpdating={updating === feature.featureKey}
                onToggle={() =>
                  handleToggle(feature.featureKey, feature.isEnabled)
                }
                onMessageUpdate={(msg) =>
                  handleMessageUpdate(feature.featureKey, msg)
                }
              />
            ))}
          </div>
        )}
      </Card>

      {/* Info Section */}
      <Card className="bg-yellow-50 border-yellow-200">
        <div className="flex items-start gap-3">
          <span className="text-2xl">⚠️</span>
          <div>
            <h3 className="font-semibold text-yellow-800">Important Notes</h3>
            <ul className="text-sm text-yellow-700 mt-2 space-y-1">
              <li>
                • <strong>Dentist Dashboard:</strong> When disabled, dentist
                registration and login are blocked for ALL users including
                admin-created accounts.
              </li>
              <li>
                • <strong>Spectral AI:</strong> When disabled, the Spectral AI
                tab is hidden from the dentist dashboard.
              </li>
              <li>
                • Changes take effect immediately across all users.
              </li>
              <li>
                • The disabled message is shown to users who try to access
                blocked features.
              </li>
            </ul>
          </div>
        </div>
      </Card>
    </div>
  );
}

interface FeatureCardProps {
  feature: FeatureFlagAdmin;
  isUpdating: boolean;
  onToggle: () => void;
  onMessageUpdate: (message: string) => void;
}

function FeatureCard({
  feature,
  isUpdating,
  onToggle,
  onMessageUpdate,
}: FeatureCardProps) {
  const [editingMessage, setEditingMessage] = useState(false);
  const [messageValue, setMessageValue] = useState(feature.disabledMessage);

  const getFeatureIcon = (key: string) => {
    switch (key) {
      case 'dentist_dashboard':
        return '🩺';
      case 'spectral_ai':
        return '🔬';
      default:
        return '⚙️';
    }
  };

  return (
    <div
      className={`border rounded-lg p-4 ${
        feature.isEnabled
          ? 'bg-green-50 border-green-200'
          : 'bg-red-50 border-red-200'
      }`}
    >
      <div className="flex items-start justify-between">
        <div className="flex items-start gap-3">
          <span className="text-3xl">{getFeatureIcon(feature.featureKey)}</span>
          <div>
            <h3 className="font-semibold text-gray-900">
              {feature.featureName}
            </h3>
            <p className="text-sm text-gray-600 mt-1">{feature.description}</p>
            <p className="text-xs text-gray-400 mt-2">
              Key: <code className="bg-gray-200 px-1 rounded">{feature.featureKey}</code>
              {feature.updatedBy && (
                <span className="ml-2">
                  • Last updated by {feature.updatedBy} on{' '}
                  {new Date(feature.updatedAt).toLocaleDateString()}
                </span>
              )}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <span
            className={`px-3 py-1 rounded-full text-sm font-medium ${
              feature.isEnabled
                ? 'bg-green-200 text-green-800'
                : 'bg-red-200 text-red-800'
            }`}
          >
            {feature.isEnabled ? '✓ Enabled' : '✗ Disabled'}
          </span>

          <button
            onClick={onToggle}
            disabled={isUpdating}
            className={`relative inline-flex h-8 w-14 items-center rounded-full transition-colors ${
              feature.isEnabled ? 'bg-green-500' : 'bg-gray-300'
            } ${isUpdating ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
          >
            <span
              className={`inline-block h-6 w-6 transform rounded-full bg-white shadow-md transition-transform ${
                feature.isEnabled ? 'translate-x-7' : 'translate-x-1'
              }`}
            />
          </button>
        </div>
      </div>

      {/* Disabled Message Editor */}
      <div className="mt-4 pt-4 border-t border-gray-200">
        <div className="flex items-center justify-between mb-2">
          <label className="text-sm font-medium text-gray-700">
            Disabled Message:
          </label>
          {!editingMessage && (
            <button
              onClick={() => setEditingMessage(true)}
              className="text-sm text-blue-600 hover:text-blue-800"
            >
              ✏️ Edit
            </button>
          )}
        </div>

        {editingMessage ? (
          <div className="flex gap-2">
            <input
              type="text"
              value={messageValue}
              onChange={(e) => setMessageValue(e.target.value)}
              className="flex-1 px-3 py-2 border border-gray-300 rounded-lg text-sm"
            />
            <Button
              size="sm"
              onClick={() => {
                onMessageUpdate(messageValue);
                setEditingMessage(false);
              }}
              disabled={isUpdating}
            >
              Save
            </Button>
            <Button
              size="sm"
              variant="secondary"
              onClick={() => {
                setMessageValue(feature.disabledMessage);
                setEditingMessage(false);
              }}
            >
              Cancel
            </Button>
          </div>
        ) : (
          <p className="text-sm text-gray-600 bg-white px-3 py-2 rounded border">
            "{feature.disabledMessage}"
          </p>
        )}
      </div>
    </div>
  );
}
