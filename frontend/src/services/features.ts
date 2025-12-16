import { api } from './api';

export interface FeatureFlag {
  name: string;
  description: string;
  enabled: boolean;
  disabledMessage: string;
}

export interface FeatureFlagAdmin {
  id: number;
  featureKey: string;
  featureName: string;
  description: string;
  isEnabled: boolean;
  disabledMessage: string;
  updatedBy: string | null;
  updatedAt: string;
  createdAt: string;
}

export interface FeaturesMap {
  [key: string]: FeatureFlag;
}

// Cache for feature flags
let featuresCache: FeaturesMap | null = null;
let cacheTimestamp = 0;
const CACHE_TTL = 60000; // 1 minute

/**
 * Get all feature flags (public endpoint)
 */
export async function getFeatures(): Promise<FeaturesMap> {
  const now = Date.now();
  if (featuresCache && now - cacheTimestamp < CACHE_TTL) {
    return featuresCache;
  }

  try {
    const response = await api.get<{ success: boolean; features: FeaturesMap }>(
      '/features'
    );
    if (response.data.success) {
      featuresCache = response.data.features;
      cacheTimestamp = now;
      return featuresCache;
    }
  } catch (error) {
    console.error('Failed to fetch features:', error);
  }

  // Return default enabled features on error
  return {
    dentist_dashboard: {
      name: 'Dentist Dashboard',
      description: '',
      enabled: true,
      disabledMessage: '',
    },
    spectral_ai: {
      name: 'Spectral AI',
      description: '',
      enabled: true,
      disabledMessage: '',
    },
  };
}

/**
 * Check if a specific feature is enabled
 */
export async function isFeatureEnabled(featureKey: string): Promise<boolean> {
  const features = await getFeatures();
  return features[featureKey]?.enabled ?? true;
}

/**
 * Get feature disabled message
 */
export async function getFeatureMessage(
  featureKey: string
): Promise<string | null> {
  const features = await getFeatures();
  const feature = features[featureKey];
  if (feature && !feature.enabled) {
    return feature.disabledMessage;
  }
  return null;
}

/**
 * Get all features for admin (requires auth)
 */
export async function getAdminFeatures(): Promise<FeatureFlagAdmin[]> {
  const response = await api.get<{
    success: boolean;
    features: FeatureFlagAdmin[];
  }>('/features/admin');
  return response.data.features || [];
}

/**
 * Update a feature flag (admin only)
 */
export async function updateFeature(
  featureKey: string,
  data: { isEnabled?: boolean; disabledMessage?: string }
): Promise<void> {
  await api.patch(`/features/${featureKey}`, data);
  // Clear cache to force refresh
  featuresCache = null;
  cacheTimestamp = 0;
}

/**
 * Clear features cache (call after updates)
 */
export function clearFeaturesCache(): void {
  featuresCache = null;
  cacheTimestamp = 0;
}
