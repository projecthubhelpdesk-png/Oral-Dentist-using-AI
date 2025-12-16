<?php
/**
 * Feature Flags Controller
 * Manages feature toggles for admin dashboard
 */

namespace OralCareAI\Controllers;

require_once __DIR__ . '/../../config/database.php';
require_once __DIR__ . '/../../middleware/AuthMiddleware.php';
require_once __DIR__ . '/../../services/AuditService.php';

use OralCareAI\Middleware\AuthMiddleware;
use OralCareAI\Services\AuditService;
use Database;

class FeatureController {
    private \PDO $db;
    private AuthMiddleware $auth;
    private AuditService $audit;
    
    public function __construct() {
        $this->db = Database::getInstance();
        $this->auth = new AuthMiddleware();
        $this->audit = new AuditService();
    }
    
    /**
     * GET /features - Get all feature flags (public, no auth required)
     */
    public function index(array $params, array $body): void {
        try {
            $stmt = $this->db->query("
                SELECT feature_key, feature_name, description, is_enabled, disabled_message 
                FROM feature_flags 
                ORDER BY id
            ");
            $features = $stmt->fetchAll(\PDO::FETCH_ASSOC);
            
            // Convert to key-value map for easy frontend access
            $featureMap = [];
            foreach ($features as $f) {
                $featureMap[$f['feature_key']] = [
                    'name' => $f['feature_name'],
                    'description' => $f['description'],
                    'enabled' => (bool) $f['is_enabled'],
                    'disabledMessage' => $f['disabled_message']
                ];
            }
            
            echo json_encode(['success' => true, 'features' => $featureMap]);
        } catch (\Exception $e) {
            http_response_code(500);
            echo json_encode(['error' => 'Server Error', 'message' => $e->getMessage()]);
        }
    }
    
    /**
     * GET /features/admin - Get all feature flags with full details (Admin only)
     */
    public function adminIndex(array $params, array $body): void {
        $user = $this->auth->authenticate();
        if (!$user) return;
        
        if ($user['role'] !== 'admin') {
            http_response_code(403);
            echo json_encode(['error' => 'Forbidden', 'message' => 'Admin access required']);
            return;
        }
        
        try {
            $stmt = $this->db->query("
                SELECT ff.*, u.email as updated_by_email
                FROM feature_flags ff
                LEFT JOIN users u ON ff.updated_by = u.id
                ORDER BY ff.id
            ");
            $features = $stmt->fetchAll(\PDO::FETCH_ASSOC);
            
            $formatted = array_map(function($f) {
                return [
                    'id' => (int) $f['id'],
                    'featureKey' => $f['feature_key'],
                    'featureName' => $f['feature_name'],
                    'description' => $f['description'],
                    'isEnabled' => (bool) $f['is_enabled'],
                    'disabledMessage' => $f['disabled_message'],
                    'updatedBy' => $f['updated_by_email'],
                    'updatedAt' => $f['updated_at'],
                    'createdAt' => $f['created_at']
                ];
            }, $features);
            
            echo json_encode(['success' => true, 'features' => $formatted]);
        } catch (\Exception $e) {
            http_response_code(500);
            echo json_encode(['error' => 'Server Error', 'message' => $e->getMessage()]);
        }
    }
    
    /**
     * PATCH /features/{key} - Update feature flag (Admin only)
     */
    public function update(array $params, array $body): void {
        $user = $this->auth->authenticate();
        if (!$user) return;
        
        if ($user['role'] !== 'admin') {
            http_response_code(403);
            echo json_encode(['error' => 'Forbidden', 'message' => 'Admin access required']);
            return;
        }
        
        $featureKey = $params['key'] ?? '';
        
        if (empty($featureKey)) {
            http_response_code(400);
            echo json_encode(['error' => 'Bad Request', 'message' => 'Feature key is required']);
            return;
        }
        
        // Check if feature exists
        $stmt = $this->db->prepare("SELECT * FROM feature_flags WHERE feature_key = ?");
        $stmt->execute([$featureKey]);
        $feature = $stmt->fetch();
        
        if (!$feature) {
            http_response_code(404);
            echo json_encode(['error' => 'Not Found', 'message' => 'Feature not found']);
            return;
        }
        
        // Build update query
        $updates = [];
        $values = [];
        
        if (isset($body['isEnabled'])) {
            $updates[] = 'is_enabled = ?';
            $values[] = $body['isEnabled'] ? 1 : 0;
        }
        
        if (isset($body['disabledMessage'])) {
            $updates[] = 'disabled_message = ?';
            $values[] = $body['disabledMessage'];
        }
        
        if (empty($updates)) {
            http_response_code(400);
            echo json_encode(['error' => 'Bad Request', 'message' => 'No valid fields to update']);
            return;
        }
        
        $updates[] = 'updated_by = ?';
        $values[] = $user['user_id'];
        $values[] = $featureKey;
        
        try {
            $sql = "UPDATE feature_flags SET " . implode(', ', $updates) . " WHERE feature_key = ?";
            $stmt = $this->db->prepare($sql);
            $stmt->execute($values);
            
            // Audit log
            $this->audit->log('FEATURE_UPDATE', $user['user_id'], 'feature', $featureKey, [
                'isEnabled' => $body['isEnabled'] ?? null
            ]);
            
            // Return updated feature
            $stmt = $this->db->prepare("SELECT * FROM feature_flags WHERE feature_key = ?");
            $stmt->execute([$featureKey]);
            $updated = $stmt->fetch();
            
            echo json_encode([
                'success' => true,
                'message' => 'Feature updated successfully',
                'feature' => [
                    'featureKey' => $updated['feature_key'],
                    'featureName' => $updated['feature_name'],
                    'isEnabled' => (bool) $updated['is_enabled'],
                    'disabledMessage' => $updated['disabled_message']
                ]
            ]);
        } catch (\Exception $e) {
            http_response_code(500);
            echo json_encode(['error' => 'Server Error', 'message' => $e->getMessage()]);
        }
    }
    
    /**
     * GET /features/check/{key} - Check if a specific feature is enabled (public)
     */
    public function check(array $params, array $body): void {
        $featureKey = $params['key'] ?? '';
        
        if (empty($featureKey)) {
            http_response_code(400);
            echo json_encode(['error' => 'Bad Request', 'message' => 'Feature key is required']);
            return;
        }
        
        try {
            $stmt = $this->db->prepare("SELECT is_enabled, disabled_message FROM feature_flags WHERE feature_key = ?");
            $stmt->execute([$featureKey]);
            $feature = $stmt->fetch();
            
            if (!$feature) {
                // Feature not found, assume enabled
                echo json_encode(['enabled' => true, 'message' => null]);
                return;
            }
            
            echo json_encode([
                'enabled' => (bool) $feature['is_enabled'],
                'message' => $feature['is_enabled'] ? null : $feature['disabled_message']
            ]);
        } catch (\Exception $e) {
            // On error, assume enabled to not block users
            echo json_encode(['enabled' => true, 'message' => null]);
        }
    }
}
