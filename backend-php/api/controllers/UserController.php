<?php
/**
 * User Controller
 * Oral Care AI - PHP Backend
 */

namespace OralCareAI\Controllers;

require_once __DIR__ . '/../../models/User.php';
require_once __DIR__ . '/../../middleware/AuthMiddleware.php';
require_once __DIR__ . '/../../services/AuditService.php';

use OralCareAI\Models\User;
use OralCareAI\Middleware\AuthMiddleware;
use OralCareAI\Services\AuditService;

class UserController {
    private User $userModel;
    private AuthMiddleware $auth;
    private AuditService $audit;
    
    public function __construct() {
        $this->userModel = new User();
        $this->auth = new AuthMiddleware();
        $this->audit = new AuditService();
    }
    
    /**
     * GET /users/me
     */
    public function me(array $params, array $body): void {
        $userData = $this->auth->authenticate();
        if (!$userData) return;
        
        $user = $this->userModel->findById($userData['user_id']);
        
        if (!$user) {
            http_response_code(404);
            echo json_encode(['error' => 'Not Found', 'message' => 'User not found']);
            return;
        }
        
        echo json_encode($user);
    }
    
    /**
     * PATCH /users/me
     */
    public function update(array $params, array $body): void {
        $userData = $this->auth->authenticate();
        if (!$userData) return;
        
        // Filter allowed fields
        $allowedFields = ['phone', 'first_name', 'last_name', 'profile_image_url'];
        $updateData = array_intersect_key($body, array_flip($allowedFields));
        
        if (empty($updateData)) {
            http_response_code(400);
            echo json_encode(['error' => 'Bad Request', 'message' => 'No valid fields to update']);
            return;
        }
        
        // Validate phone if provided
        if (isset($updateData['phone']) && !preg_match('/^[0-9+\-\s()]{10,20}$/', $updateData['phone'])) {
            http_response_code(400);
            echo json_encode(['error' => 'Bad Request', 'message' => 'Invalid phone number format']);
            return;
        }
        
        $success = $this->userModel->update($userData['user_id'], $updateData);
        
        if (!$success) {
            http_response_code(500);
            echo json_encode(['error' => 'Server Error', 'message' => 'Failed to update profile']);
            return;
        }
        
        $this->audit->logProfileUpdate($userData['user_id'], array_keys($updateData));
        
        $user = $this->userModel->findById($userData['user_id']);
        echo json_encode($user);
    }
}
