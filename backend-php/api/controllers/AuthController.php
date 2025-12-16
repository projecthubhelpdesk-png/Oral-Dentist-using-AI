<?php
/**
 * Auth Controller
 * Oral Care AI - PHP Backend
 */

namespace OralCareAI\Controllers;

require_once __DIR__ . '/../../models/User.php';
require_once __DIR__ . '/../../middleware/AuthMiddleware.php';
require_once __DIR__ . '/../../middleware/RateLimitMiddleware.php';
require_once __DIR__ . '/../../services/AuditService.php';
require_once __DIR__ . '/../../config/database.php';

use OralCareAI\Models\User;
use OralCareAI\Middleware\AuthMiddleware;
use OralCareAI\Middleware\RateLimitMiddleware;
use OralCareAI\Services\AuditService;
use Database;

class AuthController {
    private User $userModel;
    private AuthMiddleware $auth;
    private AuditService $audit;
    private \PDO $db;
    
    public function __construct() {
        $this->userModel = new User();
        $this->auth = new AuthMiddleware();
        $this->audit = new AuditService();
        $this->db = Database::getInstance();
    }
    
    /**
     * POST /auth/register
     */
    public function register(array $params, array $body): void {
        // Check if dentist registration is allowed
        $requestedRole = $body['role'] ?? 'user';
        if ($requestedRole === 'dentist') {
            $featureCheck = $this->checkFeatureEnabled('dentist_dashboard');
            if (!$featureCheck['enabled']) {
                http_response_code(403);
                echo json_encode(['error' => 'Feature Disabled', 'message' => $featureCheck['message']]);
                return;
            }
        }
        
        // Validate input
        $errors = $this->validateRegistration($body);
        if (!empty($errors)) {
            http_response_code(400);
            echo json_encode(['error' => 'Validation Error', 'details' => $errors]);
            return;
        }
        
        // Create user
        $user = $this->userModel->create([
            'email' => $body['email'],
            'password' => $body['password'],
            'phone' => $body['phone'] ?? null,
            'role' => in_array($body['role'] ?? 'user', ['user', 'dentist']) ? $body['role'] : 'user',
            'first_name' => $body['first_name'] ?? null,
            'last_name' => $body['last_name'] ?? null,
        ]);
        
        if (!$user) {
            http_response_code(409);
            echo json_encode(['error' => 'Conflict', 'message' => 'Email already exists']);
            return;
        }
        
        // Generate tokens
        $accessToken = $this->auth->generateAccessToken($user);
        $refreshToken = $this->auth->generateRefreshToken();
        
        // Store refresh token
        $this->storeRefreshToken($user['id'], $refreshToken);
        
        // Audit log
        $this->audit->log('USER_REGISTERED', $user['id'], 'user', $user['id']);
        
        http_response_code(201);
        echo json_encode([
            'accessToken' => $accessToken,
            'refreshToken' => $refreshToken,
            'expiresIn' => 900,
            'tokenType' => 'Bearer',
            'user' => $user
        ]);
    }
    
    /**
     * POST /auth/login
     */
    public function login(array $params, array $body): void {
        $email = $body['email'] ?? '';
        $password = $body['password'] ?? '';
        
        // Rate limit login attempts
        $rateLimit = new RateLimitMiddleware();
        if (!$rateLimit->checkLoginLimit($email)) {
            return;
        }
        
        // Find user
        $user = $this->userModel->findByEmail($email);
        
        if (!$user || !$this->userModel->verifyPassword($password, $user['password_hash'])) {
            $this->audit->logLogin($user['id'] ?? 'unknown', false);
            http_response_code(401);
            echo json_encode(['error' => 'Unauthorized', 'message' => 'Invalid credentials']);
            return;
        }
        
        // Check if dentist login is allowed
        if ($user['role'] === 'dentist') {
            $featureCheck = $this->checkFeatureEnabled('dentist_dashboard');
            if (!$featureCheck['enabled']) {
                http_response_code(403);
                echo json_encode(['error' => 'Feature Disabled', 'message' => $featureCheck['message']]);
                return;
            }
        }
        
        // Update last login
        $this->userModel->updateLastLogin($user['id']);
        
        // Revoke all existing refresh tokens for this user (single session)
        $stmt = $this->db->prepare("UPDATE refresh_tokens SET revoked = TRUE WHERE user_id = ?");
        $stmt->execute([$user['id']]);
        
        // Generate tokens
        $accessToken = $this->auth->generateAccessToken($user);
        $refreshToken = $this->auth->generateRefreshToken();
        
        // Store refresh token
        $this->storeRefreshToken($user['id'], $refreshToken);
        
        // Audit log
        $this->audit->logLogin($user['id'], true);
        
        // Remove password hash from response
        unset($user['password_hash']);
        
        echo json_encode([
            'accessToken' => $accessToken,
            'refreshToken' => $refreshToken,
            'expiresIn' => 900,
            'tokenType' => 'Bearer',
            'user' => $user
        ]);
    }
    
    /**
     * POST /auth/refresh
     */
    public function refresh(array $params, array $body): void {
        $refreshToken = $body['refreshToken'] ?? '';
        
        if (empty($refreshToken)) {
            http_response_code(400);
            echo json_encode(['error' => 'Bad Request', 'message' => 'Refresh token required']);
            return;
        }
        
        // Verify refresh token
        $tokenHash = hash('sha256', $refreshToken);
        $stmt = $this->db->prepare("
            SELECT rt.user_id, u.email, u.role
            FROM refresh_tokens rt
            JOIN users u ON rt.user_id = u.id
            WHERE rt.token_hash = ? AND rt.revoked = FALSE AND rt.expires_at > NOW()
        ");
        $stmt->execute([$tokenHash]);
        $tokenData = $stmt->fetch();
        
        if (!$tokenData) {
            http_response_code(401);
            echo json_encode(['error' => 'Unauthorized', 'message' => 'Invalid or expired refresh token']);
            return;
        }
        
        // Revoke old token
        $this->revokeRefreshToken($tokenHash);
        
        // Generate new tokens
        $user = [
            'id' => $tokenData['user_id'],
            'email' => $tokenData['email'],
            'role' => $tokenData['role']
        ];
        
        $newAccessToken = $this->auth->generateAccessToken($user);
        $newRefreshToken = $this->auth->generateRefreshToken();
        
        // Store new refresh token
        $this->storeRefreshToken($user['id'], $newRefreshToken);
        
        echo json_encode([
            'accessToken' => $newAccessToken,
            'refreshToken' => $newRefreshToken,
            'expiresIn' => 900,
            'tokenType' => 'Bearer'
        ]);
    }
    
    /**
     * POST /auth/logout
     */
    public function logout(array $params, array $body): void {
        $userData = $this->auth->authenticate();
        if (!$userData) {
            return;
        }
        
        // Revoke all refresh tokens for user
        $stmt = $this->db->prepare("UPDATE refresh_tokens SET revoked = TRUE WHERE user_id = ?");
        $stmt->execute([$userData['user_id']]);
        
        $this->audit->logLogout($userData['user_id']);
        
        echo json_encode(['message' => 'Logged out successfully']);
    }
    
    private function storeRefreshToken(string $userId, string $token): void {
        $tokenHash = hash('sha256', $token);
        $expiresAt = date('Y-m-d H:i:s', time() + 604800); // 7 days
        
        $stmt = $this->db->prepare("
            INSERT INTO refresh_tokens (id, user_id, token_hash, device_info, ip_address, expires_at)
            VALUES (UUID(), ?, ?, ?, ?, ?)
        ");
        
        $stmt->execute([
            $userId,
            $tokenHash,
            $_SERVER['HTTP_USER_AGENT'] ?? null,
            $_SERVER['REMOTE_ADDR'] ?? null,
            $expiresAt
        ]);
    }
    
    private function revokeRefreshToken(string $tokenHash): void {
        $stmt = $this->db->prepare("UPDATE refresh_tokens SET revoked = TRUE WHERE token_hash = ?");
        $stmt->execute([$tokenHash]);
    }
    
    private function validateRegistration(array $body): array {
        $errors = [];
        
        if (empty($body['email']) || !filter_var($body['email'], FILTER_VALIDATE_EMAIL)) {
            $errors['email'] = 'Valid email is required';
        }
        
        if (empty($body['password']) || strlen($body['password']) < 8) {
            $errors['password'] = 'Password must be at least 8 characters';
        }
        
        if (!empty($body['phone']) && !preg_match('/^[0-9+\-\s()]{10,20}$/', $body['phone'])) {
            $errors['phone'] = 'Invalid phone number format';
        }
        
        return $errors;
    }
    
    private function checkFeatureEnabled(string $featureKey): array {
        try {
            $stmt = $this->db->prepare("SELECT is_enabled, disabled_message FROM feature_flags WHERE feature_key = ?");
            $stmt->execute([$featureKey]);
            $feature = $stmt->fetch();
            
            if (!$feature) {
                return ['enabled' => true, 'message' => null];
            }
            
            return [
                'enabled' => (bool) $feature['is_enabled'],
                'message' => $feature['is_enabled'] ? null : $feature['disabled_message']
            ];
        } catch (\Exception $e) {
            return ['enabled' => true, 'message' => null];
        }
    }
}
