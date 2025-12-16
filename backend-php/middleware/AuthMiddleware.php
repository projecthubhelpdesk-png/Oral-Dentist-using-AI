<?php
/**
 * JWT Authentication Middleware
 * Oral Care AI - PHP Backend
 * 
 * Native PHP JWT implementation (no external dependencies)
 */

namespace OralCareAI\Middleware;

use Exception;

class AuthMiddleware {
    private array $config;
    
    public function __construct() {
        $this->config = require __DIR__ . '/../config/app.php';
    }
    
    /**
     * Verify JWT token and return user data
     */
    public function authenticate(): ?array {
        $token = $this->getBearerToken();
        
        if (!$token) {
            $this->sendUnauthorized('No token provided');
            return null;
        }
        
        try {
            $decoded = $this->decodeJWT($token);
            
            return [
                'user_id' => $decoded['sub'],
                'email' => $decoded['email'],
                'role' => $decoded['role'],
            ];
        } catch (Exception $e) {
            $this->sendUnauthorized('Invalid or expired token');
            return null;
        }
    }
    
    /**
     * Verify JWT token for image requests (doesn't exit on failure)
     */
    public function authenticateForImage(): ?array {
        $token = $this->getBearerToken();
        
        if (!$token) {
            return null;
        }
        
        try {
            $decoded = $this->decodeJWT($token);
            
            return [
                'user_id' => $decoded['sub'],
                'email' => $decoded['email'],
                'role' => $decoded['role'],
            ];
        } catch (Exception $e) {
            return null;
        }
    }
    
    /**
     * Require specific role(s)
     */
    public function requireRole(array $allowedRoles): bool {
        $user = $this->authenticate();
        
        if (!$user) {
            return false;
        }

        if (!in_array($user['role'], $allowedRoles)) {
            $this->sendForbidden('Insufficient permissions');
            return false;
        }
        
        return true;
    }
    
    /**
     * Generate access token (native PHP JWT)
     */
    public function generateAccessToken(array $user): string {
        $header = [
            'typ' => 'JWT',
            'alg' => 'HS256'
        ];
        
        $payload = [
            'iss' => $this->config['jwt']['issuer'],
            'sub' => $user['id'],
            'email' => $user['email'],
            'role' => $user['role'],
            'iat' => time(),
            'exp' => time() + $this->config['jwt']['access_token_ttl'],
        ];
        
        return $this->encodeJWT($header, $payload);
    }
    
    /**
     * Generate refresh token
     */
    public function generateRefreshToken(): string {
        return bin2hex(random_bytes(32));
    }
    
    /**
     * Encode JWT token
     */
    private function encodeJWT(array $header, array $payload): string {
        $secret = $this->config['jwt']['secret'];
        
        $headerEncoded = $this->base64UrlEncode(json_encode($header));
        $payloadEncoded = $this->base64UrlEncode(json_encode($payload));
        
        $signature = hash_hmac('sha256', "$headerEncoded.$payloadEncoded", $secret, true);
        $signatureEncoded = $this->base64UrlEncode($signature);
        
        return "$headerEncoded.$payloadEncoded.$signatureEncoded";
    }
    
    /**
     * Decode and verify JWT token
     */
    private function decodeJWT(string $token): array {
        $parts = explode('.', $token);
        
        if (count($parts) !== 3) {
            throw new Exception('Invalid token format');
        }
        
        [$headerEncoded, $payloadEncoded, $signatureEncoded] = $parts;
        
        // Verify signature
        $secret = $this->config['jwt']['secret'];
        $expectedSignature = $this->base64UrlEncode(
            hash_hmac('sha256', "$headerEncoded.$payloadEncoded", $secret, true)
        );
        
        if (!hash_equals($expectedSignature, $signatureEncoded)) {
            throw new Exception('Invalid signature');
        }
        
        // Decode payload
        $payload = json_decode($this->base64UrlDecode($payloadEncoded), true);
        
        if (!$payload) {
            throw new Exception('Invalid payload');
        }
        
        // Check expiration
        if (isset($payload['exp']) && $payload['exp'] < time()) {
            throw new Exception('Token expired');
        }
        
        return $payload;
    }
    
    private function base64UrlEncode(string $data): string {
        return rtrim(strtr(base64_encode($data), '+/', '-_'), '=');
    }
    
    private function base64UrlDecode(string $data): string {
        return base64_decode(strtr($data, '-_', '+/'));
    }
    
    /**
     * Extract Bearer token from Authorization header
     */
    private function getBearerToken(): ?string {
        $authHeader = '';
        
        // Try all possible sources for Authorization header
        // 1. Direct $_SERVER
        if (!empty($_SERVER['HTTP_AUTHORIZATION'])) {
            $authHeader = $_SERVER['HTTP_AUTHORIZATION'];
        }
        // 2. Apache redirect
        if (empty($authHeader) && !empty($_SERVER['REDIRECT_HTTP_AUTHORIZATION'])) {
            $authHeader = $_SERVER['REDIRECT_HTTP_AUTHORIZATION'];
        }
        // 3. getallheaders() function
        if (empty($authHeader) && function_exists('getallheaders')) {
            $headers = getallheaders();
            $authHeader = $headers['Authorization'] ?? $headers['authorization'] ?? '';
        }
        // 4. apache_request_headers() function
        if (empty($authHeader) && function_exists('apache_request_headers')) {
            $headers = apache_request_headers();
            $authHeader = $headers['Authorization'] ?? $headers['authorization'] ?? '';
        }
        
        if (preg_match('/Bearer\s+(.*)$/i', $authHeader, $matches)) {
            return $matches[1];
        }
        
        return null;
    }
    
    private function sendUnauthorized(string $message): void {
        http_response_code(401);
        echo json_encode(['error' => 'Unauthorized', 'message' => $message]);
        exit;
    }
    
    private function sendForbidden(string $message): void {
        http_response_code(403);
        echo json_encode(['error' => 'Forbidden', 'message' => $message]);
        exit;
    }
}
