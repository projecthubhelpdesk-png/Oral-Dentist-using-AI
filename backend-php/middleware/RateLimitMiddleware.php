<?php
/**
 * Rate Limiting Middleware
 * Oral Care AI - PHP Backend
 * 
 * Uses file-based storage for simplicity. In production, use Redis.
 */

namespace OralCareAI\Middleware;

class RateLimitMiddleware {
    private array $config;
    private string $storagePath;
    
    public function __construct() {
        $appConfig = require __DIR__ . '/../config/app.php';
        $this->config = $appConfig['rate_limit'];
        $this->storagePath = __DIR__ . '/../storage/rate_limits/';
        
        if (!is_dir($this->storagePath)) {
            mkdir($this->storagePath, 0755, true);
        }
    }
    
    /**
     * Check rate limit for general API requests
     */
    public function checkLimit(string $identifier = null): bool {
        if (!$this->config['enabled']) {
            return true;
        }
        
        $identifier = $identifier ?? $this->getClientIdentifier();
        $key = 'general_' . md5($identifier);
        
        return $this->isAllowed($key, $this->config['requests_per_minute'], 60);
    }
    
    /**
     * Check rate limit for login attempts
     */
    public function checkLoginLimit(string $email): bool {
        if (!$this->config['enabled']) {
            return true;
        }
        
        $key = 'login_' . md5($email . $this->getClientIdentifier());
        
        return $this->isAllowed($key, $this->config['login_attempts_per_hour'], 3600);
    }
    
    private function isAllowed(string $key, int $maxRequests, int $windowSeconds): bool {
        $file = $this->storagePath . $key . '.json';
        $now = time();
        
        $data = ['requests' => [], 'window_start' => $now];
        
        if (file_exists($file)) {
            $data = json_decode(file_get_contents($file), true) ?? $data;
        }
        
        // Reset window if expired
        if ($now - $data['window_start'] > $windowSeconds) {
            $data = ['requests' => [], 'window_start' => $now];
        }
        
        // Filter requests within window
        $data['requests'] = array_filter($data['requests'], fn($t) => $now - $t < $windowSeconds);
        
        if (count($data['requests']) >= $maxRequests) {
            $this->sendTooManyRequests($windowSeconds - ($now - $data['window_start']));
            return false;
        }
        
        // Add current request
        $data['requests'][] = $now;
        file_put_contents($file, json_encode($data));
        
        // Set rate limit headers
        header("X-RateLimit-Limit: {$maxRequests}");
        header("X-RateLimit-Remaining: " . ($maxRequests - count($data['requests'])));
        header("X-RateLimit-Reset: " . ($data['window_start'] + $windowSeconds));
        
        return true;
    }
    
    private function getClientIdentifier(): string {
        return $_SERVER['REMOTE_ADDR'] ?? 'unknown';
    }
    
    private function sendTooManyRequests(int $retryAfter): void {
        http_response_code(429);
        header("Retry-After: {$retryAfter}");
        echo json_encode([
            'error' => 'Too Many Requests',
            'message' => 'Rate limit exceeded. Please try again later.',
            'retry_after' => $retryAfter
        ]);
        exit;
    }
}
