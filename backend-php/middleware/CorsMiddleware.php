<?php
/**
 * CORS Middleware
 * Oral Care AI - PHP Backend
 */

namespace OralCareAI\Middleware;

class CorsMiddleware {
    private array $config;
    
    public function __construct() {
        $appConfig = require __DIR__ . '/../config/app.php';
        $this->config = $appConfig['cors'];
    }
    
    public function handle(): void {
        $origin = $_SERVER['HTTP_ORIGIN'] ?? '';
        $requestUri = $_SERVER['REQUEST_URI'] ?? '';
        
        // For image requests, allow all origins (images loaded via <img> tags)
        if (strpos($requestUri, '/image') !== false) {
            header("Access-Control-Allow-Origin: *");
        } elseif (in_array($origin, $this->config['allowed_origins'])) {
            // Check if origin is allowed for other requests
            header("Access-Control-Allow-Origin: {$origin}");
        } elseif ($origin) {
            // For development, allow localhost with any port
            if (strpos($origin, 'http://localhost') === 0) {
                header("Access-Control-Allow-Origin: {$origin}");
            }
        }
        
        header("Access-Control-Allow-Methods: " . implode(', ', $this->config['allowed_methods']));
        header("Access-Control-Allow-Headers: " . implode(', ', $this->config['allowed_headers']));
        header("Access-Control-Max-Age: " . $this->config['max_age']);
        header("Access-Control-Allow-Credentials: true");
        
        // Handle preflight requests
        if ($_SERVER['REQUEST_METHOD'] === 'OPTIONS') {
            http_response_code(204);
            exit;
        }
    }
}
