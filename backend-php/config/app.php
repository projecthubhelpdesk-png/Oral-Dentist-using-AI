<?php
/**
 * Application Configuration
 * Oral Care AI - PHP Backend
 */

return [
    'name' => 'Oral Care AI',
    'version' => '1.0.0',
    'debug' => getenv('APP_DEBUG') === 'true',
    
    // JWT Configuration
    'jwt' => [
        'secret' => getenv('JWT_SECRET') ?: 'change-this-secret-in-production-min-32-chars',
        'algorithm' => 'HS256',
        'access_token_ttl' => 86400,    // 24 hours (for development)
        'refresh_token_ttl' => 604800,  // 7 days
        'issuer' => 'oral-care-ai',
    ],
    
    // Encryption for PII
    'encryption' => [
        'key' => getenv('ENCRYPTION_KEY') ?: 'change-this-encryption-key-32ch',
        'cipher' => 'aes-256-gcm',
    ],
    
    // File uploads
    'uploads' => [
        'max_size' => 10 * 1024 * 1024,  // 10MB
        'allowed_types' => ['image/jpeg', 'image/png', 'image/webp'],
        'storage_path' => __DIR__ . '/../storage/scans/',
    ],
    
    // Rate limiting
    'rate_limit' => [
        'enabled' => true,
        'requests_per_minute' => 120,
        'login_attempts_per_hour' => 30,
    ],
    
    // CORS
    'cors' => [
        'allowed_origins' => ['http://localhost:5173', 'http://localhost:3000', 'http://localhost'],
        'allowed_methods' => ['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'OPTIONS'],
        'allowed_headers' => ['Content-Type', 'Authorization', 'X-Requested-With'],
        'max_age' => 86400,
    ],
    
    // ML API Configuration
    'ml' => [
        'api_url' => getenv('ML_API_URL') ?: 'http://localhost:8000',
        'timeout' => 120,
    ],
];
