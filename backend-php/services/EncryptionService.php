<?php
/**
 * Encryption Service for PII
 * Oral Care AI - PHP Backend
 * 
 * Handles encryption/decryption of sensitive data (names, addresses, etc.)
 * and hashing of phone numbers for lookup.
 */

namespace OralCareAI\Services;

class EncryptionService {
    private string $key;
    private string $cipher;
    
    public function __construct() {
        $config = require __DIR__ . '/../config/app.php';
        $this->key = $config['encryption']['key'];
        $this->cipher = $config['encryption']['cipher'];
        
        if (strlen($this->key) !== 32) {
            throw new \Exception('Encryption key must be exactly 32 characters');
        }
    }
    
    /**
     * Encrypt sensitive data (names, addresses, etc.)
     */
    public function encrypt(string $plaintext): string {
        $ivLength = openssl_cipher_iv_length($this->cipher);
        $iv = openssl_random_pseudo_bytes($ivLength);
        
        $encrypted = openssl_encrypt(
            $plaintext,
            $this->cipher,
            $this->key,
            OPENSSL_RAW_DATA,
            $iv,
            $tag
        );
        
        // Combine IV + tag + ciphertext for storage
        return base64_encode($iv . $tag . $encrypted);
    }
    
    /**
     * Decrypt sensitive data
     */
    public function decrypt(string $ciphertext): ?string {
        $data = base64_decode($ciphertext);
        
        $ivLength = openssl_cipher_iv_length($this->cipher);
        $tagLength = 16; // GCM tag is 16 bytes
        
        $iv = substr($data, 0, $ivLength);
        $tag = substr($data, $ivLength, $tagLength);
        $encrypted = substr($data, $ivLength + $tagLength);
        
        $decrypted = openssl_decrypt(
            $encrypted,
            $this->cipher,
            $this->key,
            OPENSSL_RAW_DATA,
            $iv,
            $tag
        );
        
        return $decrypted !== false ? $decrypted : null;
    }
    
    /**
     * Hash phone number for lookup (with salt)
     * Returns both the hash and the last 4 digits for display
     */
    public function hashPhone(string $phone): array {
        // Normalize phone number (remove non-digits)
        $normalized = preg_replace('/[^0-9]/', '', $phone);
        
        // Generate a unique salt per phone
        $salt = bin2hex(random_bytes(16));
        
        // Create hash with salt
        $hash = hash('sha256', $normalized . $salt);
        
        // Get last 4 digits for display
        $lastFour = substr($normalized, -4);
        
        return [
            'hash' => $hash . ':' . $salt,  // Store salt with hash
            'last_four' => $lastFour
        ];
    }
    
    /**
     * Verify phone number against stored hash
     */
    public function verifyPhone(string $phone, string $storedHash): bool {
        $normalized = preg_replace('/[^0-9]/', '', $phone);
        
        // Extract salt from stored hash
        $parts = explode(':', $storedHash);
        if (count($parts) !== 2) {
            return false;
        }
        
        [$hash, $salt] = $parts;
        
        // Recreate hash and compare
        $testHash = hash('sha256', $normalized . $salt);
        
        return hash_equals($hash, $testHash);
    }
    
    /**
     * Hash password using bcrypt
     */
    public function hashPassword(string $password): string {
        return password_hash($password, PASSWORD_BCRYPT, ['cost' => 12]);
    }
    
    /**
     * Verify password against hash
     */
    public function verifyPassword(string $password, string $hash): bool {
        return password_verify($password, $hash);
    }
}
