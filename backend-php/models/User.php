<?php
/**
 * User Model
 * Oral Care AI - PHP Backend
 */

namespace OralCareAI\Models;

require_once __DIR__ . '/../config/database.php';
require_once __DIR__ . '/../services/EncryptionService.php';

use Database;
use OralCareAI\Services\EncryptionService;

class User {
    private \PDO $db;
    private EncryptionService $encryption;
    
    public function __construct() {
        $this->db = Database::getInstance();
        $this->encryption = new EncryptionService();
    }
    
    /**
     * Create a new user
     */
    public function create(array $data): ?array {
        $id = $this->generateUuid();
        $passwordHash = $this->encryption->hashPassword($data['password']);
        
        // Handle phone number
        $phoneHash = null;
        $phoneLastFour = null;
        if (!empty($data['phone'])) {
            $phoneData = $this->encryption->hashPhone($data['phone']);
            $phoneHash = $phoneData['hash'];
            $phoneLastFour = $phoneData['last_four'];
        }
        
        // Encrypt PII
        $firstNameEncrypted = !empty($data['first_name']) 
            ? $this->encryption->encrypt($data['first_name']) 
            : null;
        $lastNameEncrypted = !empty($data['last_name']) 
            ? $this->encryption->encrypt($data['last_name']) 
            : null;
        
        $stmt = $this->db->prepare("
            INSERT INTO users (id, email, password_hash, role, phone_hash, phone_last_four, 
                               first_name_encrypted, last_name_encrypted, email_verified)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, FALSE)
        ");
        
        try {
            $stmt->execute([
                $id,
                strtolower($data['email']),
                $passwordHash,
                $data['role'] ?? 'user',
                $phoneHash,
                $phoneLastFour,
                $firstNameEncrypted,
                $lastNameEncrypted
            ]);
            
            return $this->findById($id);
        } catch (\PDOException $e) {
            if ($e->getCode() == 23000) { // Duplicate entry
                return null;
            }
            throw $e;
        }
    }
    
    /**
     * Find user by ID
     */
    public function findById(string $id): ?array {
        $stmt = $this->db->prepare("
            SELECT id, email, role, phone_last_four, profile_image_url, 
                   first_name, last_name, email_verified, is_active, created_at, last_login_at
            FROM users WHERE id = ? AND is_active = TRUE
        ");
        $stmt->execute([$id]);
        $user = $stmt->fetch();
        
        if ($user) {
            $user['firstName'] = $user['first_name'];
            $user['lastName'] = $user['last_name'];
        }
        
        return $user ?: null;
    }
    
    /**
     * Find user by email (for login)
     */
    public function findByEmail(string $email): ?array {
        $stmt = $this->db->prepare("
            SELECT id, email, password_hash, role, phone_last_four, 
                   first_name, last_name, email_verified, is_active, created_at
            FROM users WHERE email = ? AND is_active = TRUE
        ");
        $stmt->execute([strtolower($email)]);
        $user = $stmt->fetch();
        
        if ($user) {
            $user['firstName'] = $user['first_name'];
            $user['lastName'] = $user['last_name'];
        }
        
        return $user ?: null;
    }
    
    /**
     * Verify password
     */
    public function verifyPassword(string $password, string $hash): bool {
        return $this->encryption->verifyPassword($password, $hash);
    }
    
    /**
     * Update last login timestamp
     */
    public function updateLastLogin(string $id): void {
        $stmt = $this->db->prepare("UPDATE users SET last_login_at = NOW() WHERE id = ?");
        $stmt->execute([$id]);
    }
    
    /**
     * Update user profile
     */
    public function update(string $id, array $data): bool {
        $updates = [];
        $params = [];
        
        if (isset($data['phone'])) {
            $phoneData = $this->encryption->hashPhone($data['phone']);
            $updates[] = 'phone_hash = ?';
            $updates[] = 'phone_last_four = ?';
            $params[] = $phoneData['hash'];
            $params[] = $phoneData['last_four'];
        }
        
        if (isset($data['first_name'])) {
            $updates[] = 'first_name_encrypted = ?';
            $params[] = $this->encryption->encrypt($data['first_name']);
        }
        
        if (isset($data['last_name'])) {
            $updates[] = 'last_name_encrypted = ?';
            $params[] = $this->encryption->encrypt($data['last_name']);
        }
        
        if (isset($data['profile_image_url'])) {
            $updates[] = 'profile_image_url = ?';
            $params[] = $data['profile_image_url'];
        }
        
        if (empty($updates)) {
            return false;
        }
        
        $params[] = $id;
        $sql = "UPDATE users SET " . implode(', ', $updates) . " WHERE id = ?";
        
        $stmt = $this->db->prepare($sql);
        return $stmt->execute($params);
    }
    
    /**
     * Get decrypted user data (for authorized access only)
     */
    public function getFullProfile(string $id): ?array {
        $stmt = $this->db->prepare("
            SELECT id, email, role, phone_last_four, profile_image_url,
                   first_name_encrypted, last_name_encrypted, date_of_birth_encrypted,
                   email_verified, is_active, created_at, last_login_at
            FROM users WHERE id = ? AND is_active = TRUE
        ");
        $stmt->execute([$id]);
        $user = $stmt->fetch();
        
        if (!$user) {
            return null;
        }
        
        // Decrypt PII fields
        if ($user['first_name_encrypted']) {
            $user['first_name'] = $this->encryption->decrypt($user['first_name_encrypted']);
        }
        if ($user['last_name_encrypted']) {
            $user['last_name'] = $this->encryption->decrypt($user['last_name_encrypted']);
        }
        if ($user['date_of_birth_encrypted']) {
            $user['date_of_birth'] = $this->encryption->decrypt($user['date_of_birth_encrypted']);
        }
        
        // Remove encrypted fields from response
        unset($user['first_name_encrypted'], $user['last_name_encrypted'], $user['date_of_birth_encrypted']);
        
        return $user;
    }
    
    private function generateUuid(): string {
        return sprintf(
            '%04x%04x-%04x-%04x-%04x-%04x%04x%04x',
            mt_rand(0, 0xffff), mt_rand(0, 0xffff),
            mt_rand(0, 0xffff),
            mt_rand(0, 0x0fff) | 0x4000,
            mt_rand(0, 0x3fff) | 0x8000,
            mt_rand(0, 0xffff), mt_rand(0, 0xffff), mt_rand(0, 0xffff)
        );
    }
}
