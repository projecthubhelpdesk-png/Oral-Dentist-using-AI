<?php
/**
 * User Model Tests
 * Oral Care AI - PHP Backend
 * 
 * Run: ./vendor/bin/phpunit tests/UserModelTest.php
 */

use PHPUnit\Framework\TestCase;

require_once __DIR__ . '/../models/User.php';
require_once __DIR__ . '/../services/EncryptionService.php';

class UserModelTest extends TestCase {
    private $userModel;
    
    protected function setUp(): void {
        $this->userModel = new OralCareAI\Models\User();
    }
    
    public function testCreateUserWithValidData(): void {
        $userData = [
            'email' => 'test_' . time() . '@example.com',
            'password' => 'securePassword123',
            'role' => 'user',
            'phone' => '555-123-4567',
            'first_name' => 'Test',
            'last_name' => 'User'
        ];
        
        $user = $this->userModel->create($userData);
        
        $this->assertNotNull($user);
        $this->assertEquals($userData['email'], $user['email']);
        $this->assertEquals('user', $user['role']);
        $this->assertEquals('4567', $user['phone_last_four']);
    }
    
    public function testCreateUserWithDuplicateEmailReturnsNull(): void {
        $email = 'duplicate_' . time() . '@example.com';
        
        $this->userModel->create([
            'email' => $email,
            'password' => 'password123'
        ]);
        
        $duplicate = $this->userModel->create([
            'email' => $email,
            'password' => 'password456'
        ]);
        
        $this->assertNull($duplicate);
    }
    
    public function testFindByEmailReturnsUser(): void {
        $email = 'findme_' . time() . '@example.com';
        
        $this->userModel->create([
            'email' => $email,
            'password' => 'password123'
        ]);
        
        $found = $this->userModel->findByEmail($email);
        
        $this->assertNotNull($found);
        $this->assertEquals($email, $found['email']);
    }
    
    public function testFindByEmailReturnsNullForNonexistent(): void {
        $found = $this->userModel->findByEmail('nonexistent@example.com');
        $this->assertNull($found);
    }
    
    public function testVerifyPasswordWithCorrectPassword(): void {
        $password = 'correctPassword123';
        $hash = password_hash($password, PASSWORD_BCRYPT);
        
        $result = $this->userModel->verifyPassword($password, $hash);
        
        $this->assertTrue($result);
    }
    
    public function testVerifyPasswordWithIncorrectPassword(): void {
        $hash = password_hash('correctPassword', PASSWORD_BCRYPT);
        
        $result = $this->userModel->verifyPassword('wrongPassword', $hash);
        
        $this->assertFalse($result);
    }
    
    public function testUpdateUserProfile(): void {
        $email = 'update_' . time() . '@example.com';
        $user = $this->userModel->create([
            'email' => $email,
            'password' => 'password123'
        ]);
        
        $updated = $this->userModel->update($user['id'], [
            'phone' => '555-999-8888',
            'first_name' => 'Updated'
        ]);
        
        $this->assertTrue($updated);
        
        $refreshed = $this->userModel->findById($user['id']);
        $this->assertEquals('8888', $refreshed['phone_last_four']);
    }
    
    public function testDentistRoleAssignment(): void {
        $user = $this->userModel->create([
            'email' => 'dentist_' . time() . '@example.com',
            'password' => 'password123',
            'role' => 'dentist'
        ]);
        
        $this->assertEquals('dentist', $user['role']);
    }
}
