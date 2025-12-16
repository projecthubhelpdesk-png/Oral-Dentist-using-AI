<?php
/**
 * Encryption Service Tests
 * Oral Care AI - PHP Backend
 * 
 * Run: ./vendor/bin/phpunit tests/EncryptionServiceTest.php
 */

use PHPUnit\Framework\TestCase;

require_once __DIR__ . '/../services/EncryptionService.php';

class EncryptionServiceTest extends TestCase {
    private $encryption;
    
    protected function setUp(): void {
        $this->encryption = new OralCareAI\Services\EncryptionService();
    }
    
    public function testEncryptAndDecrypt(): void {
        $plaintext = 'Sensitive patient data';
        
        $encrypted = $this->encryption->encrypt($plaintext);
        $decrypted = $this->encryption->decrypt($encrypted);
        
        $this->assertNotEquals($plaintext, $encrypted);
        $this->assertEquals($plaintext, $decrypted);
    }
    
    public function testEncryptProducesDifferentOutputsForSameInput(): void {
        $plaintext = 'Same input';
        
        $encrypted1 = $this->encryption->encrypt($plaintext);
        $encrypted2 = $this->encryption->encrypt($plaintext);
        
        // Due to random IV, same plaintext should produce different ciphertext
        $this->assertNotEquals($encrypted1, $encrypted2);
        
        // But both should decrypt to same value
        $this->assertEquals($plaintext, $this->encryption->decrypt($encrypted1));
        $this->assertEquals($plaintext, $this->encryption->decrypt($encrypted2));
    }
    
    public function testHashPhoneReturnsHashAndLastFour(): void {
        $phone = '555-123-4567';
        
        $result = $this->encryption->hashPhone($phone);
        
        $this->assertArrayHasKey('hash', $result);
        $this->assertArrayHasKey('last_four', $result);
        $this->assertEquals('4567', $result['last_four']);
        $this->assertStringContainsString(':', $result['hash']); // Contains salt separator
    }
    
    public function testHashPhoneNormalizesInput(): void {
        $phone1 = '555-123-4567';
        $phone2 = '(555) 123-4567';
        $phone3 = '+1 555 123 4567';
        
        $result1 = $this->encryption->hashPhone($phone1);
        $result2 = $this->encryption->hashPhone($phone2);
        $result3 = $this->encryption->hashPhone($phone3);
        
        // All should have same last four
        $this->assertEquals('4567', $result1['last_four']);
        $this->assertEquals('4567', $result2['last_four']);
        $this->assertEquals('4567', $result3['last_four']);
    }
    
    public function testVerifyPhoneWithCorrectNumber(): void {
        $phone = '555-123-4567';
        $hashData = $this->encryption->hashPhone($phone);
        
        $result = $this->encryption->verifyPhone($phone, $hashData['hash']);
        
        $this->assertTrue($result);
    }
    
    public function testVerifyPhoneWithIncorrectNumber(): void {
        $phone = '555-123-4567';
        $hashData = $this->encryption->hashPhone($phone);
        
        $result = $this->encryption->verifyPhone('555-999-9999', $hashData['hash']);
        
        $this->assertFalse($result);
    }
    
    public function testHashPasswordProducesValidBcryptHash(): void {
        $password = 'securePassword123';
        
        $hash = $this->encryption->hashPassword($password);
        
        $this->assertStringStartsWith('$2y$', $hash);
        $this->assertTrue(password_verify($password, $hash));
    }
    
    public function testVerifyPasswordWithCorrectPassword(): void {
        $password = 'testPassword';
        $hash = $this->encryption->hashPassword($password);
        
        $result = $this->encryption->verifyPassword($password, $hash);
        
        $this->assertTrue($result);
    }
    
    public function testVerifyPasswordWithIncorrectPassword(): void {
        $hash = $this->encryption->hashPassword('correctPassword');
        
        $result = $this->encryption->verifyPassword('wrongPassword', $hash);
        
        $this->assertFalse($result);
    }
    
    public function testDecryptWithInvalidDataReturnsNull(): void {
        $result = $this->encryption->decrypt('invalid-base64-data');
        
        $this->assertNull($result);
    }
}
