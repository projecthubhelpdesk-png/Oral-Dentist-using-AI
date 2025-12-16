<?php
/**
 * Audit Logging Service
 * Oral Care AI - PHP Backend
 * 
 * Logs all security-relevant actions for compliance.
 */

namespace OralCareAI\Services;

require_once __DIR__ . '/../config/database.php';

use Database;

class AuditService {
    private \PDO $db;
    
    public function __construct() {
        $this->db = Database::getInstance();
    }
    
    /**
     * Log an action
     */
    public function log(
        string $action,
        ?string $userId = null,
        ?string $resourceType = null,
        ?string $resourceId = null,
        ?array $details = null
    ): void {
        $stmt = $this->db->prepare("
            INSERT INTO audit_logs (user_id, action, resource_type, resource_id, ip_address, user_agent, details_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ");
        
        $stmt->execute([
            $userId,
            $action,
            $resourceType,
            $resourceId,
            $_SERVER['REMOTE_ADDR'] ?? null,
            $_SERVER['HTTP_USER_AGENT'] ?? null,
            $details ? json_encode($details) : null
        ]);
    }
    
    /**
     * Common audit actions
     */
    public function logLogin(string $userId, bool $success): void {
        $this->log(
            $success ? 'LOGIN_SUCCESS' : 'LOGIN_FAILED',
            $userId,
            'user',
            $userId,
            ['success' => $success]
        );
    }
    
    public function logLogout(string $userId): void {
        $this->log('LOGOUT', $userId, 'user', $userId);
    }
    
    public function logScanUpload(string $userId, string $scanId): void {
        $this->log('SCAN_UPLOAD', $userId, 'scan', $scanId);
    }
    
    public function logScanView(string $userId, string $scanId): void {
        $this->log('SCAN_VIEW', $userId, 'scan', $scanId);
    }
    
    public function logAnalysisView(string $userId, string $analysisId): void {
        $this->log('ANALYSIS_VIEW', $userId, 'analysis', $analysisId);
    }
    
    public function logDentistReview(string $dentistId, string $scanId, string $reviewId): void {
        $this->log('DENTIST_REVIEW', $dentistId, 'review', $reviewId, ['scan_id' => $scanId]);
    }
    
    public function logDataExport(string $userId, string $dataType): void {
        $this->log('DATA_EXPORT', $userId, $dataType, null, ['type' => $dataType]);
    }
    
    public function logProfileUpdate(string $userId, array $fields): void {
        $this->log('PROFILE_UPDATE', $userId, 'user', $userId, ['fields' => $fields]);
    }
}
