<?php
/**
 * Scan Model
 * Oral Care AI - PHP Backend
 */

namespace OralCareAI\Models;

require_once __DIR__ . '/../config/database.php';

use Database;

class Scan {
    private \PDO $db;
    
    public function __construct() {
        $this->db = Database::getInstance();
    }
    
    /**
     * Create a new scan record
     */
    public function create(array $data): ?array {
        $id = $this->generateUuid();
        
        $stmt = $this->db->prepare("
            INSERT INTO scans (id, user_id, scan_type, image_storage_path, image_hash, 
                               thumbnail_path, original_filename, file_size_bytes, 
                               mime_type, capture_device, metadata_json, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'uploaded')
        ");
        
        $stmt->execute([
            $id,
            $data['user_id'],
            $data['scan_type'],
            $data['image_storage_path'],
            $data['image_hash'],
            $data['thumbnail_path'] ?? null,
            $data['original_filename'] ?? null,
            $data['file_size_bytes'] ?? 0,
            $data['mime_type'] ?? 'image/jpeg',
            $data['capture_device'] ?? null,
            isset($data['metadata']) ? json_encode($data['metadata']) : null
        ]);
        
        return $this->findById($id);
    }
    
    /**
     * Find scan by ID with analysis data
     */
    public function findById(string $id): ?array {
        $stmt = $this->db->prepare("
            SELECT s.id, s.user_id, s.scan_type, s.thumbnail_path, s.original_filename,
                   s.file_size_bytes, s.mime_type, s.capture_device, s.metadata_json,
                   s.status, s.uploaded_at, s.processed_at,
                   ar.id as analysis_id, ar.overall_score, ar.confidence_score,
                   ar.findings_json, ar.risk_areas_json, ar.recommendations_json,
                   ar.model_version, ar.created_at as analysis_date
            FROM scans s
            LEFT JOIN analysis_results ar ON s.id = ar.scan_id
            WHERE s.id = ?
            ORDER BY ar.created_at DESC
            LIMIT 1
        ");
        $stmt->execute([$id]);
        $scan = $stmt->fetch();
        
        if (!$scan) {
            return null;
        }
        
        return $this->formatScan($scan, true);
    }
    
    /**
     * Find scans by user ID with pagination
     */
    public function findByUserId(string $userId, array $filters = []): array {
        $where = ['user_id = ?'];
        $params = [$userId];
        
        if (!empty($filters['status'])) {
            $where[] = 'status = ?';
            $params[] = $filters['status'];
        }
        
        if (!empty($filters['scan_type'])) {
            $where[] = 'scan_type = ?';
            $params[] = $filters['scan_type'];
        }
        
        $limit = min($filters['limit'] ?? 20, 100);
        $offset = $filters['offset'] ?? 0;
        
        // Get total count
        $countStmt = $this->db->prepare("SELECT COUNT(*) FROM scans WHERE " . implode(' AND ', $where));
        $countStmt->execute($params);
        $total = (int) $countStmt->fetchColumn();
        
        // Get paginated results with analysis data
        $sql = "
            SELECT s.id, s.user_id, s.scan_type, s.thumbnail_path, s.original_filename,
                   s.status, s.capture_device, s.uploaded_at, s.processed_at,
                   ar.id as analysis_id, ar.overall_score, ar.confidence_score,
                   ar.findings_json, ar.risk_areas_json, ar.recommendations_json,
                   ar.model_version, ar.created_at as analysis_date
            FROM scans s
            LEFT JOIN analysis_results ar ON s.id = ar.scan_id
            WHERE " . implode(' AND ', array_map(fn($w) => 's.' . substr($w, 0, strpos($w, ' ')) . substr($w, strpos($w, ' ')), $where)) . "
            ORDER BY s.uploaded_at DESC
            LIMIT {$limit} OFFSET {$offset}
        ";
        
        $stmt = $this->db->prepare($sql);
        $stmt->execute($params);
        $scans = $stmt->fetchAll();
        
        return [
            'data' => array_map(fn($s) => $this->formatScan($s, true), $scans),
            'total' => $total,
            'limit' => $limit,
            'offset' => $offset
        ];
    }
    
    /**
     * Format scan to camelCase for frontend
     */
    private function formatScan(array $scan, bool $includeAnalysis = false): array {
        $formatted = [
            'id' => $scan['id'],
            'userId' => $scan['user_id'],
            'scanType' => $scan['scan_type'],
            'status' => $scan['status'],
            'uploadedAt' => $scan['uploaded_at'],
            'processedAt' => $scan['processed_at'],
            // Always include image URL for the scan
            'imageUrl' => '/oral-care-ai/backend-php/api/scans/' . $scan['id'] . '/image',
            'thumbnailUrl' => '/oral-care-ai/backend-php/api/scans/' . $scan['id'] . '/image',
        ];
        
        if (isset($scan['original_filename'])) {
            $formatted['originalFilename'] = $scan['original_filename'];
        }
        if (isset($scan['capture_device'])) {
            $formatted['captureDevice'] = $scan['capture_device'];
        }
        if (isset($scan['file_size_bytes'])) {
            $formatted['fileSizeBytes'] = (int)$scan['file_size_bytes'];
        }
        if (isset($scan['mime_type'])) {
            $formatted['mimeType'] = $scan['mime_type'];
        }
        if (isset($scan['metadata_json'])) {
            $formatted['metadata'] = json_decode($scan['metadata_json'], true);
        }
        
        // Include analysis data if available
        if ($includeAnalysis && !empty($scan['analysis_id'])) {
            $findingsData = json_decode($scan['findings_json'], true);
            
            $formatted['analysis'] = [
                'id' => $scan['analysis_id'],
                'overallScore' => (float) $scan['overall_score'],
                'confidenceScore' => (float) $scan['confidence_score'],
                'findings' => $findingsData['findings'] ?? [],
                'riskAreas' => json_decode($scan['risk_areas_json'], true)['regions'] ?? [],
                'recommendations' => json_decode($scan['recommendations_json'], true)['recommendations'] ?? [],
                'modelVersion' => $scan['model_version'],
                'analysisDate' => $scan['analysis_date'],
            ];
            
            // Include GPT-4o guidance if available
            if (isset($findingsData['gpt4o_guidance'])) {
                $formatted['analysis']['gpt4o_guidance'] = $findingsData['gpt4o_guidance'];
            }
            
            // Include enhanced features if available
            if (isset($findingsData['enhanced_features'])) {
                $formatted['analysis']['enhanced_features'] = $findingsData['enhanced_features'];
            }
        }
        
        return $formatted;
    }
    
    /**
     * Update scan status
     */
    public function updateStatus(string $id, string $status): bool {
        $stmt = $this->db->prepare("
            UPDATE scans 
            SET status = ?, processed_at = CASE WHEN ? IN ('analyzed', 'failed') THEN NOW() ELSE processed_at END
            WHERE id = ?
        ");
        return $stmt->execute([$status, $status, $id]);
    }
    
    /**
     * Archive scan (soft delete)
     */
    public function archive(string $id): bool {
        return $this->updateStatus($id, 'archived');
    }
    
    /**
     * Check if user owns scan
     */
    public function isOwner(string $scanId, string $userId): bool {
        $stmt = $this->db->prepare("SELECT 1 FROM scans WHERE id = ? AND user_id = ?");
        $stmt->execute([$scanId, $userId]);
        return (bool) $stmt->fetch();
    }
    
    /**
     * Get secure image path (for authorized access)
     */
    public function getImagePath(string $id): ?string {
        $stmt = $this->db->prepare("SELECT image_storage_path FROM scans WHERE id = ?");
        $stmt->execute([$id]);
        $result = $stmt->fetch();
        return $result ? $result['image_storage_path'] : null;
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
