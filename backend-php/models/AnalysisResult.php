<?php
/**
 * Analysis Result Model
 * Oral Care AI - PHP Backend
 */

namespace OralCareAI\Models;

require_once __DIR__ . '/../config/database.php';

use Database;

class AnalysisResult {
    private \PDO $db;
    
    public function __construct() {
        $this->db = Database::getInstance();
    }
    
    /**
     * Create analysis result
     */
    public function create(array $data): ?array {
        $id = $this->generateUuid();
        
        $stmt = $this->db->prepare("
            INSERT INTO analysis_results (id, scan_id, model_type, model_version, 
                                          overall_score, confidence_score, findings_json,
                                          risk_areas_json, recommendations_json, processing_time_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ");
        
        $stmt->execute([
            $id,
            $data['scan_id'],
            $data['model_type'],
            $data['model_version'],
            $data['overall_score'] ?? null,
            $data['confidence_score'] ?? null,
            json_encode($data['findings'] ?? []),
            json_encode($data['risk_areas'] ?? []),
            json_encode($data['recommendations'] ?? []),
            $data['processing_time_ms'] ?? 0
        ]);
        
        return $this->findById($id);
    }
    
    /**
     * Find by ID
     */
    public function findById(string $id): ?array {
        $stmt = $this->db->prepare("
            SELECT id, scan_id, model_type, model_version, overall_score, confidence_score,
                   findings_json, risk_areas_json, recommendations_json, processing_time_ms, created_at
            FROM analysis_results WHERE id = ?
        ");
        $stmt->execute([$id]);
        
        return $this->formatResult($stmt->fetch());
    }
    
    /**
     * Find by scan ID
     */
    public function findByScanId(string $scanId): ?array {
        $stmt = $this->db->prepare("
            SELECT id, scan_id, model_type, model_version, overall_score, confidence_score,
                   findings_json, risk_areas_json, recommendations_json, processing_time_ms, created_at
            FROM analysis_results WHERE scan_id = ?
            ORDER BY created_at DESC LIMIT 1
        ");
        $stmt->execute([$scanId]);
        
        return $this->formatResult($stmt->fetch());
    }
    
    /**
     * Get analysis history for a user
     */
    public function findByUserId(string $userId, int $limit = 10): array {
        $stmt = $this->db->prepare("
            SELECT ar.id, ar.scan_id, ar.model_type, ar.model_version, 
                   ar.overall_score, ar.confidence_score, ar.created_at,
                   s.scan_type, s.thumbnail_path
            FROM analysis_results ar
            JOIN scans s ON ar.scan_id = s.id
            WHERE s.user_id = ?
            ORDER BY ar.created_at DESC
            LIMIT ?
        ");
        $stmt->execute([$userId, $limit]);
        
        return $stmt->fetchAll();
    }
    
    /**
     * Get score trend for user
     */
    public function getScoreTrend(string $userId, int $days = 90): array {
        $stmt = $this->db->prepare("
            SELECT DATE(ar.created_at) as date, AVG(ar.overall_score) as avg_score
            FROM analysis_results ar
            JOIN scans s ON ar.scan_id = s.id
            WHERE s.user_id = ? AND ar.created_at >= DATE_SUB(NOW(), INTERVAL ? DAY)
            GROUP BY DATE(ar.created_at)
            ORDER BY date ASC
        ");
        $stmt->execute([$userId, $days]);
        
        return $stmt->fetchAll();
    }
    
    private function formatResult($result): ?array {
        if (!$result) {
            return null;
        }
        
        $findings = json_decode($result['findings_json'], true);
        $riskAreas = json_decode($result['risk_areas_json'], true);
        $recommendations = json_decode($result['recommendations_json'], true);
        
        return [
            'id' => $result['id'],
            'scanId' => $result['scan_id'],
            'modelType' => $result['model_type'],
            'modelVersion' => $result['model_version'],
            'overallScore' => (float)$result['overall_score'],
            'confidenceScore' => (float)$result['confidence_score'],
            'findings' => $findings['findings'] ?? $findings ?? [],
            'riskAreas' => $riskAreas['regions'] ?? $riskAreas ?? [],
            'recommendations' => $recommendations['recommendations'] ?? $recommendations ?? [],
            'processingTimeMs' => (int)$result['processing_time_ms'],
            'createdAt' => $result['created_at'],
        ];
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
