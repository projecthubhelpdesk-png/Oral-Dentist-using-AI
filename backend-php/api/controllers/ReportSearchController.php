<?php
/**
 * Report Search Controller
 * Allows dentists to search for patient reports by Report ID
 */

namespace OralCareAI\Controllers;

require_once __DIR__ . '/../../config/database.php';
require_once __DIR__ . '/../../middleware/AuthMiddleware.php';

use OralCareAI\Middleware\AuthMiddleware;
use Database;

class ReportSearchController {
    private \PDO $db;
    private AuthMiddleware $auth;
    
    public function __construct() {
        $this->db = Database::getInstance();
        $this->auth = new AuthMiddleware();
    }
    
    /**
     * GET /reports/search?reportId=RPT-YYYYMMDD-XXXXXXXX
     * Search for a scan by Report ID
     */
    public function search(array $params, array $body): void {
        $user = $this->auth->authenticate();
        if (!$user) return;
        
        // Only dentists can search reports
        if ($user['role'] !== 'dentist' && $user['role'] !== 'admin') {
            http_response_code(403);
            echo json_encode(['error' => 'Forbidden', 'message' => 'Only dentists can search reports']);
            return;
        }
        
        $reportId = $_GET['reportId'] ?? '';
        
        if (empty($reportId)) {
            http_response_code(400);
            echo json_encode(['error' => 'Bad Request', 'message' => 'Report ID is required']);
            return;
        }
        
        // Parse the Report ID format: RPT-YYYYMMDD-XXXXXXXX
        // Extract the analysis ID prefix (last 8 chars)
        if (!preg_match('/^RPT-(\d{8})-([A-F0-9]{8})$/i', $reportId, $matches)) {
            http_response_code(400);
            echo json_encode(['error' => 'Bad Request', 'message' => 'Invalid Report ID format. Expected: RPT-YYYYMMDD-XXXXXXXX']);
            return;
        }
        
        $dateStr = $matches[1];
        $analysisIdPrefix = strtolower($matches[2]);
        
        // Search for the analysis by ID prefix
        $stmt = $this->db->prepare("
            SELECT ar.id as analysis_id, ar.scan_id, ar.overall_score, ar.confidence_score,
                   ar.findings_json, ar.risk_areas_json, ar.recommendations_json,
                   ar.model_version, ar.created_at as analysis_date,
                   s.id, s.user_id, s.scan_type, s.status, s.uploaded_at, s.processed_at,
                   s.capture_device,
                   u.email as patient_email,
                   u.first_name as patient_first_name, u.last_name as patient_last_name
            FROM analysis_results ar
            JOIN scans s ON ar.scan_id = s.id
            JOIN users u ON s.user_id = u.id
            WHERE LOWER(ar.id) LIKE ?
            ORDER BY ar.created_at DESC
            LIMIT 1
        ");
        $stmt->execute([$analysisIdPrefix . '%']);
        $result = $stmt->fetch();
        
        if (!$result) {
            http_response_code(404);
            echo json_encode(['error' => 'Not Found', 'message' => 'No report found with this ID']);
            return;
        }
        
        // Verify the date matches
        $scanDate = new \DateTime($result['uploaded_at']);
        $expectedDate = $scanDate->format('Ymd');
        if ($expectedDate !== $dateStr) {
            http_response_code(404);
            echo json_encode(['error' => 'Not Found', 'message' => 'No report found with this ID']);
            return;
        }
        
        // Parse JSON fields
        $findingsData = json_decode($result['findings_json'], true);
        
        // Build patient name
        $patientName = trim(($result['patient_first_name'] ?? '') . ' ' . ($result['patient_last_name'] ?? ''));
        if (empty($patientName)) {
            $patientName = explode('@', $result['patient_email'])[0];
        }

        // Format response
        $scan = [
            'id' => $result['scan_id'],
            'userId' => $result['user_id'],
            'patientEmail' => $result['patient_email'],
            'patientName' => $patientName,
            'scanType' => $result['scan_type'],
            'status' => $result['status'],
            'captureDevice' => $result['capture_device'],
            'uploadedAt' => $result['uploaded_at'],
            'processedAt' => $result['processed_at'],
            'imageUrl' => '/oral-care-ai/backend-php/api/scans/' . $result['scan_id'] . '/image',
            'analysis' => [
                'id' => $result['analysis_id'],
                'overallScore' => (float) $result['overall_score'],
                'confidenceScore' => (float) $result['confidence_score'],
                'findings' => $findingsData['findings'] ?? [],
                'riskAreas' => json_decode($result['risk_areas_json'], true)['regions'] ?? [],
                'recommendations' => json_decode($result['recommendations_json'], true)['recommendations'] ?? [],
                'modelVersion' => $result['model_version'],
                'analysisDate' => $result['analysis_date'],
            ],
            'reportId' => $reportId
        ];
        
        // Include GPT-4o guidance if available
        if (isset($findingsData['gpt4o_guidance'])) {
            $scan['analysis']['gpt4o_guidance'] = $findingsData['gpt4o_guidance'];
        }
        if (isset($findingsData['enhanced_features'])) {
            $scan['analysis']['enhanced_features'] = $findingsData['enhanced_features'];
        }
        
        echo json_encode([
            'success' => true,
            'scan' => $scan
        ]);
    }
}
