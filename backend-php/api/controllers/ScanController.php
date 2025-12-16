<?php
/**
 * Scan Controller
 * Oral Care AI - PHP Backend
 */

namespace OralCareAI\Controllers;

require_once __DIR__ . '/../../models/Scan.php';
require_once __DIR__ . '/../../middleware/AuthMiddleware.php';
require_once __DIR__ . '/../../services/AuditService.php';

use OralCareAI\Models\Scan;
use OralCareAI\Middleware\AuthMiddleware;
use OralCareAI\Services\AuditService;

class ScanController {
    private Scan $scanModel;
    private AuthMiddleware $auth;
    private AuditService $audit;
    private array $config;
    
    public function __construct() {
        $this->scanModel = new Scan();
        $this->auth = new AuthMiddleware();
        $this->audit = new AuditService();
        $this->config = require __DIR__ . '/../../config/app.php';
    }
    
    /**
     * GET /scans
     */
    public function index(array $params, array $body): void {
        $user = $this->auth->authenticate();
        if (!$user) return;
        
        $filters = [
            'status' => $_GET['status'] ?? null,
            'scan_type' => $_GET['scanType'] ?? null,
            'limit' => (int) ($_GET['limit'] ?? 20),
            'offset' => (int) ($_GET['offset'] ?? 0),
        ];
        
        $result = $this->scanModel->findByUserId($user['user_id'], $filters);
        
        echo json_encode($result);
    }
    
    /**
     * GET /scans/patients - Get scans from connected patients (for dentists)
     */
    public function patientScans(array $params, array $body): void {
        $user = $this->auth->authenticate();
        if (!$user) return;
        
        if ($user['role'] !== 'dentist') {
            http_response_code(403);
            echo json_encode(['error' => 'Forbidden', 'message' => 'Only dentists can access patient scans']);
            return;
        }
        
        $limit = (int) ($_GET['limit'] ?? 20);
        $offset = (int) ($_GET['offset'] ?? 0);
        $status = $_GET['status'] ?? null;
        
        // Get scans from connected patients who share scan history with analysis data
        $sql = "
            SELECT s.*, u.email as patient_email,
                   u.first_name as patient_first_name, u.last_name as patient_last_name,
                   ar.id as analysis_id, ar.overall_score, ar.confidence_score,
                   ar.findings_json, ar.risk_areas_json, ar.recommendations_json,
                   ar.model_version, ar.created_at as analysis_date
            FROM scans s
            JOIN patient_dentist_connections pdc ON s.user_id = pdc.patient_id
            JOIN users u ON s.user_id = u.id
            LEFT JOIN analysis_results ar ON s.id = ar.scan_id
            WHERE pdc.dentist_id = ? 
            AND pdc.status = 'active'
            AND pdc.share_scan_history = 1
        ";
        $sqlParams = [$user['user_id']];
        
        if ($status) {
            $sql .= " AND s.status = ?";
            $sqlParams[] = $status;
        }
        
        $sql .= " ORDER BY s.uploaded_at DESC LIMIT ? OFFSET ?";
        $sqlParams[] = $limit;
        $sqlParams[] = $offset;
        
        require_once __DIR__ . '/../../config/database.php';
        $db = \Database::getInstance();
        
        $stmt = $db->prepare($sql);
        $stmt->execute($sqlParams);
        $scans = $stmt->fetchAll();
        
        // Format scans with analysis
        $formatted = array_map(function($scan) {
            $result = [
                'id' => $scan['id'],
                'userId' => $scan['user_id'],
                'patientEmail' => $scan['patient_email'],
                'patientName' => trim(($scan['patient_first_name'] ?? '') . ' ' . ($scan['patient_last_name'] ?? '')) ?: null,
                'scanType' => $scan['scan_type'],
                'status' => $scan['status'],
                'captureDevice' => $scan['capture_device'],
                'uploadedAt' => $scan['uploaded_at'],
                'processedAt' => $scan['processed_at'],
                'imageUrl' => '/oral-care-ai/backend-php/api/scans/' . $scan['id'] . '/image',
            ];
            
            // Include analysis if available
            if (!empty($scan['analysis_id'])) {
                $findingsData = json_decode($scan['findings_json'], true);
                
                $result['analysis'] = [
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
                    $result['analysis']['gpt4o_guidance'] = $findingsData['gpt4o_guidance'];
                }
                
                // Include enhanced features if available
                if (isset($findingsData['enhanced_features'])) {
                    $result['analysis']['enhanced_features'] = $findingsData['enhanced_features'];
                }
            }
            
            return $result;
        }, $scans);
        
        echo json_encode(['data' => $formatted]);
    }
    
    /**
     * POST /scans
     */
    public function store(array $params, array $body): void {
        $user = $this->auth->authenticate();
        if (!$user) return;
        
        // Validate file upload
        if (!isset($_FILES['image']) || $_FILES['image']['error'] !== UPLOAD_ERR_OK) {
            http_response_code(400);
            echo json_encode(['error' => 'Bad Request', 'message' => 'Image file is required']);
            return;
        }
        
        $file = $_FILES['image'];
        $scanType = $_POST['scanType'] ?? 'basic_rgb';
        
        // Validate file type
        $finfo = finfo_open(FILEINFO_MIME_TYPE);
        $mimeType = finfo_file($finfo, $file['tmp_name']);
        finfo_close($finfo);
        
        if (!in_array($mimeType, $this->config['uploads']['allowed_types'])) {
            http_response_code(400);
            echo json_encode(['error' => 'Bad Request', 'message' => 'Invalid file type. Allowed: JPEG, PNG, WebP']);
            return;
        }
        
        // Validate file size
        if ($file['size'] > $this->config['uploads']['max_size']) {
            http_response_code(400);
            echo json_encode(['error' => 'Bad Request', 'message' => 'File too large. Max: 10MB']);
            return;
        }
        
        // Generate secure filename and path
        $imageHash = hash_file('sha256', $file['tmp_name']);
        $extension = pathinfo($file['name'], PATHINFO_EXTENSION);
        $secureFilename = $imageHash . '_' . time() . '.' . $extension;
        $storagePath = $this->config['uploads']['storage_path'];
        $datePath = date('Y/m/');
        
        // Create directory if not exists
        $fullPath = $storagePath . $datePath;
        if (!is_dir($fullPath)) {
            mkdir($fullPath, 0755, true);
        }
        
        // Move uploaded file
        $finalPath = $fullPath . $secureFilename;
        if (!move_uploaded_file($file['tmp_name'], $finalPath)) {
            http_response_code(500);
            echo json_encode(['error' => 'Server Error', 'message' => 'Failed to save file']);
            return;
        }
        
        // Create scan record
        $scan = $this->scanModel->create([
            'user_id' => $user['user_id'],
            'scan_type' => in_array($scanType, ['basic_rgb', 'advanced_spectral']) ? $scanType : 'basic_rgb',
            'image_storage_path' => $datePath . $secureFilename,
            'image_hash' => $imageHash,
            'original_filename' => $file['name'],
            'file_size_bytes' => $file['size'],
            'mime_type' => $mimeType,
            'capture_device' => $_POST['captureDevice'] ?? null,
            'metadata' => [
                'user_agent' => $_SERVER['HTTP_USER_AGENT'] ?? null,
            ]
        ]);
        
        $this->audit->logScanUpload($user['user_id'], $scan['id']);
        
        http_response_code(201);
        echo json_encode($scan);
    }
    
    /**
     * GET /scans/{id}
     */
    public function show(array $params, array $body): void {
        $user = $this->auth->authenticate();
        if (!$user) return;
        
        $scanId = $params['id'];
        $scan = $this->scanModel->findById($scanId);
        
        if (!$scan) {
            http_response_code(404);
            echo json_encode(['error' => 'Not Found', 'message' => 'Scan not found']);
            return;
        }
        
        // Check ownership or dentist connection
        if ($scan['userId'] !== $user['user_id'] && $user['role'] !== 'admin') {
            // TODO: Check if dentist has connection with patient
            if ($user['role'] !== 'dentist') {
                http_response_code(403);
                echo json_encode(['error' => 'Forbidden', 'message' => 'Access denied']);
                return;
            }
        }
        
        $this->audit->logScanView($user['user_id'], $scanId);
        
        echo json_encode($scan);
    }
    
    /**
     * DELETE /scans/{id}
     */
    public function archive(array $params, array $body): void {
        $user = $this->auth->authenticate();
        if (!$user) return;
        
        $scanId = $params['id'];
        
        // Verify ownership
        if (!$this->scanModel->isOwner($scanId, $user['user_id'])) {
            http_response_code(403);
            echo json_encode(['error' => 'Forbidden', 'message' => 'Access denied']);
            return;
        }
        
        $this->scanModel->archive($scanId);
        $this->audit->log('SCAN_ARCHIVED', $user['user_id'], 'scan', $scanId);
        
        echo json_encode(['message' => 'Scan archived successfully']);
    }
    
    /**
     * GET /scans/{id}/image - Serve scan image
     */
    public function image(array $params, array $body): void {
        // Clear any output buffers first
        while (ob_get_level()) {
            ob_end_clean();
        }
        
        // CORS headers for image requests - set early
        header('Access-Control-Allow-Origin: *');
        header('Access-Control-Allow-Methods: GET, OPTIONS');
        header('Access-Control-Allow-Headers: Authorization, Content-Type');
        
        // Accept token from query param for image requests
        $token = $_GET['token'] ?? null;
        if ($token) {
            // URL decode the token in case it was encoded
            $token = urldecode($token);
            $_SERVER['HTTP_AUTHORIZATION'] = 'Bearer ' . $token;
        }
        
        $user = $this->auth->authenticateForImage();
        if (!$user) {
            http_response_code(401);
            header('Content-Type: application/json');
            echo json_encode(['error' => 'Unauthorized', 'message' => 'Invalid or missing token']);
            return;
        }
        
        $scanId = $params['id'];
        $imagePath = $this->scanModel->getImagePath($scanId);
        
        if (!$imagePath) {
            http_response_code(404);
            header('Content-Type: application/json');
            echo json_encode(['error' => 'Not Found', 'message' => 'Image not found in database']);
            return;
        }
        
        // Check ownership or dentist connection
        $scan = $this->scanModel->findById($scanId);
        if ($scan['userId'] !== $user['user_id'] && $user['role'] !== 'admin' && $user['role'] !== 'dentist') {
            http_response_code(403);
            header('Content-Type: application/json');
            echo json_encode(['error' => 'Forbidden', 'message' => 'Access denied']);
            return;
        }
        
        $fullPath = $this->config['uploads']['storage_path'] . $imagePath;
        
        if (!file_exists($fullPath)) {
            http_response_code(404);
            header('Content-Type: application/json');
            echo json_encode(['error' => 'Not Found', 'message' => 'Image file not found on disk: ' . $imagePath]);
            return;
        }
        
        // Serve the image
        $finfo = finfo_open(FILEINFO_MIME_TYPE);
        $mimeType = finfo_file($finfo, $fullPath);
        finfo_close($finfo);
        
        header_remove('Content-Type');
        header('Content-Type: ' . $mimeType);
        header('Content-Length: ' . filesize($fullPath));
        header('Cache-Control: private, max-age=3600');
        readfile($fullPath);
        exit;
    }

    /**
     * POST /scans/{id}/analyze
     */
    public function analyze(array $params, array $body): void {
        $user = $this->auth->authenticate();
        if (!$user) return;
        
        $scanId = $params['id'];
        $scan = $this->scanModel->findById($scanId);
        
        if (!$scan) {
            http_response_code(404);
            echo json_encode(['error' => 'Not Found', 'message' => 'Scan not found']);
            return;
        }
        
        if (!$this->scanModel->isOwner($scanId, $user['user_id'])) {
            http_response_code(403);
            echo json_encode(['error' => 'Forbidden', 'message' => 'Access denied']);
            return;
        }
        
        if ($scan['status'] !== 'uploaded') {
            http_response_code(400);
            echo json_encode(['error' => 'Bad Request', 'message' => 'Scan already processed or processing']);
            return;
        }
        
        // Update status to processing
        $this->scanModel->updateStatus($scanId, 'processing');
        
        try {
            // Run AI analysis
            require_once __DIR__ . '/../../services/AIAnalysisService.php';
            $aiService = new \OralCareAI\Services\AIAnalysisService();
            $result = $aiService->analyzeScan($scanId);
            
            $this->audit->log('ANALYSIS_COMPLETED', $user['user_id'], 'scan', $scanId);
            
            http_response_code(200);
            echo json_encode([
                'message' => 'Analysis complete',
                'scanId' => $scanId,
                'analysisId' => $result['id'],
                'overallScore' => $result['overallScore'],
            ]);
        } catch (\Exception $e) {
            $this->scanModel->updateStatus($scanId, 'failed');
            http_response_code(500);
            echo json_encode(['error' => 'Analysis Failed', 'message' => $e->getMessage()]);
        }
    }
}
