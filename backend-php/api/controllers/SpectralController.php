<?php
/**
 * Spectral Analysis Controller
 * Handles advanced spectral imaging AI analysis for dentists
 */

namespace OralCareAI\Controllers;

require_once __DIR__ . '/../../config/database.php';
require_once __DIR__ . '/../../middleware/AuthMiddleware.php';
require_once __DIR__ . '/../../services/AuditService.php';

use OralCareAI\Middleware\AuthMiddleware;
use OralCareAI\Services\AuditService;
use Database;

class SpectralController {
    private \PDO $db;
    private AuthMiddleware $auth;
    private AuditService $audit;
    private array $config;
    
    public function __construct() {
        $this->db = Database::getInstance();
        $this->auth = new AuthMiddleware();
        $this->audit = new AuditService();
        $this->config = require __DIR__ . '/../../config/app.php';
    }
    
    /**
     * POST /spectral/analyze - Run spectral AI analysis (Dentist only)
     */
    public function analyze(array $params, array $body): void {
        $user = $this->auth->authenticate();
        if (!$user) return;
        
        // Only dentists can use spectral analysis
        if ($user['role'] !== 'dentist' && $user['role'] !== 'admin') {
            http_response_code(403);
            echo json_encode(['error' => 'Forbidden', 'message' => 'Spectral analysis is only available for verified dentists']);
            return;
        }
        
        // Validate file upload
        if (!isset($_FILES['image']) || $_FILES['image']['error'] !== UPLOAD_ERR_OK) {
            http_response_code(400);
            echo json_encode(['error' => 'Bad Request', 'message' => 'Spectral image file is required']);
            return;
        }
        
        $file = $_FILES['image'];
        $imageType = $_POST['imageType'] ?? 'nir'; // nir, fluorescence, intraoral
        $patientId = $_POST['patientId'] ?? null;
        
        // Validate file type
        $finfo = finfo_open(FILEINFO_MIME_TYPE);
        $mimeType = finfo_file($finfo, $file['tmp_name']);
        finfo_close($finfo);
        
        $allowedTypes = ['image/jpeg', 'image/png', 'image/webp', 'image/tiff'];
        if (!in_array($mimeType, $allowedTypes)) {
            http_response_code(400);
            echo json_encode(['error' => 'Bad Request', 'message' => 'Invalid file type. Allowed: JPEG, PNG, WebP, TIFF']);
            return;
        }
        
        // Validate file size (25MB for spectral images)
        if ($file['size'] > 25 * 1024 * 1024) {
            http_response_code(400);
            echo json_encode(['error' => 'Bad Request', 'message' => 'File too large. Max: 25MB']);
            return;
        }
        
        try {
            // Call ML API for spectral analysis
            $mlApiUrl = $this->config['ml']['api_url'] ?? 'http://localhost:8000';
            
            $curl = curl_init();
            $cfile = new \CURLFile($file['tmp_name'], $mimeType, $file['name']);
            
            curl_setopt_array($curl, [
                CURLOPT_URL => $mlApiUrl . '/analyze/spectral?image_type=' . urlencode($imageType) . '&use_llm=true',
                CURLOPT_RETURNTRANSFER => true,
                CURLOPT_POST => true,
                CURLOPT_POSTFIELDS => ['file' => $cfile],
                CURLOPT_TIMEOUT => 120,
                CURLOPT_HTTPHEADER => ['Accept: application/json'],
            ]);
            
            $response = curl_exec($curl);
            $httpCode = curl_getinfo($curl, CURLINFO_HTTP_CODE);
            $error = curl_error($curl);
            curl_close($curl);
            
            if ($error) {
                throw new \Exception("ML API connection failed: $error");
            }
            
            if ($httpCode !== 200) {
                throw new \Exception("ML API returned error: $httpCode");
            }
            
            $result = json_decode($response, true);
            
            if (!$result || !$result['success']) {
                throw new \Exception("ML API analysis failed");
            }
            
            // Store original image
            $storageDir = __DIR__ . '/../../storage/spectral';
            if (!is_dir($storageDir)) {
                mkdir($storageDir, 0755, true);
            }
            $originalImagePath = 'spectral/' . uniqid() . '_' . preg_replace('/[^a-zA-Z0-9._-]/', '', $file['name']);
            $copyResult = copy($file['tmp_name'], __DIR__ . '/../../storage/' . $originalImagePath);
            if (!$copyResult) {
                error_log("Failed to copy original image to: " . __DIR__ . '/../../storage/' . $originalImagePath);
                $originalImagePath = null;
            }
            
            // Get spectral image base64
            $spectralImageBase64 = $result['spectral_overlay'] ?? $result['spectral_image'] ?? null;
            
            // Store spectral analysis result
            $analysisId = $this->generateUuid();
            $stmt = $this->db->prepare("
                INSERT INTO spectral_analyses (id, dentist_id, patient_id, image_type, 
                    original_image_path, spectral_image_base64, analysis_json, health_score, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending_review', NOW())
            ");
            $stmt->execute([
                $analysisId,
                $user['user_id'],
                $patientId,
                $imageType,
                $originalImagePath,
                $spectralImageBase64,
                json_encode($result),
                $result['spectral_analysis']['overall_health_score'] ?? 0
            ]);
            
            // Audit log
            $this->audit->log('SPECTRAL_ANALYSIS', $user['user_id'], 'spectral', $analysisId);
            
            echo json_encode([
                'success' => true,
                'analysisId' => $analysisId,
                'imageType' => $imageType,
                'spectralAnalysis' => $result['spectral_analysis'],
                'standardAnalysis' => $result['standard_analysis'],
                'guidance' => [
                    'exactComplaint' => $result['exact_complaint'] ?? '',
                    'detailedFindings' => $result['detailed_findings'] ?? '',
                    'whatThisMeans' => $result['what_this_means'] ?? '',
                    'immediateActions' => $result['immediate_actions'] ?? '',
                    'treatmentOptions' => $result['treatment_options'] ?? '',
                    'homeCareRoutine' => $result['home_care_routine'] ?? '',
                    'preventionTips' => $result['prevention_tips'] ?? '',
                ],
                'spectralRecommendations' => $result['spectral_recommendations'] ?? [],
                'disclaimer' => $result['disclaimer'] ?? '',
                // Spectral visualization images (base64 encoded PNG)
                'spectral_image' => $result['spectral_image'] ?? null,
                'spectral_overlay' => $result['spectral_overlay'] ?? null,
                'color_legend' => $result['color_legend'] ?? null,
            ]);
            
        } catch (\Exception $e) {
            error_log("Spectral analysis error: " . $e->getMessage());
            http_response_code(500);
            echo json_encode(['error' => 'Analysis Failed', 'message' => $e->getMessage()]);
        }
    }
    
    /**
     * POST /spectral/{id}/review - Dentist review of spectral analysis
     */
    public function review(array $params, array $body): void {
        $user = $this->auth->authenticate();
        if (!$user) return;
        
        if ($user['role'] !== 'dentist' && $user['role'] !== 'admin') {
            http_response_code(403);
            echo json_encode(['error' => 'Forbidden', 'message' => 'Only dentists can review analyses']);
            return;
        }
        
        $analysisId = $params['id'];
        $action = $body['action'] ?? ''; // accept, edit, reject
        $clinicalNotes = $body['clinicalNotes'] ?? '';
        $editedDiagnosis = $body['editedDiagnosis'] ?? null;
        
        if (!in_array($action, ['accept', 'edit', 'reject'])) {
            http_response_code(400);
            echo json_encode(['error' => 'Bad Request', 'message' => 'Invalid action. Use: accept, edit, reject']);
            return;
        }
        
        // Verify analysis exists and belongs to this dentist
        $stmt = $this->db->prepare("SELECT * FROM spectral_analyses WHERE id = ? AND dentist_id = ?");
        $stmt->execute([$analysisId, $user['user_id']]);
        $analysis = $stmt->fetch();
        
        if (!$analysis) {
            http_response_code(404);
            echo json_encode(['error' => 'Not Found', 'message' => 'Analysis not found']);
            return;
        }
        
        // Update analysis with review
        $status = $action === 'accept' ? 'approved' : ($action === 'reject' ? 'rejected' : 'edited');
        
        $stmt = $this->db->prepare("
            UPDATE spectral_analyses 
            SET status = ?, review_action = ?, clinical_notes = ?, 
                edited_diagnosis = ?, reviewed_at = NOW()
            WHERE id = ?
        ");
        $stmt->execute([$status, $action, $clinicalNotes, $editedDiagnosis, $analysisId]);
        
        // Audit log
        $this->audit->log('SPECTRAL_REVIEW', $user['user_id'], 'spectral', $analysisId, [
            'action' => $action
        ]);
        
        echo json_encode([
            'success' => true,
            'analysisId' => $analysisId,
            'status' => $status,
            'action' => $action,
            'message' => "Analysis $action" . "ed successfully"
        ]);
    }
    
    /**
     * POST /spectral/{id}/report - Generate report after review
     */
    public function generateReport(array $params, array $body): void {
        $user = $this->auth->authenticate();
        if (!$user) return;
        
        if ($user['role'] !== 'dentist' && $user['role'] !== 'admin') {
            http_response_code(403);
            echo json_encode(['error' => 'Forbidden', 'message' => 'Only dentists can generate reports']);
            return;
        }
        
        $analysisId = $params['id'];
        $reportType = $body['reportType'] ?? 'both'; // clinical, patient, both
        
        // Verify analysis is reviewed
        $stmt = $this->db->prepare("
            SELECT sa.*, u.email as patient_email, u.first_name, u.last_name
            FROM spectral_analyses sa
            LEFT JOIN users u ON sa.patient_id = u.id
            WHERE sa.id = ? AND sa.dentist_id = ?
        ");
        $stmt->execute([$analysisId, $user['user_id']]);
        $analysis = $stmt->fetch();
        
        if (!$analysis) {
            http_response_code(404);
            echo json_encode(['error' => 'Not Found', 'message' => 'Analysis not found']);
            return;
        }
        
        if ($analysis['status'] === 'pending_review') {
            http_response_code(400);
            echo json_encode(['error' => 'Bad Request', 'message' => 'Analysis must be reviewed before generating report']);
            return;
        }
        
        $analysisData = json_decode($analysis['analysis_json'], true);
        
        // Use patient name from request body if provided, otherwise from database
        $patientName = $body['patientName'] ?? null;
        $patientPhone = $body['patientPhone'] ?? null;
        
        if (empty($patientName)) {
            $patientName = trim(($analysis['first_name'] ?? '') . ' ' . ($analysis['last_name'] ?? ''));
        }
        if (empty($patientName)) {
            $patientName = $analysis['patient_email'] ? explode('@', $analysis['patient_email'])[0] : 'Patient';
        }
        
        // Update analysis with patient info if provided
        if (!empty($body['patientName'])) {
            $stmt = $this->db->prepare("
                UPDATE spectral_analyses 
                SET patient_name = ?, patient_phone = ?
                WHERE id = ?
            ");
            $stmt->execute([$patientName, $patientPhone, $analysisId]);
        }
        
        // Generate report ID
        $reportId = 'SPR-' . date('Ymd') . '-' . strtoupper(substr($analysisId, 0, 8));
        
        $response = [
            'success' => true,
            'reportId' => $reportId,
            'analysisId' => $analysisId,
            'generatedAt' => date('c'),
        ];
        
        if ($reportType === 'clinical' || $reportType === 'both') {
            $response['clinicalReport'] = [
                'type' => 'clinical',
                'reportId' => $reportId,
                'patientName' => $patientName,
                'dentistReview' => [
                    'action' => $analysis['review_action'],
                    'clinicalNotes' => $analysis['clinical_notes'],
                    'editedDiagnosis' => $analysis['edited_diagnosis'],
                    'reviewedAt' => $analysis['reviewed_at']
                ],
                'spectralFindings' => $analysisData['spectral_analysis'] ?? [],
                'aiAnalysis' => $analysisData['standard_analysis'] ?? [],
                'recommendations' => $analysisData['spectral_recommendations'] ?? []
            ];
        }
        
        if ($reportType === 'patient' || $reportType === 'both') {
            $response['patientReport'] = [
                'type' => 'patient',
                'reportId' => $reportId,
                'patientName' => $patientName,
                'healthScore' => $analysis['health_score'],
                'summary' => $analysisData['exact_complaint'] ?? 'Your dental scan has been analyzed.',
                'whatThisMeans' => $analysisData['what_this_means'] ?? '',
                'nextSteps' => $analysisData['immediate_actions'] ?? '',
                'homeCare' => $analysisData['home_care_routine'] ?? '',
                'preventionTips' => $analysisData['prevention_tips'] ?? ''
            ];
        }
        
        // Store report reference
        $stmt = $this->db->prepare("
            UPDATE spectral_analyses SET report_id = ?, report_generated_at = NOW() WHERE id = ?
        ");
        $stmt->execute([$reportId, $analysisId]);
        
        echo json_encode($response);
    }
    
    /**
     * GET /spectral/history - Get dentist's spectral analysis history
     */
    public function history(array $params, array $body): void {
        $user = $this->auth->authenticate();
        if (!$user) return;
        
        if ($user['role'] !== 'dentist' && $user['role'] !== 'admin') {
            http_response_code(403);
            echo json_encode(['error' => 'Forbidden']);
            return;
        }
        
        $limit = (int) ($_GET['limit'] ?? 20);
        $offset = (int) ($_GET['offset'] ?? 0);
        $status = $_GET['status'] ?? null;
        
        $sql = "
            SELECT sa.id, sa.patient_id, sa.patient_name, sa.patient_phone, sa.image_type, sa.health_score, sa.status,
                   sa.review_action, sa.report_id, sa.created_at, sa.reviewed_at,
                   u.email as patient_email, u.first_name, u.last_name
            FROM spectral_analyses sa
            LEFT JOIN users u ON sa.patient_id = u.id
            WHERE sa.dentist_id = ?
        ";
        $sqlParams = [$user['user_id']];
        
        if ($status) {
            $sql .= " AND sa.status = ?";
            $sqlParams[] = $status;
        }
        
        $sql .= " ORDER BY sa.created_at DESC LIMIT ? OFFSET ?";
        $sqlParams[] = $limit;
        $sqlParams[] = $offset;
        
        $stmt = $this->db->prepare($sql);
        $stmt->execute($sqlParams);
        $analyses = $stmt->fetchAll();
        
        $formatted = array_map(function($a) {
            // Use patient_name from spectral_analyses first, then fall back to users table
            $patientName = $a['patient_name'] ?? null;
            if (empty($patientName)) {
                $patientName = trim(($a['first_name'] ?? '') . ' ' . ($a['last_name'] ?? ''));
            }
            if (empty($patientName) && $a['patient_email']) {
                $patientName = explode('@', $a['patient_email'])[0];
            }
            
            return [
                'id' => $a['id'],
                'patientId' => $a['patient_id'],
                'patientName' => $patientName ?: null,
                'patientPhone' => $a['patient_phone'] ?? null,
                'imageType' => $a['image_type'],
                'healthScore' => (float) $a['health_score'],
                'status' => $a['status'],
                'reviewAction' => $a['review_action'],
                'reportId' => $a['report_id'],
                'createdAt' => $a['created_at'],
                'reviewedAt' => $a['reviewed_at']
            ];
        }, $analyses);
        
        echo json_encode(['data' => $formatted]);
    }
    
    /**
     * GET /spectral/{id}/report/download - Download PDF report
     */
    public function downloadReport(array $params, array $body): void {
        $user = $this->auth->authenticate();
        if (!$user) return;
        
        if ($user['role'] !== 'dentist' && $user['role'] !== 'admin') {
            http_response_code(403);
            echo json_encode(['error' => 'Forbidden']);
            return;
        }
        
        $analysisId = $params['id'];
        
        // Get analysis with all data
        $stmt = $this->db->prepare("
            SELECT sa.*, d.first_name as dentist_first, d.last_name as dentist_last, d.email as dentist_email
            FROM spectral_analyses sa
            LEFT JOIN users d ON sa.dentist_id = d.id
            WHERE sa.id = ? AND sa.dentist_id = ?
        ");
        $stmt->execute([$analysisId, $user['user_id']]);
        $analysis = $stmt->fetch();
        
        if (!$analysis) {
            http_response_code(404);
            echo json_encode(['error' => 'Not Found']);
            return;
        }
        
        $analysisData = json_decode($analysis['analysis_json'], true);
        $reportId = $analysis['report_id'] ?? 'SPR-' . date('Ymd') . '-' . strtoupper(substr($analysisId, 0, 8));
        
        // Generate HTML for PDF
        $html = $this->generateReportHtml($analysis, $analysisData, $reportId);
        
        // Convert to PDF using simple HTML output (or use a PDF library if available)
        header('Content-Type: text/html; charset=utf-8');
        header('Content-Disposition: attachment; filename="' . $reportId . '.html"');
        echo $html;
    }
    
    private function generateReportHtml(array $analysis, array $analysisData, string $reportId): string {
        $patientName = $analysis['patient_name'] ?? 'N/A';
        $patientPhone = $analysis['patient_phone'] ?? 'N/A';
        $dentistName = trim(($analysis['dentist_first'] ?? '') . ' ' . ($analysis['dentist_last'] ?? ''));
        $healthScore = $analysis['health_score'] ?? 0;
        $imageType = strtoupper($analysis['image_type'] ?? 'NIR');
        $createdAt = date('F j, Y g:i A', strtotime($analysis['created_at']));
        $reviewedAt = $analysis['reviewed_at'] ? date('F j, Y g:i A', strtotime($analysis['reviewed_at'])) : 'N/A';
        $status = ucfirst($analysis['status'] ?? 'pending');
        $clinicalNotes = $analysis['clinical_notes'] ?? '';
        $editedDiagnosis = $analysis['edited_diagnosis'] ?? '';
        
        // Get detections
        $detections = $analysisData['spectral_analysis']['detections'] ?? [];
        $recommendations = $analysisData['spectral_recommendations'] ?? [];
        
        // Images
        $originalImagePath = $analysis['original_image_path'] ?? null;
        $spectralImageBase64 = $analysis['spectral_image_base64'] ?? null;
        
        $originalImageHtml = '';
        if ($originalImagePath && file_exists(__DIR__ . '/../../storage/' . $originalImagePath)) {
            $imageData = base64_encode(file_get_contents(__DIR__ . '/../../storage/' . $originalImagePath));
            $mimeType = mime_content_type(__DIR__ . '/../../storage/' . $originalImagePath);
            $originalImageHtml = '<img src="data:' . $mimeType . ';base64,' . $imageData . '" style="max-width:300px;max-height:250px;border-radius:8px;">';
        }
        
        $spectralImageHtml = '';
        if ($spectralImageBase64) {
            $spectralImageHtml = '<img src="data:image/png;base64,' . $spectralImageBase64 . '" style="max-width:300px;max-height:250px;border-radius:8px;">';
        }
        
        // Build detections HTML
        $detectionsHtml = '';
        foreach ($detections as $d) {
            $severity = $d['severity'] ?? 'mild';
            $color = $severity === 'severe' ? '#dc2626' : ($severity === 'moderate' ? '#f59e0b' : '#22c55e');
            $detectionsHtml .= '
            <tr>
                <td style="padding:10px;border-bottom:1px solid #e5e7eb;">' . htmlspecialchars($d['condition'] ?? '') . '</td>
                <td style="padding:10px;border-bottom:1px solid #e5e7eb;">' . htmlspecialchars($d['location'] ?? '') . '</td>
                <td style="padding:10px;border-bottom:1px solid #e5e7eb;text-align:center;">
                    <span style="background:' . $color . ';color:white;padding:2px 8px;border-radius:4px;font-size:12px;">' . ucfirst($severity) . '</span>
                </td>
                <td style="padding:10px;border-bottom:1px solid #e5e7eb;text-align:center;font-weight:bold;">' . round($d['confidence'] ?? 0) . '%</td>
            </tr>';
        }
        
        // Build recommendations HTML
        $recommendationsHtml = '';
        foreach ($recommendations as $rec) {
            $recommendationsHtml .= '<li style="margin-bottom:8px;">' . htmlspecialchars($rec) . '</li>';
        }
        
        $scoreColor = $healthScore >= 70 ? '#22c55e' : ($healthScore >= 40 ? '#f59e0b' : '#dc2626');
        
        return '<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Spectral Analysis Report - ' . $reportId . '</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; color: #1f2937; }
        .header { background: linear-gradient(135deg, #7c3aed, #4f46e5); color: white; padding: 30px; border-radius: 12px; margin-bottom: 30px; }
        .header h1 { margin: 0 0 10px 0; font-size: 28px; }
        .header p { margin: 5px 0; opacity: 0.9; }
        .section { background: #f9fafb; border-radius: 12px; padding: 20px; margin-bottom: 20px; }
        .section h2 { color: #4f46e5; margin-top: 0; border-bottom: 2px solid #e5e7eb; padding-bottom: 10px; }
        .info-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; }
        .info-item { background: white; padding: 15px; border-radius: 8px; border: 1px solid #e5e7eb; }
        .info-item label { font-size: 12px; color: #6b7280; display: block; margin-bottom: 5px; }
        .info-item span { font-size: 16px; font-weight: 600; }
        .images-container { display: flex; gap: 20px; justify-content: center; flex-wrap: wrap; }
        .image-box { text-align: center; background: #1f2937; padding: 15px; border-radius: 12px; }
        .image-box p { color: #9ca3af; margin: 0 0 10px 0; font-size: 14px; }
        table { width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; }
        th { background: #f3f4f6; padding: 12px; text-align: left; font-weight: 600; color: #374151; }
        .score-box { text-align: center; padding: 30px; background: white; border-radius: 12px; border: 3px solid ' . $scoreColor . '; }
        .score-value { font-size: 64px; font-weight: bold; color: ' . $scoreColor . '; }
        .footer { text-align: center; margin-top: 30px; padding-top: 20px; border-top: 1px solid #e5e7eb; color: #6b7280; font-size: 12px; }
        @media print { body { padding: 0; } .header { break-inside: avoid; } }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔬 Spectral Dental Analysis Report</h1>
        <p><strong>Report ID:</strong> ' . $reportId . '</p>
        <p><strong>Generated:</strong> ' . date('F j, Y g:i A') . '</p>
    </div>
    
    <div class="section">
        <h2>📋 Patient & Analysis Information</h2>
        <div class="info-grid">
            <div class="info-item">
                <label>Patient Name</label>
                <span>' . htmlspecialchars($patientName) . '</span>
            </div>
            <div class="info-item">
                <label>Phone Number</label>
                <span>' . htmlspecialchars($patientPhone) . '</span>
            </div>
            <div class="info-item">
                <label>Image Type</label>
                <span>' . $imageType . '</span>
            </div>
            <div class="info-item">
                <label>Analysis Date</label>
                <span>' . $createdAt . '</span>
            </div>
            <div class="info-item">
                <label>Review Status</label>
                <span>' . $status . '</span>
            </div>
            <div class="info-item">
                <label>Reviewed At</label>
                <span>' . $reviewedAt . '</span>
            </div>
            <div class="info-item">
                <label>Reviewing Dentist</label>
                <span>Dr. ' . htmlspecialchars($dentistName) . '</span>
            </div>
        </div>
    </div>
    
    <div class="section">
        <h2>🖼️ Image Analysis</h2>
        <div class="images-container">
            <div class="image-box">
                <p>Original Image</p>
                ' . ($originalImageHtml ?: '<p style="color:#6b7280;">No image available</p>') . '
            </div>
            <div class="image-box">
                <p>Spectral Analysis</p>
                ' . ($spectralImageHtml ?: '<p style="color:#6b7280;">No spectral image</p>') . '
            </div>
        </div>
        <div style="margin-top:15px;padding:10px;background:#f3f4f6;border-radius:8px;">
            <p style="margin:0;font-size:12px;color:#6b7280;"><strong>Color Legend:</strong> 
                <span style="color:#00FF00;">■</span> Healthy Enamel &nbsp;
                <span style="color:#008000;">■</span> Healthy Gingiva &nbsp;
                <span style="color:#FF0000;">■</span> Caries/Decay &nbsp;
                <span style="color:#FFA500;">■</span> Early Caries &nbsp;
                <span style="color:#00FFFF;">■</span> Calculus &nbsp;
                <span style="color:#800080;">■</span> Inflammation &nbsp;
                <span style="color:#FF00FF;">■</span> Demineralization
            </p>
        </div>
    </div>
    
    <div class="section">
        <h2>📊 Overall Health Score</h2>
        <div class="score-box">
            <div class="score-value">' . round($healthScore) . '%</div>
            <p style="margin:10px 0 0 0;color:#6b7280;">Based on CNN + PCA + Ensemble Classifier Analysis</p>
        </div>
    </div>
    
    <div class="section">
        <h2>🎯 AI Detection Results</h2>
        ' . ($detectionsHtml ? '
        <table>
            <thead>
                <tr>
                    <th>Condition</th>
                    <th>Location</th>
                    <th style="text-align:center;">Severity</th>
                    <th style="text-align:center;">Confidence</th>
                </tr>
            </thead>
            <tbody>' . $detectionsHtml . '</tbody>
        </table>' : '<p style="color:#6b7280;">No detections found</p>') . '
    </div>
    
    ' . ($recommendationsHtml ? '
    <div class="section">
        <h2>💡 AI Recommendations</h2>
        <ul style="margin:0;padding-left:20px;">' . $recommendationsHtml . '</ul>
    </div>' : '') . '
    
    ' . ($clinicalNotes ? '
    <div class="section">
        <h2>🩺 Clinical Notes</h2>
        <p style="background:white;padding:15px;border-radius:8px;border-left:4px solid #4f46e5;margin:0;">' . nl2br(htmlspecialchars($clinicalNotes)) . '</p>
    </div>' : '') . '
    
    ' . ($editedDiagnosis ? '
    <div class="section">
        <h2>✏️ Edited Diagnosis</h2>
        <p style="background:white;padding:15px;border-radius:8px;border-left:4px solid #f59e0b;margin:0;">' . nl2br(htmlspecialchars($editedDiagnosis)) . '</p>
    </div>' : '') . '
    
    <div class="footer">
        <p><strong>DISCLAIMER:</strong> This AI-assisted analysis is for informational purposes only and should not replace professional dental examination. 
        Always consult with a qualified dental professional for diagnosis and treatment.</p>
        <p>Generated by Oral Care AI Spectral Analysis System | Report ID: ' . $reportId . '</p>
    </div>
</body>
</html>';
    }
    
    private function generateUuid(): string {
        return sprintf('%04x%04x-%04x-%04x-%04x-%04x%04x%04x',
            mt_rand(0, 0xffff), mt_rand(0, 0xffff),
            mt_rand(0, 0xffff),
            mt_rand(0, 0x0fff) | 0x4000,
            mt_rand(0, 0x3fff) | 0x8000,
            mt_rand(0, 0xffff), mt_rand(0, 0xffff), mt_rand(0, 0xffff)
        );
    }
}
