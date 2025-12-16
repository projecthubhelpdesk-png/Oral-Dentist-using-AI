<?php
/**
 * Analysis Controller
 * Oral Care AI - PHP Backend
 */

namespace OralCareAI\Controllers;

require_once __DIR__ . '/../../models/Scan.php';
require_once __DIR__ . '/../../models/AnalysisResult.php';
require_once __DIR__ . '/../../middleware/AuthMiddleware.php';
require_once __DIR__ . '/../../services/AuditService.php';

use OralCareAI\Models\Scan;
use OralCareAI\Models\AnalysisResult;
use OralCareAI\Middleware\AuthMiddleware;
use OralCareAI\Services\AuditService;

class AnalysisController {
    private Scan $scanModel;
    private AnalysisResult $analysisModel;
    private AuthMiddleware $auth;
    private AuditService $audit;
    
    public function __construct() {
        $this->scanModel = new Scan();
        $this->analysisModel = new AnalysisResult();
        $this->auth = new AuthMiddleware();
        $this->audit = new AuditService();
    }
    
    /**
     * GET /scans/{scanId}/analysis
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
        
        // Check access (owner or connected dentist)
        if ($scan['userId'] !== $user['user_id'] && $user['role'] !== 'admin') {
            // TODO: Check dentist connection
            if ($user['role'] !== 'dentist') {
                http_response_code(403);
                echo json_encode(['error' => 'Forbidden', 'message' => 'Access denied']);
                return;
            }
        }
        
        $analysis = $this->analysisModel->findByScanId($scanId);
        
        if (!$analysis) {
            http_response_code(404);
            echo json_encode(['error' => 'Not Found', 'message' => 'No analysis found for this scan']);
            return;
        }
        
        $this->audit->logAnalysisView($user['user_id'], $analysis['id']);
        
        echo json_encode($analysis);
    }
    
    /**
     * GET /users/me/analysis-history
     */
    public function history(array $params, array $body): void {
        $user = $this->auth->authenticate();
        if (!$user) return;
        
        $limit = min((int) ($_GET['limit'] ?? 10), 50);
        
        $history = $this->analysisModel->findByUserId($user['user_id'], $limit);
        
        echo json_encode(['data' => $history]);
    }
    
    /**
     * GET /users/me/score-trend
     */
    public function scoreTrend(array $params, array $body): void {
        $user = $this->auth->authenticate();
        if (!$user) return;
        
        $days = min((int) ($_GET['days'] ?? 90), 365);
        
        $trend = $this->analysisModel->getScoreTrend($user['user_id'], $days);
        
        echo json_encode(['data' => $trend]);
    }
}
