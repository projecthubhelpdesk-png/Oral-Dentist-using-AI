<?php
/**
 * Enhanced Teeth Analyzer - PHP Integration Example
 * =================================================
 * 
 * Example of how to integrate the Enhanced Teeth Analyzer API
 * with the existing PHP backend.
 * 
 * This replaces the existing AIAnalysisService.php calls to use
 * the new enhanced API with tooth-level detection and comprehensive reports.
 */

class EnhancedTeethAnalyzerClient {
    private string $apiUrl;
    private int $timeout;
    
    public function __construct(string $apiUrl = 'http://localhost:8000', int $timeout = 30) {
        $this->apiUrl = rtrim($apiUrl, '/');
        $this->timeout = $timeout;
    }
    
    /**
     * Analyze dental image using the enhanced API
     */
    public function analyzeDentalImage(string $imagePath): array {
        $url = $this->apiUrl . '/analyze-teeth';
        
        // Prepare file for upload
        $cfile = new CURLFile($imagePath, mime_content_type($imagePath), basename($imagePath));
        
        $ch = curl_init();
        curl_setopt_array($ch, [
            CURLOPT_URL => $url,
            CURLOPT_POST => true,
            CURLOPT_POSTFIELDS => ['file' => $cfile],
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_TIMEOUT => $this->timeout,
            CURLOPT_HTTPHEADER => [
                'Accept: application/json'
            ]
        ]);
        
        $response = curl_exec($ch);
        $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
        $error = curl_error($ch);
        curl_close($ch);
        
        if ($error) {
            throw new Exception("cURL error: $error");
        }
        
        if ($httpCode !== 200) {
            throw new Exception("API error: HTTP $httpCode - $response");
        }
        
        $result = json_decode($response, true);
        if (!$result) {
            throw new Exception("Invalid JSON response");
        }
        
        return $result;
    }
    
    /**
     * Get quick analysis summary
     */
    public function getAnalysisSummary(string $imagePath): array {
        $url = $this->apiUrl . '/analyze-teeth/summary';
        
        $cfile = new CURLFile($imagePath, mime_content_type($imagePath), basename($imagePath));
        
        $ch = curl_init();
        curl_setopt_array($ch, [
            CURLOPT_URL => $url,
            CURLOPT_POST => true,
            CURLOPT_POSTFIELDS => ['file' => $cfile],
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_TIMEOUT => $this->timeout,
        ]);
        
        $response = curl_exec($ch);
        $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
        curl_close($ch);
        
        if ($httpCode !== 200) {
            throw new Exception("API error: HTTP $httpCode");
        }
        
        return json_decode($response, true);
    }
    
    /**
     * Check API health
     */
    public function checkHealth(): array {
        $url = $this->apiUrl . '/health';
        
        $ch = curl_init();
        curl_setopt_array($ch, [
            CURLOPT_URL => $url,
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_TIMEOUT => 10,
        ]);
        
        $response = curl_exec($ch);
        $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
        curl_close($ch);
        
        if ($httpCode !== 200) {
            return ['status' => 'unhealthy', 'error' => "HTTP $httpCode"];
        }
        
        return json_decode($response, true);
    }
    
    /**
     * Convert enhanced API result to format compatible with existing backend
     */
    public function convertToLegacyFormat(array $enhancedResult): array {
        $diseaseClassification = $enhancedResult['disease_classification'];
        $severityAnalysis = $enhancedResult['severity_analysis'];
        $dentalReport = $enhancedResult['dental_report'];
        $toothDetections = $enhancedResult['tooth_detections'];
        
        // Map to existing format expected by backend
        return [
            'success' => true,
            'disease' => $diseaseClassification['primary_condition'],
            'confidence' => $diseaseClassification['confidence'],
            'severity' => $severityAnalysis['severity'],
            'description' => $diseaseClassification['description'],
            'recommendations' => $dentalReport['professional_advice'],
            'all_predictions' => array_map(function($pred) {
                return [
                    'disease' => $pred['disease'],
                    'confidence' => $pred['confidence']
                ];
            }, $diseaseClassification['all_predictions']),
            
            // Enhanced features
            'enhanced_features' => [
                'tooth_detections' => $toothDetections,
                'affected_teeth' => $dentalReport['affected_teeth'],
                'home_care_tips' => $dentalReport['home_care_tips'],
                'dental_report' => $dentalReport,
                'analysis_id' => $enhancedResult['analysis_id']
            ],
            
            'disclaimer' => $enhancedResult['disclaimer']
        ];
    }
}

/**
 * Updated AIAnalysisService that uses the Enhanced Teeth Analyzer
 */
class EnhancedAIAnalysisService {
    private EnhancedTeethAnalyzerClient $client;
    private \PDO $db;
    private array $config;
    
    public function __construct() {
        $this->db = Database::getInstance();
        $this->config = require __DIR__ . '/../config/app.php';
        
        // Initialize enhanced client
        $apiUrl = getenv('ENHANCED_ML_API_URL') ?: 'http://localhost:8000';
        $this->client = new EnhancedTeethAnalyzerClient($apiUrl);
    }
    
    /**
     * Analyze scan using Enhanced Teeth Analyzer
     */
    public function analyzeScan(string $scanId): array {
        // Get scan info
        $stmt = $this->db->prepare("SELECT * FROM scans WHERE id = ?");
        $stmt->execute([$scanId]);
        $scan = $stmt->fetch();
        
        if (!$scan) {
            throw new \Exception("Scan not found");
        }
        
        // Check if enhanced API is available
        $health = $this->client->checkHealth();
        if ($health['status'] !== 'healthy') {
            // Fallback to original analysis
            return $this->fallbackAnalysis($scan);
        }
        
        try {
            // Use enhanced analysis
            $imagePath = $this->config['uploads']['storage_path'] . $scan['image_storage_path'];
            
            if (!file_exists($imagePath)) {
                throw new \Exception("Image file not found: $imagePath");
            }
            
            // Run enhanced analysis
            $enhancedResult = $this->client->analyzeDentalImage($imagePath);
            
            // Convert to legacy format for compatibility
            $legacyResult = $this->client->convertToLegacyFormat($enhancedResult);
            
            // Store enhanced results
            $resultId = $this->storeEnhancedResults($scanId, $scan['scan_type'], $legacyResult, $enhancedResult);
            
            // Update scan status
            $stmt = $this->db->prepare("UPDATE scans SET status = 'analyzed', processed_at = NOW() WHERE id = ?");
            $stmt->execute([$scanId]);
            
            return array_merge(['id' => $resultId], $legacyResult);
            
        } catch (\Exception $e) {
            // Log error and fallback
            error_log("Enhanced analysis failed: " . $e->getMessage());
            return $this->fallbackAnalysis($scan);
        }
    }
    
    /**
     * Store enhanced analysis results
     */
    private function storeEnhancedResults(string $scanId, string $modelType, array $legacyResult, array $enhancedResult): string {
        $id = $this->generateUuid();
        
        // Calculate overall score from severity and confidence
        $overallScore = $this->calculateOverallScore(
            $legacyResult['severity'],
            $legacyResult['confidence'],
            count($enhancedResult['tooth_detections'])
        );
        
        // Prepare findings from tooth detections
        $findings = [];
        foreach ($enhancedResult['tooth_detections'] as $detection) {
            if ($detection['issue'] !== 'healthy_tooth') {
                $findings[] = [
                    'type' => $this->mapIssueToFindingType($detection['issue']),
                    'severity' => strtolower($legacyResult['severity']),
                    'location' => $detection['tooth_id'],
                    'confidence' => $detection['confidence']
                ];
            }
        }
        
        // Add primary disease finding
        if ($legacyResult['disease'] !== 'Healthy') {
            array_unshift($findings, [
                'type' => $this->mapDiseaseToFindingType($legacyResult['disease']),
                'severity' => strtolower($legacyResult['severity']),
                'location' => 'general',
                'confidence' => $legacyResult['confidence']
            ]);
        }
        
        // Prepare risk areas from tooth detections
        $riskAreas = [];
        foreach ($enhancedResult['tooth_detections'] as $detection) {
            if ($detection['issue'] !== 'healthy_tooth') {
                $bbox = $detection['bbox'];
                $riskAreas[] = [
                    'x' => (int)$bbox[0],
                    'y' => (int)$bbox[1], 
                    'width' => (int)$bbox[2],
                    'height' => (int)$bbox[3],
                    'label' => $detection['issue'],
                    'tooth_id' => $detection['tooth_id']
                ];
            }
        }
        
        $stmt = $this->db->prepare("
            INSERT INTO analysis_results 
            (id, scan_id, model_type, model_version, overall_score, confidence_score, 
             findings_json, risk_areas_json, recommendations_json, processing_time_ms, enhanced_data_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ");
        
        $stmt->execute([
            $id,
            $scanId,
            $modelType,
            'enhanced-2.0.0',
            $overallScore,
            $legacyResult['confidence'],
            json_encode(['findings' => $findings]),
            json_encode(['regions' => $riskAreas]),
            json_encode(['recommendations' => $legacyResult['recommendations']]),
            1000, // Mock processing time
            json_encode($enhancedResult) // Store full enhanced results
        ]);
        
        return $id;
    }
    
    /**
     * Calculate overall score from enhanced analysis
     */
    private function calculateOverallScore(string $severity, float $confidence, int $affectedTeeth): float {
        $baseScores = [
            'Low' => 75,
            'Medium' => 50, 
            'High' => 25
        ];
        
        $baseScore = $baseScores[$severity] ?? 60;
        
        // Adjust for confidence and affected teeth
        $confidenceAdjustment = $confidence * 20;
        $teethAdjustment = min($affectedTeeth * 5, 25);
        
        $finalScore = max(5, $baseScore - $confidenceAdjustment - $teethAdjustment);
        
        return round($finalScore, 2);
    }
    
    /**
     * Map tooth issues to finding types
     */
    private function mapIssueToFindingType(string $issue): string {
        $mapping = [
            'cavity' => 'early_cavity',
            'plaque' => 'plaque_buildup',
            'crooked_tooth' => 'tooth_alignment',
            'missing_tooth' => 'missing_teeth'
        ];
        
        return $mapping[$issue] ?? 'unknown';
    }
    
    /**
     * Map diseases to finding types
     */
    private function mapDiseaseToFindingType(string $disease): string {
        $mapping = [
            'Caries' => 'early_cavity',
            'Gingivitis' => 'gum_inflammation',
            'Calculus' => 'tartar_buildup',
            'Mouth_Ulcer' => 'oral_ulcer'
        ];
        
        return $mapping[$disease] ?? 'unknown';
    }
    
    /**
     * Fallback to original analysis if enhanced API fails
     */
    private function fallbackAnalysis(array $scan): array {
        // Use original mock analysis as fallback
        $isAdvanced = $scan['scan_type'] === 'advanced_spectral';
        
        // Simple fallback analysis
        return [
            'id' => $this->generateUuid(),
            'success' => true,
            'disease' => 'Caries',
            'confidence' => 0.65,
            'severity' => 'Medium',
            'description' => 'Fallback analysis - enhanced API unavailable',
            'recommendations' => [
                'Enhanced analysis temporarily unavailable',
                'Schedule dental checkup for professional evaluation',
                'Continue regular oral hygiene practices'
            ],
            'all_predictions' => [
                ['disease' => 'Caries', 'confidence' => 0.65],
                ['disease' => 'Gingivitis', 'confidence' => 0.25]
            ],
            'disclaimer' => 'This is a fallback analysis. Enhanced features unavailable.'
        ];
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

// Example usage:
/*
// Replace in your existing AIAnalysisService.php:

// Old way:
// $aiService = new \OralCareAI\Services\AIAnalysisService();

// New way:
$aiService = new EnhancedAIAnalysisService();

// The rest of the code remains the same
$result = $aiService->analyzeScan($scanId);
*/

?>