<?php
/**
 * AI Analysis Service
 * Oral Care AI - PHP Backend
 * 
 * Supports both mock analysis (development) and real ML model (production).
 * Set USE_REAL_ML=true in environment to use the Python ML model.
 */

namespace OralCareAI\Services;

require_once __DIR__ . '/../config/database.php';

use Database;

class AIAnalysisService {
    private \PDO $db;
    private bool $useRealML;
    private string $mlApiUrl;
    private array $config;
    
    public function __construct() {
        $this->db = Database::getInstance();
        $this->config = require __DIR__ . '/../config/app.php';
        
        // Check if real ML model should be used
        $this->useRealML = getenv('USE_REAL_ML') === 'true';
        $this->mlApiUrl = getenv('ML_API_URL') ?: 'http://localhost:8000';
    }
    
    /**
     * Analyze a scan and store results
     */
    public function analyzeScan(string $scanId): array {
        // Get scan info
        $stmt = $this->db->prepare("SELECT * FROM scans WHERE id = ?");
        $stmt->execute([$scanId]);
        $scan = $stmt->fetch();
        
        if (!$scan) {
            throw new \Exception("Scan not found");
        }
        
        // Determine analysis method
        $isAdvanced = $scan['scan_type'] === 'advanced_spectral';
        
        if ($this->useRealML) {
            // Use real ML model
            $analysis = $this->analyzeWithMLModel($scan);
        } else {
            // Use mock analysis for development
            usleep(500000); // 0.5 seconds delay
            $analysis = $this->generateMockAnalysis($isAdvanced);
        }
        
        // Store results
        $resultId = $this->storeResults($scanId, $scan['scan_type'], $analysis);
        
        // Update scan status
        $stmt = $this->db->prepare("UPDATE scans SET status = 'analyzed', processed_at = NOW() WHERE id = ?");
        $stmt->execute([$scanId]);
        
        // URGENT NOTIFICATION: If health score < 50%, notify dentists
        if ($analysis['overallScore'] < 50) {
            $this->notifyDentistsOfUrgentCase($scan, $analysis);
        }
        
        return array_merge(['id' => $resultId], $analysis);
    }
    
    /**
     * Notify dentists about urgent cases (health score < 50%)
     */
    private function notifyDentistsOfUrgentCase(array $scan, array $analysis): void {
        try {
            // Get all dentists
            $stmt = $this->db->prepare("SELECT id, email FROM users WHERE role = 'dentist'");
            $stmt->execute();
            $dentists = $stmt->fetchAll();
            
            if (empty($dentists)) {
                return;
            }
            
            // Get patient info
            $stmt = $this->db->prepare("SELECT email FROM users WHERE id = ?");
            $stmt->execute([$scan['user_id']]);
            $patient = $stmt->fetch();
            
            $score = round($analysis['overallScore']);
            $severity = $analysis['findings'][0]['severity'] ?? 'unknown';
            $condition = $analysis['findings'][0]['type'] ?? 'unknown condition';
            $safetyFlags = $analysis['safetyFlags'] ?? [];
            
            // Build notification message
            $title = "⚠️ URGENT: Patient Requires Attention";
            $message = "A patient scan shows concerning results:\n";
            $message .= "• Health Score: {$score}%\n";
            $message .= "• Primary Finding: " . str_replace('_', ' ', $condition) . " ({$severity})\n";
            
            if (!empty($safetyFlags)) {
                $message .= "• Safety Flags: " . implode(', ', $safetyFlags) . "\n";
            }
            
            $message .= "\nPlease review this case and provide guidance.";
            
            // Create notification for each dentist
            foreach ($dentists as $dentist) {
                $notificationId = $this->generateUuid();
                $stmt = $this->db->prepare("
                    INSERT INTO notifications (id, user_id, type, title, message, data_json, is_read, created_at)
                    VALUES (?, ?, 'urgent_case', ?, ?, ?, FALSE, NOW())
                ");
                $stmt->execute([
                    $notificationId,
                    $dentist['id'],
                    $title,
                    $message,
                    json_encode([
                        'scan_id' => $scan['id'],
                        'patient_id' => $scan['user_id'],
                        'health_score' => $score,
                        'severity' => $severity,
                        'condition' => $condition,
                        'safety_flags' => $safetyFlags,
                        'requires_immediate_review' => $score < 30
                    ])
                ]);
            }
            
            error_log("Urgent case notification sent to " . count($dentists) . " dentists for scan " . $scan['id']);
            
        } catch (\Exception $e) {
            error_log("Failed to send urgent case notification: " . $e->getMessage());
        }
    }
    
    /**
     * Analyze scan using real ML model API
     */
    private function analyzeWithMLModel(array $scan): array {
        $imagePath = $this->config['uploads']['storage_path'] . $scan['image_storage_path'];
        
        if (!file_exists($imagePath)) {
            throw new \Exception("Image file not found: $imagePath");
        }
        
        // Try GPT-4o analysis first (new PyTorch API with detailed guidance)
        $gptResult = $this->tryGPT4oAnalysis($imagePath);
        if ($gptResult) {
            return $this->convertGPT4oResponse($gptResult, $scan['scan_type'] === 'advanced_spectral');
        }
        
        // Try enhanced analysis second
        $enhancedResult = $this->tryEnhancedAnalysis($imagePath);
        if ($enhancedResult) {
            return $this->convertEnhancedResponse($enhancedResult, $scan['scan_type'] === 'advanced_spectral');
        }
        
        // Fallback to basic analysis
        return $this->performBasicAnalysis($imagePath, $scan['scan_type'] === 'advanced_spectral');
    }
    
    /**
     * Try GPT-4o analysis with detailed guidance (PyTorch API)
     */
    private function tryGPT4oAnalysis(string $imagePath): ?array {
        try {
            $ch = curl_init();
            $cfile = new \CURLFile($imagePath, 'image/jpeg', 'scan.jpg');
            
            curl_setopt_array($ch, [
                CURLOPT_URL => $this->mlApiUrl . '/analyze?use_llm=true',
                CURLOPT_POST => true,
                CURLOPT_POSTFIELDS => ['file' => $cfile],
                CURLOPT_RETURNTRANSFER => true,
                CURLOPT_TIMEOUT => 60, // Longer timeout for GPT-4o analysis
                CURLOPT_HTTPHEADER => ['Accept: application/json']
            ]);
            
            $response = curl_exec($ch);
            $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
            curl_close($ch);
            
            if ($httpCode === 200) {
                $result = json_decode($response, true);
                if ($result && $result['success']) {
                    error_log("GPT-4o analysis successful for: $imagePath");
                    return $result;
                }
            }
            
            error_log("GPT-4o analysis failed with HTTP code: $httpCode");
            return null;
        } catch (\Exception $e) {
            error_log("GPT-4o analysis failed: " . $e->getMessage());
            return null;
        }
    }
    
    /**
     * Convert GPT-4o response to our analysis format
     */
    private function convertGPT4oResponse(array $gptResult, bool $isAdvanced): array {
        $summary = $gptResult['summary'] ?? [];
        $report = $gptResult['report'] ?? [];
        
        $disease = $summary['disease'] ?? 'Unknown';
        $confidence = $summary['confidence'] ?? 0.5;
        $severity = $summary['severity'] ?? 'Medium';
        
        // Map disease to finding type
        $findingType = $this->mapDiseaseToFindingType($disease);
        $severityMapped = strtolower($severity);
        
        // Build findings
        $findings = [[
            'type' => $findingType,
            'severity' => $severityMapped === 'high' ? 'severe' : ($severityMapped === 'medium' ? 'moderate' : 'mild'),
            'location' => 'detected_region',
            'confidence' => $confidence,
        ]];
        
        // Calculate overall score based on severity
        $scoreMapping = [
            'High' => 20,
            'Medium' => 45,
            'Low' => 70
        ];
        $overallScore = $scoreMapping[$severity] ?? 50;
        
        // Adjust score based on confidence
        $overallScore = max(10, $overallScore - ($confidence * 20));
        
        // Build recommendations from GPT-4o guidance
        $recommendations = $report['recommendations'] ?? [];
        if (empty($recommendations)) {
            $recommendations = $this->getMLRecommendations($disease, $severity);
        }
        
        return [
            'overallScore' => round($overallScore, 2),
            'confidenceScore' => $confidence,
            'findings' => $findings,
            'riskAreas' => [[
                'x' => 100,
                'y' => 100,
                'width' => 200,
                'height' => 150,
                'label' => $findingType,
            ]],
            'recommendations' => $recommendations,
            'modelVersion' => $isAdvanced ? '3.0.0-gpt4o-advanced' : '3.0.0-gpt4o',
            
            // GPT-4o detailed guidance sections
            'gpt4o_guidance' => [
                'exact_complaint' => $gptResult['exact_complaint'] ?? '',
                'detailed_findings' => $gptResult['detailed_findings'] ?? '',
                'what_this_means' => $gptResult['what_this_means'] ?? '',
                'immediate_actions' => $gptResult['immediate_actions'] ?? '',
                'treatment_options' => $gptResult['treatment_options'] ?? '',
                'home_care_routine' => $gptResult['home_care_routine'] ?? '',
                'prevention_tips' => $gptResult['prevention_tips'] ?? '',
                'llm_analysis' => $gptResult['llm_analysis'] ?? '',
            ],
            
            'enhanced_features' => [
                'analysis_id' => $gptResult['analysis_id'] ?? null,
                'home_care_tips' => $this->extractHomeCareList($gptResult['home_care_routine'] ?? ''),
                'dental_report' => [
                    'summary' => $gptResult['exact_complaint'] ?? '',
                    'severity' => $severity,
                    'recommendation' => $summary['recommendation'] ?? '',
                    'professional_advice' => $this->extractActionsList($gptResult['immediate_actions'] ?? ''),
                    'next_steps' => $this->extractActionsList($gptResult['treatment_options'] ?? ''),
                ]
            ]
        ];
    }
    
    /**
     * Extract list items from text
     */
    private function extractHomeCareList(string $text): array {
        if (empty($text)) return [];
        
        $items = [];
        $lines = explode("\n", $text);
        foreach ($lines as $line) {
            $line = trim($line);
            // Remove bullet points and numbering
            $line = preg_replace('/^[\-\*\•\d\.]+\s*/', '', $line);
            if (!empty($line) && strlen($line) > 5) {
                $items[] = $line;
            }
        }
        return array_slice($items, 0, 10); // Max 10 items
    }
    
    /**
     * Extract action items from text
     */
    private function extractActionsList(string $text): array {
        return $this->extractHomeCareList($text);
    }
    
    /**
     * Try enhanced analysis with tooth-level detection
     */
    private function tryEnhancedAnalysis(string $imagePath): ?array {
        try {
            $ch = curl_init();
            $cfile = new \CURLFile($imagePath, 'image/jpeg', 'scan.jpg');
            
            curl_setopt_array($ch, [
                CURLOPT_URL => $this->mlApiUrl . '/analyze-teeth',
                CURLOPT_POST => true,
                CURLOPT_POSTFIELDS => ['file' => $cfile],
                CURLOPT_RETURNTRANSFER => true,
                CURLOPT_TIMEOUT => 45, // Longer timeout for enhanced analysis
                CURLOPT_HTTPHEADER => ['Accept: application/json']
            ]);
            
            $response = curl_exec($ch);
            $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
            curl_close($ch);
            
            if ($httpCode === 200) {
                $result = json_decode($response, true);
                if ($result && $result['success']) {
                    return $result;
                }
            }
            
            return null;
        } catch (\Exception $e) {
            error_log("Enhanced analysis failed: " . $e->getMessage());
            return null;
        }
    }
    
    /**
     * Perform basic analysis (fallback)
     */
    private function performBasicAnalysis(string $imagePath, bool $isAdvanced): array {
        $ch = curl_init();
        $cfile = new \CURLFile($imagePath, 'image/jpeg', 'scan.jpg');
        
        curl_setopt_array($ch, [
            CURLOPT_URL => $this->mlApiUrl . '/predict',
            CURLOPT_POST => true,
            CURLOPT_POSTFIELDS => ['file' => $cfile],
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_TIMEOUT => 30,
        ]);
        
        $response = curl_exec($ch);
        $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
        curl_close($ch);
        
        if ($httpCode !== 200) {
            throw new \Exception("ML API error: HTTP $httpCode");
        }
        
        $result = json_decode($response, true);
        
        if (!$result || !$result['success']) {
            throw new \Exception("ML API returned invalid response");
        }
        
        // Convert basic ML response to our format
        return $this->convertMLResponse($result, $isAdvanced);
    }
    
    /**
     * Convert enhanced ML response to our analysis format
     */
    private function convertEnhancedResponse(array $enhancedResult, bool $isAdvanced): array {
        $disease = $enhancedResult['disease'];
        $confidence = $enhancedResult['confidence'];
        $severity = $enhancedResult['severity'];
        $toothDetections = $enhancedResult['tooth_detections'] ?? [];
        $dentalReport = $enhancedResult['dental_report'] ?? [];
        
        // Build findings from tooth detections and primary disease
        $findings = [];
        
        // Add primary disease finding
        if ($disease !== 'Healthy') {
            $findings[] = [
                'type' => $this->mapDiseaseToFindingType($disease),
                'severity' => strtolower($severity),
                'location' => 'general_oral_cavity',
                'confidence' => $confidence,
            ];
        }
        
        // Add tooth-level findings
        foreach ($toothDetections as $detection) {
            if ($detection['issue'] !== 'healthy_tooth') {
                $findings[] = [
                    'type' => $this->mapIssueToFindingType($detection['issue']),
                    'severity' => strtolower($severity),
                    'location' => $detection['tooth_id'],
                    'confidence' => $detection['confidence'],
                ];
            }
        }
        
        // Build risk areas from tooth detections
        $riskAreas = [];
        foreach ($toothDetections as $detection) {
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
        
        // Calculate enhanced overall score
        $overallScore = $this->calculateEnhancedScore($disease, $severity, $confidence, count($toothDetections));
        
        // Use enhanced recommendations
        $recommendations = $enhancedResult['recommendations'] ?? $this->getMLRecommendations($disease, $severity);
        
        return [
            'overallScore' => $overallScore,
            'confidenceScore' => $confidence,
            'findings' => $findings,
            'riskAreas' => $riskAreas,
            'recommendations' => $recommendations,
            'modelVersion' => $isAdvanced ? '2.0.0-enhanced-advanced' : '2.0.0-enhanced',
            'enhanced_features' => [
                'tooth_detections' => $toothDetections,
                'affected_teeth' => $dentalReport['affected_teeth'] ?? [],
                'home_care_tips' => $dentalReport['home_care_tips'] ?? [],
                'analysis_id' => $enhancedResult['analysis_id'] ?? null,
                'dental_report' => $dentalReport
            ]
        ];
    }
    
    /**
     * Calculate enhanced overall score
     */
    private function calculateEnhancedScore(string $disease, string $severity, float $confidence, int $affectedTeeth): float {
        // Enhanced scoring that considers tooth-level detections
        $diseaseScoreMapping = [
            'Caries' => ['Low' => 45, 'Medium' => 25, 'High' => 15],
            'Gingivitis' => ['Low' => 55, 'Medium' => 35, 'High' => 20],
            'Calculus' => ['Low' => 50, 'Medium' => 30, 'High' => 18],
            'Mouth_Ulcer' => ['Low' => 60, 'Medium' => 40, 'High' => 25],
            'Hypodontia' => ['Low' => 40, 'Medium' => 25, 'High' => 15],
            'Tooth Discoloration' => ['Low' => 65, 'Medium' => 45, 'High' => 30],
        ];
        
        // Get base score
        $baseScore = $diseaseScoreMapping[$disease][$severity] ?? 40;
        
        // Enhanced adjustments
        $confidenceAdjustment = $confidence * 25;
        $teethAdjustment = min($affectedTeeth * 3, 20); // More granular tooth-level penalty
        
        // Tooth-level severity bonus/penalty
        $toothLevelBonus = $affectedTeeth > 0 ? -10 : 5; // Penalty for detected tooth issues
        
        $finalScore = max(5, $baseScore - $confidenceAdjustment - $teethAdjustment + $toothLevelBonus);
        
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
        
        return $mapping[$issue] ?? 'unknown_issue';
    }
    
    /**
     * Map diseases to finding types
     */
    private function mapDiseaseToFindingType(string $disease): string {
        $mapping = [
            'Caries' => 'early_cavity',
            'Gingivitis' => 'gum_inflammation',
            'Calculus' => 'tartar_buildup',
            'Mouth_Ulcer' => 'oral_ulcer',
            'Hypodontia' => 'missing_teeth',
            'Tooth Discoloration' => 'tooth_discoloration'
        ];
        
        return $mapping[$disease] ?? 'unknown_condition';
    }
    
    /**
     * Convert ML model response to our analysis format
     */
    private function convertMLResponse(array $mlResult, bool $isAdvanced): array {
        // Map disease names to finding types
        $diseaseMapping = [
            'Calculus' => 'tartar_buildup',
            'Caries' => 'early_cavity',
            'Gingivitis' => 'gum_inflammation',
            'Mouth_Ulcer' => 'oral_ulcer',
            'Hypodontia' => 'missing_teeth',
            'Mouth Ulcer' => 'oral_ulcer',
            'Tooth Discoloration' => 'tooth_discoloration',
            'Healthy' => 'healthy',
        ];
        
        // Map severity
        $severityMapping = [
            'Low' => 'mild',
            'Medium' => 'moderate',
            'High' => 'severe',
        ];
        
        $findingType = $diseaseMapping[$mlResult['disease']] ?? 'unknown';
        $severity = $severityMapping[$mlResult['severity']] ?? 'moderate';
        
        // Build findings - include primary finding
        $findings = [[
            'type' => $findingType,
            'severity' => $severity,
            'location' => 'detected_region',
            'confidence' => $mlResult['confidence'],
        ]];
        
        // IMPROVED: Add ALL significant secondary findings (threshold lowered to 0.05)
        // This helps detect multiple conditions like tartar + cavity
        $secondaryFindings = 0;
        $significantSecondaryConfidence = 0;
        foreach (array_slice($mlResult['all_predictions'], 1, 3) as $pred) {
            // Lower threshold to catch more conditions
            if ($pred['confidence'] > 0.05 && $pred['disease'] !== 'Healthy') {
                // Determine severity based on confidence
                $secondarySeverity = 'mild';
                if ($pred['confidence'] > 0.3) {
                    $secondarySeverity = 'moderate';
                }
                if ($pred['confidence'] > 0.5) {
                    $secondarySeverity = 'severe';
                }
                
                $findings[] = [
                    'type' => $diseaseMapping[$pred['disease']] ?? 'unknown',
                    'severity' => $secondarySeverity,
                    'location' => 'secondary_region_' . ($secondaryFindings + 1),
                    'confidence' => $pred['confidence'],
                ];
                $significantSecondaryConfidence += $pred['confidence'];
                $secondaryFindings++;
            }
        }
        
        // Calculate overall score - MUCH more aggressive for dental diseases
        // Dental diseases are serious and should reflect urgency
        $diseaseScoreMapping = [
            'Caries' => ['Low' => 40, 'Medium' => 20, 'High' => 10],           // Cavities are very serious
            'Gingivitis' => ['Low' => 50, 'Medium' => 30, 'High' => 15],       // Gum disease progression
            'Calculus' => ['Low' => 45, 'Medium' => 25, 'High' => 12],         // Tartar buildup - serious
            'Mouth_Ulcer' => ['Low' => 55, 'Medium' => 35, 'High' => 20],      // Ulcers can indicate issues
            'Hypodontia' => ['Low' => 35, 'Medium' => 20, 'High' => 10],       // Missing teeth
            'Tooth Discoloration' => ['Low' => 60, 'Medium' => 40, 'High' => 25], // Cosmetic but can indicate decay
        ];
        
        $disease = $mlResult['disease'];
        $severityLevel = $mlResult['severity'];
        
        // Get base score for this disease and severity
        if (isset($diseaseScoreMapping[$disease][$severityLevel])) {
            $baseScore = $diseaseScoreMapping[$disease][$severityLevel];
        } else {
            // Default aggressive scoring for unknown diseases
            $defaultMapping = ['Low' => 35, 'Medium' => 20, 'High' => 10];
            $baseScore = $defaultMapping[$severityLevel] ?? 25;
        }
        
        // IMPROVED: Stronger confidence adjustment - higher confidence = much lower score
        $confidenceAdjustment = $mlResult['confidence'] * 30; // Up to 30 point reduction
        
        // IMPROVED: Heavier penalty for multiple conditions
        $multipleConditionsPenalty = $secondaryFindings * 12; // 12 points per additional condition
        
        // IMPROVED: Additional penalty based on secondary condition confidence
        $secondaryConfidencePenalty = $significantSecondaryConfidence * 15;
        
        // Apply all adjustments
        $overallScore = max(5, $baseScore - $confidenceAdjustment - $multipleConditionsPenalty - $secondaryConfidencePenalty);
        
        // IMPROVED: More aggressive capping for multiple conditions
        // If ANY secondary finding has confidence > 0.1, this is a multi-condition case
        if ($secondaryFindings >= 1 && $significantSecondaryConfidence > 0.1) {
            $overallScore = min($overallScore, 35); // Cap at 35% for multiple conditions
        }
        
        // If multiple high-confidence findings, cap even lower
        if ($secondaryFindings >= 2 && $mlResult['confidence'] > 0.2) {
            $overallScore = min($overallScore, 20); // Cap at 20% for multiple serious conditions
        }
        
        // NEW: Use visual health score from ML API if available (more accurate)
        if (isset($mlResult['visual_health_score'])) {
            $visualScore = $mlResult['visual_health_score'];
            // Use visual score but cap it based on detected disease
            if ($disease === 'Healthy') {
                $overallScore = $visualScore;
            } else {
                // For diseases, use the lower of calculated or visual score
                $overallScore = min($overallScore, $visualScore);
            }
        }
        
        // IMPROVED: Determine the most severe condition for recommendations
        // Check if secondary conditions are more severe than primary
        $mostSevereDisease = $disease;
        $mostSevereSeverity = $severityLevel;
        
        // Severity ranking for diseases (higher = more urgent)
        $diseaseSeverityRank = [
            'Caries' => 5,
            'Calculus' => 4,
            'Gingivitis' => 3,
            'Mouth_Ulcer' => 2,
            'Tooth Discoloration' => 1,
            'Healthy' => 0
        ];
        
        foreach ($mlResult['all_predictions'] as $pred) {
            if ($pred['confidence'] > 0.15 && $pred['disease'] !== 'Healthy') {
                $predRank = $diseaseSeverityRank[$pred['disease']] ?? 0;
                $currentRank = $diseaseSeverityRank[$mostSevereDisease] ?? 0;
                
                // If this condition is more severe OR has higher confidence for same severity
                if ($predRank > $currentRank || 
                    ($predRank === $currentRank && $pred['confidence'] > $mlResult['confidence'])) {
                    $mostSevereDisease = $pred['disease'];
                }
            }
        }
        
        // Upgrade severity if multiple conditions detected
        if ($secondaryFindings >= 1 && $mostSevereSeverity === 'Low') {
            $mostSevereSeverity = 'Medium';
        }
        if ($secondaryFindings >= 2 || $significantSecondaryConfidence > 0.3) {
            $mostSevereSeverity = 'High';
        }
        
        // Get recommendations based on most severe condition
        $recommendations = $this->getMLRecommendations($mostSevereDisease, $mostSevereSeverity);
        
        // Add recommendations for secondary conditions
        foreach (array_slice($mlResult['all_predictions'], 1, 2) as $pred) {
            if ($pred['confidence'] > 0.1 && $pred['disease'] !== 'Healthy' && $pred['disease'] !== $mostSevereDisease) {
                $secondaryRecs = $this->getMLRecommendations($pred['disease'], 'Medium');
                // Add first recommendation from secondary condition
                if (!empty($secondaryRecs) && count($secondaryRecs) > 1) {
                    $recommendations[] = "Also detected: " . $secondaryRecs[1];
                }
            }
        }
        
        // CLINICAL SAFETY: Apply safety flags from ML API - VERY AGGRESSIVE
        $safetyFlags = $mlResult['safety_flags'] ?? [];
        $visualRiskLevel = $mlResult['visual_risk_level'] ?? 'LOW';
        $urgentWarning = $mlResult['urgent_warning'] ?? null;
        
        // Override score if safety flags indicate high risk - MUCH LOWER CAPS
        if (in_array('BLEEDING_DETECTED', $safetyFlags) || in_array('BLEEDING_REQUIRES_ATTENTION', $safetyFlags)) {
            $overallScore = min($overallScore, 15);  // Bleeding = max 15%
            $mostSevereSeverity = 'High';
        }
        if (in_array('HIGH_VISUAL_RISK', $safetyFlags) || in_array('CRITICAL_MULTIPLE_ISSUES', $safetyFlags)) {
            $overallScore = min($overallScore, 12);  // Critical = max 12%
            $mostSevereSeverity = 'High';
        }
        if (in_array('SEVERE_DECAY_DETECTED', $safetyFlags)) {
            $overallScore = min($overallScore, 12);  // Severe decay = max 12%
            $mostSevereSeverity = 'High';
        }
        if (in_array('SEVERE_TARTAR_BUILDUP', $safetyFlags)) {
            $overallScore = min($overallScore, 12);  // Severe tartar = max 12%
            $mostSevereSeverity = 'High';
        }
        if (in_array('MULTIPLE_ISSUES_DETECTED', $safetyFlags)) {
            $overallScore = min($overallScore, 18);  // Multiple issues = max 18%
            $mostSevereSeverity = 'High';
        }
        if (in_array('TARTAR_BUILDUP', $safetyFlags)) {
            $overallScore = min($overallScore, 25);  // Tartar = max 25%
            if ($mostSevereSeverity !== 'High') $mostSevereSeverity = 'Medium';
        }
        if (in_array('HIGH_INFLAMMATION', $safetyFlags)) {
            $overallScore = min($overallScore, 20);  // High inflammation = max 20%
            $mostSevereSeverity = 'High';
        }
        
        // Visual risk level caps - MUCH LOWER
        if ($visualRiskLevel === 'HIGH') {
            $overallScore = min($overallScore, 15);  // High risk = max 15%
            $mostSevereSeverity = 'High';
        } elseif ($visualRiskLevel === 'MEDIUM') {
            $overallScore = min($overallScore, 35);  // Medium risk = max 35%
            if ($mostSevereSeverity === 'Low') $mostSevereSeverity = 'Medium';
        }
        
        // Add urgent warning to recommendations
        if ($urgentWarning) {
            array_unshift($recommendations, $urgentWarning);
        }
        
        // FINAL SAFETY CHECK: If disease is NOT Healthy and severity is High, cap at 20%
        if ($disease !== 'Healthy' && $mostSevereSeverity === 'High') {
            $overallScore = min($overallScore, 20);
        }
        
        // If Calculus/Caries detected with any confidence, cap based on severity
        if (in_array($disease, ['Calculus', 'Caries']) && $mlResult['confidence'] > 0.3) {
            if ($mostSevereSeverity === 'High') {
                $overallScore = min($overallScore, 15);
            } elseif ($mostSevereSeverity === 'Medium') {
                $overallScore = min($overallScore, 30);
            }
        }
        
        return [
            'overallScore' => round($overallScore, 2),
            'confidenceScore' => $mlResult['confidence'],
            'findings' => $findings,
            'riskAreas' => [[
                'x' => 100,
                'y' => 100,
                'width' => 200,
                'height' => 150,
                'label' => $findingType,
            ]],
            'recommendations' => $recommendations,
            'modelVersion' => $isAdvanced ? '1.5.2-ml' : '2.1.0-ml',
            'detectedConditions' => $secondaryFindings + 1,
            'safetyFlags' => $safetyFlags,
            'visualRiskLevel' => $visualRiskLevel,
        ];
    }
    
    /**
     * Get recommendations for ML-detected disease
     */
    private function getMLRecommendations(string $disease, string $severity): array {
        $urgentRecommendations = [
            'Caries' => [
                'Low' => ['🦷 Schedule dental appointment within 2 weeks', 'Avoid sugary foods and drinks', 'Use fluoride toothpaste twice daily'],
                'Medium' => ['🚨 URGENT: See dentist within 1 week', 'Stop all sugary intake immediately', 'Use prescription fluoride if available'],
                'High' => ['🚨 EMERGENCY: See dentist immediately', 'Severe decay detected - treatment needed now', 'Risk of infection and tooth loss'],
            ],
            'Gingivitis' => [
                'Low' => ['⚠️ Improve oral hygiene immediately', 'Floss daily without fail', 'Use antiseptic mouthwash'],
                'Medium' => ['🚨 Schedule dental cleaning within 1 week', 'Gum disease is progressing', 'Risk of tooth loss if untreated'],
                'High' => ['🚨 URGENT: Advanced gum disease detected', 'See periodontist immediately', 'Risk of systemic health issues'],
            ],
            'Calculus' => [
                'Low' => ['Schedule professional cleaning soon', 'Improve brushing technique', 'Use tartar-control toothpaste'],
                'Medium' => ['🚨 Heavy tartar buildup - professional cleaning needed', 'Risk of gum disease progression', 'May require deep cleaning'],
                'High' => ['🚨 SEVERE tartar buildup detected', 'Immediate professional intervention required', 'Risk of tooth loss and gum disease'],
            ],
            'Hypodontia' => [
                'Low' => ['Consult orthodontist for missing teeth', 'Discuss replacement options', 'Prevent further tooth loss'],
                'Medium' => ['🚨 Multiple missing teeth detected', 'Urgent orthodontic consultation needed', 'Risk of bite problems and bone loss'],
                'High' => ['🚨 SEVERE tooth loss detected', 'Immediate comprehensive dental evaluation', 'Risk of facial structure changes'],
            ],
            'Mouth_Ulcer' => [
                'Low' => ['Monitor ulcer closely', 'Avoid irritating foods', 'See dentist if persists >2 weeks'],
                'Medium' => ['⚠️ Persistent or large ulcer detected', 'Schedule dental appointment soon', 'May indicate underlying condition'],
                'High' => ['🚨 Severe oral lesion detected', 'Immediate dental/medical evaluation needed', 'Rule out serious conditions'],
            ],
            'Tooth Discoloration' => [
                'Low' => ['Professional cleaning recommended', 'May indicate early decay', 'Reduce staining substances'],
                'Medium' => ['⚠️ Significant discoloration detected', 'May indicate decay or trauma', 'Professional evaluation needed'],
                'High' => ['🚨 Severe discoloration detected', 'Possible nerve damage or advanced decay', 'Urgent dental evaluation required'],
            ],
        ];
        
        // Get disease-specific recommendations
        $recs = $urgentRecommendations[$disease][$severity] ?? [
            'Consult a dental professional immediately',
            'Condition detected requires professional evaluation',
            'Do not delay treatment'
        ];
        
        // Add general urgent disclaimer
        if ($severity === 'High') {
            array_unshift($recs, '🚨 CRITICAL: This condition requires immediate attention');
        } elseif ($severity === 'Medium') {
            array_unshift($recs, '⚠️ WARNING: This condition needs prompt treatment');
        }
        
        $recs[] = '⚠️ AI screening only - professional diagnosis required';
        
        return $recs;
    }

    
    /**
     * Generate mock AI analysis results
     */
    private function generateMockAnalysis(bool $isAdvanced): array {
        $possibleFindings = [
            ['type' => 'plaque_buildup', 'locations' => ['lower_molars', 'upper_molars', 'front_teeth']],
            ['type' => 'gum_inflammation', 'locations' => ['upper_front', 'lower_front', 'back_gums']],
            ['type' => 'early_cavity', 'locations' => ['upper_right_molar', 'lower_left_molar', 'upper_left_premolar']],
            ['type' => 'enamel_erosion', 'locations' => ['front_teeth', 'canines', 'incisors']],
            ['type' => 'tartar_buildup', 'locations' => ['lower_front', 'behind_lower_teeth']],
            ['type' => 'gum_recession', 'locations' => ['lower_front', 'upper_canines']],
            ['type' => 'tooth_discoloration', 'locations' => ['front_teeth', 'molars']],
        ];
        
        $severities = ['none', 'minimal', 'mild', 'moderate', 'severe'];
        
        // Randomly select 1-4 findings
        $numFindings = rand(1, 4);
        shuffle($possibleFindings);
        $selectedFindings = array_slice($possibleFindings, 0, $numFindings);
        
        $findings = [];
        $riskAreas = [];
        $totalSeverityScore = 0;
        
        foreach ($selectedFindings as $index => $finding) {
            $severityIndex = rand(0, 3); // Mostly mild issues for demo
            $severity = $severities[$severityIndex];
            $confidence = round(rand(70, 98) / 100, 2);
            $location = $finding['locations'][array_rand($finding['locations'])];
            
            $findings[] = [
                'type' => $finding['type'],
                'severity' => $severity,
                'location' => $location,
                'confidence' => $confidence,
            ];
            
            // Generate risk area coordinates
            $riskAreas[] = [
                'x' => rand(50, 400),
                'y' => rand(50, 300),
                'width' => rand(40, 100),
                'height' => rand(30, 80),
                'label' => $finding['type'],
            ];
            
            $totalSeverityScore += $severityIndex;
        }
        
        // Calculate overall score (100 = perfect, lower = more issues)
        $maxPossibleSeverity = $numFindings * 4;
        $overallScore = round(100 - ($totalSeverityScore / $maxPossibleSeverity * 40) - rand(0, 10), 2);
        $overallScore = max(50, min(98, $overallScore));
        
        // Confidence score
        $confidenceScore = round(rand(85, 98) / 100, 4);
        if ($isAdvanced) {
            $confidenceScore = round(rand(92, 99) / 100, 4);
        }
        
        // Generate recommendations
        $recommendations = $this->generateRecommendations($findings);
        
        return [
            'overallScore' => $overallScore,
            'confidenceScore' => $confidenceScore,
            'findings' => $findings,
            'riskAreas' => $riskAreas,
            'recommendations' => $recommendations,
            'modelVersion' => $isAdvanced ? '1.5.2' : '2.1.0',
        ];
    }
    
    /**
     * Generate recommendations based on findings
     */
    private function generateRecommendations(array $findings): array {
        $recommendationMap = [
            'plaque_buildup' => [
                'Brush teeth at least twice daily for 2 minutes',
                'Use an electric toothbrush for better plaque removal',
                'Consider using an antimicrobial mouthwash',
            ],
            'gum_inflammation' => [
                'Floss daily to reduce gum inflammation',
                'Use a soft-bristled toothbrush',
                'Consider using an anti-gingivitis toothpaste',
            ],
            'early_cavity' => [
                'Schedule a dental appointment within 2 weeks',
                'Use fluoride toothpaste and mouthwash',
                'Reduce sugar intake and acidic beverages',
            ],
            'enamel_erosion' => [
                'Avoid acidic foods and drinks',
                'Wait 30 minutes after eating before brushing',
                'Use enamel-strengthening toothpaste',
            ],
            'tartar_buildup' => [
                'Schedule a professional cleaning',
                'Use tartar-control toothpaste',
                'Brush along the gumline carefully',
            ],
            'gum_recession' => [
                'Use a soft-bristled toothbrush with gentle pressure',
                'Consider a gum graft consultation if severe',
                'Avoid aggressive brushing techniques',
            ],
            'tooth_discoloration' => [
                'Consider professional whitening treatment',
                'Reduce coffee, tea, and red wine consumption',
                'Brush after consuming staining foods',
            ],
        ];
        
        $recommendations = [];
        foreach ($findings as $finding) {
            if (isset($recommendationMap[$finding['type']])) {
                $recs = $recommendationMap[$finding['type']];
                $recommendations[] = $recs[array_rand($recs)];
            }
        }
        
        // Add general recommendation
        $recommendations[] = 'Schedule your next dental checkup in 6 months';
        
        return array_unique($recommendations);
    }
    
    /**
     * Store analysis results in database
     */
    private function storeResults(string $scanId, string $modelType, array $analysis): string {
        $id = $this->generateUuid();
        
        // Include GPT-4o guidance in findings JSON if available
        $findingsData = ['findings' => $analysis['findings']];
        if (isset($analysis['gpt4o_guidance'])) {
            $findingsData['gpt4o_guidance'] = $analysis['gpt4o_guidance'];
        }
        if (isset($analysis['enhanced_features'])) {
            $findingsData['enhanced_features'] = $analysis['enhanced_features'];
        }
        
        $stmt = $this->db->prepare("
            INSERT INTO analysis_results 
            (id, scan_id, model_type, model_version, overall_score, confidence_score, 
             findings_json, risk_areas_json, recommendations_json, processing_time_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ");
        
        $stmt->execute([
            $id,
            $scanId,
            $modelType,
            $analysis['modelVersion'],
            $analysis['overallScore'],
            $analysis['confidenceScore'],
            json_encode($findingsData),
            json_encode(['regions' => $analysis['riskAreas']]),
            json_encode(['recommendations' => $analysis['recommendations']]),
            rand(800, 3500), // Mock processing time
        ]);
        
        return $id;
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
