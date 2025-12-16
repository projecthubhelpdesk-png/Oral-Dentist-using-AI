<?php
/**
 * Review Controller
 * Oral Care AI - PHP Backend
 */

namespace OralCareAI\Controllers;

require_once __DIR__ . '/../../config/database.php';
require_once __DIR__ . '/../../middleware/AuthMiddleware.php';

use OralCareAI\Middleware\AuthMiddleware;
use Database;

class ReviewController {
    private \PDO $db;
    private AuthMiddleware $auth;

    public function __construct() {
        $this->db = Database::getInstance();
        $this->auth = new AuthMiddleware();
    }

    /**
     * GET /scans/{id}/reviews - Get reviews for a scan
     */
    public function index(array $params, array $body): void {
        $userData = $this->auth->authenticate();
        if (!$userData) return;

        $scanId = $params['id'] ?? null;

        // Verify user has access to this scan
        $stmt = $this->db->prepare("SELECT user_id FROM scans WHERE id = ?");
        $stmt->execute([$scanId]);
        $scan = $stmt->fetch();

        if (!$scan) {
            http_response_code(404);
            echo json_encode(['error' => 'Not Found', 'message' => 'Scan not found']);
            return;
        }

        // Check access: owner, connected dentist, or admin
        $hasAccess = $scan['user_id'] === $userData['user_id'] || $userData['role'] === 'admin';
        
        if (!$hasAccess && $userData['role'] === 'dentist') {
            $stmt = $this->db->prepare("
                SELECT 1 FROM patient_dentist_connections 
                WHERE patient_id = ? AND dentist_id = ? AND status = 'active'
            ");
            $stmt->execute([$scan['user_id'], $userData['user_id']]);
            $hasAccess = $stmt->fetch() !== false;
        }

        if (!$hasAccess) {
            http_response_code(403);
            echo json_encode(['error' => 'Forbidden', 'message' => 'Not authorized to view this scan']);
            return;
        }

        $stmt = $this->db->prepare("
            SELECT dr.*, u.email as dentist_email, dp.specialty, dp.clinic_name
            FROM dentist_reviews dr
            JOIN users u ON dr.dentist_id = u.id
            LEFT JOIN dentist_profiles dp ON dr.dentist_id = dp.user_id
            WHERE dr.scan_id = ? AND dr.is_draft = FALSE
            ORDER BY dr.reviewed_at DESC
        ");
        $stmt->execute([$scanId]);
        $reviews = $stmt->fetchAll();

        echo json_encode(array_map([$this, 'formatReview'], $reviews));
    }


    /**
     * POST /scans/{id}/reviews - Create dentist review
     */
    public function store(array $params, array $body): void {
        $userData = $this->auth->authenticate();
        if (!$userData) return;

        if ($userData['role'] !== 'dentist') {
            http_response_code(403);
            echo json_encode(['error' => 'Forbidden', 'message' => 'Only dentists can create reviews']);
            return;
        }

        $scanId = $params['id'] ?? null;
        $dentistId = $userData['user_id'];

        // Verify scan exists and dentist has access
        $stmt = $this->db->prepare("SELECT s.*, ar.id as analysis_id FROM scans s LEFT JOIN analysis_results ar ON s.id = ar.scan_id WHERE s.id = ?");
        $stmt->execute([$scanId]);
        $scan = $stmt->fetch();

        if (!$scan) {
            http_response_code(404);
            echo json_encode(['error' => 'Not Found', 'message' => 'Scan not found']);
            return;
        }

        // Check if dentist is connected to patient
        $stmt = $this->db->prepare("
            SELECT 1 FROM patient_dentist_connections 
            WHERE patient_id = ? AND dentist_id = ? AND status = 'active'
        ");
        $stmt->execute([$scan['user_id'], $dentistId]);
        
        if (!$stmt->fetch()) {
            http_response_code(403);
            echo json_encode(['error' => 'Forbidden', 'message' => 'Not connected to this patient']);
            return;
        }

        // Check for existing review
        $stmt = $this->db->prepare("SELECT id FROM dentist_reviews WHERE scan_id = ? AND dentist_id = ?");
        $stmt->execute([$scanId, $dentistId]);
        if ($stmt->fetch()) {
            http_response_code(409);
            echo json_encode(['error' => 'Conflict', 'message' => 'Review already exists']);
            return;
        }

        $id = $this->generateUuid();
        $stmt = $this->db->prepare("
            INSERT INTO dentist_reviews (
                id, scan_id, dentist_id, ai_result_id, agrees_with_ai, 
                professional_assessment, diagnosis_codes, treatment_recommendations,
                urgency_level, follow_up_days, is_draft, reviewed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, FALSE, NOW())
        ");

        $stmt->execute([
            $id,
            $scanId,
            $dentistId,
            $scan['analysis_id'],
            isset($body['agreesWithAi']) ? ($body['agreesWithAi'] ? 1 : 0) : null,
            $body['professionalAssessment'] ?? null,
            isset($body['diagnosisCodes']) ? json_encode($body['diagnosisCodes']) : null,
            $body['treatmentRecommendations'] ?? null,
            $body['urgencyLevel'] ?? 'routine',
            $body['followUpDays'] ?? null,
        ]);

        http_response_code(201);
        echo json_encode(['id' => $id, 'message' => 'Review created']);
    }

    private function formatReview(array $review): array {
        return [
            'id' => $review['id'],
            'scanId' => $review['scan_id'],
            'dentistId' => $review['dentist_id'],
            'dentist' => [
                'email' => $review['dentist_email'],
                'specialty' => $review['specialty'],
                'clinicName' => $review['clinic_name'],
            ],
            'agreesWithAi' => $review['agrees_with_ai'] === null ? null : (bool)$review['agrees_with_ai'],
            'professionalAssessment' => $review['professional_assessment'],
            'diagnosisCodes' => $review['diagnosis_codes'] ? json_decode($review['diagnosis_codes'], true) : [],
            'treatmentRecommendations' => $review['treatment_recommendations'],
            'urgencyLevel' => $review['urgency_level'],
            'followUpDays' => $review['follow_up_days'] ? (int)$review['follow_up_days'] : null,
            'reviewedAt' => $review['reviewed_at'],
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
