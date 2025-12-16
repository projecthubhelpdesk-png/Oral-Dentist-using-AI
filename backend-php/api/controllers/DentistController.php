<?php
/**
 * Dentist Controller
 * Oral Care AI - PHP Backend
 */

namespace OralCareAI\Controllers;

require_once __DIR__ . '/../../config/database.php';
require_once __DIR__ . '/../../middleware/AuthMiddleware.php';

use OralCareAI\Middleware\AuthMiddleware;
use Database;

class DentistController {
    private \PDO $db;
    private AuthMiddleware $auth;

    public function __construct() {
        $this->db = Database::getInstance();
        $this->auth = new AuthMiddleware();
    }

    /**
     * GET /dentists - List all dentists
     */
    public function index(array $params, array $body): void {
        $specialty = $_GET['specialty'] ?? null;
        $acceptingPatients = $_GET['acceptingPatients'] ?? null;
        $limit = min((int)($_GET['limit'] ?? 20), 100);
        $offset = (int)($_GET['offset'] ?? 0);

        $sql = "
            SELECT dp.*, u.email, u.profile_image_url, u.first_name, u.last_name
            FROM dentist_profiles dp
            JOIN users u ON dp.user_id = u.id
            WHERE u.is_active = TRUE AND dp.license_verified = TRUE
        ";
        $params = [];

        if ($specialty) {
            $sql .= " AND dp.specialty = ?";
            $params[] = $specialty;
        }

        if ($acceptingPatients !== null) {
            $sql .= " AND dp.accepting_patients = ?";
            $params[] = $acceptingPatients === 'true' ? 1 : 0;
        }

        $sql .= " ORDER BY dp.average_rating DESC, dp.total_reviews DESC LIMIT ? OFFSET ?";
        $params[] = $limit;
        $params[] = $offset;

        $stmt = $this->db->prepare($sql);
        $stmt->execute($params);
        $dentists = $stmt->fetchAll();

        // Get total count
        $countSql = "SELECT COUNT(*) FROM dentist_profiles dp JOIN users u ON dp.user_id = u.id WHERE u.is_active = TRUE";
        $total = $this->db->query($countSql)->fetchColumn();

        echo json_encode([
            'data' => array_map([$this, 'formatDentist'], $dentists),
            'total' => (int)$total,
            'limit' => $limit,
            'offset' => $offset
        ]);
    }


    /**
     * GET /dentists/{id} - Get dentist profile
     */
    public function show(array $params, array $body): void {
        $dentistId = $params['id'] ?? null;

        $stmt = $this->db->prepare("
            SELECT dp.*, u.email, u.profile_image_url, u.first_name, u.last_name
            FROM dentist_profiles dp
            JOIN users u ON dp.user_id = u.id
            WHERE dp.user_id = ? AND u.is_active = TRUE
        ");
        $stmt->execute([$dentistId]);
        $dentist = $stmt->fetch();

        if (!$dentist) {
            http_response_code(404);
            echo json_encode(['error' => 'Not Found', 'message' => 'Dentist not found']);
            return;
        }

        echo json_encode($this->formatDentist($dentist));
    }

    /**
     * GET /dentists/me - Get current dentist's profile
     */
    public function me(array $params, array $body): void {
        $userData = $this->auth->authenticate();
        if (!$userData) return;

        if ($userData['role'] !== 'dentist') {
            http_response_code(403);
            echo json_encode(['error' => 'Forbidden', 'message' => 'Not a dentist account']);
            return;
        }

        $stmt = $this->db->prepare("
            SELECT dp.*, u.email, u.profile_image_url, u.first_name, u.last_name
            FROM dentist_profiles dp
            JOIN users u ON dp.user_id = u.id
            WHERE dp.user_id = ?
        ");
        $stmt->execute([$userData['user_id']]);
        $dentist = $stmt->fetch();

        if (!$dentist) {
            http_response_code(404);
            echo json_encode(['error' => 'Not Found', 'message' => 'Dentist profile not found']);
            return;
        }

        echo json_encode($this->formatDentist($dentist));
    }

    /**
     * PATCH /dentists/me - Update dentist profile
     */
    public function update(array $params, array $body): void {
        $userData = $this->auth->authenticate();
        if (!$userData) return;

        if ($userData['role'] !== 'dentist') {
            http_response_code(403);
            echo json_encode(['error' => 'Forbidden', 'message' => 'Not a dentist account']);
            return;
        }

        $allowedFields = ['specialty', 'clinic_name', 'bio', 'accepting_patients', 'consultation_fee_cents', 'years_experience'];
        $updates = [];
        $values = [];

        foreach ($allowedFields as $field) {
            $camelCase = lcfirst(str_replace('_', '', ucwords($field, '_')));
            if (isset($body[$camelCase])) {
                $updates[] = "$field = ?";
                $values[] = $body[$camelCase];
            }
        }

        if (empty($updates)) {
            http_response_code(400);
            echo json_encode(['error' => 'Bad Request', 'message' => 'No valid fields to update']);
            return;
        }

        $values[] = $userData['user_id'];
        $sql = "UPDATE dentist_profiles SET " . implode(', ', $updates) . " WHERE user_id = ?";
        
        $stmt = $this->db->prepare($sql);
        $stmt->execute($values);

        $this->me($params, $body);
    }

    private function formatDentist(array $dentist): array {
        // Build dentist name
        $name = trim(($dentist['first_name'] ?? '') . ' ' . ($dentist['last_name'] ?? ''));
        if (empty($name)) {
            $name = explode('@', $dentist['email'])[0];
        }
        
        return [
            'id' => $dentist['id'],
            'userId' => $dentist['user_id'],
            'email' => $dentist['email'],
            'name' => $name,
            'profileImageUrl' => $dentist['profile_image_url'],
            'licenseState' => $dentist['license_state'],
            'licenseVerified' => (bool)$dentist['license_verified'],
            'specialty' => $dentist['specialty'],
            'clinicName' => $dentist['clinic_name'],
            'yearsExperience' => (int)$dentist['years_experience'],
            'bio' => $dentist['bio'],
            'acceptingPatients' => (bool)$dentist['accepting_patients'],
            'consultationFeeCents' => (int)$dentist['consultation_fee_cents'],
            'averageRating' => (float)$dentist['average_rating'],
            'totalReviews' => (int)$dentist['total_reviews'],
        ];
    }
}
