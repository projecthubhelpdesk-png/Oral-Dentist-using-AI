<?php
/**
 * Connection Controller
 * Oral Care AI - PHP Backend
 */

namespace OralCareAI\Controllers;

require_once __DIR__ . '/../../config/database.php';
require_once __DIR__ . '/../../middleware/AuthMiddleware.php';

use OralCareAI\Middleware\AuthMiddleware;
use Database;

class ConnectionController {
    private \PDO $db;
    private AuthMiddleware $auth;

    public function __construct() {
        $this->db = Database::getInstance();
        $this->auth = new AuthMiddleware();
    }

    /**
     * GET /connections - List user's connections
     */
    public function index(array $params, array $body): void {
        $userData = $this->auth->authenticate();
        if (!$userData) return;

        $status = $_GET['status'] ?? null;
        $userId = $userData['user_id'];
        $role = $userData['role'];

        $sql = "
            SELECT pdc.*, 
                   u_patient.email as patient_email,
                   u_patient.first_name as patient_first_name, u_patient.last_name as patient_last_name,
                   u_dentist.email as dentist_email,
                   u_dentist.first_name as dentist_first_name, u_dentist.last_name as dentist_last_name,
                   dp.specialty, dp.clinic_name
            FROM patient_dentist_connections pdc
            JOIN users u_patient ON pdc.patient_id = u_patient.id
            JOIN users u_dentist ON pdc.dentist_id = u_dentist.id
            LEFT JOIN dentist_profiles dp ON pdc.dentist_id = dp.user_id
            WHERE (pdc.patient_id = ? OR pdc.dentist_id = ?)
        ";
        $sqlParams = [$userId, $userId];

        if ($status) {
            $sql .= " AND pdc.status = ?";
            $sqlParams[] = $status;
        }

        $sql .= " ORDER BY pdc.created_at DESC";

        $stmt = $this->db->prepare($sql);
        $stmt->execute($sqlParams);
        $connections = $stmt->fetchAll();

        echo json_encode([
            'data' => array_map(function($conn) use ($role, $userId) {
                return $this->formatConnection($conn, $role, $userId);
            }, $connections)
        ]);
    }


    /**
     * POST /connections - Create connection request
     */
    public function store(array $params, array $body): void {
        $userData = $this->auth->authenticate();
        if (!$userData) return;

        $targetUserId = $body['targetUserId'] ?? null;
        $shareScanHistory = $body['shareScanHistory'] ?? false;

        if (!$targetUserId) {
            http_response_code(400);
            echo json_encode(['error' => 'Bad Request', 'message' => 'targetUserId is required']);
            return;
        }

        // Determine patient and dentist based on roles
        $userId = $userData['user_id'];
        $role = $userData['role'];

        // Get target user's role
        $stmt = $this->db->prepare("SELECT role FROM users WHERE id = ? AND is_active = TRUE");
        $stmt->execute([$targetUserId]);
        $targetUser = $stmt->fetch();

        if (!$targetUser) {
            http_response_code(404);
            echo json_encode(['error' => 'Not Found', 'message' => 'Target user not found']);
            return;
        }

        if ($role === 'dentist' && $targetUser['role'] === 'user') {
            $dentistId = $userId;
            $patientId = $targetUserId;
            $initiatedBy = 'dentist';
        } elseif ($role === 'user' && $targetUser['role'] === 'dentist') {
            $patientId = $userId;
            $dentistId = $targetUserId;
            $initiatedBy = 'patient';
        } else {
            http_response_code(400);
            echo json_encode(['error' => 'Bad Request', 'message' => 'Invalid connection: must be between patient and dentist']);
            return;
        }

        // Check for existing connection
        $stmt = $this->db->prepare("SELECT id, status FROM patient_dentist_connections WHERE patient_id = ? AND dentist_id = ?");
        $stmt->execute([$patientId, $dentistId]);
        $existing = $stmt->fetch();

        if ($existing) {
            if (in_array($existing['status'], ['pending', 'active'])) {
                http_response_code(409);
                echo json_encode(['error' => 'Conflict', 'message' => 'Connection already exists']);
                return;
            }
            // Update existing declined/terminated connection to pending
            $stmt = $this->db->prepare("UPDATE patient_dentist_connections SET status = 'pending', initiated_by = ?, share_scan_history = ? WHERE id = ?");
            $stmt->execute([$initiatedBy, $shareScanHistory ? 1 : 0, $existing['id']]);
            
            http_response_code(200);
            echo json_encode(['id' => $existing['id'], 'status' => 'pending', 'message' => 'Connection request sent']);
            return;
        }

        try {
            $id = $this->generateUuid();
            $stmt = $this->db->prepare("
                INSERT INTO patient_dentist_connections (id, patient_id, dentist_id, status, initiated_by, share_scan_history)
                VALUES (?, ?, ?, 'pending', ?, ?)
            ");
            $stmt->execute([$id, $patientId, $dentistId, $initiatedBy, $shareScanHistory ? 1 : 0]);

            http_response_code(201);
            echo json_encode(['id' => $id, 'status' => 'pending', 'message' => 'Connection request sent']);
        } catch (\PDOException $e) {
            http_response_code(500);
            echo json_encode(['error' => 'Database Error', 'message' => $e->getMessage()]);
        }
    }

    /**
     * PATCH /connections/{id} - Update connection status
     */
    public function update(array $params, array $body): void {
        $userData = $this->auth->authenticate();
        if (!$userData) return;

        $connectionId = $params['id'] ?? null;
        $newStatus = $body['status'] ?? null;
        $shareScanHistory = $body['shareScanHistory'] ?? null;

        $stmt = $this->db->prepare("SELECT * FROM patient_dentist_connections WHERE id = ?");
        $stmt->execute([$connectionId]);
        $connection = $stmt->fetch();

        if (!$connection) {
            http_response_code(404);
            echo json_encode(['error' => 'Not Found', 'message' => 'Connection not found']);
            return;
        }

        // Verify user is part of this connection
        $userId = $userData['user_id'];
        if ($connection['patient_id'] !== $userId && $connection['dentist_id'] !== $userId) {
            http_response_code(403);
            echo json_encode(['error' => 'Forbidden', 'message' => 'Not authorized']);
            return;
        }

        $updates = [];
        $values = [];

        if ($newStatus && in_array($newStatus, ['active', 'declined', 'terminated'])) {
            $updates[] = "status = ?";
            $values[] = $newStatus;
            
            if ($newStatus === 'active') {
                $updates[] = "connected_at = NOW()";
            }
        }

        if ($shareScanHistory !== null) {
            $updates[] = "share_scan_history = ?";
            $values[] = $shareScanHistory ? 1 : 0;
        }

        if (empty($updates)) {
            http_response_code(400);
            echo json_encode(['error' => 'Bad Request', 'message' => 'No valid fields to update']);
            return;
        }

        $values[] = $connectionId;
        $sql = "UPDATE patient_dentist_connections SET " . implode(', ', $updates) . " WHERE id = ?";
        $stmt = $this->db->prepare($sql);
        $stmt->execute($values);

        echo json_encode(['message' => 'Connection updated']);
    }

    private function formatConnection(array $conn, string $role, string $userId): array {
        $isPatient = $conn['patient_id'] === $userId;
        
        // Build names
        $patientName = trim(($conn['patient_first_name'] ?? '') . ' ' . ($conn['patient_last_name'] ?? ''));
        if (empty($patientName)) {
            $patientName = explode('@', $conn['patient_email'])[0];
        }
        $dentistName = trim(($conn['dentist_first_name'] ?? '') . ' ' . ($conn['dentist_last_name'] ?? ''));
        if (empty($dentistName)) {
            $dentistName = explode('@', $conn['dentist_email'])[0];
        }
        
        return [
            'id' => $conn['id'],
            'status' => $conn['status'],
            'initiatedBy' => $conn['initiated_by'],
            'shareScanHistory' => (bool)$conn['share_scan_history'],
            'connectedAt' => $conn['connected_at'],
            'createdAt' => $conn['created_at'],
            'otherUser' => [
                'id' => $isPatient ? $conn['dentist_id'] : $conn['patient_id'],
                'email' => $isPatient ? $conn['dentist_email'] : $conn['patient_email'],
                'name' => $isPatient ? $dentistName : $patientName,
                'role' => $isPatient ? 'dentist' : 'user',
                'specialty' => $isPatient ? $conn['specialty'] : null,
                'clinicName' => $isPatient ? $conn['clinic_name'] : null,
            ]
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
