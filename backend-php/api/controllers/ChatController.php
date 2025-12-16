<?php
/**
 * Chat Controller
 * Handles scan-related chat between users and dentists
 */

namespace OralCareAI\Controllers;

require_once __DIR__ . '/../../config/database.php';
require_once __DIR__ . '/../../middleware/AuthMiddleware.php';

use OralCareAI\Middleware\AuthMiddleware;
use Database;

class ChatController {
    private \PDO $db;
    private AuthMiddleware $auth;
    
    public function __construct() {
        $this->db = Database::getInstance();
        $this->auth = new AuthMiddleware();
    }
    
    /**
     * GET /scans/{id}/chat - Get chat messages for a scan
     */
    public function getMessages(array $params, array $body): void {
        $user = $this->auth->authenticate();
        if (!$user) return;
        
        $scanId = $params['id'];
        
        // Verify access to this scan
        if (!$this->canAccessScan($scanId, $user)) {
            http_response_code(403);
            echo json_encode(['error' => 'Forbidden', 'message' => 'Access denied']);
            return;
        }
        
        $limit = (int) ($_GET['limit'] ?? 50);
        $before = $_GET['before'] ?? null;
        
        $sql = "
            SELECT m.id, m.scan_id, m.sender_id, m.message, m.message_type, 
                   m.is_read, m.created_at,
                   u.email as sender_email, u.role as sender_role,
                   u.first_name as sender_first_name, u.last_name as sender_last_name
            FROM scan_chat_messages m
            JOIN users u ON m.sender_id = u.id
            WHERE m.scan_id = ?
        ";
        $sqlParams = [$scanId];
        
        if ($before) {
            $sql .= " AND m.created_at < ?";
            $sqlParams[] = $before;
        }
        
        $sql .= " ORDER BY m.created_at DESC LIMIT ?";
        $sqlParams[] = $limit;
        
        $stmt = $this->db->prepare($sql);
        $stmt->execute($sqlParams);
        $messages = $stmt->fetchAll();
        
        // Mark messages as read
        $this->markAsRead($scanId, $user['user_id']);
        
        // Format and reverse to get chronological order
        $formatted = array_map(function($m) {
            $senderName = trim(($m['sender_first_name'] ?? '') . ' ' . ($m['sender_last_name'] ?? ''));
            if (empty($senderName)) {
                $senderName = explode('@', $m['sender_email'])[0];
            }
            return [
                'id' => $m['id'],
                'scanId' => $m['scan_id'],
                'senderId' => $m['sender_id'],
                'senderEmail' => $m['sender_email'],
                'senderName' => $senderName,
                'senderRole' => $m['sender_role'],
                'message' => $m['message'],
                'messageType' => $m['message_type'],
                'isRead' => (bool) $m['is_read'],
                'createdAt' => $m['created_at'],
            ];
        }, array_reverse($messages));
        
        echo json_encode(['messages' => $formatted]);
    }
    
    /**
     * POST /scans/{id}/chat - Send a chat message
     */
    public function sendMessage(array $params, array $body): void {
        $user = $this->auth->authenticate();
        if (!$user) return;
        
        $scanId = $params['id'];
        
        // Verify access to this scan
        if (!$this->canAccessScan($scanId, $user)) {
            http_response_code(403);
            echo json_encode(['error' => 'Forbidden', 'message' => 'Access denied']);
            return;
        }
        
        $message = trim($body['message'] ?? '');
        if (empty($message)) {
            http_response_code(400);
            echo json_encode(['error' => 'Bad Request', 'message' => 'Message is required']);
            return;
        }
        
        if (strlen($message) > 2000) {
            http_response_code(400);
            echo json_encode(['error' => 'Bad Request', 'message' => 'Message too long (max 2000 chars)']);
            return;
        }
        
        $messageId = $this->generateUuid();
        $messageType = $body['messageType'] ?? 'text';
        
        $stmt = $this->db->prepare("
            INSERT INTO scan_chat_messages (id, scan_id, sender_id, message, message_type)
            VALUES (?, ?, ?, ?, ?)
        ");
        $stmt->execute([$messageId, $scanId, $user['user_id'], $message, $messageType]);
        
        // Get the created message with sender info
        $stmt = $this->db->prepare("
            SELECT m.*, u.email as sender_email, u.role as sender_role,
                   u.first_name as sender_first_name, u.last_name as sender_last_name
            FROM scan_chat_messages m
            JOIN users u ON m.sender_id = u.id
            WHERE m.id = ?
        ");
        $stmt->execute([$messageId]);
        $m = $stmt->fetch();
        
        // Build sender name
        $senderName = trim(($m['sender_first_name'] ?? '') . ' ' . ($m['sender_last_name'] ?? ''));
        if (empty($senderName)) {
            $senderName = explode('@', $m['sender_email'])[0];
        }
        
        // Notify the other party
        $this->notifyNewMessage($scanId, $user, $message);
        
        http_response_code(201);
        echo json_encode([
            'id' => $m['id'],
            'scanId' => $m['scan_id'],
            'senderId' => $m['sender_id'],
            'senderEmail' => $m['sender_email'],
            'senderName' => $senderName,
            'senderRole' => $m['sender_role'],
            'message' => $m['message'],
            'messageType' => $m['message_type'],
            'isRead' => false,
            'createdAt' => $m['created_at'],
        ]);
    }
    
    /**
     * GET /scans/{id}/chat/unread - Get unread message count
     */
    public function getUnreadCount(array $params, array $body): void {
        $user = $this->auth->authenticate();
        if (!$user) return;
        
        $scanId = $params['id'];
        
        $stmt = $this->db->prepare("
            SELECT COUNT(*) as count FROM scan_chat_messages 
            WHERE scan_id = ? AND sender_id != ? AND is_read = FALSE
        ");
        $stmt->execute([$scanId, $user['user_id']]);
        $result = $stmt->fetch();
        
        echo json_encode(['unreadCount' => (int) $result['count']]);
    }
    
    /**
     * Check if user can access scan chat
     */
    private function canAccessScan(string $scanId, array $user): bool {
        // Check if user owns the scan
        $stmt = $this->db->prepare("SELECT user_id FROM scans WHERE id = ?");
        $stmt->execute([$scanId]);
        $scan = $stmt->fetch();
        
        if (!$scan) return false;
        
        // Owner can always access
        if ($scan['user_id'] === $user['user_id']) return true;
        
        // Dentists with active connection can access
        if ($user['role'] === 'dentist') {
            $stmt = $this->db->prepare("
                SELECT 1 FROM patient_dentist_connections 
                WHERE patient_id = ? AND dentist_id = ? AND status = 'active'
            ");
            $stmt->execute([$scan['user_id'], $user['user_id']]);
            return (bool) $stmt->fetch();
        }
        
        // Admin can access
        if ($user['role'] === 'admin') return true;
        
        return false;
    }
    
    /**
     * Mark messages as read
     */
    private function markAsRead(string $scanId, string $userId): void {
        $stmt = $this->db->prepare("
            UPDATE scan_chat_messages 
            SET is_read = TRUE 
            WHERE scan_id = ? AND sender_id != ? AND is_read = FALSE
        ");
        $stmt->execute([$scanId, $userId]);
    }
    
    /**
     * Notify the other party about new message
     */
    private function notifyNewMessage(string $scanId, array $sender, string $message): void {
        try {
            // Get scan owner
            $stmt = $this->db->prepare("SELECT user_id FROM scans WHERE id = ?");
            $stmt->execute([$scanId]);
            $scan = $stmt->fetch();
            
            $recipientId = null;
            $title = '';
            
            if ($sender['role'] === 'dentist') {
                // Notify patient
                $recipientId = $scan['user_id'];
                $title = '💬 New message from your dentist';
            } else {
                // Notify connected dentist(s)
                $stmt = $this->db->prepare("
                    SELECT dentist_id FROM patient_dentist_connections 
                    WHERE patient_id = ? AND status = 'active' LIMIT 1
                ");
                $stmt->execute([$scan['user_id']]);
                $conn = $stmt->fetch();
                if ($conn) {
                    $recipientId = $conn['dentist_id'];
                    $title = '💬 New message from patient';
                }
            }
            
            if ($recipientId && $recipientId !== $sender['user_id']) {
                $notifId = $this->generateUuid();
                $stmt = $this->db->prepare("
                    INSERT INTO notifications (id, user_id, type, title, message, data_json)
                    VALUES (?, ?, 'chat_message', ?, ?, ?)
                ");
                $stmt->execute([
                    $notifId,
                    $recipientId,
                    $title,
                    substr($message, 0, 100) . (strlen($message) > 100 ? '...' : ''),
                    json_encode(['scan_id' => $scanId])
                ]);
            }
        } catch (\Exception $e) {
            error_log("Failed to send chat notification: " . $e->getMessage());
        }
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
