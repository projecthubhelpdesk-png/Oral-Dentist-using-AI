<?php
/**
 * Database Test Script
 * Delete this file after testing!
 */

header('Content-Type: application/json');

require_once __DIR__ . '/../config/database.php';

try {
    $db = Database::getInstance();
    echo json_encode(['status' => 'connected']);
    
    // Check if users table exists and has data
    $stmt = $db->query("SELECT COUNT(*) as count FROM users");
    $result = $stmt->fetch();
    echo "\n" . json_encode(['users_count' => $result['count']]);
    
    // Try to find the demo user
    $stmt = $db->prepare("SELECT id, email, role FROM users WHERE email = ?");
    $stmt->execute(['john.doe@example.com']);
    $user = $stmt->fetch();
    echo "\n" . json_encode(['demo_user' => $user ?: 'NOT FOUND']);
    
} catch (Exception $e) {
    echo json_encode([
        'status' => 'error',
        'message' => $e->getMessage()
    ]);
}
