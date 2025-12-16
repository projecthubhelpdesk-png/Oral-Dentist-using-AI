<?php
/**
 * Migration: Add image storage columns to spectral_analyses table
 */

require_once __DIR__ . '/config/database.php';

try {
    $db = Database::getInstance();
    
    // Check if columns exist
    $stmt = $db->query("SHOW COLUMNS FROM spectral_analyses LIKE 'original_image_path'");
    if ($stmt->rowCount() === 0) {
        $db->exec("ALTER TABLE spectral_analyses ADD COLUMN original_image_path VARCHAR(500) DEFAULT NULL AFTER image_type");
        echo "Added original_image_path column\n";
    } else {
        echo "original_image_path column already exists\n";
    }
    
    $stmt = $db->query("SHOW COLUMNS FROM spectral_analyses LIKE 'spectral_image_base64'");
    if ($stmt->rowCount() === 0) {
        $db->exec("ALTER TABLE spectral_analyses ADD COLUMN spectral_image_base64 LONGTEXT DEFAULT NULL AFTER original_image_path");
        echo "Added spectral_image_base64 column\n";
    } else {
        echo "spectral_image_base64 column already exists\n";
    }
    
    // Also ensure patient_name and patient_phone exist
    $stmt = $db->query("SHOW COLUMNS FROM spectral_analyses LIKE 'patient_name'");
    if ($stmt->rowCount() === 0) {
        $db->exec("ALTER TABLE spectral_analyses ADD COLUMN patient_name VARCHAR(255) DEFAULT NULL AFTER patient_id");
        echo "Added patient_name column\n";
    }
    
    $stmt = $db->query("SHOW COLUMNS FROM spectral_analyses LIKE 'patient_phone'");
    if ($stmt->rowCount() === 0) {
        $db->exec("ALTER TABLE spectral_analyses ADD COLUMN patient_phone VARCHAR(50) DEFAULT NULL AFTER patient_name");
        echo "Added patient_phone column\n";
    }
    
    echo "\nMigration completed successfully!\n";
    
} catch (Exception $e) {
    echo "Migration failed: " . $e->getMessage() . "\n";
    exit(1);
}
