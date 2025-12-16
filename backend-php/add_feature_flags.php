<?php
/**
 * Migration: Create feature_flags table
 */

require_once __DIR__ . '/config/database.php';

try {
    $db = Database::getInstance();
    
    // Check if table exists
    $stmt = $db->query("SHOW TABLES LIKE 'feature_flags'");
    if ($stmt->rowCount() === 0) {
        $db->exec("
            CREATE TABLE feature_flags (
                id INT AUTO_INCREMENT PRIMARY KEY,
                feature_key VARCHAR(50) NOT NULL UNIQUE,
                feature_name VARCHAR(100) NOT NULL,
                description TEXT DEFAULT NULL,
                is_enabled BOOLEAN DEFAULT TRUE,
                disabled_message VARCHAR(255) DEFAULT 'This feature is coming soon.',
                updated_by CHAR(36) DEFAULT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_feature_key (feature_key)
            ) ENGINE=InnoDB
        ");
        echo "Created feature_flags table\n";
        
        // Insert default flags
        $db->exec("
            INSERT INTO feature_flags (feature_key, feature_name, description, is_enabled, disabled_message) VALUES
            ('dentist_dashboard', 'Dentist Dashboard', 'Controls dentist account creation and login. When disabled, no one can create or login as dentist.', TRUE, 'This feature is coming soon.'),
            ('spectral_ai', 'Spectral AI Analysis', 'Controls the Spectral AI tab visibility and functionality for dentists.', TRUE, 'This feature is coming soon.')
        ");
        echo "Inserted default feature flags\n";
    } else {
        echo "feature_flags table already exists\n";
    }
    
    echo "\nMigration completed successfully!\n";
    
} catch (Exception $e) {
    echo "Migration failed: " . $e->getMessage() . "\n";
    exit(1);
}
