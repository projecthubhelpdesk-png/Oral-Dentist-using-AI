<?php
/**
 * Add spectral_analyses table for advanced dentist AI features
 */
$pdo = new PDO('mysql:host=localhost;dbname=oral_care_ai', 'root', '');

// Create spectral_analyses table
$pdo->exec("
    CREATE TABLE IF NOT EXISTS spectral_analyses (
        id CHAR(36) PRIMARY KEY,
        dentist_id CHAR(36) NOT NULL,
        patient_id CHAR(36) DEFAULT NULL,
        patient_name VARCHAR(255) DEFAULT NULL,
        patient_phone VARCHAR(50) DEFAULT NULL,
        image_type ENUM('nir', 'fluorescence', 'intraoral') NOT NULL DEFAULT 'nir',
        analysis_json JSON NOT NULL,
        health_score DECIMAL(5,2) DEFAULT 0,
        status ENUM('pending_review', 'approved', 'edited', 'rejected') DEFAULT 'pending_review',
        review_action VARCHAR(20) DEFAULT NULL,
        clinical_notes TEXT DEFAULT NULL,
        edited_diagnosis TEXT DEFAULT NULL,
        report_id VARCHAR(50) DEFAULT NULL,
        reviewed_at TIMESTAMP NULL,
        report_generated_at TIMESTAMP NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_dentist_id (dentist_id),
        INDEX idx_patient_id (patient_id),
        INDEX idx_status (status),
        INDEX idx_created_at (created_at),
        FOREIGN KEY (dentist_id) REFERENCES users(id) ON DELETE CASCADE,
        FOREIGN KEY (patient_id) REFERENCES users(id) ON DELETE SET NULL
    ) ENGINE=InnoDB
");

// Add columns if table already exists
try {
    $pdo->exec("ALTER TABLE spectral_analyses ADD COLUMN patient_name VARCHAR(255) DEFAULT NULL AFTER patient_id");
    echo "Added patient_name column\n";
} catch (Exception $e) {
    // Column might already exist
}

try {
    $pdo->exec("ALTER TABLE spectral_analyses ADD COLUMN patient_phone VARCHAR(50) DEFAULT NULL AFTER patient_name");
    echo "Added patient_phone column\n";
} catch (Exception $e) {
    // Column might already exist
}

echo "Spectral analyses table created/updated successfully!\n";
