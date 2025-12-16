-- Oral Care AI Database Schema
-- MySQL 8.0+

CREATE DATABASE IF NOT EXISTS oral_care_ai CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE oral_care_ai;

-- ============================================
-- USERS TABLE
-- ============================================
CREATE TABLE users (
    id CHAR(36) PRIMARY KEY DEFAULT (UUID()),
    email VARCHAR(255) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    role ENUM('user', 'dentist', 'admin') NOT NULL DEFAULT 'user',
    phone_hash VARCHAR(255) DEFAULT NULL,          -- Salted hash of phone number
    phone_last_four VARCHAR(4) DEFAULT NULL,       -- Last 4 digits for display
    first_name VARCHAR(100) DEFAULT NULL,          -- Display name
    last_name VARCHAR(100) DEFAULT NULL,           -- Display name
    first_name_encrypted VARBINARY(512) DEFAULT NULL,
    last_name_encrypted VARBINARY(512) DEFAULT NULL,
    date_of_birth_encrypted VARBINARY(256) DEFAULT NULL,
    profile_image_url VARCHAR(500) DEFAULT NULL,
    email_verified BOOLEAN DEFAULT FALSE,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    last_login_at TIMESTAMP NULL,
    INDEX idx_email (email),
    INDEX idx_role (role),
    INDEX idx_phone_hash (phone_hash)
) ENGINE=InnoDB;

-- ============================================
-- DENTIST PROFILES (extends users with role='dentist')
-- ============================================
CREATE TABLE dentist_profiles (
    id CHAR(36) PRIMARY KEY DEFAULT (UUID()),
    user_id CHAR(36) NOT NULL UNIQUE,
    license_number_encrypted VARBINARY(256) NOT NULL,
    license_state VARCHAR(50) NOT NULL,
    license_verified BOOLEAN DEFAULT FALSE,
    specialty VARCHAR(100) DEFAULT 'General Dentistry',
    clinic_name VARCHAR(255) DEFAULT NULL,
    clinic_address_encrypted VARBINARY(1024) DEFAULT NULL,
    years_experience INT DEFAULT 0,
    bio TEXT DEFAULT NULL,
    accepting_patients BOOLEAN DEFAULT TRUE,
    consultation_fee_cents INT DEFAULT 0,
    average_rating DECIMAL(3,2) DEFAULT 0.00,
    total_reviews INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
) ENGINE=InnoDB;

-- ============================================
-- SCANS / IMAGE UPLOADS
-- ============================================
CREATE TABLE scans (
    id CHAR(36) PRIMARY KEY DEFAULT (UUID()),
    user_id CHAR(36) NOT NULL,
    scan_type ENUM('basic_rgb', 'advanced_spectral') NOT NULL,
    image_storage_path VARCHAR(500) NOT NULL,      -- Encrypted path in object storage
    image_hash VARCHAR(64) NOT NULL,               -- SHA-256 for integrity
    thumbnail_path VARCHAR(500) DEFAULT NULL,
    original_filename VARCHAR(255) DEFAULT NULL,
    file_size_bytes INT DEFAULT 0,
    mime_type VARCHAR(100) DEFAULT 'image/jpeg',
    capture_device VARCHAR(100) DEFAULT NULL,
    metadata_json JSON DEFAULT NULL,               -- Non-PII metadata
    status ENUM('uploaded', 'processing', 'analyzed', 'failed', 'archived') DEFAULT 'uploaded',
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP NULL,
    INDEX idx_user_id (user_id),
    INDEX idx_status (status),
    INDEX idx_scan_type (scan_type),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
) ENGINE=InnoDB;

-- ============================================
-- AI ANALYSIS RESULTS
-- ============================================
CREATE TABLE analysis_results (
    id CHAR(36) PRIMARY KEY DEFAULT (UUID()),
    scan_id CHAR(36) NOT NULL,
    model_type ENUM('basic_rgb', 'advanced_spectral') NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    overall_score DECIMAL(5,2) DEFAULT NULL,       -- 0-100 health score
    confidence_score DECIMAL(5,4) DEFAULT NULL,    -- 0-1 confidence
    findings_json JSON NOT NULL,                   -- Structured findings
    risk_areas_json JSON DEFAULT NULL,             -- Highlighted regions
    recommendations_json JSON DEFAULT NULL,
    raw_output_encrypted VARBINARY(8000) DEFAULT NULL,
    processing_time_ms INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_scan_id (scan_id),
    INDEX idx_model_type (model_type),
    FOREIGN KEY (scan_id) REFERENCES scans(id) ON DELETE CASCADE
) ENGINE=InnoDB;

-- ============================================
-- DENTIST REVIEWS OF SCANS
-- ============================================
CREATE TABLE dentist_reviews (
    id CHAR(36) PRIMARY KEY DEFAULT (UUID()),
    scan_id CHAR(36) NOT NULL,
    dentist_id CHAR(36) NOT NULL,
    ai_result_id CHAR(36) DEFAULT NULL,
    agrees_with_ai BOOLEAN DEFAULT NULL,
    professional_assessment TEXT DEFAULT NULL,
    diagnosis_codes JSON DEFAULT NULL,             -- ICD/dental codes
    treatment_recommendations TEXT DEFAULT NULL,
    urgency_level ENUM('routine', 'soon', 'urgent', 'emergency') DEFAULT 'routine',
    follow_up_days INT DEFAULT NULL,
    is_draft BOOLEAN DEFAULT TRUE,
    reviewed_at TIMESTAMP NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_scan_id (scan_id),
    INDEX idx_dentist_id (dentist_id),
    FOREIGN KEY (scan_id) REFERENCES scans(id) ON DELETE CASCADE,
    FOREIGN KEY (dentist_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (ai_result_id) REFERENCES analysis_results(id) ON DELETE SET NULL
) ENGINE=InnoDB;

-- ============================================
-- USER-DENTIST CONNECTIONS
-- ============================================
CREATE TABLE patient_dentist_connections (
    id CHAR(36) PRIMARY KEY DEFAULT (UUID()),
    patient_id CHAR(36) NOT NULL,
    dentist_id CHAR(36) NOT NULL,
    status ENUM('pending', 'active', 'declined', 'terminated') DEFAULT 'pending',
    initiated_by ENUM('patient', 'dentist') NOT NULL,
    share_scan_history BOOLEAN DEFAULT FALSE,
    notes_encrypted VARBINARY(2048) DEFAULT NULL,
    connected_at TIMESTAMP NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY unique_connection (patient_id, dentist_id),
    INDEX idx_patient_id (patient_id),
    INDEX idx_dentist_id (dentist_id),
    FOREIGN KEY (patient_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (dentist_id) REFERENCES users(id) ON DELETE CASCADE
) ENGINE=InnoDB;

-- ============================================
-- REFRESH TOKENS
-- ============================================
CREATE TABLE refresh_tokens (
    id CHAR(36) PRIMARY KEY DEFAULT (UUID()),
    user_id CHAR(36) NOT NULL,
    token_hash VARCHAR(255) NOT NULL UNIQUE,
    device_info VARCHAR(255) DEFAULT NULL,
    ip_address VARCHAR(45) DEFAULT NULL,
    expires_at TIMESTAMP NOT NULL,
    revoked BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_user_id (user_id),
    INDEX idx_token_hash (token_hash),
    INDEX idx_expires_at (expires_at),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
) ENGINE=InnoDB;

-- ============================================
-- AUDIT LOG
-- ============================================
CREATE TABLE audit_logs (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id CHAR(36) DEFAULT NULL,
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(50) DEFAULT NULL,
    resource_id CHAR(36) DEFAULT NULL,
    ip_address VARCHAR(45) DEFAULT NULL,
    user_agent VARCHAR(500) DEFAULT NULL,
    details_json JSON DEFAULT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_user_id (user_id),
    INDEX idx_action (action),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB;

-- ============================================
-- NOTIFICATIONS
-- ============================================
CREATE TABLE notifications (
    id CHAR(36) PRIMARY KEY DEFAULT (UUID()),
    user_id CHAR(36) NOT NULL,
    type VARCHAR(50) NOT NULL,
    title VARCHAR(255) NOT NULL,
    message TEXT DEFAULT NULL,
    data_json JSON DEFAULT NULL,
    is_read BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_user_id (user_id),
    INDEX idx_is_read (is_read),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
) ENGINE=InnoDB;

-- ============================================
-- SCAN CHAT MESSAGES (User-Dentist Communication)
-- ============================================
CREATE TABLE scan_chat_messages (
    id CHAR(36) PRIMARY KEY DEFAULT (UUID()),
    scan_id CHAR(36) NOT NULL,
    sender_id CHAR(36) NOT NULL,
    message TEXT NOT NULL,
    message_type ENUM('text', 'image', 'system') DEFAULT 'text',
    is_read BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_scan_id (scan_id),
    INDEX idx_sender_id (sender_id),
    INDEX idx_created_at (created_at),
    FOREIGN KEY (scan_id) REFERENCES scans(id) ON DELETE CASCADE,
    FOREIGN KEY (sender_id) REFERENCES users(id) ON DELETE CASCADE
) ENGINE=InnoDB;


-- ============================================
-- SPECTRAL ANALYSES (Advanced Dentist AI)
-- ============================================
-- ============================================
-- FEATURE FLAGS (Admin controlled)
-- ============================================
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
    INDEX idx_feature_key (feature_key),
    FOREIGN KEY (updated_by) REFERENCES users(id) ON DELETE SET NULL
) ENGINE=InnoDB;

-- Insert default feature flags
INSERT INTO feature_flags (feature_key, feature_name, description, is_enabled, disabled_message) VALUES
('dentist_dashboard', 'Dentist Dashboard', 'Controls dentist account creation and login. When disabled, no one can create or login as dentist.', TRUE, 'This feature is coming soon.'),
('spectral_ai', 'Spectral AI Analysis', 'Controls the Spectral AI tab visibility and functionality for dentists.', TRUE, 'This feature is coming soon.');

CREATE TABLE spectral_analyses (
    id CHAR(36) PRIMARY KEY DEFAULT (UUID()),
    dentist_id CHAR(36) NOT NULL,
    patient_id CHAR(36) DEFAULT NULL,
    patient_name VARCHAR(255) DEFAULT NULL,
    patient_phone VARCHAR(50) DEFAULT NULL,
    image_type ENUM('nir', 'fluorescence', 'intraoral') NOT NULL DEFAULT 'nir',
    original_image_path VARCHAR(500) DEFAULT NULL,
    spectral_image_base64 LONGTEXT DEFAULT NULL,
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
) ENGINE=InnoDB;
