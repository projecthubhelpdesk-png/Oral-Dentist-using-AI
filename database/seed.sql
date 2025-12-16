-- Oral Care AI Seed Data
-- Test data for development (passwords are 'password123' hashed with bcrypt)

USE oral_care_ai;

-- ============================================
-- USERS
-- ============================================
-- Password for all users: password123
-- Hash generated with: password_hash('password123', PASSWORD_BCRYPT)
INSERT INTO users (id, email, password_hash, role, phone_hash, phone_last_four, email_verified, is_active) VALUES
-- Admin user
('a0000000-0000-0000-0000-000000000001', 'admin@oralcare.ai', '$2y$10$TKh8H1.PfQx37YgCzwiKb.KjNyWgaHb9cbcoQgdIVFlYg7B77UdFm', 'admin', NULL, NULL, TRUE, TRUE),

-- Regular users (patients)
('u0000000-0000-0000-0000-000000000001', 'john.doe@example.com', '$2y$10$TKh8H1.PfQx37YgCzwiKb.KjNyWgaHb9cbcoQgdIVFlYg7B77UdFm', 'user', 'hash_5551234567_salt1', '4567', TRUE, TRUE),
('u0000000-0000-0000-0000-000000000002', 'jane.smith@example.com', '$2y$10$TKh8H1.PfQx37YgCzwiKb.KjNyWgaHb9cbcoQgdIVFlYg7B77UdFm', 'user', 'hash_5559876543_salt2', '6543', TRUE, TRUE),
('u0000000-0000-0000-0000-000000000003', 'bob.wilson@example.com', '$2y$10$TKh8H1.PfQx37YgCzwiKb.KjNyWgaHb9cbcoQgdIVFlYg7B77UdFm', 'user', 'hash_5555551234_salt3', '1234', FALSE, TRUE),

-- Dentists
('d0000000-0000-0000-0000-000000000001', 'dr.sarah.chen@dental.com', '$2y$10$TKh8H1.PfQx37YgCzwiKb.KjNyWgaHb9cbcoQgdIVFlYg7B77UdFm', 'dentist', 'hash_5552223333_salt4', '3333', TRUE, TRUE),
('d0000000-0000-0000-0000-000000000002', 'dr.mike.johnson@dental.c om', '$2y$10$TKh8H1.PfQx37YgCzwiKb.KjNyWgaHb9cbcoQgdIVFlYg7B77UdFm', 'dentist', 'hash_5554445555_salt5', '5555', TRUE, TRUE),
('d0000000-0000-0000-0000-000000000003', 'dr.emily.wong@dental.com', '$2y$10$TKh8H1.PfQx37YgCzwiKb.KjNyWgaHb9cbcoQgdIVFlYg7B77UdFm', 'dentist', 'hash_5556667777_salt6', '7777', TRUE, TRUE),
('d0000000-0000-0000-0000-000000000004', 'dr.james.patel@dental.com', '$2y$10$TKh8H1.PfQx37YgCzwiKb.KjNyWgaHb9cbcoQgdIVFlYg7B77UdFm', 'dentist', 'hash_5558889999_salt7', '9999', TRUE, TRUE),
('d0000000-0000-0000-0000-000000000005', 'dr.lisa.martinez@dental.com', '$2y$10$TKh8H1.PfQx37YgCzwiKb.KjNyWgaHb9cbcoQgdIVFlYg7B77UdFm', 'dentist', 'hash_5551112222_salt8', '2222', TRUE, TRUE);

-- ============================================
-- DENTIST PROFILES
-- ============================================
INSERT INTO dentist_profiles (id, user_id, license_number_encrypted, license_state, license_verified, specialty, clinic_name, years_experience, bio, accepting_patients, consultation_fee_cents, average_rating, total_reviews) VALUES
('dp000000-0000-0000-0000-000000000001', 'd0000000-0000-0000-0000-000000000001', 'encrypted_DDS12345', 'CA', TRUE, 'Cosmetic Dentistry', 'Bright Smile Dental', 12, 'Dr. Chen specializes in cosmetic dentistry with over 12 years of experience. She is passionate about using AI technology to improve patient outcomes.', TRUE, 7500, 4.85, 127),
('dp000000-0000-0000-0000-000000000002', 'd0000000-0000-0000-0000-000000000002', 'encrypted_DDS67890', 'NY', TRUE, 'Periodontics', 'Manhattan Dental Care', 8, 'Dr. Johnson is a board-certified periodontist focusing on gum health and implant dentistry.', TRUE, 10000, 4.72, 89),
('dp000000-0000-0000-0000-000000000003', 'd0000000-0000-0000-0000-000000000003', 'encrypted_DDS11111', 'CA', TRUE, 'General Dentistry', 'Family Dental Center', 15, 'Dr. Wong provides comprehensive family dental care with a gentle approach. Specializes in preventive care and patient education.', TRUE, 5000, 4.92, 203),
('dp000000-0000-0000-0000-000000000004', 'd0000000-0000-0000-0000-000000000004', 'encrypted_DDS22222', 'TX', TRUE, 'Orthodontics', 'Patel Orthodontics', 10, 'Dr. Patel is an expert in Invisalign and traditional braces. Known for creating beautiful smiles for patients of all ages.', TRUE, 8500, 4.68, 156),
('dp000000-0000-0000-0000-000000000005', 'd0000000-0000-0000-0000-000000000005', 'encrypted_DDS33333', 'FL', TRUE, 'Endodontics', 'Root Canal Specialists', 7, 'Dr. Martinez specializes in root canal therapy and dental trauma. Uses the latest technology for pain-free procedures.', TRUE, 12000, 4.55, 78);

-- ============================================
-- SCANS
-- ============================================
INSERT INTO scans (id, user_id, scan_type, image_storage_path, image_hash, status, capture_device, metadata_json, uploaded_at, processed_at) VALUES
('s0000000-0000-0000-0000-000000000001', 'u0000000-0000-0000-0000-000000000001', 'basic_rgb', 'sample1.jpg', 'a1b2c3d4e5f6789012345678901234567890123456789012345678901234abcd', 'analyzed', 'iPhone 14 Pro', '{"resolution": "4032x3024", "flash": true}', '2024-12-01 10:30:00', '2024-12-01 10:31:15'),
('s0000000-0000-0000-0000-000000000002', 'u0000000-0000-0000-0000-000000000001', 'basic_rgb', 'sample2.jpg', 'b2c3d4e5f67890123456789012345678901234567890123456789012345bcde', 'analyzed', 'iPhone 14 Pro', '{"resolution": "4032x3024", "flash": true}', '2024-12-05 14:20:00', '2024-12-05 14:21:30'),
('s0000000-0000-0000-0000-000000000003', 'u0000000-0000-0000-0000-000000000002', 'advanced_spectral', 'sample3.jpg', 'c3d4e5f678901234567890123456789012345678901234567890123456cdef', 'analyzed', 'OralScan Pro 3000', '{"spectral_bands": 8, "resolution": "2048x2048"}', '2024-12-08 09:15:00', '2024-12-08 09:18:45'),
('s0000000-0000-0000-0000-000000000004', 'u0000000-0000-0000-0000-000000000003', 'basic_rgb', 'sample4.jpg', 'd4e5f6789012345678901234567890123456789012345678901234567defg', 'uploaded', 'Samsung Galaxy S23', '{"resolution": "4000x3000", "flash": false}', '2024-12-10 16:45:00', NULL);

-- ============================================
-- ANALYSIS RESULTS
-- ============================================
INSERT INTO analysis_results (id, scan_id, model_type, model_version, overall_score, confidence_score, findings_json, risk_areas_json, recommendations_json, processing_time_ms) VALUES
('ar00000-0000-0000-0000-000000000001', 's0000000-0000-0000-0000-000000000001', 'basic_rgb', '2.1.0', 78.50, 0.8923, 
'{"findings": [{"type": "plaque_buildup", "severity": "mild", "location": "lower_molars", "confidence": 0.87}, {"type": "gum_inflammation", "severity": "minimal", "location": "upper_front", "confidence": 0.72}]}',
'{"regions": [{"x": 120, "y": 340, "width": 80, "height": 60, "label": "plaque_buildup"}, {"x": 200, "y": 150, "width": 100, "height": 40, "label": "gum_inflammation"}]}',
'{"recommendations": ["Increase brushing frequency to twice daily", "Consider using an electric toothbrush", "Schedule professional cleaning within 3 months"]}',
1150),

('ar00000-0000-0000-0000-000000000002', 's0000000-0000-0000-0000-000000000002', 'basic_rgb', '2.1.0', 82.30, 0.9145,
'{"findings": [{"type": "healthy", "severity": "none", "location": "overall", "confidence": 0.91}]}',
'{"regions": []}',
'{"recommendations": ["Continue current oral hygiene routine", "Next scan recommended in 6 months"]}',
980),

('ar00000-0000-0000-0000-000000000003', 's0000000-0000-0000-0000-000000000003', 'advanced_spectral', '1.5.2', 65.20, 0.9567,
'{"findings": [{"type": "early_cavity", "severity": "moderate", "location": "upper_right_molar", "confidence": 0.95, "spectral_signature": "demineralization"}, {"type": "enamel_erosion", "severity": "mild", "location": "front_teeth", "confidence": 0.88}]}',
'{"regions": [{"x": 450, "y": 280, "width": 40, "height": 40, "label": "early_cavity", "spectral_data": {"band_4": 0.72, "band_6": 0.45}}, {"x": 180, "y": 120, "width": 120, "height": 30, "label": "enamel_erosion"}]}',
'{"recommendations": ["Schedule dentist appointment within 2 weeks", "Avoid acidic foods and beverages", "Use fluoride mouthwash daily", "Consider dental sealants"]}',
3420);

-- ============================================
-- DENTIST REVIEWS
-- ============================================
INSERT INTO dentist_reviews (id, scan_id, dentist_id, ai_result_id, agrees_with_ai, professional_assessment, diagnosis_codes, treatment_recommendations, urgency_level, follow_up_days, is_draft, reviewed_at) VALUES
('dr00000-0000-0000-0000-000000000001', 's0000000-0000-0000-0000-000000000001', 'd0000000-0000-0000-0000-000000000001', 'ar00000-0000-0000-0000-000000000001', TRUE, 'AI analysis is accurate. Patient shows mild plaque accumulation on lower molars consistent with irregular brushing patterns. Gum inflammation is minimal and should resolve with improved hygiene.', '["K03.0", "K05.10"]', 'Professional cleaning recommended. Patient should switch to soft-bristle brush and consider water flosser for interdental cleaning.', 'routine', 90, FALSE, '2024-12-02 11:00:00'),

('dr00000-0000-0000-0000-000000000002', 's0000000-0000-0000-0000-000000000003', 'd0000000-0000-0000-0000-000000000002', 'ar00000-0000-0000-0000-000000000003', TRUE, 'Spectral analysis correctly identified early-stage cavity on tooth #3. Demineralization pattern is consistent with initial carious lesion. Enamel erosion on anterior teeth likely due to dietary acids.', '["K02.51", "K03.2"]', 'Recommend fluoride varnish application and monitoring. If progression noted in 4 weeks, minimally invasive restoration may be needed. Dietary counseling for acid reduction.', 'soon', 28, FALSE, '2024-12-09 14:30:00');

-- ============================================
-- PATIENT-DENTIST CONNECTIONS
-- ============================================
INSERT INTO patient_dentist_connections (id, patient_id, dentist_id, status, initiated_by, share_scan_history, connected_at) VALUES
('pc00000-0000-0000-0000-000000000001', 'u0000000-0000-0000-0000-000000000001', 'd0000000-0000-0000-0000-000000000001', 'active', 'patient', TRUE, '2024-11-15 09:00:00'),
('pc00000-0000-0000-0000-000000000002', 'u0000000-0000-0000-0000-000000000002', 'd0000000-0000-0000-0000-000000000002', 'active', 'dentist', TRUE, '2024-12-01 10:00:00'),
('pc00000-0000-0000-0000-000000000003', 'u0000000-0000-0000-0000-000000000003', 'd0000000-0000-0000-0000-000000000001', 'pending', 'patient', FALSE, NULL);

-- ============================================
-- NOTIFICATIONS
-- ============================================
INSERT INTO notifications (id, user_id, type, title, message, data_json, is_read) VALUES
('n0000000-0000-0000-0000-000000000001', 'u0000000-0000-0000-0000-000000000001', 'analysis_complete', 'Scan Analysis Ready', 'Your dental scan from Dec 5 has been analyzed. View your results now.', '{"scan_id": "s0000000-0000-0000-0000-000000000002"}', FALSE),
('n0000000-0000-0000-0000-000000000002', 'u0000000-0000-0000-0000-000000000001', 'dentist_review', 'Dr. Chen Reviewed Your Scan', 'Dr. Sarah Chen has provided a professional assessment of your scan.', '{"review_id": "dr00000-0000-0000-0000-000000000001"}', TRUE),
('n0000000-0000-0000-0000-000000000003', 'd0000000-0000-0000-0000-000000000001', 'new_patient_request', 'New Patient Connection Request', 'Bob Wilson has requested to connect with you.', '{"patient_id": "u0000000-0000-0000-0000-000000000003"}', FALSE);
