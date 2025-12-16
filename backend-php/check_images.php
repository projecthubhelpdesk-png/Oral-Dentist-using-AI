<?php
require_once __DIR__ . '/config/database.php';

$db = Database::getInstance();
$stmt = $db->query('SELECT id, original_image_path, LEFT(spectral_image_base64, 100) as spectral_preview FROM spectral_analyses ORDER BY created_at DESC LIMIT 5');
$results = $stmt->fetchAll(PDO::FETCH_ASSOC);

echo "Recent spectral analyses:\n";
foreach ($results as $row) {
    echo "ID: " . $row['id'] . "\n";
    echo "  Original path: " . ($row['original_image_path'] ?: 'NULL') . "\n";
    echo "  Spectral preview: " . ($row['spectral_preview'] ? substr($row['spectral_preview'], 0, 50) . '...' : 'NULL') . "\n";
    echo "\n";
}
