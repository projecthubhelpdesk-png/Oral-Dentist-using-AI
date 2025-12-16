<?php
/**
 * Add display name columns to users table
 */
$pdo = new PDO('mysql:host=localhost;dbname=oral_care_ai', 'root', '');

// Add first_name and last_name columns if they don't exist
$pdo->exec("
    ALTER TABLE users 
    ADD COLUMN IF NOT EXISTS first_name VARCHAR(100) DEFAULT NULL,
    ADD COLUMN IF NOT EXISTS last_name VARCHAR(100) DEFAULT NULL
");

echo "Columns added successfully!\n";

// Update existing users with names from email (temporary)
$stmt = $pdo->query("SELECT id, email FROM users WHERE first_name IS NULL");
$users = $stmt->fetchAll();

foreach ($users as $user) {
    $emailPart = explode('@', $user['email'])[0];
    // Convert email prefix to name (e.g., john.doe -> John Doe)
    $nameParts = preg_split('/[._-]/', $emailPart);
    $firstName = ucfirst($nameParts[0] ?? 'User');
    $lastName = isset($nameParts[1]) ? ucfirst($nameParts[1]) : '';
    
    $update = $pdo->prepare("UPDATE users SET first_name = ?, last_name = ? WHERE id = ?");
    $update->execute([$firstName, $lastName, $user['id']]);
    echo "Updated user: {$user['email']} -> $firstName $lastName\n";
}

echo "Done!\n";
