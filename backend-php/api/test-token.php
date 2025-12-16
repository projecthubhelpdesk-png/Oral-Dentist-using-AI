<?php
/**
 * Token Debug Script
 * Run this to test if tokens are working
 */

header('Content-Type: application/json');
header('Access-Control-Allow-Origin: *');
header('Access-Control-Allow-Headers: Authorization, Content-Type');

if ($_SERVER['REQUEST_METHOD'] === 'OPTIONS') {
    exit;
}

// Get Authorization header from various sources
$authHeader = '';
if (!empty($_SERVER['HTTP_AUTHORIZATION'])) {
    $authHeader = $_SERVER['HTTP_AUTHORIZATION'];
} elseif (!empty($_SERVER['REDIRECT_HTTP_AUTHORIZATION'])) {
    $authHeader = $_SERVER['REDIRECT_HTTP_AUTHORIZATION'];
} elseif (function_exists('apache_request_headers')) {
    $headers = apache_request_headers();
    $authHeader = $headers['Authorization'] ?? $headers['authorization'] ?? '';
}

echo json_encode([
    'authHeader' => $authHeader ? substr($authHeader, 0, 50) . '...' : 'NOT FOUND',
    'HTTP_AUTHORIZATION' => isset($_SERVER['HTTP_AUTHORIZATION']) ? 'SET' : 'NOT SET',
    'REDIRECT_HTTP_AUTHORIZATION' => isset($_SERVER['REDIRECT_HTTP_AUTHORIZATION']) ? 'SET' : 'NOT SET',
    'apache_request_headers' => function_exists('apache_request_headers') ? 'AVAILABLE' : 'NOT AVAILABLE',
    'allHeaders' => function_exists('getallheaders') ? array_keys(getallheaders()) : [],
]);
