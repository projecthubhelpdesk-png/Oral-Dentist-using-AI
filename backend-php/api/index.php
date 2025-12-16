<?php
/**
 * API Router
 * Oral Care AI - PHP Backend
 * 
 * Main entry point for all API requests.
 */

// Ensure Authorization header is available (Apache workaround)
if (!isset($_SERVER['HTTP_AUTHORIZATION'])) {
    if (isset($_SERVER['REDIRECT_HTTP_AUTHORIZATION'])) {
        $_SERVER['HTTP_AUTHORIZATION'] = $_SERVER['REDIRECT_HTTP_AUTHORIZATION'];
    } elseif (function_exists('apache_request_headers')) {
        $headers = apache_request_headers();
        if (isset($headers['Authorization'])) {
            $_SERVER['HTTP_AUTHORIZATION'] = $headers['Authorization'];
        } elseif (isset($headers['authorization'])) {
            $_SERVER['HTTP_AUTHORIZATION'] = $headers['authorization'];
        }
    }
}

// Don't set Content-Type for image requests - let the controller handle it
$requestUri = $_SERVER['REQUEST_URI'] ?? '';
if (strpos($requestUri, '/image') === false) {
    header('Content-Type: application/json');
}

// Autoload
spl_autoload_register(function ($class) {
    $prefix = 'OralCareAI\\';
    $baseDir = __DIR__ . '/../';
    
    $len = strlen($prefix);
    if (strncmp($prefix, $class, $len) !== 0) {
        return;
    }
    
    $relativeClass = substr($class, $len);
    $file = $baseDir . str_replace('\\', '/', $relativeClass) . '.php';
    
    if (file_exists($file)) {
        require $file;
    }
});

// Load middleware
require_once __DIR__ . '/../middleware/CorsMiddleware.php';
require_once __DIR__ . '/../middleware/RateLimitMiddleware.php';
require_once __DIR__ . '/../middleware/AuthMiddleware.php';

use OralCareAI\Middleware\CorsMiddleware;
use OralCareAI\Middleware\RateLimitMiddleware;

// Handle CORS
$cors = new CorsMiddleware();
$cors->handle();

// Rate limiting
$rateLimit = new RateLimitMiddleware();
if (!$rateLimit->checkLimit()) {
    exit;
}

// Parse request
$method = $_SERVER['REQUEST_METHOD'];
$uri = parse_url($_SERVER['REQUEST_URI'], PHP_URL_PATH);

// Remove base path (handle both with and without URL encoding)
$basePath = '/oral-care-ai/backend-php/api';
$basePathSpaces = '/oral care ai/backend-php/api';
$uri = str_replace($basePath, '', $uri);
$uri = str_replace($basePathSpaces, '', $uri);
$uri = trim($uri, '/');

// Route the request
$routes = [
    // Auth routes
    'POST auth/register' => 'AuthController@register',
    'POST auth/login' => 'AuthController@login',
    'POST auth/refresh' => 'AuthController@refresh',
    'POST auth/logout' => 'AuthController@logout',
    
    // User routes
    'GET users/me' => 'UserController@me',
    'PATCH users/me' => 'UserController@update',
    
    // Scan routes
    'GET scans' => 'ScanController@index',
    'GET scans/patients' => 'ScanController@patientScans',
    'POST scans' => 'ScanController@store',
    'GET scans/{id}' => 'ScanController@show',
    'DELETE scans/{id}' => 'ScanController@archive',
    'POST scans/{id}/analyze' => 'ScanController@analyze',
    'GET scans/{id}/image' => 'ScanController@image',
    'GET scans/{id}/analysis' => 'AnalysisController@show',
    'GET scans/{id}/reviews' => 'ReviewController@index',
    'POST scans/{id}/reviews' => 'ReviewController@store',
    
    // Chat routes
    'GET scans/{id}/chat' => 'ChatController@getMessages',
    'POST scans/{id}/chat' => 'ChatController@sendMessage',
    'GET scans/{id}/chat/unread' => 'ChatController@getUnreadCount',
    
    // Report search (for dentists)
    'GET reports/search' => 'ReportSearchController@search',
    
    // Dentist routes
    'GET dentists' => 'DentistController@index',
    'GET dentists/me' => 'DentistController@me',
    'PATCH dentists/me' => 'DentistController@update',
    'GET dentists/{id}' => 'DentistController@show',
    
    // Connection routes
    'GET connections' => 'ConnectionController@index',
    'POST connections' => 'ConnectionController@store',
    'PATCH connections/{id}' => 'ConnectionController@update',
    
    // Spectral Analysis routes (Dentist only)
    'POST spectral/analyze' => 'SpectralController@analyze',
    'POST spectral/{id}/review' => 'SpectralController@review',
    'POST spectral/{id}/report' => 'SpectralController@generateReport',
    'GET spectral/{id}/report/download' => 'SpectralController@downloadReport',
    'GET spectral/history' => 'SpectralController@history',
    
    // Analytics routes
    'GET analytics/dentist' => 'AnalyticsController@dentistAnalytics',
    'GET analytics/user' => 'AnalyticsController@userAnalytics',
    
    // Feature flags routes
    'GET features' => 'FeatureController@index',
    'GET features/admin' => 'FeatureController@adminIndex',
    'PATCH features/{key}' => 'FeatureController@update',
    'GET features/check/{key}' => 'FeatureController@check',
];

// Debug mode - add ?debug=1 to see routing info
if (isset($_GET['debug'])) {
    echo json_encode([
        'method' => $method,
        'raw_uri' => $_SERVER['REQUEST_URI'],
        'parsed_uri' => $uri,
        'base_path' => $basePath,
    ], JSON_PRETTY_PRINT);
    exit;
}

// Match route
$matchedRoute = null;
$params = [];

foreach ($routes as $route => $handler) {
    [$routeMethod, $routePath] = explode(' ', $route);
    
    if ($method !== $routeMethod) {
        continue;
    }
    
    // Convert route pattern to regex
    $pattern = preg_replace('/\{(\w+)\}/', '(?P<$1>[^/]+)', $routePath);
    $pattern = '#^' . $pattern . '$#';
    
    if (preg_match($pattern, $uri, $matches)) {
        $matchedRoute = $handler;
        $params = array_filter($matches, 'is_string', ARRAY_FILTER_USE_KEY);
        break;
    }
}

if (!$matchedRoute) {
    http_response_code(404);
    echo json_encode(['error' => 'Not Found', 'message' => 'Endpoint not found', 'debug_uri' => $uri, 'debug_method' => $method]);
    exit;
}

// Load and execute controller
[$controllerName, $action] = explode('@', $matchedRoute);
$controllerFile = __DIR__ . "/../api/controllers/{$controllerName}.php";

if (!file_exists($controllerFile)) {
    http_response_code(500);
    echo json_encode(['error' => 'Server Error', 'message' => 'Controller not found']);
    exit;
}

require_once $controllerFile;
$controllerClass = "OralCareAI\\Controllers\\{$controllerName}";
$controller = new $controllerClass();

// Get request body
$body = json_decode(file_get_contents('php://input'), true) ?? [];

// Execute action
try {
    $controller->$action($params, $body);
} catch (Exception $e) {
    $config = require __DIR__ . '/../config/app.php';
    
    http_response_code(500);
    $response = ['error' => 'Server Error', 'message' => 'An unexpected error occurred'];
    
    if ($config['debug']) {
        $response['debug'] = $e->getMessage();
    }
    
    echo json_encode($response);
}
