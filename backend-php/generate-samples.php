<?php
/**
 * Generate sample dental scan images using GD or fallback to downloading
 */

$storageDir = __DIR__ . '/storage/scans/';

if (!is_dir($storageDir)) {
    mkdir($storageDir, 0755, true);
}

// Try to use placeholder image service
$samples = [
    'sample1.jpg' => 'https://placehold.co/800x600/e3f2fd/1976d2?text=Dental+Scan+1',
    'sample2.jpg' => 'https://placehold.co/800x600/e8f5e9/388e3c?text=Dental+Scan+2', 
    'sample3.jpg' => 'https://placehold.co/800x600/fff3e0/f57c00?text=Spectral+Scan',
    'sample4.jpg' => 'https://placehold.co/800x600/f3e5f5/7b1fa2?text=Dental+Scan+4',
];

foreach ($samples as $filename => $url) {
    $filepath = $storageDir . $filename;
    
    // Download from placeholder service
    $context = stream_context_create([
        'http' => [
            'timeout' => 10,
            'user_agent' => 'Mozilla/5.0'
        ]
    ]);
    
    $imageData = @file_get_contents($url, false, $context);
    
    if ($imageData) {
        file_put_contents($filepath, $imageData);
        echo "Downloaded: $filename (" . strlen($imageData) . " bytes)\n";
    } else {
        // Fallback: create a simple valid PNG
        $png = createSimplePNG($filename);
        file_put_contents($filepath, $png);
        echo "Created fallback: $filename\n";
    }
}

function createSimplePNG($name) {
    // Create a minimal valid PNG (8x8 pixels, solid color)
    $width = 8;
    $height = 8;
    
    // PNG signature
    $png = "\x89PNG\r\n\x1a\n";
    
    // IHDR chunk
    $ihdr = pack('N', $width) . pack('N', $height) . "\x08\x02\x00\x00\x00";
    $png .= pack('N', 13) . 'IHDR' . $ihdr . pack('N', crc32('IHDR' . $ihdr));
    
    // IDAT chunk (compressed image data - solid light blue)
    $raw = '';
    for ($y = 0; $y < $height; $y++) {
        $raw .= "\x00"; // filter byte
        for ($x = 0; $x < $width; $x++) {
            $raw .= "\xcc\xe5\xff"; // RGB light blue
        }
    }
    $compressed = gzcompress($raw);
    $png .= pack('N', strlen($compressed)) . 'IDAT' . $compressed . pack('N', crc32('IDAT' . $compressed));
    
    // IEND chunk
    $png .= pack('N', 0) . 'IEND' . pack('N', crc32('IEND'));
    
    return $png;
}

echo "\nDone!\n";
