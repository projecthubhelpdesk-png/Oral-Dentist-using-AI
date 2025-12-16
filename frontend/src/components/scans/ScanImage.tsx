import { useState, useEffect } from 'react';

interface ScanImageProps {
  imageUrl: string;
  alt?: string;
  className?: string;
}

export function ScanImage({ imageUrl, alt = 'Scan Image', className = '' }: ScanImageProps) {
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(false);

  useEffect(() => {
    let objectUrl: string | null = null;
    
    const fetchImage = async () => {
      if (!imageUrl) {
        setError(true);
        setIsLoading(false);
        return;
      }

      try {
        setIsLoading(true);
        setError(false);

        const token = localStorage.getItem('accessToken');
        
        // Build full URL - use the API base URL from env
        const apiBaseUrl = import.meta.env.VITE_API_URL || 'http://localhost/oral-care-ai/backend-php/api';
        
        let fullUrl = imageUrl;
        // If it's a relative URL starting with /oral-care-ai, use localhost
        if (imageUrl.startsWith('/oral-care-ai')) {
          fullUrl = `http://localhost${imageUrl}`;
        } else if (imageUrl.startsWith('/')) {
          // If it starts with /, prepend the API base (without /api)
          const baseWithoutApi = apiBaseUrl.replace('/api', '');
          fullUrl = `${baseWithoutApi}${imageUrl}`;
        }
        
        // Add token as query param
        fullUrl = fullUrl.includes('?') 
          ? `${fullUrl}&token=${token}` 
          : `${fullUrl}?token=${token}`;

        console.log('Fetching image from:', fullUrl);

        const response = await fetch(fullUrl, {
          headers: {
            'Authorization': `Bearer ${token}`
          }
        });

        if (!response.ok) {
          console.error('Image fetch failed:', response.status, response.statusText);
          throw new Error(`Failed to fetch image: ${response.status}`);
        }

        const blob = await response.blob();
        objectUrl = URL.createObjectURL(blob);
        setImageSrc(objectUrl);
      } catch (err) {
        console.error('Failed to load scan image:', err);
        setError(true);
      } finally {
        setIsLoading(false);
      }
    };

    fetchImage();

    // Cleanup object URL on unmount
    return () => {
      if (objectUrl) {
        URL.revokeObjectURL(objectUrl);
      }
    };
  }, [imageUrl]);

  if (isLoading) {
    return (
      <div className={`flex items-center justify-center bg-gray-100 rounded-lg ${className}`} style={{ minHeight: '200px' }}>
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600" />
      </div>
    );
  }

  if (error || !imageSrc) {
    return (
      <div className={`flex flex-col items-center justify-center bg-gray-100 rounded-lg py-12 ${className}`}>
        <svg className="w-16 h-16 text-gray-300 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
        </svg>
        <p className="text-sm text-gray-400">Image not available</p>
      </div>
    );
  }

  return (
    <img
      src={imageSrc}
      alt={alt}
      className={`rounded-lg border border-gray-200 shadow-sm ${className}`}
    />
  );
}
