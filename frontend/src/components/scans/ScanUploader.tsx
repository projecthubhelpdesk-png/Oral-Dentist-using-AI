import { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { clsx } from 'clsx';
import { Button } from '@/components/ui/Button';
import type { ScanType } from '@/types';

interface ScanUploaderProps {
  onUpload: (file: File, scanType: ScanType, captureDevice?: string) => Promise<void>;
  isUploading?: boolean;
}

export function ScanUploader({ onUpload, isUploading = false }: ScanUploaderProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [scanType, setScanType] = useState<ScanType>('basic_rgb');
  const [preview, setPreview] = useState<string | null>(null);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (file) {
      setSelectedFile(file);
      const reader = new FileReader();
      reader.onload = () => setPreview(reader.result as string);
      reader.readAsDataURL(file);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/jpeg': ['.jpg', '.jpeg'],
      'image/png': ['.png'],
      'image/webp': ['.webp'],
    },
    maxSize: 10 * 1024 * 1024, // 10MB
    multiple: false,
  });

  const handleUpload = async () => {
    if (!selectedFile) return;
    await onUpload(selectedFile, scanType);
    setSelectedFile(null);
    setPreview(null);
  };

  const clearSelection = () => {
    setSelectedFile(null);
    setPreview(null);
  };

  return (
    <div className="space-y-4">
      {/* Scan Type Selection */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">Scan Type</label>
        <div className="flex gap-4">
          <label className="flex items-center">
            <input
              type="radio"
              name="scanType"
              value="basic_rgb"
              checked={scanType === 'basic_rgb'}
              onChange={() => setScanType('basic_rgb')}
              className="mr-2 text-primary-600 focus:ring-primary-500"
            />
            <span className="text-sm">
              <span className="font-medium">Basic RGB</span>
              <span className="text-gray-500 ml-1">(Phone camera)</span>
            </span>
          </label>
          <label className="flex items-center">
            <input
              type="radio"
              name="scanType"
              value="advanced_spectral"
              checked={scanType === 'advanced_spectral'}
              onChange={() => setScanType('advanced_spectral')}
              className="mr-2 text-primary-600 focus:ring-primary-500"
            />
            <span className="text-sm">
              <span className="font-medium">Advanced Spectral</span>
              <span className="text-gray-500 ml-1">(Multispectral device)</span>
            </span>
          </label>
        </div>
      </div>

      {/* Dropzone */}
      {!selectedFile ? (
        <div
          {...getRootProps()}
          className={clsx(
            'border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-colors',
            isDragActive
              ? 'border-primary-500 bg-primary-50'
              : 'border-gray-300 hover:border-primary-400 hover:bg-gray-50'
          )}
        >
          <input {...getInputProps()} />
          <div className="space-y-2">
            <svg
              className="mx-auto h-12 w-12 text-gray-400"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1.5}
                d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
              />
            </svg>
            <p className="text-gray-600">
              {isDragActive ? 'Drop your image here' : 'Drag & drop your dental scan, or click to select'}
            </p>
            <p className="text-sm text-gray-500">JPEG, PNG, or WebP up to 10MB</p>
          </div>
        </div>
      ) : (
        <div className="border rounded-xl p-4">
          <div className="flex items-start gap-4">
            {preview && (
              <img
                src={preview}
                alt="Scan preview"
                className="w-32 h-32 object-cover rounded-lg"
              />
            )}
            <div className="flex-1">
              <p className="font-medium text-gray-900">{selectedFile.name}</p>
              <p className="text-sm text-gray-500">
                {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
              </p>
              <p className="text-sm text-gray-500 mt-1">
                Type: {scanType === 'basic_rgb' ? 'Basic RGB' : 'Advanced Spectral'}
              </p>
            </div>
            <button
              onClick={clearSelection}
              className="text-gray-400 hover:text-gray-600"
              aria-label="Remove file"
            >
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
          <div className="mt-4 flex gap-2">
            <Button onClick={handleUpload} isLoading={isUploading} disabled={isUploading}>
              Upload & Analyze
            </Button>
            <Button variant="secondary" onClick={clearSelection} disabled={isUploading}>
              Cancel
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}
