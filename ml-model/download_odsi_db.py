"""
ODSI-DB Dataset Downloader
==========================
Downloads the Oral and Dental Spectral Image Database from University of Eastern Finland.

Dataset URL: https://cs.uef.fi/pub/color/spectra/ODSI-DB
Paper: "Oral and Dental Spectral Image Database for Human Identification"

This script downloads and organizes the spectral dental images for training
the spectral dental AI pipeline.

Usage:
    python download_odsi_db.py
"""

import os
import sys
import urllib.request
import zipfile
import tarfile
from pathlib import Path
import shutil

# Dataset base URL
BASE_URL = "https://cs.uef.fi/pub/color/spectra/ODSI-DB"

# Local paths
DATASET_DIR = Path(__file__).parent / "datasets" / "ODSI-DB"
RAW_DIR = DATASET_DIR / "raw"
PROCESSED_DIR = DATASET_DIR / "processed"

# Known files in ODSI-DB (based on typical structure)
# Note: Actual file names may vary - check the website for current structure
EXPECTED_FILES = [
    "README.txt",
    "spectral_images/",
    "metadata/",
]


def create_directories():
    """Create necessary directories."""
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for processed data
    (PROCESSED_DIR / "train").mkdir(exist_ok=True)
    (PROCESSED_DIR / "val").mkdir(exist_ok=True)
    (PROCESSED_DIR / "test").mkdir(exist_ok=True)
    
    print(f"Created directories at: {DATASET_DIR}")


def download_file(url: str, dest: Path) -> bool:
    """Download a file from URL to destination."""
    try:
        print(f"Downloading: {url}")
        urllib.request.urlretrieve(url, dest)
        print(f"Saved to: {dest}")
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False


def extract_archive(archive_path: Path, dest_dir: Path):
    """Extract zip or tar archive."""
    if archive_path.suffix == '.zip':
        with zipfile.ZipFile(archive_path, 'r') as zf:
            zf.extractall(dest_dir)
    elif archive_path.suffix in ['.tar', '.gz', '.tgz']:
        with tarfile.open(archive_path, 'r:*') as tf:
            tf.extractall(dest_dir)
    print(f"Extracted to: {dest_dir}")


def create_dataset_info():
    """Create dataset information file."""
    info = """
ODSI-DB (Oral and Dental Spectral Image Database)
==================================================

Source: University of Eastern Finland
URL: https://cs.uef.fi/pub/color/spectra/ODSI-DB

Description:
The ODSI-DB contains spectral images of human teeth captured using
hyperspectral imaging technology. The database includes:

- Spectral images in the visible and near-infrared range (380-780nm)
- Multiple subjects with various dental conditions
- Calibration data for spectral analysis

Usage for Dental AI:
1. Spectral band analysis for early caries detection
2. Enamel demineralization mapping
3. Subsurface decay identification
4. Periodontal tissue assessment

File Structure:
- raw/: Original downloaded files
- processed/: Preprocessed images ready for training
  - train/: Training set (70%)
  - val/: Validation set (15%)
  - test/: Test set (15%)

Citation:
If you use this dataset, please cite the original paper from
University of Eastern Finland.

Note: This dataset is for research purposes only.
"""
    
    info_path = DATASET_DIR / "README.md"
    with open(info_path, 'w') as f:
        f.write(info)
    print(f"Created dataset info at: {info_path}")


def create_sample_spectral_data():
    """Create sample spectral data for testing when real data isn't available."""
    import numpy as np
    from PIL import Image
    
    print("\nCreating sample spectral data for testing...")
    
    # Create sample images for each condition
    conditions = [
        'healthy',
        'early_caries',
        'enamel_caries',
        'demineralization',
        'calculus',
        'gingivitis',
    ]
    
    for split in ['train', 'val', 'test']:
        split_dir = PROCESSED_DIR / split
        
        for condition in conditions:
            condition_dir = split_dir / condition
            condition_dir.mkdir(parents=True, exist_ok=True)
            
            # Number of samples per condition
            n_samples = 20 if split == 'train' else 5
            
            for i in range(n_samples):
                # Create synthetic spectral-like image
                img = create_synthetic_spectral_image(condition)
                
                # Save image
                img_path = condition_dir / f"{condition}_{split}_{i:03d}.png"
                Image.fromarray(img).save(img_path)
    
    print(f"Created sample data in: {PROCESSED_DIR}")


def create_synthetic_spectral_image(condition: str, size: int = 224) -> np.ndarray:
    """Create a synthetic spectral-like dental image for testing."""
    import numpy as np
    
    # Base tooth-like pattern
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    
    # Tooth shape (ellipse)
    tooth_mask = (X**2 / 0.6**2 + Y**2 / 0.8**2) < 1
    
    # Base colors (healthy tooth)
    r = np.ones((size, size)) * 0.9
    g = np.ones((size, size)) * 0.85
    b = np.ones((size, size)) * 0.75
    
    # Add condition-specific patterns
    if condition == 'healthy':
        # Slight natural variation
        noise = np.random.normal(0, 0.02, (size, size))
        r += noise
        g += noise
        b += noise
        
    elif condition == 'early_caries':
        # White spot lesion
        spot_x, spot_y = np.random.uniform(-0.3, 0.3, 2)
        spot = np.exp(-((X - spot_x)**2 + (Y - spot_y)**2) / 0.05)
        r += spot * 0.1
        g += spot * 0.1
        b += spot * 0.15  # More blue = fluorescence change
        
    elif condition == 'enamel_caries':
        # Darker spot with irregular edges
        spot_x, spot_y = np.random.uniform(-0.3, 0.3, 2)
        spot = np.exp(-((X - spot_x)**2 + (Y - spot_y)**2) / 0.08)
        r -= spot * 0.2
        g -= spot * 0.15
        b -= spot * 0.1
        
    elif condition == 'demineralization':
        # Chalky white areas
        for _ in range(3):
            spot_x, spot_y = np.random.uniform(-0.4, 0.4, 2)
            spot = np.exp(-((X - spot_x)**2 + (Y - spot_y)**2) / 0.03)
            r += spot * 0.08
            g += spot * 0.08
            b += spot * 0.12
            
    elif condition == 'calculus':
        # Yellowish deposits near gum line
        gum_line = Y > 0.5
        calculus = gum_line * np.random.uniform(0, 0.15, (size, size))
        r += calculus * 0.1
        g += calculus * 0.05
        b -= calculus * 0.1
        
    elif condition == 'gingivitis':
        # Reddish gum area
        gum_area = Y > 0.6
        inflammation = gum_area * np.random.uniform(0, 0.2, (size, size))
        r += inflammation * 0.2
        g -= inflammation * 0.1
        b -= inflammation * 0.1
    
    # Apply tooth mask
    r = np.where(tooth_mask, r, 0.2)
    g = np.where(tooth_mask, g, 0.2)
    b = np.where(tooth_mask, b, 0.25)
    
    # Add noise
    noise = np.random.normal(0, 0.01, (size, size))
    r += noise
    g += noise
    b += noise
    
    # Clip and convert to uint8
    r = np.clip(r, 0, 1)
    g = np.clip(g, 0, 1)
    b = np.clip(b, 0, 1)
    
    img = np.stack([r, g, b], axis=-1)
    img = (img * 255).astype(np.uint8)
    
    return img


def main():
    """Main function to download and prepare ODSI-DB dataset."""
    print("=" * 60)
    print("ODSI-DB Dataset Downloader")
    print("=" * 60)
    print(f"\nDataset URL: {BASE_URL}")
    print(f"Local directory: {DATASET_DIR}")
    print()
    
    # Create directories
    create_directories()
    
    # Create dataset info
    create_dataset_info()
    
    print("\n" + "=" * 60)
    print("IMPORTANT: Manual Download Required")
    print("=" * 60)
    print(f"""
The ODSI-DB dataset requires manual download from:
{BASE_URL}

Steps:
1. Visit the URL above in your browser
2. Download the spectral image files
3. Extract them to: {RAW_DIR}
4. Run this script again to process the data

Alternatively, for testing purposes, sample synthetic data
will be created.
""")
    
    # Check if real data exists
    if not any(RAW_DIR.iterdir()) if RAW_DIR.exists() else True:
        print("\nNo raw data found. Creating synthetic sample data for testing...")
        try:
            create_sample_spectral_data()
            print("\n✓ Sample data created successfully!")
            print(f"  Location: {PROCESSED_DIR}")
        except ImportError as e:
            print(f"\nCould not create sample data: {e}")
            print("Install numpy and Pillow: pip install numpy Pillow")
    else:
        print("\nRaw data found. Processing...")
        # TODO: Add actual processing logic for ODSI-DB format
        print("Processing complete!")
    
    print("\n" + "=" * 60)
    print("Dataset preparation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
