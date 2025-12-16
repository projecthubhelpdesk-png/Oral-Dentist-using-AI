"""
Data Preparation Script for Oral Disease Detection
===================================================
Organizes the Kaggle dataset into a clean structure for training.
"""

import os
import shutil
from pathlib import Path

# Source: Kaggle cache location
KAGGLE_CACHE = Path(r"C:\Users\ASUS\.cache\kagglehub\datasets\salmansajid05\oral-diseases\versions\3")

# Destination: Local data folder
DATA_DIR = Path(__file__).parent / "data"

# Class mappings (source folders -> clean class name)
# Each class can have multiple source folders
CLASS_MAPPINGS = {
    "Calculus": [
        "Calculus/Calculus",
    ],
    "Gingivitis": [
        "Gingivitis/Gingivitis",
    ],
    "Mouth_Ulcer": [
        "Mouth Ulcer/Mouth Ulcer/Mouth_Ulcer_augmented_DataSet/preview",
        "Mouth Ulcer/Mouth Ulcer/ulcer original dataset/ulcer original dataset",
    ],
    "Caries": [
        "Data caries/Data caries/caries augmented data set/preview",
        "Data caries/Data caries/caries orignal data set/done",
    ],
}

def prepare_dataset():
    """Copy and organize dataset from Kaggle cache."""
    print("="*60)
    print("PREPARING ORAL DISEASE DATASET")
    print("="*60)
    
    # Create data directory
    DATA_DIR.mkdir(exist_ok=True)
    
    total_images = 0
    
    for class_name, src_paths in CLASS_MAPPINGS.items():
        dst_full = DATA_DIR / class_name
        dst_full.mkdir(exist_ok=True)
        
        class_images = []
        
        for src_path in src_paths:
            src_full = KAGGLE_CACHE / src_path
            
            if not src_full.exists():
                print(f"⚠️  Source not found: {src_full}")
                continue
            
            # Collect images
            images = list(src_full.glob("*.jpg")) + list(src_full.glob("*.jpeg")) + list(src_full.glob("*.png"))
            images += list(src_full.glob("*.JPG")) + list(src_full.glob("*.JPEG")) + list(src_full.glob("*.PNG"))
            class_images.extend(images)
        
        print(f"\n📁 {class_name}: {len(class_images)} images")
        
        for i, img in enumerate(class_images):
            dst_file = dst_full / f"{class_name}_{i:04d}{img.suffix.lower()}"
            if not dst_file.exists():
                shutil.copy2(img, dst_file)
        
        total_images += len(class_images)
    
    print(f"\n{'='*60}")
    print(f"✅ Dataset prepared: {total_images} total images")
    print(f"📂 Location: {DATA_DIR}")
    print(f"{'='*60}")
    
    # List final structure
    print("\nFinal dataset structure:")
    for class_dir in sorted(DATA_DIR.iterdir()):
        if class_dir.is_dir():
            count = len(list(class_dir.glob("*")))
            print(f"  - {class_dir.name}: {count} images")

if __name__ == "__main__":
    prepare_dataset()
