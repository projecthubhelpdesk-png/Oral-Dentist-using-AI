"""
Prepare Classification Dataset from Kaggle Oral Diseases
=========================================================

Organizes images into train/val folders for EfficientNet training.
"""

import os
import shutil
from pathlib import Path
import random

# Source paths
KAGGLE_CACHE = Path(r"C:\Users\ASUS\.cache\kagglehub\datasets\salmansajid05\oral-diseases\versions\3")
OUTPUT_DIR = Path(__file__).parent / "datasets" / "oral-diseases-classification"

# Class mapping - multiple source folders per class
CLASS_FOLDERS = {
    'Calculus': [
        KAGGLE_CACHE / 'Calculus' / 'Calculus',
    ],
    'Caries': [
        KAGGLE_CACHE / 'Data caries' / 'Data caries' / 'caries augmented data set',
        KAGGLE_CACHE / 'Data caries' / 'Data caries' / 'caries orignal data set',
    ],
    'Gingivitis': [
        KAGGLE_CACHE / 'Gingivitis' / 'Gingivitis',
    ],
    'Mouth_Ulcer': [
        KAGGLE_CACHE / 'Mouth Ulcer' / 'Mouth Ulcer',
    ],
}

TRAIN_SPLIT = 0.8  # 80% train, 20% val

def prepare_dataset():
    """Prepare classification dataset."""
    print("🦷 Preparing Oral Disease Classification Dataset")
    print("=" * 50)
    
    # Create output directories
    train_dir = OUTPUT_DIR / 'train'
    val_dir = OUTPUT_DIR / 'val'
    
    for class_name in CLASS_FOLDERS.keys():
        (train_dir / class_name).mkdir(parents=True, exist_ok=True)
        (val_dir / class_name).mkdir(parents=True, exist_ok=True)
    
    total_train = 0
    total_val = 0
    
    for class_name, source_dirs in CLASS_FOLDERS.items():
        print(f"\n📁 Processing {class_name}...")
        
        # Collect images from all source directories
        images = []
        for source_dir in source_dirs:
            if not source_dir.exists():
                print(f"   ⚠️ Source not found: {source_dir}")
                continue
            
            # Get all images from this source
            images.extend(list(source_dir.glob('*.jpg')))
            images.extend(list(source_dir.glob('*.png')))
            images.extend(list(source_dir.glob('*.jpeg')))
            # Also check subdirectories
            images.extend(list(source_dir.rglob('*.jpg')))
            images.extend(list(source_dir.rglob('*.png')))
            images.extend(list(source_dir.rglob('*.jpeg')))
        
        # Remove duplicates
        images = list(set(images))
        
        if not images:
            print(f"   ⚠️ No images found")
            continue
        
        # Shuffle and split
        random.shuffle(images)
        split_idx = int(len(images) * TRAIN_SPLIT)
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        # Copy to train
        for img in train_images:
            dest = train_dir / class_name / img.name
            if not dest.exists():
                shutil.copy2(img, dest)
        
        # Copy to val
        for img in val_images:
            dest = val_dir / class_name / img.name
            if not dest.exists():
                shutil.copy2(img, dest)
        
        print(f"   ✅ Train: {len(train_images)}, Val: {len(val_images)}")
        total_train += len(train_images)
        total_val += len(val_images)
    
    print(f"\n{'=' * 50}")
    print(f"✅ Dataset prepared!")
    print(f"   Total train: {total_train}")
    print(f"   Total val: {total_val}")
    print(f"   Output: {OUTPUT_DIR}")
    
    # Create class labels JSON
    import json
    labels = {
        "class_labels": {str(i): name for i, name in enumerate(CLASS_FOLDERS.keys())},
        "num_classes": len(CLASS_FOLDERS),
        "image_size": [224, 224],
        "model_version": "2.0.0",
        "dataset": "oral-diseases-kaggle"
    }
    
    with open(OUTPUT_DIR / 'class_labels.json', 'w') as f:
        json.dump(labels, f, indent=2)
    
    print(f"\n📋 Class labels saved to: {OUTPUT_DIR / 'class_labels.json'}")
    
    return OUTPUT_DIR

if __name__ == '__main__':
    prepare_dataset()
