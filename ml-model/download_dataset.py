"""
Download and prepare the Dental Anatomy Dataset for YOLOv8 training.

Dataset: https://www.kaggle.com/datasets/saisiddartha69/dental-anatomy-dataset-yolov8

Instructions:
1. Download the dataset manually from Kaggle (requires account)
2. Extract to: ml-model/datasets/dental-anatomy/
3. Run this script to prepare the data

Or use Kaggle API:
    pip install kaggle
    kaggle datasets download -d saisiddartha69/dental-anatomy-dataset-yolov8
"""

import os
import shutil
from pathlib import Path
import json

# Paths
SCRIPT_DIR = Path(__file__).parent
DATASETS_DIR = SCRIPT_DIR / 'datasets'
DENTAL_ANATOMY_DIR = DATASETS_DIR / 'dental-anatomy'
OUTPUT_DIR = DATASETS_DIR / 'dental-detection-trained'

def check_dataset():
    """Check if dataset exists."""
    if not DENTAL_ANATOMY_DIR.exists():
        print("❌ Dataset not found!")
        print(f"\nPlease download from Kaggle and extract to:")
        print(f"   {DENTAL_ANATOMY_DIR}")
        print("\nSteps:")
        print("1. Go to: https://www.kaggle.com/datasets/saisiddartha69/dental-anatomy-dataset-yolov8")
        print("2. Click 'Download' (requires Kaggle account)")
        print("3. Extract the ZIP file to the path above")
        print("4. Run this script again")
        return False
    
    # Check for expected structure
    train_dir = DENTAL_ANATOMY_DIR / 'train' / 'images'
    if not train_dir.exists():
        # Try alternate structure
        train_dir = DENTAL_ANATOMY_DIR / 'Train' / 'images'
    
    if not train_dir.exists():
        print("❌ Dataset structure not recognized!")
        print(f"Expected: {DENTAL_ANATOMY_DIR}/train/images/ or {DENTAL_ANATOMY_DIR}/Train/images/")
        print("\nContents found:")
        for item in DENTAL_ANATOMY_DIR.iterdir():
            print(f"  - {item.name}")
        return False
    
    print(f"✅ Dataset found at: {DENTAL_ANATOMY_DIR}")
    return True

def count_images(directory):
    """Count images in directory."""
    count = 0
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        count += len(list(directory.glob(ext)))
    return count

def prepare_dataset():
    """Prepare dataset for training."""
    print("\n📊 Analyzing dataset structure...")
    
    # Find the actual structure
    possible_structures = [
        ('train', 'valid', 'test'),
        ('Train', 'Valid', 'Test'),
        ('train', 'val', 'test'),
    ]
    
    train_dir = val_dir = test_dir = None
    
    for train_name, val_name, test_name in possible_structures:
        t = DENTAL_ANATOMY_DIR / train_name
        v = DENTAL_ANATOMY_DIR / val_name
        if t.exists():
            train_dir = t
            val_dir = v if v.exists() else None
            test_dir = DENTAL_ANATOMY_DIR / test_name if (DENTAL_ANATOMY_DIR / test_name).exists() else None
            break
    
    if not train_dir:
        print("❌ Could not find training directory!")
        return False
    
    # Count images
    train_images = train_dir / 'images' if (train_dir / 'images').exists() else train_dir
    val_images = (val_dir / 'images' if (val_dir / 'images').exists() else val_dir) if val_dir else None
    
    print(f"\n📁 Dataset structure:")
    print(f"   Train: {count_images(train_images)} images")
    if val_images:
        print(f"   Val: {count_images(val_images)} images")
    if test_dir:
        test_images = test_dir / 'images' if (test_dir / 'images').exists() else test_dir
        print(f"   Test: {count_images(test_images)} images")
    
    # Check for data.yaml
    yaml_path = DENTAL_ANATOMY_DIR / 'data.yaml'
    if yaml_path.exists():
        print(f"\n✅ data.yaml found!")
        with open(yaml_path, 'r') as f:
            print(f.read())
    else:
        print("\n⚠️ No data.yaml found - will need to create one")
    
    return True

def create_training_config():
    """Create training configuration."""
    config = {
        'dataset_path': str(DENTAL_ANATOMY_DIR),
        'model': 'yolov8n.pt',  # Start with nano model
        'epochs': 100,
        'imgsz': 640,
        'batch': 16,
        'patience': 20,
    }
    
    config_path = SCRIPT_DIR / 'training_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✅ Training config saved to: {config_path}")
    return config

if __name__ == '__main__':
    print("🦷 Dental Dataset Preparation Tool")
    print("=" * 50)
    
    if check_dataset():
        if prepare_dataset():
            create_training_config()
            print("\n" + "=" * 50)
            print("✅ Dataset ready for training!")
            print("\nNext step - run training:")
            print("   py -3.11 train_yolo.py")
    else:
        print("\n" + "=" * 50)
        print("📥 Please download the dataset first!")
