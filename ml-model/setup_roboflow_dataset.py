"""
Setup Roboflow Dental Dataset for Training
==========================================

This script downloads a dental dataset from Roboflow and prepares it for YOLOv8 training.

Usage:
    py -3.11 setup_roboflow_dataset.py

Requirements:
    pip install roboflow
"""

import os
from pathlib import Path

def setup_roboflow():
    """Download dataset from Roboflow."""
    try:
        from roboflow import Roboflow
    except ImportError:
        print("❌ Roboflow not installed. Installing...")
        os.system("py -3.11 -m pip install roboflow")
        from roboflow import Roboflow
    
    print("🦷 Roboflow Dental Dataset Setup")
    print("=" * 50)
    
    # You'll need a Roboflow API key
    # Get one free at: https://app.roboflow.com/
    api_key = input("Enter your Roboflow API key (get free at roboflow.com): ").strip()
    
    if not api_key:
        print("\n❌ API key required!")
        print("1. Go to https://app.roboflow.com/")
        print("2. Create free account")
        print("3. Go to Settings > API Key")
        print("4. Copy your API key")
        return False
    
    rf = Roboflow(api_key=api_key)
    
    # List of good dental datasets on Roboflow
    datasets = [
        ("teeth-detection-yolov8", "teeth-detection-yolov8", 1),  # Example
        ("dental-disease", "dental-disease-detection", 1),
    ]
    
    print("\n📥 Downloading dental dataset...")
    
    # Try to download a dental dataset
    # You can find datasets at: https://universe.roboflow.com/
    try:
        # This is an example - you'll need to find the actual workspace/project
        project = rf.workspace().project("teeth-detection")
        dataset = project.version(1).download("yolov8")
        
        print(f"\n✅ Dataset downloaded to: {dataset.location}")
        return dataset.location
        
    except Exception as e:
        print(f"\n⚠️ Could not download: {e}")
        print("\nManual steps:")
        print("1. Go to https://universe.roboflow.com/")
        print("2. Search for 'dental' or 'teeth'")
        print("3. Find a dataset with YOLOv8 format")
        print("4. Click 'Download' > 'YOLOv8' format")
        print("5. Extract to: ml-model/datasets/dental-roboflow/")
        return None

def download_kaggle_dataset():
    """Download from Kaggle (alternative)."""
    print("\n📥 Kaggle Dataset Download")
    print("=" * 50)
    
    try:
        import kaggle
        print("✅ Kaggle API available")
        
        # Download the dental anatomy dataset
        print("\nDownloading: saisiddartha69/dental-anatomy-dataset-yolov8")
        kaggle.api.dataset_download_files(
            'saisiddartha69/dental-anatomy-dataset-yolov8',
            path='datasets/dental-anatomy',
            unzip=True
        )
        print("✅ Dataset downloaded!")
        return 'datasets/dental-anatomy'
        
    except ImportError:
        print("❌ Kaggle API not installed")
        print("\nTo install:")
        print("  py -3.11 -m pip install kaggle")
        print("\nThen setup API key:")
        print("  1. Go to kaggle.com > Account > Create API Token")
        print("  2. Save kaggle.json to ~/.kaggle/")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def manual_download_instructions():
    """Show manual download instructions."""
    print("\n" + "=" * 60)
    print("📥 MANUAL DOWNLOAD INSTRUCTIONS")
    print("=" * 60)
    
    print("\n🔹 Option 1: Kaggle (Recommended)")
    print("-" * 40)
    print("1. Go to: https://www.kaggle.com/datasets/saisiddartha69/dental-anatomy-dataset-yolov8")
    print("2. Click 'Download' button")
    print("3. Extract ZIP to: ml-model/datasets/dental-anatomy/")
    print("4. Run: py -3.11 train_yolo.py --data datasets/dental-anatomy/data.yaml")
    
    print("\n🔹 Option 2: Roboflow Universe")
    print("-" * 40)
    print("1. Go to: https://universe.roboflow.com/")
    print("2. Search: 'dental disease' or 'teeth detection'")
    print("3. Good datasets:")
    print("   - https://universe.roboflow.com/search?q=dental")
    print("   - https://universe.roboflow.com/search?q=teeth%20cavity")
    print("4. Download in 'YOLOv8' format")
    print("5. Extract to: ml-model/datasets/")
    
    print("\n🔹 Option 3: Use Roboflow API")
    print("-" * 40)
    print("1. Create free account at roboflow.com")
    print("2. Get API key from Settings")
    print("3. Run this script again with API key")
    
    print("\n" + "=" * 60)

if __name__ == '__main__':
    print("🦷 Dental Dataset Setup Tool")
    print("=" * 50)
    
    print("\nChoose download method:")
    print("1. Roboflow (requires free API key)")
    print("2. Kaggle (requires Kaggle account)")
    print("3. Show manual download instructions")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == '1':
        setup_roboflow()
    elif choice == '2':
        download_kaggle_dataset()
    else:
        manual_download_instructions()
