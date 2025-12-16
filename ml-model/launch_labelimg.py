#!/usr/bin/env python3
"""
Launch LabelImg for Dental Image Annotation
============================================
Opens LabelImg with predefined dental classes for YOLO annotation.

Usage:
    py -3.11 launch_labelimg.py
    py -3.11 launch_labelimg.py --images path/to/images
"""

import os
import sys
import subprocess
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
DATASET_DIR = SCRIPT_DIR / "datasets" / "teeth_detection"
IMAGES_DIR = DATASET_DIR / "images" / "train"
CLASSES_FILE = DATASET_DIR / "predefined_classes.txt"

def setup_directories():
    """Ensure all directories exist."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    (DATASET_DIR / "images" / "val").mkdir(parents=True, exist_ok=True)
    (DATASET_DIR / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (DATASET_DIR / "labels" / "val").mkdir(parents=True, exist_ok=True)
    
    print(f"✅ Dataset directories ready at: {DATASET_DIR}")

def print_instructions():
    """Print annotation instructions."""
    print("""
🦷 DENTAL IMAGE ANNOTATION GUIDE
================================

CLASSES TO ANNOTATE:
  0: healthy_tooth  - Normal, healthy teeth
  1: cavity         - Visible decay, dark spots, holes
  2: plaque         - Yellowish/white buildup on teeth
  3: crooked_tooth  - Misaligned or crooked teeth
  4: missing_tooth  - Gap where tooth should be

LABELIMG SHORTCUTS:
  W     - Create a rectangle box
  D     - Next image
  A     - Previous image
  Del   - Delete selected box
  Ctrl+S - Save annotation
  Ctrl+U - Load images from directory

WORKFLOW:
  1. Click "Open Dir" and select: datasets/teeth_detection/images/train
  2. Click "Change Save Dir" and select: datasets/teeth_detection/labels/train
  3. Make sure "YOLO" format is selected (bottom left)
  4. Draw boxes around teeth/issues and select the class
  5. Press Ctrl+S to save, then D for next image

TIPS:
  - Draw tight boxes around individual teeth or issues
  - Label ALL visible teeth (healthy or not)
  - Be consistent with box sizes
  - Aim for 100+ annotated images for good results
  - Split 80% train, 20% validation

After annotation, run:
  py -3.11 train_yolo.py --data datasets/teeth_detection/data.yaml --epochs 100
""")

def launch_labelimg(images_path=None):
    """Launch LabelImg with dental classes."""
    setup_directories()
    print_instructions()
    
    # Determine images path
    if images_path:
        img_dir = Path(images_path)
    else:
        img_dir = IMAGES_DIR
    
    print(f"\n🚀 Launching LabelImg...")
    print(f"   Images: {img_dir}")
    print(f"   Classes: {CLASSES_FILE}")
    
    # Try different ways to launch labelImg
    python_exe = sys.executable
    
    # Method 1: Try labelImg module
    try:
        cmd = [python_exe, "-m", "labelImg", str(img_dir), str(CLASSES_FILE)]
        print(f"\nRunning: {' '.join(cmd)}")
        subprocess.run(cmd)
        return
    except Exception as e:
        print(f"Method 1 failed: {e}")
    
    # Method 2: Try labelImg directly
    try:
        scripts_dir = Path(python_exe).parent / "Scripts"
        labelimg_exe = scripts_dir / "labelImg.exe"
        if labelimg_exe.exists():
            cmd = [str(labelimg_exe), str(img_dir), str(CLASSES_FILE)]
            print(f"\nRunning: {' '.join(cmd)}")
            subprocess.run(cmd)
            return
    except Exception as e:
        print(f"Method 2 failed: {e}")
    
    # Method 3: Just try labelImg command
    try:
        cmd = ["labelImg", str(img_dir), str(CLASSES_FILE)]
        print(f"\nRunning: {' '.join(cmd)}")
        subprocess.run(cmd, shell=True)
        return
    except Exception as e:
        print(f"Method 3 failed: {e}")
    
    print("\n❌ Could not launch LabelImg automatically.")
    print("\nManual launch instructions:")
    print(f"  1. Open command prompt")
    print(f"  2. Run: labelImg \"{img_dir}\" \"{CLASSES_FILE}\"")
    print(f"\nOr run: py -3.11 -m labelImg")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Launch LabelImg for dental annotation")
    parser.add_argument("--images", "-i", help="Path to images directory")
    args = parser.parse_args()
    
    launch_labelimg(args.images)
