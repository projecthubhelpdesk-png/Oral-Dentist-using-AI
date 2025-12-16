#!/usr/bin/env python3
"""
YOLOv8 Tooth Detection Training Script
======================================
Train YOLOv8 model for dental tooth detection and issue localization.

Usage:
    # Train from scratch with custom dataset:
    py -3.11 train_yolo.py --data datasets/teeth_detection/data.yaml --epochs 100
    
    # Fine-tune from pretrained:
    py -3.11 train_yolo.py --data datasets/teeth_detection/data.yaml --weights yolov8n.pt --epochs 50
    
    # Quick test with sample:
    py -3.11 train_yolo.py --test
"""

import os
import argparse
from pathlib import Path

def check_ultralytics():
    """Check if ultralytics is installed."""
    try:
        from ultralytics import YOLO
        print("✅ Ultralytics YOLOv8 is installed!")
        return True
    except ImportError:
        print("❌ Ultralytics not installed.")
        print("\nTo install, run:")
        print("  py -3.11 -m pip install ultralytics")
        return False

def create_sample_dataset():
    """Create a sample dataset structure for reference."""
    base_path = Path("datasets/teeth_detection")
    
    # Create directories
    (base_path / "images/train").mkdir(parents=True, exist_ok=True)
    (base_path / "images/val").mkdir(parents=True, exist_ok=True)
    (base_path / "labels/train").mkdir(parents=True, exist_ok=True)
    (base_path / "labels/val").mkdir(parents=True, exist_ok=True)
    
    # Create data.yaml
    data_yaml = """# Teeth Detection Dataset
# For YOLOv8 training

path: ./datasets/teeth_detection
train: images/train
val: images/val

# Classes
names:
  0: healthy_tooth
  1: cavity
  2: plaque
  3: crooked_tooth
  4: missing_tooth

# Number of classes
nc: 5
"""
    
    with open(base_path / "data.yaml", "w") as f:
        f.write(data_yaml)
    
    # Create sample annotation format reference
    annotation_guide = """# YOLO Annotation Format
# Each line: class_id x_center y_center width height
# All values normalized (0-1) relative to image dimensions

# Example for a 640x480 image with a cavity at pixel (200, 150) with size 50x40:
# class_id = 1 (cavity)
# x_center = 200/640 = 0.3125
# y_center = 150/480 = 0.3125  
# width = 50/640 = 0.078125
# height = 40/480 = 0.083333

# Sample annotation (img001.txt):
1 0.3125 0.3125 0.078125 0.083333
0 0.6 0.4 0.1 0.12
2 0.2 0.7 0.08 0.1
"""
    
    with open(base_path / "ANNOTATION_GUIDE.txt", "w") as f:
        f.write(annotation_guide)
    
    print(f"✅ Sample dataset structure created at: {base_path}")
    print("\nNext steps:")
    print("1. Add your dental images to images/train/ and images/val/")
    print("2. Create corresponding .txt annotation files in labels/")
    print("3. Run training with: py -3.11 train_yolo.py --data datasets/teeth_detection/data.yaml")
    
    return base_path

def train_model(data_path: str, weights: str = "yolov8n.pt", epochs: int = 100, 
                imgsz: int = 640, batch: int = 16, device: str = "0"):
    """Train YOLOv8 model."""
    from ultralytics import YOLO
    
    print(f"\n🦷 Starting YOLOv8 Training for Tooth Detection")
    print("=" * 50)
    print(f"Dataset: {data_path}")
    print(f"Base weights: {weights}")
    print(f"Epochs: {epochs}")
    print(f"Image size: {imgsz}")
    print(f"Batch size: {batch}")
    print("=" * 50)
    
    # Load model
    model = YOLO(weights)
    
    # Train
    results = model.train(
        data=data_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project="runs/teeth_detection",
        name="train",
        exist_ok=True,
        pretrained=True,
        optimizer="Adam",
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        patience=50,  # Early stopping patience
        save=True,
        save_period=10,
        val=True,
        plots=True,
        verbose=True,
    )
    
    print("\n✅ Training completed!")
    print(f"Best model saved to: runs/teeth_detection/train/weights/best.pt")
    
    # Copy best model to models directory
    import shutil
    best_model = Path("runs/teeth_detection/train/weights/best.pt")
    target_path = Path("models/yolo_tooth_detector.pt")
    
    if best_model.exists():
        shutil.copy(best_model, target_path)
        print(f"✅ Model copied to: {target_path}")
    
    return results

def export_model(model_path: str = "models/yolo_tooth_detector.pt"):
    """Export model to ONNX format for deployment."""
    from ultralytics import YOLO
    
    print(f"\n📦 Exporting model to ONNX...")
    
    model = YOLO(model_path)
    
    # Export to ONNX
    model.export(format="onnx", imgsz=640, simplify=True)
    
    print("✅ Model exported to ONNX format!")

def test_detection(image_path: str = None, model_path: str = "models/yolo_tooth_detector.pt"):
    """Test detection on an image."""
    from ultralytics import YOLO
    import cv2
    
    # Use sample image if not provided
    if image_path is None:
        # Try to find a sample image
        sample_paths = [
            "../backend-php/storage/scans/sample1.jpg",
            "test_image.jpg",
        ]
        for p in sample_paths:
            if Path(p).exists():
                image_path = p
                break
    
    if image_path is None or not Path(image_path).exists():
        print("❌ No test image found. Please provide an image path.")
        return
    
    print(f"\n🔍 Testing detection on: {image_path}")
    
    # Check if trained model exists
    if not Path(model_path).exists():
        print(f"⚠️ Trained model not found at {model_path}")
        print("Using base YOLOv8n model for demonstration...")
        model_path = "yolov8n.pt"
    
    model = YOLO(model_path)
    
    # Run detection
    results = model.predict(
        source=image_path,
        conf=0.25,
        save=True,
        project="runs/detect",
        name="test",
        exist_ok=True,
    )
    
    print("\n📊 Detection Results:")
    for r in results:
        if r.boxes is not None and len(r.boxes) > 0:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model.names[cls]
                print(f"  - {class_name}: {conf:.2%} confidence")
        else:
            print("  No detections found")
    
    print(f"\n✅ Results saved to: runs/detect/test/")

def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Tooth Detection Training")
    parser.add_argument("--data", type=str, help="Path to data.yaml")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="Initial weights")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="0", help="Device (0 for GPU, cpu for CPU)")
    parser.add_argument("--setup", action="store_true", help="Create sample dataset structure")
    parser.add_argument("--test", action="store_true", help="Test detection")
    parser.add_argument("--export", action="store_true", help="Export to ONNX")
    parser.add_argument("--check", action="store_true", help="Check installation")
    parser.add_argument("--image", type=str, help="Image path for testing")
    
    args = parser.parse_args()
    
    # Check installation
    if args.check or not check_ultralytics():
        if not check_ultralytics():
            return
    
    # Setup sample dataset
    if args.setup:
        create_sample_dataset()
        return
    
    # Test detection
    if args.test:
        if check_ultralytics():
            test_detection(args.image)
        return
    
    # Export model
    if args.export:
        if check_ultralytics():
            export_model()
        return
    
    # Train model
    if args.data:
        if check_ultralytics():
            train_model(
                data_path=args.data,
                weights=args.weights,
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device,
            )
    else:
        parser.print_help()
        print("\n📋 Quick Start:")
        print("  1. Check installation:  py -3.11 train_yolo.py --check")
        print("  2. Create dataset:      py -3.11 train_yolo.py --setup")
        print("  3. Train model:         py -3.11 train_yolo.py --data datasets/teeth_detection/data.yaml")
        print("  4. Test detection:      py -3.11 train_yolo.py --test")
        print("  5. Export to ONNX:      py -3.11 train_yolo.py --export")

if __name__ == "__main__":
    main()
