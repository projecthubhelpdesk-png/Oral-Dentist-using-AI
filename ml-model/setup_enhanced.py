#!/usr/bin/env python3
"""
Enhanced Teeth Analyzer - Setup Script
======================================
Easy setup and installation script for the enhanced AI teeth analysis system.
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def print_header():
    """Print setup header."""
    print("🦷 Enhanced Teeth Analyzer - Setup")
    print("=" * 50)
    print("Setting up complete AI teeth analysis system...")
    print()

def check_python_version():
    """Check Python version compatibility."""
    print("🐍 Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} detected")
        print("💡 Python 3.8+ required. Please upgrade Python.")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
    return True

def install_requirements():
    """Install required packages."""
    print("\n📦 Installing requirements...")
    
    requirements_file = Path(__file__).parent / 'requirements_enhanced.txt'
    
    if not requirements_file.exists():
        print("❌ requirements_enhanced.txt not found")
        return False
    
    try:
        # Install requirements
        cmd = [sys.executable, '-m', 'pip', 'install', '-r', str(requirements_file)]
        print(f"Running: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Requirements installed successfully")
            return True
        else:
            print(f"❌ Installation failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Installation error: {e}")
        return False

def setup_directories():
    """Create necessary directories."""
    print("\n📁 Setting up directories...")
    
    base_dir = Path(__file__).parent
    directories = [
        'models',
        'config', 
        'data',
        'results',
        'notebooks'
    ]
    
    for dir_name in directories:
        dir_path = base_dir / dir_name
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created directory: {dir_name}")
        else:
            print(f"✅ Directory exists: {dir_name}")
    
    return True

def check_model_files():
    """Check for required model files."""
    print("\n🤖 Checking model files...")
    
    base_dir = Path(__file__).parent
    
    # Check EfficientNet model
    efficientnet_path = base_dir / 'models' / 'oral_disease_rgb_model.h5'
    if efficientnet_path.exists():
        print("✅ EfficientNet model found")
    else:
        print("⚠️  EfficientNet model not found")
        print("💡 Run 'python train.py' to train the model")
    
    # Check class labels
    labels_path = base_dir / 'config' / 'class_labels.json'
    if labels_path.exists():
        print("✅ Class labels found")
    else:
        print("⚠️  Class labels not found")
        print("💡 Labels will be created during training")
    
    # Check FDI mapping
    fdi_path = base_dir / 'config' / 'fdi_mapping.json'
    if fdi_path.exists():
        print("✅ FDI mapping found")
    else:
        print("⚠️  FDI mapping not found (should have been created)")
    
    return True

def test_installation():
    """Test the installation."""
    print("\n🧪 Testing installation...")
    
    try:
        # Test basic imports
        print("  Testing imports...")
        import tensorflow as tf
        print(f"  ✅ TensorFlow {tf.__version__}")
        
        import numpy as np
        print(f"  ✅ NumPy {np.__version__}")
        
        from PIL import Image
        print("  ✅ Pillow")
        
        import cv2
        print(f"  ✅ OpenCV {cv2.__version__}")
        
        try:
            import fastapi
            print(f"  ✅ FastAPI {fastapi.__version__}")
        except ImportError:
            print("  ❌ FastAPI not available")
        
        try:
            import ultralytics
            print("  ✅ YOLOv8 (Ultralytics)")
        except ImportError:
            print("  ⚠️  YOLOv8 not available (optional)")
        
        # Test enhanced analyzer import
        print("  Testing Enhanced Teeth Analyzer...")
        from enhanced_teeth_analyzer import EnhancedTeethAnalyzer
        print("  ✅ Enhanced Teeth Analyzer imported successfully")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Test failed: {e}")
        return False

def create_sample_config():
    """Create sample configuration files."""
    print("\n⚙️  Creating sample configurations...")
    
    base_dir = Path(__file__).parent
    
    # Create sample environment file
    env_sample = base_dir / '.env.sample'
    if not env_sample.exists():
        env_content = """# Enhanced Teeth Analyzer Configuration
# =====================================

# API Configuration
ENHANCED_ML_API_URL=http://localhost:8000
API_TIMEOUT=30

# Model Paths
EFFICIENTNET_MODEL_PATH=models/oral_disease_rgb_model.h5
YOLO_MODEL_PATH=models/yolo_tooth_detector.pt

# Processing Configuration
MAX_IMAGE_SIZE=10485760  # 10MB
SUPPORTED_FORMATS=jpg,jpeg,png,webp

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/enhanced_analyzer.log
"""
        with open(env_sample, 'w') as f:
            f.write(env_content)
        print("✅ Created .env.sample")
    
    return True

def print_next_steps():
    """Print next steps for the user."""
    print("\n🎯 Next Steps:")
    print("=" * 30)
    
    print("\n1. 📚 Train the EfficientNet model (if not done):")
    print("   python train.py")
    
    print("\n2. 🧪 Test the system:")
    print("   python test_enhanced_system.py")
    
    print("\n3. 🚀 Start the API server:")
    print("   python enhanced_api.py")
    print("   # Server will run on http://localhost:8000")
    print("   # API docs at http://localhost:8000/docs")
    
    print("\n4. 🔍 Analyze a dental image:")
    print("   python enhanced_teeth_analyzer.py your_image.jpg --summary")
    
    print("\n5. 🌐 Integration with existing backend:")
    print("   # Update your .env file:")
    print("   ENHANCED_ML_API_URL=http://localhost:8000")
    print("   # Use php_integration_example.php as reference")
    
    print("\n📖 Documentation:")
    print("   - README_Enhanced.md - Complete documentation")
    print("   - API docs: http://localhost:8000/docs (when server running)")
    
    print("\n🆘 Troubleshooting:")
    print("   - Run: python test_enhanced_system.py")
    print("   - Check logs for errors")
    print("   - Ensure all dependencies are installed")

def main():
    """Main setup function."""
    print_header()
    
    # Setup steps
    steps = [
        ("Python Version", check_python_version),
        ("Install Requirements", install_requirements),
        ("Setup Directories", setup_directories),
        ("Check Model Files", check_model_files),
        ("Create Configurations", create_sample_config),
        ("Test Installation", test_installation)
    ]
    
    success_count = 0
    
    for step_name, step_func in steps:
        print(f"\n{'='*20}")
        try:
            if step_func():
                success_count += 1
            else:
                print(f"⚠️  {step_name} completed with warnings")
        except Exception as e:
            print(f"❌ {step_name} failed: {e}")
    
    # Summary
    print(f"\n{'='*50}")
    print("📋 SETUP SUMMARY")
    print(f"{'='*50}")
    print(f"Completed: {success_count}/{len(steps)} steps")
    
    if success_count >= len(steps) - 1:  # Allow for some warnings
        print("🎉 Setup completed successfully!")
        print_next_steps()
        return 0
    else:
        print("⚠️  Setup completed with issues. Check the output above.")
        return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)