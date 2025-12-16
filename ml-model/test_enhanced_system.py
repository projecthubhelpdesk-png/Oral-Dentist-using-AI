#!/usr/bin/env python3
"""
Test Script for Enhanced Teeth Analyzer
=======================================
Comprehensive testing of the enhanced AI teeth analysis system.
"""

import os
import sys
import json
import time
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

try:
    from enhanced_teeth_analyzer import EnhancedTeethAnalyzer, create_analysis_summary
    print("✅ Enhanced Teeth Analyzer imported successfully")
except ImportError as e:
    print(f"❌ Failed to import Enhanced Teeth Analyzer: {e}")
    sys.exit(1)

def test_basic_functionality():
    """Test basic analyzer functionality."""
    print("\n🧪 Testing Basic Functionality...")
    
    try:
        # Initialize analyzer
        print("  Initializing analyzer...")
        analyzer = EnhancedTeethAnalyzer()
        print("  ✅ Analyzer initialized successfully")
        
        # Test with a sample image (create a dummy image if none exists)
        test_image_path = Path(__file__).parent / 'test_image.jpg'
        
        if not test_image_path.exists():
            print("  Creating test image...")
            from PIL import Image
            import numpy as np
            
            # Create a simple test image
            img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(test_image_path)
            print("  ✅ Test image created")
        
        # Run analysis
        print("  Running analysis...")
        start_time = time.time()
        result = analyzer.analyze_teeth(str(test_image_path))
        analysis_time = time.time() - start_time
        
        print(f"  ✅ Analysis completed in {analysis_time:.2f} seconds")
        
        # Verify result structure
        required_keys = [
            'analysis_id', 'timestamp', 'disease_classification',
            'tooth_detections', 'severity_analysis', 'dental_report'
        ]
        
        for key in required_keys:
            if key not in result:
                print(f"  ❌ Missing key in result: {key}")
                return False
            else:
                print(f"  ✅ Found key: {key}")
        
        # Print summary
        print("\n📊 Analysis Summary:")
        print(f"  Disease: {result['disease_classification']['primary_condition']}")
        print(f"  Confidence: {result['disease_classification']['confidence']:.3f}")
        print(f"  Severity: {result['severity_analysis']['severity']}")
        print(f"  Affected Teeth: {len(result['dental_report']['affected_teeth'])}")
        print(f"  Tooth Detections: {len(result['tooth_detections'])}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Basic functionality test failed: {e}")
        return False

def test_api_compatibility():
    """Test API server startup."""
    print("\n🌐 Testing API Compatibility...")
    
    try:
        # Try to import FastAPI components
        from enhanced_api import app
        print("  ✅ FastAPI app imported successfully")
        
        # Test health endpoint logic
        from enhanced_api import analyzer as api_analyzer
        if api_analyzer is None:
            print("  ⚠️  API analyzer not loaded (expected during import)")
        else:
            print("  ✅ API analyzer loaded")
        
        return True
        
    except ImportError as e:
        print(f"  ❌ API import failed: {e}")
        print("  💡 Install FastAPI: pip install fastapi uvicorn")
        return False
    except Exception as e:
        print(f"  ❌ API test failed: {e}")
        return False

def test_model_files():
    """Test model file availability."""
    print("\n📁 Testing Model Files...")
    
    models_dir = Path(__file__).parent / 'models'
    config_dir = Path(__file__).parent / 'config'
    
    # Check EfficientNet model
    efficientnet_path = models_dir / 'oral_disease_rgb_model.h5'
    if efficientnet_path.exists():
        print(f"  ✅ EfficientNet model found: {efficientnet_path}")
    else:
        print(f"  ⚠️  EfficientNet model not found: {efficientnet_path}")
        print("  💡 Run train.py to create the model")
    
    # Check class labels
    labels_path = config_dir / 'class_labels.json'
    if labels_path.exists():
        print(f"  ✅ Class labels found: {labels_path}")
        
        # Verify labels content
        try:
            with open(labels_path, 'r') as f:
                labels = json.load(f)
            print(f"  ✅ Labels loaded: {labels['num_classes']} classes")
        except Exception as e:
            print(f"  ❌ Labels file corrupted: {e}")
    else:
        print(f"  ❌ Class labels not found: {labels_path}")
    
    # Check FDI mapping
    fdi_path = config_dir / 'fdi_mapping.json'
    if fdi_path.exists():
        print(f"  ✅ FDI mapping found: {fdi_path}")
    else:
        print(f"  ⚠️  FDI mapping not found: {fdi_path}")
    
    # Check YOLO model (optional)
    yolo_path = models_dir / 'yolo_tooth_detector.pt'
    if yolo_path.exists():
        print(f"  ✅ YOLO model found: {yolo_path}")
    else:
        print(f"  ⚠️  YOLO model not found: {yolo_path} (will use mock detector)")
    
    return True

def test_dependencies():
    """Test required dependencies."""
    print("\n📦 Testing Dependencies...")
    
    dependencies = [
        ('tensorflow', 'TensorFlow'),
        ('PIL', 'Pillow'),
        ('numpy', 'NumPy'),
        ('cv2', 'OpenCV'),
        ('fastapi', 'FastAPI'),
        ('uvicorn', 'Uvicorn')
    ]
    
    missing_deps = []
    
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"  ✅ {name} available")
        except ImportError:
            print(f"  ❌ {name} missing")
            missing_deps.append(name)
    
    # Test optional dependencies
    optional_deps = [
        ('ultralytics', 'YOLOv8 (Ultralytics)')
    ]
    
    for module, name in optional_deps:
        try:
            __import__(module)
            print(f"  ✅ {name} available (optional)")
        except ImportError:
            print(f"  ⚠️  {name} missing (optional)")
    
    if missing_deps:
        print(f"\n  💡 Install missing dependencies:")
        print(f"     pip install {' '.join(missing_deps.lower())}")
        return False
    
    return True

def test_performance():
    """Test performance benchmarks."""
    print("\n⚡ Testing Performance...")
    
    try:
        analyzer = EnhancedTeethAnalyzer()
        
        # Create test image
        from PIL import Image
        import numpy as np
        
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        # Run multiple analyses
        times = []
        for i in range(3):
            start_time = time.time()
            result = analyzer.analyze_teeth(img)
            analysis_time = time.time() - start_time
            times.append(analysis_time)
            print(f"  Run {i+1}: {analysis_time:.2f}s")
        
        avg_time = sum(times) / len(times)
        print(f"  📊 Average analysis time: {avg_time:.2f}s")
        
        if avg_time < 2.0:
            print("  ✅ Performance: Excellent")
        elif avg_time < 5.0:
            print("  ✅ Performance: Good")
        else:
            print("  ⚠️  Performance: Slow (consider optimization)")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Performance test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🦷 Enhanced Teeth Analyzer - System Test")
    print("=" * 50)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Model Files", test_model_files),
        ("Basic Functionality", test_basic_functionality),
        ("API Compatibility", test_api_compatibility),
        ("Performance", test_performance)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above.")
        return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)