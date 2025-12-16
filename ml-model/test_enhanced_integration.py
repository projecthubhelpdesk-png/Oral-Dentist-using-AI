#!/usr/bin/env python3
"""
Test Enhanced Integration
========================
Test the enhanced teeth analyzer integration with the existing API.
"""

import requests
import json
from pathlib import Path

def test_enhanced_api():
    """Test the enhanced API endpoints."""
    base_url = "http://localhost:8000"
    
    print("🦷 Testing Enhanced Teeth Analyzer Integration")
    print("=" * 50)
    
    # Test 1: Check models info
    print("\n1. Testing /models/info endpoint...")
    try:
        response = requests.get(f"{base_url}/models/info")
        if response.status_code == 200:
            data = response.json()
            print("✅ Models info retrieved successfully!")
            print(f"   Basic predictor loaded: {data['basic_predictor']['loaded']}")
            print(f"   Enhanced analyzer loaded: {data['enhanced_analyzer']['loaded']}")
            print(f"   Available classes: {len(data['basic_predictor']['classes'])}")
        else:
            print(f"❌ Failed: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    # Test 2: Check health endpoint
    print("\n2. Testing /health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            data = response.json()
            print("✅ Health check passed!")
            print(f"   Status: {data['status']}")
            print(f"   Model loaded: {data['model_loaded']}")
        else:
            print(f"❌ Failed: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    # Test 3: Test enhanced analysis with sample image
    print("\n3. Testing enhanced analysis...")
    
    # Find a sample image
    sample_images = list(Path("samples").glob("*.jpg")) if Path("samples").exists() else []
    if not sample_images:
        # Create a simple test image
        from PIL import Image
        import numpy as np
        
        # Create a simple test image
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        test_image_path = Path("test_sample.jpg")
        img.save(test_image_path)
        sample_images = [test_image_path]
    
    if sample_images:
        test_image = sample_images[0]
        print(f"   Using test image: {test_image}")
        
        try:
            with open(test_image, 'rb') as f:
                files = {'file': f}
                response = requests.post(f"{base_url}/analyze-teeth", files=files)
            
            if response.status_code == 200:
                data = response.json()
                print("✅ Enhanced analysis completed successfully!")
                print(f"   Analysis ID: {data['analysis_id']}")
                print(f"   Primary condition: {data['disease']}")
                print(f"   Confidence: {data['confidence']:.2%}")
                print(f"   Severity: {data['severity']}")
                print(f"   Tooth detections: {len(data['tooth_detections'])}")
                print(f"   Affected teeth: {len(data['affected_teeth'])}")
                print(f"   Recommendations: {len(data['recommendations'])}")
                
                # Show some tooth detections
                if data['tooth_detections']:
                    print("\n   Tooth Detections:")
                    for detection in data['tooth_detections'][:3]:  # Show first 3
                        print(f"     - {detection['tooth_id']}: {detection['issue']} ({detection['confidence']:.2%})")
                
            else:
                print(f"❌ Enhanced analysis failed: HTTP {response.status_code}")
                print(f"   Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Error during enhanced analysis: {e}")
            return False
    else:
        print("⚠️  No sample images found, skipping enhanced analysis test")
    
    # Test 4: Test summary endpoint
    print("\n4. Testing enhanced summary...")
    if sample_images:
        try:
            with open(sample_images[0], 'rb') as f:
                files = {'file': f}
                response = requests.post(f"{base_url}/analyze-teeth/summary", files=files)
            
            if response.status_code == 200:
                data = response.json()
                print("✅ Enhanced summary completed successfully!")
                print(f"   Analysis ID: {data['analysis_id']}")
                print(f"   Key findings:")
                for key, value in data['key_findings'].items():
                    print(f"     {key}: {value}")
                
            else:
                print(f"❌ Enhanced summary failed: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Error during enhanced summary: {e}")
            return False
    
    print("\n🎉 All tests passed! Enhanced integration is working correctly.")
    print("\n📋 Available Endpoints:")
    print("   • GET  /models/info - Model information")
    print("   • GET  /health - Health check")
    print("   • POST /predict - Basic disease prediction")
    print("   • POST /analyze-teeth - Enhanced analysis with tooth detection")
    print("   • POST /analyze-teeth/summary - Quick enhanced summary")
    
    return True

if __name__ == "__main__":
    success = test_enhanced_api()
    exit(0 if success else 1)