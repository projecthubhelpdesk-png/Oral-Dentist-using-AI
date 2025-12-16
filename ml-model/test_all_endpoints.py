#!/usr/bin/env python3
"""Test all API endpoints."""
import requests
from pathlib import Path

BASE_URL = "http://localhost:8000"
TEST_IMAGE = "../backend-php/storage/scans/sample1.jpg"

def test_endpoint(name, method, endpoint, files=None):
    """Test an endpoint and print results."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"Endpoint: {method} {endpoint}")
    print('='*60)
    
    try:
        if method == "GET":
            response = requests.get(f"{BASE_URL}{endpoint}")
        else:
            response = requests.post(f"{BASE_URL}{endpoint}", files=files)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ SUCCESS (HTTP {response.status_code})")
            return data
        else:
            print(f"❌ FAILED (HTTP {response.status_code})")
            print(f"Response: {response.text[:200]}")
            return None
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return None

def main():
    print("\n🦷 ORAL CARE AI - API ENDPOINT TESTS")
    print("="*60)
    
    # Test 1: Health check
    data = test_endpoint("Health Check", "GET", "/health")
    if data:
        print(f"  Status: {data.get('status')}")
        print(f"  Model loaded: {data.get('model_loaded')}")
    
    # Test 2: Models info
    data = test_endpoint("Models Info", "GET", "/models/info")
    if data:
        print(f"  Basic predictor: {data.get('basic_predictor', {}).get('loaded')}")
        print(f"  Enhanced analyzer: {data.get('enhanced_analyzer', {}).get('loaded')}")
    
    # Test 3: Basic prediction
    with open(TEST_IMAGE, 'rb') as f:
        data = test_endpoint("Basic Disease Prediction", "POST", "/predict", files={'file': f})
    if data:
        print(f"  Disease: {data.get('disease')}")
        print(f"  Confidence: {data.get('confidence', 0):.2%}")
        print(f"  Severity: {data.get('severity')}")
    
    # Test 4: Enhanced analysis
    with open(TEST_IMAGE, 'rb') as f:
        data = test_endpoint("Enhanced Teeth Analysis", "POST", "/analyze-teeth", files={'file': f})
    if data:
        print(f"  Disease: {data.get('disease')}")
        print(f"  Tooth detections: {len(data.get('tooth_detections', []))}")
        print(f"  Affected teeth: {data.get('affected_teeth', [])}")
    
    # Test 5: Condition analysis
    with open(TEST_IMAGE, 'rb') as f:
        data = test_endpoint("Teeth Condition Analysis", "POST", "/analyze-condition", files={'file': f})
    if data:
        print(f"  Overall Score: {data.get('overall_score')}/100")
        print(f"  Summary: {data.get('summary', '')[:80]}...")
        if data.get('condition_breakdown'):
            print("  Conditions:")
            for cat, info in list(data['condition_breakdown'].items())[:3]:
                print(f"    - {info['name']}: {info['level']}")
    
    # Test 6: Complete analysis
    with open(TEST_IMAGE, 'rb') as f:
        data = test_endpoint("Complete Dental Analysis", "POST", "/analyze-complete", files={'file': f})
    if data:
        print(f"  Disease: {data.get('disease_analysis', {}).get('primary_condition')}")
        print(f"  Condition Score: {data.get('condition_analysis', {}).get('overall_score')}/100")
        print(f"  Tooth detections: {len(data.get('tooth_detections', []))}")
        print(f"  Follow-up: {data.get('summary', {}).get('follow_up', 'N/A')}")
    
    print("\n" + "="*60)
    print("🎉 All endpoint tests completed!")
    print("="*60)
    
    print("\n📋 Available API Endpoints:")
    print("  GET  /health              - Health check")
    print("  GET  /models/info         - Model information")
    print("  POST /predict             - Basic disease prediction")
    print("  POST /analyze-teeth       - Enhanced analysis with tooth detection")
    print("  POST /analyze-teeth/summary - Quick enhanced summary")
    print("  POST /analyze-condition   - EfficientNet teeth condition analysis")
    print("  POST /analyze-complete    - Complete combined analysis")

if __name__ == "__main__":
    main()
