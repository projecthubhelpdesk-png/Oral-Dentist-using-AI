#!/usr/bin/env python3
"""Test with real sample image."""
import requests

with open('../backend-php/storage/scans/sample1.jpg', 'rb') as f:
    response = requests.post('http://localhost:8000/analyze-teeth', files={'file': f})
    data = response.json()
    
    print("🦷 Enhanced Analysis Results")
    print("=" * 50)
    print(f"Analysis ID: {data.get('analysis_id')}")
    print(f"Disease: {data.get('disease')}")
    print(f"Confidence: {data.get('confidence', 0):.2%}")
    print(f"Severity: {data.get('severity')}")
    print(f"Tooth detections: {len(data.get('tooth_detections', []))}")
    print(f"Affected teeth: {data.get('affected_teeth', [])}")
    print()
    print("Recommendations:")
    for rec in data.get('recommendations', [])[:4]:
        print(f"  - {rec}")
    print()
    print("Home Care Tips:")
    for tip in data.get('home_care_tips', [])[:3]:
        print(f"  - {tip}")