"""
Test Advanced Dental Pipeline
=============================
Quick test script to verify the advanced pipeline works.

Usage:
    python test_advanced_pipeline.py [image_path]
"""

import sys
import asyncio
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))


async def test_pipeline(image_path: str = None):
    """Test the advanced dental pipeline."""
    print("="*60)
    print("Testing Advanced Dental AI Pipeline")
    print("="*60)
    
    # Use test image if not provided
    if not image_path:
        test_images = [
            Path(__file__).parent / 'test_image.jpg',
            Path(__file__).parent / 'data' / 'test.jpg',
        ]
        for img in test_images:
            if img.exists():
                image_path = str(img)
                break
    
    if not image_path or not Path(image_path).exists():
        print("No test image found. Please provide an image path.")
        print("Usage: python test_advanced_pipeline.py <image_path>")
        return
    
    print(f"\nTest image: {image_path}")
    
    # Import pipeline
    print("\n1. Importing Advanced Pipeline...")
    try:
        from advanced_dental_pipeline import (
            AdvancedDentalPipeline,
            DentalReportFormatter,
            CNNEnsemble,
            LLMProvider
        )
        print("   ✅ Import successful")
    except ImportError as e:
        print(f"   ❌ Import failed: {e}")
        return
    
    # Test CNN Ensemble
    print("\n2. Testing CNN Ensemble...")
    try:
        cnn = CNNEnsemble()
        cnn_result = cnn.predict(image_path)
        print(f"   ✅ CNN prediction successful")
        print(f"   Model used: {cnn_result['model_used']}")
        print(f"   Predictions: {[f'{p:.3f}' for p in cnn_result['predictions']]}")
    except Exception as e:
        print(f"   ❌ CNN failed: {e}")
    
    # Test LLM Provider
    print("\n3. Testing LLM Provider...")
    try:
        llm = LLMProvider()
        if llm.openai_client:
            print("   ✅ OpenAI client available")
        else:
            print("   ⚠️ OpenAI not configured (set OPENAI_API_KEY)")
    except Exception as e:
        print(f"   ❌ LLM init failed: {e}")
    
    # Test Full Pipeline
    print("\n4. Testing Full Pipeline...")
    try:
        pipeline = AdvancedDentalPipeline()
        print("   ✅ Pipeline initialized")
        
        # Run analysis (without LLM for quick test)
        print("\n5. Running analysis (CNN only)...")
        result = await pipeline.analyze(image_path, use_llm=False, use_vlm=False)
        
        print(f"   ✅ Analysis complete!")
        print(f"\n   Analysis ID: {result['analysis_id']}")
        print(f"   Disease: {result['summary']['disease']}")
        print(f"   Confidence: {result['summary']['confidence']*100:.1f}%")
        print(f"   Severity: {result['summary']['severity']}")
        print(f"   Recommendation: {result['summary']['recommendation']}")
        
        # Test report formatter
        print("\n6. Testing Report Formatter...")
        formatter = DentalReportFormatter()
        text_report = formatter.to_text(result)
        print("   ✅ Text report generated")
        print("\n" + "="*60)
        print("SAMPLE REPORT (truncated):")
        print("="*60)
        print(text_report[:1500] + "...")
        
    except Exception as e:
        print(f"   ❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test with LLM if available
    import os
    if os.environ.get('OPENAI_API_KEY'):
        print("\n7. Testing with LLM (GPT-4o)...")
        try:
            result_llm = await pipeline.analyze(image_path, use_llm=True, use_vlm=False)
            print("   ✅ LLM analysis complete!")
            if result_llm.get('llm_analysis'):
                llm_text = result_llm['llm_analysis'].get('llm_analysis', '')[:500]
                print(f"\n   LLM Response (truncated):\n   {llm_text}...")
        except Exception as e:
            print(f"   ❌ LLM analysis failed: {e}")
    else:
        print("\n7. Skipping LLM test (OPENAI_API_KEY not set)")
    
    print("\n" + "="*60)
    print("Test Complete!")
    print("="*60)


if __name__ == '__main__':
    image_path = sys.argv[1] if len(sys.argv) > 1 else None
    asyncio.run(test_pipeline(image_path))
