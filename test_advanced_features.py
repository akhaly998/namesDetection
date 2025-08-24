#!/usr/bin/env python3
"""
Advanced test script for the improved Arabic Names Detection API
Tests new features like Arabic text processing, phonetic matching, etc.
"""

import requests
import json

# API base URL
BASE_URL = "http://localhost:8000"

def test_advanced_features():
    """Test the advanced Arabic processing features"""
    
    print("Testing Advanced Arabic Names Detection API")
    print("=" * 60)
    
    # Test cases with different Arabic text variations
    test_cases = [
        {
            "name": "صدام حسين العراقي",
            "description": "Name with additional terms",
            "expected_suspicious": True
        },
        {
            "name": "صَدَّام حُسَين",
            "description": "Name with diacritics (tashkeel)",
            "expected_suspicious": True
        },
        {
            "name": "اسامه بن لادن", 
            "description": "Different spelling variation",
            "expected_suspicious": True
        },
        {
            "name": "أبوبكر البغدادي",
            "description": "Name without spaces",
            "expected_suspicious": True
        },
        {
            "name": "محمد أحمد علي",
            "description": "Common safe name",
            "expected_suspicious": False
        },
        {
            "name": "فاطمة الزهراء",
            "description": "Female safe name",
            "expected_suspicious": False
        },
        {
            "name": "خالد محمد شيخ",
            "description": "Partial match test",
            "expected_suspicious": False  # Should be below 70%
        }
    ]
    
    print("\n1. Testing Advanced Name Processing")
    print("-" * 40)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['description']}")
        print(f"Input: '{test_case['name']}'")
        
        # Test with detailed endpoint
        payload = {"name": test_case['name']}
        response = requests.post(f"{BASE_URL}/check-name", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            print(f"Normalized: '{result['normalized_input']}'")
            print(f"Suspicious: {result['is_suspicious']} (expected: {test_case['expected_suspicious']})")
            print(f"Max Similarity: {result['max_similarity']}%")
            print(f"Confidence: {result['confidence_level']}")
            
            if result['matches']:
                best_match = result['matches'][0]
                print(f"Best Match: '{best_match['blacklisted_name']}'")
                if 'similarity_details' in best_match and best_match['similarity_details']:
                    details = best_match['similarity_details']
                    print("Similarity Details:")
                    for method, score in details.items():
                        if score > 0:
                            print(f"  - {method}: {score:.2f}%")
            
            # Validate expectations
            if result['is_suspicious'] == test_case['expected_suspicious']:
                print("✅ Test passed")
            else:
                print("❌ Test failed - suspicion level mismatch")
        else:
            print(f"❌ API Error: {response.status_code}")
    
    print("\n\n2. Testing Text Normalization Features")
    print("-" * 40)
    
    # Test normalization with simple endpoint
    normalization_tests = [
        ("صَدَّام حُسَين", "Name with tashkeel"),
        ("صدام    حسين", "Name with extra spaces"),
        ("أبو بكر البغدادي123", "Name with numbers"),
        ("صدام-حسين", "Name with punctuation")
    ]
    
    for test_name, description in normalization_tests:
        print(f"\nTesting: {description}")
        print(f"Input: '{test_name}'")
        
        payload = {"name": test_name}
        response = requests.post(f"{BASE_URL}/check-name-simple", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            print(f"Normalized: '{result['normalized_input']}'")
            print(f"Match: {result['matched_name']}")
            print(f"Similarity: {result['similarity_percentage']}%")
            
            if 'similarity_details' in result:
                details = result['similarity_details']
                phonetic_score = details.get('phonetic', 0)
                normalized_score = details.get('normalized', 0)
                if phonetic_score > 0:
                    print(f"Phonetic Score: {phonetic_score:.2f}%")
                if normalized_score > 0:
                    print(f"Normalized Score: {normalized_score:.2f}%")
        else:
            print(f"❌ API Error: {response.status_code}")
    
    print("\n\n3. Testing Performance with Larger Blacklist")
    print("-" * 40)
    
    # Check updated blacklist
    response = requests.get(f"{BASE_URL}/blacklist")
    if response.status_code == 200:
        blacklist_data = response.json()
        print(f"Blacklist now contains {blacklist_data['count']} names")
        
        # Test performance with a few names
        performance_tests = ["محمد علي أحمد", "فاطمة محمد", "عبدالله حسن"]
        
        for test_name in performance_tests:
            payload = {"name": test_name}
            response = requests.post(f"{BASE_URL}/check-name", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                processing_info = result.get('processing_info', {})
                print(f"Processed '{test_name}': {processing_info.get('total_comparisons', 'N/A')} comparisons")
    
    print("\n\n4. Testing API Root Information")
    print("-" * 40)
    
    response = requests.get(f"{BASE_URL}/")
    if response.status_code == 200:
        api_info = response.json()
        print(f"API: {api_info['message']}")
        print(f"Version: {api_info['version']}")
        print("Features:")
        for feature in api_info.get('features', []):
            print(f"  ✓ {feature}")
    
    print("\n" + "=" * 60)
    print("✅ Advanced features testing completed!")

if __name__ == "__main__":
    test_advanced_features()