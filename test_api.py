#!/usr/bin/env python3
"""
Simple test script for the Names Detection API
"""

import requests
import json

# API base URL
BASE_URL = "http://localhost:8000"

def test_api():
    """Test the main API endpoints"""
    
    print("Testing Names Detection API...")
    print("=" * 50)
    
    # Test health endpoint
    print("\n1. Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except requests.exceptions.ConnectionError:
        print("❌ Server is not running. Please start the server first with: python main.py")
        return
    
    # Test the example name from requirements
    print("\n2. Testing with the example name: 'صدام حسن سعيد مجيد'")
    test_name = "صدام حسن سعيد مجيد"
    payload = {"name": test_name}
    
    response = requests.post(f"{BASE_URL}/check-name-simple", json=payload)
    result = response.json()
    print(f"Input: {result['input_name']}")
    print(f"Matched: {result['matched_name']}")
    print(f"Similarity: {result['similarity_percentage']}%")
    print(f"Suspicious: {result['is_suspicious']}")
    
    # Test with a safe name
    print("\n3. Testing with a safe name: 'محمد علي'")
    test_name = "محمد علي"
    payload = {"name": test_name}
    
    response = requests.post(f"{BASE_URL}/check-name-simple", json=payload)
    result = response.json()
    print(f"Input: {result['input_name']}")
    print(f"Matched: {result['matched_name']}")
    print(f"Similarity: {result['similarity_percentage']}%")
    print(f"Suspicious: {result['is_suspicious']}")
    
    # Test detailed endpoint
    print("\n4. Testing detailed endpoint with 'أسامة بن لادن'")
    test_name = "أسامة بن لادن"
    payload = {"name": test_name}
    
    response = requests.post(f"{BASE_URL}/check-name", json=payload)
    result = response.json()
    print(f"Input: {result['input_name']}")
    print(f"Is Suspicious: {result['is_suspicious']}")
    print(f"Max Similarity: {result['max_similarity']}%")
    print(f"Number of matches: {len(result['matches'])}")
    
    # Show blacklist count
    print("\n5. Checking blacklist...")
    response = requests.get(f"{BASE_URL}/blacklist")
    result = response.json()
    print(f"Blacklist contains {result['count']} names")
    
    print("\n" + "=" * 50)
    print("✅ All tests completed successfully!")

if __name__ == "__main__":
    test_api()