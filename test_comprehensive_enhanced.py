#!/usr/bin/env python3
"""
Comprehensive test suite for the enhanced Arabic Names Detection API
Tests all new features, edge cases, and security measures
"""

import requests
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor

# API base URL
BASE_URL = "http://localhost:8000"

def test_security_validation():
    """Test security features and input validation"""
    print("\n🔒 Testing Security & Validation Features")
    print("-" * 50)
    
    # Test cases for security validation
    security_tests = [
        {
            "name": "",
            "description": "Empty input",
            "should_fail": True
        },
        {
            "name": "a",
            "description": "Too short input",
            "should_fail": True
        },
        {
            "name": "x" * 201,
            "description": "Too long input",
            "should_fail": True
        },
        {
            "name": "<script>alert('xss')</script>",
            "description": "XSS attempt",
            "should_fail": True
        },
        {
            "name": "'; DROP TABLE users; --",
            "description": "SQL injection attempt",
            "should_fail": True
        },
        {
            "name": "صدام حسين@#$%^&*()",
            "description": "Too many special characters",
            "should_fail": True
        },
        {
            "name": "صدام حسين",
            "description": "Valid Arabic name",
            "should_fail": False
        }
    ]
    
    for test in security_tests:
        try:
            response = requests.post(f"{BASE_URL}/check-name", 
                                   json={"name": test["name"]},
                                   timeout=10)
            
            if test["should_fail"]:
                if response.status_code >= 400:
                    print(f"✅ {test['description']}: Correctly rejected")
                else:
                    print(f"❌ {test['description']}: Should have been rejected")
            else:
                if response.status_code == 200:
                    print(f"✅ {test['description']}: Correctly accepted")
                else:
                    print(f"❌ {test['description']}: Should have been accepted")
                    
        except Exception as e:
            print(f"🔥 Error testing {test['description']}: {e}")

def test_enhanced_similarity():
    """Test enhanced similarity algorithms"""
    print("\n🧠 Testing Enhanced Similarity Algorithms")
    print("-" * 50)
    
    test_cases = [
        {
            "name": "صدام حسين",
            "description": "Exact match",
            "expected_high_similarity": True
        },
        {
            "name": "صَدَّام حُسَين",
            "description": "With diacritics",
            "expected_high_similarity": True
        },
        {
            "name": "saddam hussein",
            "description": "Latin transliteration",
            "expected_high_similarity": True
        },
        {
            "name": "صدام    حسين",
            "description": "Extra spaces",
            "expected_high_similarity": True
        },
        {
            "name": "محمد علي أحمد",
            "description": "Common safe name",
            "expected_high_similarity": False
        },
        {
            "name": "فاطمة الزهراء",
            "description": "Female name",
            "expected_high_similarity": False
        }
    ]
    
    for test in test_cases:
        try:
            response = requests.post(f"{BASE_URL}/check-name", 
                                   json={"name": test["name"]},
                                   timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                similarity = result.get("max_similarity", 0)
                is_suspicious = result.get("is_suspicious", False)
                language = result.get("language_detected", "unknown")
                
                print(f"\nTest: {test['description']}")
                print(f"  Input: '{test['name']}'")
                print(f"  Language: {language}")
                print(f"  Similarity: {similarity:.2f}%")
                print(f"  Suspicious: {is_suspicious}")
                
                if test["expected_high_similarity"]:
                    if similarity >= 75:
                        print("  ✅ High similarity correctly detected")
                    else:
                        print("  ❌ Expected high similarity but got low")
                else:
                    if similarity < 75:
                        print("  ✅ Low similarity correctly detected")
                    else:
                        print("  ❌ Expected low similarity but got high")
            else:
                print(f"❌ API Error for {test['description']}: {response.status_code}")
                
        except Exception as e:
            print(f"🔥 Error testing {test['description']}: {e}")

def test_batch_processing():
    """Test batch processing capabilities"""
    print("\n📦 Testing Batch Processing")
    print("-" * 50)
    
    batch_names = [
        "صدام حسين",
        "أسامة بن لادن", 
        "محمد أحمد علي",
        "فاطمة الزهراء",
        "علي محمد حسن"
    ]
    
    try:
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/check-names-batch", 
                               json={"names": batch_names},
                               timeout=30)
        processing_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            summary = result.get("summary", {})
            
            print(f"✅ Batch processing completed in {processing_time:.3f}s")
            print(f"   Total names: {summary.get('total_names', 0)}")
            print(f"   Suspicious: {summary.get('suspicious_names', 0)}")
            print(f"   Safe: {summary.get('safe_names', 0)}")
            print(f"   Average time per name: {summary.get('average_processing_time', 0):.4f}s")
            
        else:
            print(f"❌ Batch processing failed: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except Exception as e:
        print(f"🔥 Error in batch processing: {e}")

def test_performance_monitoring():
    """Test performance monitoring and analytics"""
    print("\n📊 Testing Performance Monitoring")
    print("-" * 50)
    
    try:
        # Get health status
        health_response = requests.get(f"{BASE_URL}/health", timeout=10)
        if health_response.status_code == 200:
            health = health_response.json()
            print("✅ Health check passed")
            print(f"   Status: {health.get('status')}")
            print(f"   Blacklist count: {health.get('blacklist', {}).get('count', 0)}")
            print(f"   Version: {health.get('version')}")
        
        # Get detailed stats
        stats_response = requests.get(f"{BASE_URL}/stats", timeout=10)
        if stats_response.status_code == 200:
            stats = stats_response.json()
            performance = stats.get("performance", {})
            print("\n📈 Performance Statistics:")
            print(f"   Total requests: {performance.get('total_requests', 0)}")
            print(f"   Cache hit rate: {performance.get('cache_hit_rate', 0):.2f}%")
            print(f"   Error count: {performance.get('error_count', 0)}")
            print(f"   Uptime: {performance.get('uptime', 0):.2f}s")
        
        # Get analytics
        analytics_response = requests.get(f"{BASE_URL}/analytics", timeout=10)
        if analytics_response.status_code == 200:
            analytics = analytics_response.json()
            summary = analytics.get("summary", {})
            print("\n🔍 Analytics Summary:")
            print(f"   Success rate: {summary.get('success_rate', 0):.2f}%")
            print(f"   Cache efficiency: {summary.get('cache_efficiency', 0):.2f}%")
            
            insights = analytics.get("insights", [])
            if insights:
                print("   Insights:")
                for insight in insights:
                    print(f"     • {insight}")
                    
    except Exception as e:
        print(f"🔥 Error testing monitoring: {e}")

def test_rate_limiting():
    """Test rate limiting functionality"""
    print("\n⚡ Testing Rate Limiting")
    print("-" * 50)
    
    def make_request():
        try:
            response = requests.post(f"{BASE_URL}/check-name", 
                                   json={"name": "test"}, 
                                   timeout=5)
            return response.status_code
        except:
            return 0
    
    # Make rapid requests
    start_time = time.time()
    results = []
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(make_request) for _ in range(50)]
        results = [future.result() for future in futures]
    
    elapsed = time.time() - start_time
    
    rate_limited = sum(1 for code in results if code == 429)
    successful = sum(1 for code in results if code == 200)
    
    print(f"Made 50 rapid requests in {elapsed:.2f}s")
    print(f"  Successful: {successful}")
    print(f"  Rate limited: {rate_limited}")
    
    if rate_limited > 0:
        print("✅ Rate limiting is working")
    else:
        print("⚠️ Rate limiting may not be active")

def test_edge_cases():
    """Test various edge cases and unusual inputs"""
    print("\n🎯 Testing Edge Cases")
    print("-" * 50)
    
    edge_cases = [
        {
            "name": "ابو123بكر",
            "description": "Mixed Arabic with numbers"
        },
        {
            "name": "صدام-حسين",
            "description": "Arabic with hyphen"
        },
        {
            "name": "محمد  احمد",
            "description": "Multiple spaces"
        },
        {
            "name": "Ṣaddām Ḥusayn",
            "description": "Latin with diacritics"
        },
        {
            "name": "صدام حسين الصغير",
            "description": "Extended name variant"
        }
    ]
    
    for test in edge_cases:
        try:
            response = requests.post(f"{BASE_URL}/check-name", 
                                   json={"name": test["name"]},
                                   timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ {test['description']}: Processed successfully")
                print(f"   Normalized: '{result.get('normalized_input', '')}'")
                print(f"   Language: {result.get('language_detected', 'unknown')}")
                print(f"   Similarity: {result.get('max_similarity', 0):.2f}%")
            else:
                print(f"❌ {test['description']}: Failed ({response.status_code})")
                
        except Exception as e:
            print(f"🔥 Error testing {test['description']}: {e}")

def main():
    """Run comprehensive tests"""
    print("🚀 Starting Comprehensive Enhanced API Tests")
    print("=" * 60)
    
    # Test API availability
    try:
        response = requests.get(BASE_URL, timeout=10)
        if response.status_code == 200:
            api_info = response.json()
            print(f"✅ API is running - Version: {api_info.get('version', 'unknown')}")
            print(f"   Features: {len(api_info.get('features', []))} enhanced features")
        else:
            print("❌ API is not responding correctly")
            return
    except Exception as e:
        print(f"🔥 Cannot connect to API: {e}")
        return
    
    # Run all test suites
    test_security_validation()
    test_enhanced_similarity()
    test_batch_processing()
    test_performance_monitoring()
    test_rate_limiting()
    test_edge_cases()
    
    print("\n" + "=" * 60)
    print("🎉 Comprehensive testing completed!")
    print("   Check the results above for any issues")

if __name__ == "__main__":
    main()