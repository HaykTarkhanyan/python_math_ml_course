"""
Example usage of the Kargin Haghordum API

This script demonstrates how to interact with the API endpoints
using the requests library.
"""

import requests
import json
from typing import Dict, Any


# API base URL (assuming the server is running locally)
BASE_URL = "http://localhost:8000"


def test_health_check():
    """Test the health check endpoint"""
    print("🩺 Testing health check...")
    response = requests.get(f"{BASE_URL}/health")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ API is healthy!")
        print(f"   Total sketches: {data['total_sketches']}")
        print(f"   Status: {data['status']}")
        print(f"   Message: {data['message']}")
    else:
        print(f"❌ Health check failed: {response.status_code}")
    
    print("-" * 50)


def test_search(query: str):
    """Test the search endpoint"""
    print(f"🔍 Searching for: '{query}'")
    
    params = {
        "q": query,
        "limit": 5,
        "include_incomplete": True
    }
    
    response = requests.get(f"{BASE_URL}/search", params=params)
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Found {data['matches_found']} matches")
        
        for i, sketch in enumerate(data['results'], 1):
            print(f"   {i}. {sketch['title']}")
            print(f"      Actors: {sketch['main_actors']}")
            print(f"      Location: {sketch['location']}")
            if sketch['text_common']:
                preview = sketch['text_common'][:100] + "..." if len(sketch['text_common']) > 100 else sketch['text_common']
                print(f"      Preview: {preview}")
            print(f"      Link: {sketch['link']}")
            print()
    else:
        print(f"❌ Search failed: {response.status_code}")
        print(f"   Error: {response.text}")
    
    print("-" * 50)


def test_random_sketch():
    """Test the random sketch endpoint"""
    print("🎲 Getting random sketch...")
    
    response = requests.get(f"{BASE_URL}/random")
    
    if response.status_code == 200:
        data = response.json()
        sketch = data['sketch']
        print(f"✅ {data['message']}")
        print(f"   Title: {sketch['title']}")
        print(f"   Actors: {sketch['main_actors']} ({sketch['main_actors_count']} actors)")
        print(f"   Location: {sketch['location']}")
        print(f"   Languages: {sketch['languages']}")
        print(f"   Link: {sketch['link']}")
    else:
        print(f"❌ Random sketch failed: {response.status_code}")
        print(f"   Error: {response.text}")
    
    print("-" * 50)


def test_update_sketch(sketch_index: int):
    """Test updating a sketch"""
    print(f"✏️ Updating sketch {sketch_index}...")
    
    # First, get the current sketch data
    response = requests.get(f"{BASE_URL}/sketches/{sketch_index}")
    if response.status_code != 200:
        print(f"❌ Sketch {sketch_index} not found")
        return
    
    current_sketch = response.json()
    print(f"   Current title: {current_sketch['title']}")
    
    # Update with new data
    update_data = {
        "text_common": "Updated via API example script",
        "roles_names": "API Test Role"
    }
    
    response = requests.put(f"{BASE_URL}/sketches/{sketch_index}", json=update_data)
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ {data['message']}")
        print(f"   Updated sketch: {data['updated_sketch']['title']}")
    else:
        print(f"❌ Update failed: {response.status_code}")
        print(f"   Error: {response.text}")
    
    print("-" * 50)


def test_get_statistics():
    """Test the statistics endpoint"""
    print("📊 Getting statistics...")
    
    response = requests.get(f"{BASE_URL}/stats")
    
    if response.status_code == 200:
        stats = response.json()
        print("✅ Statistics:")
        print(f"   Total sketches: {stats['total_sketches']}")
        print(f"   Completed: {stats['completed_sketches']}")
        print(f"   Incomplete: {stats['incomplete_sketches']}")
        print(f"   Completion rate: {stats['completion_rate']:.1f}%")
        print(f"   Most common location: {stats['most_common_location']}")
        print(f"   Average actors per sketch: {stats['average_actors_per_sketch']:.1f}")
    else:
        print(f"❌ Statistics failed: {response.status_code}")
        print(f"   Error: {response.text}")
    
    print("-" * 50)


def main():
    """Run all API tests"""
    print("🎭 Kargin Haghordum API Testing Script")
    print("=" * 50)
    
    try:
        # Test each endpoint
        test_health_check()
        test_get_statistics()
        test_search("շուն")  # Search for "dog" in Armenian
        test_search("բժիշկ")  # Search for "doctor" in Armenian  
        test_random_sketch()
        
        # Note: Update test is commented out to avoid modifying data
        # Uncomment the line below if you want to test updates:
        # test_update_sketch(0)
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API server!")
        print("   Make sure the API is running at http://localhost:8000")
        print("   Start it with: python kargin_api.py")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()
