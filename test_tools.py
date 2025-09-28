"""
Test script for converted airline tools
"""

import json
from tools import get_airline_tools, get_airline_tool_info


def test_tools():
    """Test all converted airline tools."""
    print("Testing Converted Airline Tools")
    print("=" * 50)
    
    # Load sample data
    from dataset import load_tau_bench_data
    
    print("\n=== Loading Sample Data ===")
    airline_data = load_tau_bench_data("airline", "train")
    print(f"✓ Loaded airline data with {len(airline_data['data']['flights'])} flights")
    
    # Get tools
    tools = get_airline_tools()
    tool_info = get_airline_tool_info()
    
    print(f"\n=== Available Tools ===")
    print(f"✓ Found {len(tools)} tools:")
    for name, tool in tools.items():
        print(f"  - {name}: {tool['info']['function']['description']}")
    
    print(f"\n=== Testing Individual Tools ===")
    
    # Test calculate tool
    print("\n--- Testing Calculate Tool ---")
    try:
        result = tools["calculate"]["function"]("2 + 2")
        print(f"✓ calculate('2 + 2') = {result}")
        
        result = tools["calculate"]["function"]("10 * 5")
        print(f"✓ calculate('10 * 5') = {result}")
    except Exception as e:
        print(f"✗ Error testing calculate: {e}")
    
    # Test list_all_airports tool
    print("\n--- Testing List All Airports Tool ---")
    try:
        result = tools["list_all_airports"]["function"]()
        airports = json.loads(result)
        print(f"✓ list_all_airports() returned {len(airports)} airports")
        print(f"  Sample: {list(airports.items())[:3]}")
    except Exception as e:
        print(f"✗ Error testing list_all_airports: {e}")
    
    # Test search_direct_flight tool
    print("\n--- Testing Search Direct Flight Tool ---")
    try:
        result = tools["search_direct_flight"]["function"](
            "JFK", "LAX", "2024-05-20", airline_data["data"]
        )
        flights = json.loads(result)
        print(f"✓ search_direct_flight('JFK', 'LAX', '2024-05-20') returned {len(flights)} flights")
        if flights:
            print(f"  Sample flight: {flights[0]}")
    except Exception as e:
        print(f"✗ Error testing search_direct_flight: {e}")
    
    # Test get_user_details tool
    print("\n--- Testing Get User Details Tool ---")
    try:
        # Get a sample user ID
        users = airline_data["data"]["users"]
        sample_user_id = list(users.keys())[0]
        
        result = tools["get_user_details"]["function"](sample_user_id, airline_data["data"])
        user_details = json.loads(result)
        print(f"✓ get_user_details('{sample_user_id}') returned user details")
        print(f"  User: {user_details.get('first_name', 'Unknown')} {user_details.get('last_name', 'Unknown')}")
    except Exception as e:
        print(f"✗ Error testing get_user_details: {e}")
    
    # Test think tool
    print("\n--- Testing Think Tool ---")
    try:
        result = tools["think"]["function"]("This is a test thought")
        print(f"✓ think('This is a test thought') = '{result}'")
    except Exception as e:
        print(f"✗ Error testing think: {e}")
    
    print(f"\n=== Tool Information ===")
    print(f"✓ Tool info available for {len(tool_info)} tools")
    for i, info in enumerate(tool_info[:3]):  # Show first 3
        print(f"  {i+1}. {info['function']['name']}: {info['function']['description']}")
    
    print(f"\n" + "=" * 50)
    print("Test completed!")


if __name__ == "__main__":
    test_tools()
