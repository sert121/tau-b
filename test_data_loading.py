#!/usr/bin/env python3
"""
Test script to verify that we're actually loading data from tau-bench.

This script tests the data loading functions to ensure they're working
with the actual tau-bench data files.
"""

import json
from pathlib import Path
from dataset import load_tau_bench_data, load_tau_bench_tools, load_tau_bench_tasks


def test_data_loading():
    """Test loading tau-bench data."""
    print("Testing tau-bench data loading...")
    
    # Test airline data loading
    print("\n=== Testing Airline Data Loading ===")
    try:
        airline_data = load_tau_bench_data("airline", "train")
        print(f"✓ Loaded airline data")
        print(f"  - Domain: {airline_data['domain']}")
        print(f"  - Split: {airline_data['split']}")
        print(f"  - Data keys: {list(airline_data['data'].keys())}")
        print(f"  - Number of tasks: {len(airline_data['tasks'])}")
        
        # Check if we have the expected data files
        if 'flights' in airline_data['data']:
            flights = airline_data['data']['flights']
            print(f"  - Flights data: {len(flights)} flights")
            if flights:
                sample_flight = list(flights.keys())[0]
                print(f"  - Sample flight: {sample_flight}")
        
        if 'reservations' in airline_data['data']:
            reservations = airline_data['data']['reservations']
            print(f"  - Reservations data: {len(reservations)} reservations")
            if reservations:
                sample_reservation = list(reservations.keys())[0]
                print(f"  - Sample reservation: {sample_reservation}")
        
        if 'users' in airline_data['data']:
            users = airline_data['data']['users']
            print(f"  - Users data: {len(users)} users")
            if users:
                sample_user = list(users.keys())[0]
                print(f"  - Sample user: {sample_user}")
        
    except Exception as e:
        print(f"✗ Error loading airline data: {e}")
    
    # Test retail data loading
    print("\n=== Testing Retail Data Loading ===")
    try:
        retail_data = load_tau_bench_data("retail", "train")
        print(f"✓ Loaded retail data")
        print(f"  - Domain: {retail_data['domain']}")
        print(f"  - Split: {retail_data['split']}")
        print(f"  - Data keys: {list(retail_data['data'].keys())}")
        print(f"  - Number of tasks: {len(retail_data['tasks'])}")
        
        # Check if we have the expected data files
        if 'orders' in retail_data['data']:
            orders = retail_data['data']['orders']
            print(f"  - Orders data: {len(orders)} orders")
            if orders:
                sample_order = list(orders.keys())[0]
                print(f"  - Sample order: {sample_order}")
        
        if 'products' in retail_data['data']:
            products = retail_data['data']['products']
            print(f"  - Products data: {len(products)} products")
            if products:
                sample_product = list(products.keys())[0]
                print(f"  - Sample product: {sample_product}")
        
        if 'users' in retail_data['data']:
            users = retail_data['data']['users']
            print(f"  - Users data: {len(users)} users")
            if users:
                sample_user = list(users.keys())[0]
                print(f"  - Sample user: {sample_user}")
        
    except Exception as e:
        print(f"✗ Error loading retail data: {e}")


def test_tools_loading():
    """Test loading tau-bench tools."""
    print("\n=== Testing Tools Loading ===")
    
    # Test airline tools
    print("\n--- Airline Tools ---")
    try:
        airline_tools = load_tau_bench_tools("airline")
        print(f"✓ Loaded {len(airline_tools)} airline tools")
        
        for i, tool in enumerate(airline_tools):
            function_info = tool.get("function", {})
            name = function_info.get("name", "unknown")
            description = function_info.get("description", "No description")
            print(f"  {i+1}. {name}: {description}")
            
    except Exception as e:
        print(f"✗ Error loading airline tools: {e}")
    
    # Test retail tools
    print("\n--- Retail Tools ---")
    try:
        retail_tools = load_tau_bench_tools("retail")
        print(f"✓ Loaded {len(retail_tools)} retail tools")
        
        for i, tool in enumerate(retail_tools):
            function_info = tool.get("function", {})
            name = function_info.get("name", "unknown")
            description = function_info.get("description", "No description")
            print(f"  {i+1}. {name}: {description}")
            
    except Exception as e:
        print(f"✗ Error loading retail tools: {e}")


def test_tasks_loading():
    """Test loading tau-bench tasks."""
    print("\n=== Testing Tasks Loading ===")
    
    # Test airline tasks
    print("\n--- Airline Tasks ---")
    try:
        airline_tasks = load_tau_bench_tasks("airline", "train")
        print(f"✓ Loaded {len(airline_tasks)} airline tasks")
        
        if airline_tasks:
            for i, task in enumerate(airline_tasks[:3]):  # Show first 3 tasks
                # Handle both dict and Task object
                if hasattr(task, 'user_id'):
                    user_id = task.user_id
                    instruction = task.instruction
                else:
                    user_id = task.get('user_id', 'unknown')
                    instruction = task.get('instruction', 'No instruction')
                print(f"  Task {i+1}: {user_id} - {instruction[:50]}...")
        
    except Exception as e:
        print(f"✗ Error loading airline tasks: {e}")
    
    # Test retail tasks
    print("\n--- Retail Tasks ---")
    try:
        retail_tasks = load_tau_bench_tasks("retail", "train")
        print(f"✓ Loaded {len(retail_tasks)} retail tasks")
        
        if retail_tasks:
            for i, task in enumerate(retail_tasks[:3]):  # Show first 3 tasks
                # Handle both dict and Task object
                if hasattr(task, 'user_id'):
                    user_id = task.user_id
                    instruction = task.instruction
                else:
                    user_id = task.get('user_id', 'unknown')
                    instruction = task.get('instruction', 'No instruction')
                print(f"  Task {i+1}: {user_id} - {instruction[:50]}...")
        
    except Exception as e:
        print(f"✗ Error loading retail tasks: {e}")


def test_file_paths():
    """Test that tau-bench files exist."""
    print("\n=== Testing File Paths ===")
    
    tau_bench_path = Path(__file__).parent.parent / "tau-bench" / "tau_bench"
    print(f"Tau-bench path: {tau_bench_path}")
    print(f"Path exists: {tau_bench_path.exists()}")
    
    if tau_bench_path.exists():
        # Check airline files
        print("\n--- Airline Files ---")
        airline_data_path = tau_bench_path / "envs" / "airline" / "data"
        print(f"Airline data path: {airline_data_path}")
        print(f"Airline data exists: {airline_data_path.exists()}")
        
        if airline_data_path.exists():
            for file_name in ["users.json", "flights.json", "reservations.json"]:
                file_path = airline_data_path / file_name
                print(f"  {file_name}: {file_path.exists()}")
                if file_path.exists():
                    print(f"    Size: {file_path.stat().st_size} bytes")
        
        airline_tools_path = tau_bench_path / "envs" / "airline" / "tools"
        print(f"Airline tools path: {airline_tools_path}")
        print(f"Airline tools exists: {airline_tools_path.exists()}")
        
        if airline_tools_path.exists():
            tool_files = list(airline_tools_path.glob("*.py"))
            print(f"  Tool files: {len(tool_files)}")
            for tool_file in tool_files[:5]:  # Show first 5
                print(f"    {tool_file.name}")
        
        # Check retail files
        print("\n--- Retail Files ---")
        retail_data_path = tau_bench_path / "envs" / "retail" / "data"
        print(f"Retail data path: {retail_data_path}")
        print(f"Retail data exists: {retail_data_path.exists()}")
        
        if retail_data_path.exists():
            for file_name in ["users.json", "orders.json", "products.json"]:
                file_path = retail_data_path / file_name
                print(f"  {file_name}: {file_path.exists()}")
                if file_path.exists():
                    print(f"    Size: {file_path.stat().st_size} bytes")
        
        retail_tools_path = tau_bench_path / "envs" / "retail" / "tools"
        print(f"Retail tools path: {retail_tools_path}")
        print(f"Retail tools exists: {retail_tools_path.exists()}")
        
        if retail_tools_path.exists():
            tool_files = list(retail_tools_path.glob("*.py"))
            print(f"  Tool files: {len(tool_files)}")
            for tool_file in tool_files[:5]:  # Show first 5
                print(f"    {tool_file.name}")


def main():
    """Main test function."""
    print("Tau-bench Data Loading Test")
    print("=" * 50)
    
    # Test file paths first
    test_file_paths()
    
    # Test data loading
    test_data_loading()
    
    # Test tools loading
    test_tools_loading()
    
    # Test tasks loading
    test_tasks_loading()
    
    print("\n" + "=" * 50)
    print("Test completed!")


if __name__ == "__main__":
    main()
