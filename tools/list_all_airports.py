"""
List all airports tool for inspect_ai - converts tau-bench list_all_airports tool
"""

import json
from typing import Any, Dict


def list_all_airports() -> str:
    """List all airports and their cities.
    
    Returns:
        JSON string containing airport codes and their corresponding cities.
    """
    airports = [
        "SFO", "JFK", "LAX", "ORD", "DFW", "DEN", "SEA", "ATL", "MIA", "BOS",
        "PHX", "IAH", "LAS", "MCO", "EWR", "CLT", "MSP", "DTW", "PHL", "LGA",
    ]
    cities = [
        "San Francisco", "New York", "Los Angeles", "Chicago", "Dallas",
        "Denver", "Seattle", "Atlanta", "Miami", "Boston", "Phoenix",
        "Houston", "Las Vegas", "Orlando", "Newark", "Charlotte",
        "Minneapolis", "Detroit", "Philadelphia", "LaGuardia",
    ]
    return json.dumps({airport: city for airport, city in zip(airports, cities)})


def get_list_all_airports_tool_info() -> Dict[str, Any]:
    """Get tool information for inspect_ai integration."""
    return {
        "type": "function",
        "function": {
            "name": "list_all_airports",
            "description": "List all airports and their cities.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    }
