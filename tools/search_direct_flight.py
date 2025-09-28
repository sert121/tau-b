"""
Search direct flight tool for inspect_ai - converts tau-bench search_direct_flight tool
"""

import json
from typing import Any, Dict


def search_direct_flight(origin: str, destination: str, date: str, flights_data: Dict[str, Any]) -> str:
    """Search direct flights between two cities on a specific date.
    
    Args:
        origin: The origin city airport in three letters, such as 'JFK'.
        destination: The destination city airport in three letters, such as 'LAX'.
        date: The date of the flight in the format 'YYYY-MM-DD', such as '2024-01-01'.
        flights_data: The flights data dictionary containing flight information.
    
    Returns:
        JSON string containing available direct flights matching the criteria.
    """
    flights = flights_data.get("flights", {})
    results = []
    
    for flight in flights.values():
        if flight["origin"] == origin and flight["destination"] == destination:
            if (
                date in flight["dates"]
                and flight["dates"][date]["status"] == "available"
            ):
                # Add flight except dates, but add flight["dates"][date]
                flight_result = {k: v for k, v in flight.items() if k != "dates"}
                flight_result.update(flight["dates"][date])
                results.append(flight_result)
    
    return json.dumps(results)


def get_search_direct_flight_tool_info() -> Dict[str, Any]:
    """Get tool information for inspect_ai integration."""
    return {
        "type": "function",
        "function": {
            "name": "search_direct_flight",
            "description": "Search direct flights between two cities on a specific date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {
                        "type": "string",
                        "description": "The origin city airport in three letters, such as 'JFK'.",
                    },
                    "destination": {
                        "type": "string",
                        "description": "The destination city airport in three letters, such as 'LAX'.",
                    },
                    "date": {
                        "type": "string",
                        "description": "The date of the flight in the format 'YYYY-MM-DD', such as '2024-01-01'.",
                    },
                },
                "required": ["origin", "destination", "date"],
            },
        },
    }
