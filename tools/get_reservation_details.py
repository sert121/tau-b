"""
Get reservation details tool for inspect_ai - converts tau-bench get_reservation_details tool
"""

import json
from typing import Any, Dict


def get_reservation_details(reservation_id: str, data: Dict[str, Any]) -> str:
    """Get the details of a reservation.
    
    Args:
        reservation_id: The reservation id, such as '8JX2WO'.
        data: The data dictionary containing reservations information.
    
    Returns:
        JSON string containing reservation details, or error message if not found.
    """
    reservations = data.get("reservations", {})
    if reservation_id in reservations:
        return json.dumps(reservations[reservation_id])
    return "Error: reservation not found"


def get_get_reservation_details_tool_info() -> Dict[str, Any]:
    """Get tool information for inspect_ai integration."""
    return {
        "type": "function",
        "function": {
            "name": "get_reservation_details",
            "description": "Get the details of a reservation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reservation_id": {
                        "type": "string",
                        "description": "The reservation id, such as '8JX2WO'.",
                    },
                },
                "required": ["reservation_id"],
            },
        },
    }
