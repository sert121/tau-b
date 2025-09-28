"""
Get user details tool for inspect_ai - converts tau-bench get_user_details tool
"""

import json
from typing import Any, Dict


def get_user_details(user_id: str, data: Dict[str, Any]) -> str:
    """Get the details of a user, including their reservations.
    
    Args:
        user_id: The user id, such as 'sara_doe_496'.
        data: The data dictionary containing users information.
    
    Returns:
        JSON string containing user details, or error message if user not found.
    """
    users = data.get("users", {})
    if user_id in users:
        return json.dumps(users[user_id])
    return "Error: user not found"


def get_get_user_details_tool_info() -> Dict[str, Any]:
    """Get tool information for inspect_ai integration."""
    return {
        "type": "function",
        "function": {
            "name": "get_user_details",
            "description": "Get the details of an user, including their reservations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "The user id, such as 'sara_doe_496'.",
                    },
                },
                "required": ["user_id"],
            },
        },
    }
