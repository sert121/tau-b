"""
Think tool for inspect_ai - converts tau-bench think tool
"""

from typing import Any, Dict


def think(thought: str) -> str:
    """Use the tool to think about something. It will not obtain new information or change the database, but just append the thought to the log. Use it when complex reasoning is needed.
    
    Args:
        thought: A thought to think about.
    
    Returns:
        Empty string (this tool is for internal reasoning only).
    """
    return ""


def get_think_tool_info() -> Dict[str, Any]:
    """Get tool information for inspect_ai integration."""
    return {
        "type": "function",
        "function": {
            "name": "think",
            "description": "Use the tool to think about something. It will not obtain new information or change the database, but just append the thought to the log. Use it when complex reasoning is needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "thought": {
                        "type": "string",
                        "description": "A thought to think about.",
                    },
                },
                "required": ["thought"],
            },
        },
    }
