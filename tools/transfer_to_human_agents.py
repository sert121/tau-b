"""
Transfer to human agents tool for inspect_ai - converts tau-bench transfer_to_human_agents tool
"""

from typing import Any, Dict


def transfer_to_human_agents(summary: str) -> str:
    """Transfer the user to a human agent, with a summary of the user's issue. Only transfer if the user explicitly asks for a human agent, or if the user's issue cannot be resolved by the agent with the available tools.
    
    Args:
        summary: A summary of the user's issue.
    
    Returns:
        Transfer confirmation message.
    """
    return "Transfer successful"


def get_transfer_to_human_agents_tool_info() -> Dict[str, Any]:
    """Get tool information for inspect_ai integration."""
    return {
        "type": "function",
        "function": {
            "name": "transfer_to_human_agents",
            "description": "Transfer the user to a human agent, with a summary of the user's issue. Only transfer if the user explicitly asks for a human agent, or if the user's issue cannot be resolved by the agent with the available tools.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "A summary of the user's issue.",
                    },
                },
                "required": ["summary"],
            },
        },
    }
