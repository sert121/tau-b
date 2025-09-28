"""
Calculate tool for inspect_ai - converts tau-bench calculate tool
"""

from typing import Any, Dict
from inspect_ai.tool import Tool, tool  

@tool
def calculate() -> Tool:  
    async def run(expression: str) -> str:  
        # Your existing calculate logic here  
        if not all(char in "0123456789+-*/(). " for char in expression):  
            return "Error: invalid characters in expression"  
        try:  
            return str(round(float(eval(expression, {"__builtins__": None}, {})), 2))  
        except Exception as e:  
            return f"Error: {e}"  
      
    return run


def get_calculate_tool_info() -> Dict[str, Any]:
    """Get tool information for inspect_ai integration."""
    return {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Calculate the result of a mathematical expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to calculate, such as '2 + 2'. The expression can contain numbers, operators (+, -, *, /), parentheses, and spaces.",
                    },
                },
                "required": ["expression"],
            },
        },
    }
