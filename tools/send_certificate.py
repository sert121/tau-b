"""
Send certificate tool for inspect_ai - converts tau-bench send_certificate tool
"""

from typing import Any, Dict


def send_certificate(user_id: str, amount: int, data: Dict[str, Any]) -> str:
    """Send a certificate to a user. Be careful!
    
    Args:
        user_id: The ID of the user to send the certificate to, such as 'sara_doe_496'.
        amount: Certificate amount to send.
        data: The data dictionary containing users information.
    
    Returns:
        Success message with certificate details, or error message if user not found.
    """
    users = data.get("users", {})
    if user_id not in users:
        return "Error: user not found"
    
    user = users[user_id]
    
    # Add a certificate, assume at most 3 cases per task
    for id in [3221322, 3221323, 3221324]:
        payment_id = f"certificate_{id}"
        if payment_id not in user["payment_methods"]:
            user["payment_methods"][payment_id] = {
                "source": "certificate",
                "amount": amount,
                "id": payment_id,
            }
            return f"Certificate {payment_id} added to user {user_id} with amount {amount}."
    
    return "Error: No available certificate slots"


def get_send_certificate_tool_info() -> Dict[str, Any]:
    """Get tool information for inspect_ai integration."""
    return {
        "type": "function",
        "function": {
            "name": "send_certificate",
            "description": "Send a certificate to a user. Be careful!",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "The ID of the user to send the certificate to, such as 'sara_doe_496'.",
                    },
                    "amount": {
                        "type": "number",
                        "description": "Certificate amount to send.",
                    },
                },
                "required": ["user_id", "amount"],
            },
        },
    }
