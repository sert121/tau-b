"""
Cancel reservation tool for inspect_ai - converts tau-bench cancel_reservation tool
"""

import json
from typing import Any, Dict


def cancel_reservation(reservation_id: str, data: Dict[str, Any]) -> str:
    """Cancel the whole reservation.
    
    Args:
        reservation_id: The reservation ID, such as 'ZFA04Y'.
        data: The data dictionary containing reservations information.
    
    Returns:
        JSON string containing the cancelled reservation details, or error message if not found.
    """
    reservations = data.get("reservations", {})
    if reservation_id not in reservations:
        return "Error: reservation not found"
    
    reservation = reservations[reservation_id]
    
    # Reverse the payment
    refunds = []
    for payment in reservation["payment_history"]:
        refunds.append({
            "payment_id": payment["payment_id"],
            "amount": -payment["amount"],
        })
    reservation["payment_history"].extend(refunds)
    reservation["status"] = "cancelled"
    
    return json.dumps(reservation)


def get_cancel_reservation_tool_info() -> Dict[str, Any]:
    """Get tool information for inspect_ai integration."""
    return {
        "type": "function",
        "function": {
            "name": "cancel_reservation",
            "description": "Cancel the whole reservation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reservation_id": {
                        "type": "string",
                        "description": "The reservation ID, such as 'ZFA04Y'.",
                    },
                },
                "required": ["reservation_id"],
            },
        },
    }
