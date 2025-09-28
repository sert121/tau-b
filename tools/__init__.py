"""
Tools package for tau_bench_inspect - converted airline tools for inspect_ai
"""

from .calculate import calculate, get_calculate_tool_info
from .list_all_airports import list_all_airports, get_list_all_airports_tool_info
from .search_direct_flight import search_direct_flight, get_search_direct_flight_tool_info
from .search_onestop_flight import search_onestop_flight, get_search_onestop_flight_tool_info
from .get_user_details import get_user_details, get_get_user_details_tool_info
from .get_reservation_details import get_reservation_details, get_get_reservation_details_tool_info
from .cancel_reservation import cancel_reservation, get_cancel_reservation_tool_info
from .book_reservation import book_reservation, get_book_reservation_tool_info
from .think import think, get_think_tool_info
from .send_certificate import send_certificate, get_send_certificate_tool_info
from .transfer_to_human_agents import transfer_to_human_agents, get_transfer_to_human_agents_tool_info

# Registry of all available tools
AIRLINE_TOOLS = {
    "calculate": {
        "function": calculate,
        "info": get_calculate_tool_info(),
    },
    "list_all_airports": {
        "function": list_all_airports,
        "info": get_list_all_airports_tool_info(),
    },
    "search_direct_flight": {
        "function": search_direct_flight,
        "info": get_search_direct_flight_tool_info(),
    },
    "search_onestop_flight": {
        "function": search_onestop_flight,
        "info": get_search_onestop_flight_tool_info(),
    },
    "get_user_details": {
        "function": get_user_details,
        "info": get_get_user_details_tool_info(),
    },
    "get_reservation_details": {
        "function": get_reservation_details,
        "info": get_get_reservation_details_tool_info(),
    },
    "cancel_reservation": {
        "function": cancel_reservation,
        "info": get_cancel_reservation_tool_info(),
    },
    "book_reservation": {
        "function": book_reservation,
        "info": get_book_reservation_tool_info(),
    },
    "think": {
        "function": think,
        "info": get_think_tool_info(),
    },
    "send_certificate": {
        "function": send_certificate,
        "info": get_send_certificate_tool_info(),
    },
    "transfer_to_human_agents": {
        "function": transfer_to_human_agents,
        "info": get_transfer_to_human_agents_tool_info(),
    },
}


def get_airline_tools():
    """Get all airline tools for inspect_ai integration."""
    return AIRLINE_TOOLS


def get_airline_tool_info():
    """Get tool information for all airline tools."""
    return [tool["info"] for tool in AIRLINE_TOOLS.values()]


__all__ = [
    "AIRLINE_TOOLS",
    "get_airline_tools",
    "get_airline_tool_info",
    # Individual tools
    "calculate",
    "list_all_airports", 
    "search_direct_flight",
    "search_onestop_flight",
    "get_user_details",
    "get_reservation_details",
    "cancel_reservation",
    "book_reservation",
    "think",
    "send_certificate",
    "transfer_to_human_agents",
]
