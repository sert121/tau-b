"""
Solver implementations for tau-bench evaluations.

This module provides different solver strategies that mirror the tau-bench agent types:
- Tool calling agents
- ReAct agents  
- Act agents
- Few-shot agents
"""

from typing import Any, Dict, List, Optional, Union
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.model import ChatMessage, ChatMessageAssistant, ChatMessageUser
from inspect_ai.tool import Tool

