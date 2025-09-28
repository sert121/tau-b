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


def create_tau_bench_tools(domain: str) -> List[Tool]:
    """Create tool definitions for the given domain.
    
    Args:
        domain: Either "retail" or "airline"
    
    Returns:
        List of Tool objects
    """
    from .dataset import load_tau_bench_tools
    
    # Load actual tool information from tau-bench
    tools_info = load_tau_bench_tools(domain)
    
    # Convert to inspect_ai Tool objects
    tools = []
    for tool_info in tools_info:
        # Create a simple tool wrapper
        tool = create_tool_from_info(tool_info)
        if tool:
            tools.append(tool)
    
    return tools


def create_tool_from_info(tool_info: Dict[str, Any]) -> Optional[Tool]:
    """Create an inspect_ai Tool from tau-bench tool info.
    
    Args:
        tool_info: Tool information dictionary from tau-bench
    
    Returns:
        inspect_ai Tool object or None if creation fails
    """
    try:
        from inspect_ai.tool import Tool
        
        # Extract tool details
        function_info = tool_info.get("function", {})
        name = function_info.get("name", "unknown_tool")
        description = function_info.get("description", "No description")
        parameters = function_info.get("parameters", {})
        
        # Create a simple tool wrapper
        class TauBenchTool(Tool):
            def __init__(self, name: str, description: str, parameters: Dict[str, Any]):
                self.name = name
                self.description = description
                self.parameters = parameters
            
            async def __call__(self, **kwargs):
                # This is a placeholder - in practice, you'd call the actual tau-bench tool
                return f"Tool {self.name} called with {kwargs}"
        
        return TauBenchTool(name, description, parameters)
        
    except Exception as e:
        print(f"Warning: Could not create tool from info: {e}")
        return None


@solver
def tool_calling_solver(
    tools: Optional[List[Tool]] = None,
    max_iterations: int = 10,
    domain: str = "retail"
) -> Solver:
    """Solver that uses tool calling strategy.
    
    This mirrors the tau-bench tool-calling agent that can directly
    call tools to fulfill user requests.
    """
    if tools is None:
        tools = create_tau_bench_tools(domain)
    
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Initialize conversation
        conversation_history = []
        
        # Add system message about available tools
        system_msg = f"""You are a helpful customer service agent for a {domain} company.
You have access to various tools to help customers. Use these tools when appropriate.

Available tools: {[tool.name for tool in tools]}"""
        
        conversation_history.append(ChatMessageUser(content=system_msg))
        
        # Get the user's request
        user_request = state.input[-1].text if hasattr(state.input[-1], 'text') else str(state.input[-1])
        conversation_history.append(ChatMessageUser(content=user_request))
        
        # Simulate tool calling behavior
        for iteration in range(max_iterations):
            # Generate response
            await generate(state)
            
            # Check if we need to call tools
            response = state.output.message.text if hasattr(state.output, 'message') else ""
            
            # Simple heuristic: if response mentions needing to look something up,
            # simulate tool calling
            if any(keyword in response.lower() for keyword in ["let me check", "i'll look", "i need to"]):
                # Simulate tool call
                tool_call_msg = "I'm checking that information for you..."
                conversation_history.append(ChatMessageAssistant(content=tool_call_msg))
                conversation_history.append(ChatMessageUser(content="Please continue."))
            else:
                # Final response
                break
        
        return state
    
    return solve


@solver
def react_solver(
    tools: Optional[List[Tool]] = None,
    max_iterations: int = 10,
    domain: str = "retail"
) -> Solver:
    """Solver that uses ReAct (Reasoning + Acting) strategy.
    
    This mirrors the tau-bench ReAct agent that reasons about what to do
    before taking actions.
    """
    if tools is None:
        tools = create_tau_bench_tools(domain)
    
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # ReAct pattern: Reason -> Act -> Observe
        for iteration in range(max_iterations):
            # Reason: Think about what to do
            reasoning_prompt = f"""Let me think about this step by step.
User request: {state.input[-1].text if hasattr(state.input[-1], 'text') else str(state.input[-1])}

What should I do next?"""
            
            # Act: Take action or provide response
            await generate(state)
            
            # Check if we should continue reasoning
            response = state.output.message.text if hasattr(state.output, 'message') else ""
            
            # Simple stopping condition
            if "final answer" in response.lower() or "that's all" in response.lower():
                break
        
        return state
    
    return solve


@solver
def act_solver(
    tools: Optional[List[Tool]] = None,
    max_iterations: int = 10,
    domain: str = "retail"
) -> Solver:
    """Solver that uses Act strategy.
    
    This mirrors the tau-bench Act agent that focuses on taking actions
    rather than extensive reasoning.
    """
    if tools is None:
        tools = create_tau_bench_tools(domain)
    
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Act pattern: Focus on taking actions
        user_request = state.input[-1].text if hasattr(state.input[-1], 'text') else str(state.input[-1])
        
        # Direct action-oriented response
        action_prompt = f"""I'll help you with that right away.
Request: {user_request}

Let me take the necessary actions:"""
        
        await generate(state)
        return state
    
    return solve


@solver
def few_shot_solver(
    examples: Optional[List[Dict[str, Any]]] = None,
    domain: str = "retail"
) -> Solver:
    """Solver that uses few-shot examples.
    
    This mirrors the tau-bench few-shot agent that learns from examples.
    """
    if examples is None:
        examples = get_few_shot_examples(domain)
    
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Add few-shot examples to context
        context_messages = []
        
        for example in examples:
            context_messages.append(ChatMessageUser(content=example["input"]))
            context_messages.append(ChatMessageAssistant(content=example["output"]))
        
        # Add the current request
        context_messages.append(ChatMessageUser(content=state.input[-1].text if hasattr(state.input[-1], 'text') else str(state.input[-1])))
        
        # Generate response based on examples
        await generate(state)
        return state
    
    return solve


def get_few_shot_examples(domain: str) -> List[Dict[str, Any]]:
    """Get few-shot examples for the given domain."""
    if domain == "retail":
        return [
            {
                "input": "I want to return an item I ordered last week.",
                "output": "I'd be happy to help you with your return. Let me look up your order details."
            },
            {
                "input": "Can you check my order status?",
                "output": "Of course! Let me check the status of your recent orders."
            }
        ]
    elif domain == "airline":
        return [
            {
                "input": "I need to change my flight.",
                "output": "I can help you modify your reservation. Let me look up your booking details."
            },
            {
                "input": "What's the status of my flight?",
                "output": "Let me check the current status of your flight for you."
            }
        ]
    else:
        return []


def create_conversation_solver(
    strategy: str = "tool_calling",
    domain: str = "retail",
    **kwargs
) -> Solver:
    """Create a solver based on the specified strategy.
    
    Args:
        strategy: Solver strategy ("tool_calling", "react", "act", "few_shot")
        domain: Domain for the solver
        **kwargs: Additional arguments for the solver
    
    Returns:
        Configured Solver object
    """
    if strategy == "tool_calling":
        return tool_calling_solver(domain=domain, **kwargs)
    elif strategy == "react":
        return react_solver(domain=domain, **kwargs)
    elif strategy == "act":
        return act_solver(domain=domain, **kwargs)
    elif strategy == "few_shot":
        return few_shot_solver(domain=domain, **kwargs)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def create_multi_turn_solver(
    strategy: str = "tool_calling",
    domain: str = "retail",
    max_turns: int = 5,
    **kwargs
) -> Solver:
    """Create a solver for multi-turn conversations.
    
    This is more complex than single-turn solvers as it needs to
    maintain conversation state and handle multiple exchanges.
    """
    base_solver = create_conversation_solver(strategy, domain, **kwargs)
    
    @solver
    def multi_turn_solver() -> Solver:
        async def solve(state: TaskState, generate: Generate) -> TaskState:
            # Handle multi-turn conversation
            conversation_history = []
            
            # Process each turn
            for turn in range(max_turns):
                # Add current input to conversation
                current_input = state.input[-1] if isinstance(state.input, list) else state.input
                conversation_history.append(current_input)
                
                # Generate response
                await generate(state)
                
                # Add response to conversation
                response = state.output.message if hasattr(state.output, 'message') else None
                if response:
                    conversation_history.append(response)
                
                # Check if conversation should continue
                if should_end_conversation(state, turn):
                    break
            
            return state
        
        return solve
    
    return multi_turn_solver()


def should_end_conversation(state: TaskState, turn: int) -> bool:
    """Determine if the conversation should end."""
    # Simple heuristics for ending conversation
    response = state.output.message.text if hasattr(state.output, 'message') else ""
    
    # End if response indicates completion
    end_indicators = [
        "is there anything else",
        "anything else i can help",
        "have a great day",
        "thank you for calling"
    ]
    
    return any(indicator in response.lower() for indicator in end_indicators) or turn >= 5
