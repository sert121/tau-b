import json
from typing import Any, Dict, List, Optional
from inspect_ai.solver import Solver, TaskState, Generate, solver
from inspect_ai.tool import Tool, ToolInfo, ToolParam, ToolParams
from inspect_ai.model import ChatMessage, ChatMessageSystem, ChatMessageUser
from tau_bench_dataclasses import Action, RESPOND_ACTION_NAME
from envs.base import Env


def message_to_action_inspect(message: ChatMessage) -> Action:  
    """Convert inspect_ai message to tau-bench Action (matching original logic)."""  
    if hasattr(message, 'tool_calls') and message.tool_calls:  
        tool_call = message.tool_calls[0]  
        return Action(  
            name=tool_call.function,  
            kwargs=tool_call.arguments if tool_call.arguments else {}  
        )  
    else:  
        return Action(  
            name=RESPOND_ACTION_NAME,   
            kwargs={"content": message.content}  
        )


def create_tool_param(param_dict: Dict[str, Any]) -> Optional[ToolParam]:
    """Helper function to create ToolParam instances recursively"""
    if not param_dict:
        return None
    
    # Handle nested properties
    properties = {}
    if param_dict.get("properties"):
        properties = {
            key: create_tool_param(value)
            for key, value in param_dict["properties"].items()
            if value is not None
        }
    
    # Handle array items
    items = None
    if param_dict.get("items"):
        items = create_tool_param(param_dict["items"])
    
    return ToolParam(
        type=param_dict.get("type", "string"),
        description=param_dict.get("description"),
        default=param_dict.get("default"),
        enum=param_dict.get("enum"),
        items=items,
        properties=properties,
        additionalProperties=param_dict.get("additionalProperties"),
        required=param_dict.get("required"),
    )


def create_tool_info_from_dict(tool_dict: Dict[str, Any]) -> ToolInfo:
    """
    Create a ToolInfo instance from a dictionary.

    Args:
        tool_dict: Dictionary containing tool information

    Returns:
        ToolInfo instance
    """
    # Extract function information from the tool structure
    if "function" in tool_dict:
        func_info = tool_dict["function"]
        name = func_info.get("name", "")
        description = func_info.get("description", "")
        parameters_dict = func_info.get("parameters", {})
    else:
        # Fallback for direct structure
        name = tool_dict.get("name", "")
        description = tool_dict.get("description", "")
        parameters_dict = tool_dict.get("parameters", {})

    # Create the parameters object
    parameters = None
    if parameters_dict:
        parameters = create_tool_param(parameters_dict)

    # Handle case where parameters is None
    if parameters is None:
        parameters = create_tool_param({"type": "object", "properties": {}, "required": []})

    tool_params = ToolParams(
        properties=parameters.properties,
        required=parameters.required or [],
    )
    # Create and return the ToolInfo instance
    return ToolInfo(
        name=name,
        description=description,
        parameters=tool_params,
    )


@solver  
def tool_calling_agent(  
    env: Env,  
    tools_info: List[Dict[str, Any]],  
    wiki: str,  
    max_steps: int = 30,  
    temperature: float = 0.0  
) -> Solver:  
    """  
    Tool calling agent that replicates tau-bench's ToolCallingAgent logic.  
      
    Args:  
        env: Tau-bench environment instance  
        tools_info: List of tool information dictionaries  
        wiki: System prompt/wiki content for the agent  
        max_steps: Maximum number of steps to take  
        temperature: Temperature for generation  
      
    Returns:  
        Solver that can be used with inspect_ai  
    """  
    async def solve(state: TaskState, generate: Generate) -> TaskState:  
        # Initialize environment (matching tau-bench logic)  
        task_index = state.metadata.get("task_index")  
        env_reset_res = env.reset(task_index=task_index)  
        obs = env_reset_res.observation  
        info = env_reset_res.info.model_dump()  
        reward = 0.0  
          
        # Set up initial messages (matching tau-bench format)  
        state.messages = [  
            ChatMessageSystem(content=wiki),  
            ChatMessageUser(content=obs)  
        ]  
          
        # Convert tau-bench tools to inspect_ai ToolInfo objects  
        tool_infos = []  
        for tool_spec in tools_info:  
            tool_info = create_tool_info_from_dict(tool_spec)  
            tool_infos.append(tool_info)  
            
        state.tools.extend(tool_infos)  
          
        # Main interaction loop (matching tau-bench logic)  
        for _ in range(max_steps):  
            # Generate response with tools  
            await generate(state, tool_calls="none", temperature=temperature)  
              
            # Convert inspect_ai output to tau-bench action  
            action = message_to_action_inspect(state.output.message)  
              
            # Step environment (core tau-bench logic)  
            env_response = env.step(action)  
            reward = env_response.reward  
            info = {**info, **env_response.info.model_dump()}  
              
            # Handle tool call responses (matching tau-bench message handling)  
            if action.name != RESPOND_ACTION_NAME:  
                # Add tool response to conversation  
                state.messages.append(  
                    ChatMessageUser(content=env_response.observation)  
                )  
            else:  
                # Add user response to conversation  
                state.messages.append(  
                    ChatMessageUser(content=env_response.observation)  
                )  
              
            # Store environment info in state  
            state.metadata["env_info"] = info  
            state.metadata["reward"] = reward  
              
            if env_response.done:  
                break  
          
        return state  
      
    return solve  
  
