from typing import Any, Dict, List
from inspect_ai.solver import Solver, TaskState, Generate
from inspect_ai.tool import Tool
from inspect_ai.model import ChatMessage, ChatMessageSystem, ChatMessageUser
from tau_bench_dataclasses import Action, RESPOND_ACTION_NAME
from envs.base import Env

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
            await generate(state, tool_calls="auto", temperature=temperature)  
              
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
  
def message_to_action_inspect(message: ChatMessage) -> Action:  
    """Convert inspect_ai message to tau-bench Action (matching original logic)."""  
    if hasattr(message, 'tool_calls') and message.tool_calls:  
        tool_call = message.tool_calls[0]  
        return Action(  
            name=tool_call.function,  
            kwargs=json.loads(tool_call.arguments) if tool_call.arguments else {}  
        )  
    else:  
        return Action(  
            name=RESPOND_ACTION_NAME,   
            kwargs={"content": message.content}  
        )