import json
from typing import Any, Dict, List, Optional
from inspect_ai.solver import Solver, TaskState, Generate, solver
from inspect_ai.model import ChatMessage, ChatMessageSystem, ChatMessageUser
from tau_bench_dataclasses import Action, RESPOND_ACTION_NAME
from envs.base import Env
from litellm import completion


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


def message_to_action_litellm(message: Dict[str, Any]) -> Action:
    """
    Convert litellm message to tau-bench Action (matching original logic).
    
    Args:
        message: Litellm message dictionary
        
    Returns:
        Action object for execution
    """
    if ("tool_calls" in message and 
        message["tool_calls"] is not None and 
        len(message["tool_calls"]) > 0 and 
        message["tool_calls"][0]["function"] is not None):
        tool_call = message["tool_calls"][0]
        return Action(
            name=tool_call["function"]["name"],
            kwargs=json.loads(tool_call["function"]["arguments"]),
        )
    else:
        return Action(name=RESPOND_ACTION_NAME, kwargs={"content": message["content"]})




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
          
        # Convert inspect_ai messages to litellm format
        messages = []
        for msg in state.messages:
            if isinstance(msg, ChatMessageSystem):
                messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, ChatMessageUser):
                messages.append({"role": "user", "content": msg.content})
        
        # Main interaction loop (matching tau-bench logic)  
        for _ in range(max_steps):  
            # Use litellm completion like original implementation
            res = completion(
                messages=messages,
                model="openai/gpt-4o-mini",  # You might want to make this configurable
                tools=tools_info,
                temperature=temperature,
            )
            next_message = res.choices[0].message.model_dump()
            
            # Convert to tau-bench action using original logic
            action = message_to_action_litellm(next_message)
              
            # Step environment (core tau-bench logic)  
            env_response = env.step(action)  
            reward = env_response.reward  
            info = {**info, **env_response.info.model_dump()}  
              
            # Handle tool call responses (matching tau-bench message handling)  
            if action.name != RESPOND_ACTION_NAME:  
                next_message["tool_calls"] = next_message["tool_calls"][:1]
                messages.extend([
                    next_message,
                    {
                        "role": "tool",
                        "tool_call_id": next_message["tool_calls"][0]["id"],
                        "name": next_message["tool_calls"][0]["function"]["name"],
                        "content": env_response.observation,
                    },
                ])
            else:  
                messages.extend([
                    next_message,
                    {"role": "user", "content": env_response.observation},
                ])
              
            # Store environment info in state  
            state.metadata["env_info"] = info  
            state.metadata["reward"] = reward  
              
            if env_response.done:  
                break  
          
        return state  
      
    return solve  
  
