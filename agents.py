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
    temperature: float = 0.0,
    base_model: str = "openai/gpt-4o-mini"
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
                model=base_model, 
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


def create_react_prompt(wiki: str, tools_info: List[Dict[str, Any]], use_reasoning: bool = True) -> str:
    """
    Create a ReAct-style prompt with tools information.
    
    Args:
        wiki: Base system prompt/wiki content
        tools_info: List of tool specifications
        use_reasoning: Whether to include reasoning steps
        
    Returns:
        ReAct prompt string
    """
    prompt = wiki + "\n#Available tools\n" + json.dumps(tools_info)
    
    if use_reasoning:
        prompt += REACT_INSTRUCTION
    else:
        prompt += ACT_INSTRUCTION
        
    return prompt


def parse_react_response(content: str) -> Action:
    """
    Parse ReAct response to extract action.
    
    Args:
        content: The model's response content
        
    Returns:
        Action object for execution
    """
    try:
        # Extract action from "Action:" section
        action_str = content.split("Action:")[-1].strip()
        action_parsed = json.loads(action_str)
        
        # Validate required fields
        if "name" not in action_parsed or "arguments" not in action_parsed:
            raise ValueError("Missing required fields in action")
            
        return Action(
            name=action_parsed["name"],
            kwargs=action_parsed["arguments"]
        )
    except (json.JSONDecodeError, KeyError, ValueError):
        # Fallback to treating as regular response
        return Action(
            name=RESPOND_ACTION_NAME,
            kwargs={"content": content}
        )


@solver  
def chat_react_agent(  
    env: Env,  
    tools_info: List[Dict[str, Any]],  
    wiki: str,  
    max_steps: int = 30,  
    temperature: float = 0.0,
    use_reasoning: bool = True,
    base_model: str = "openai/gpt-4o-mini"
) -> Solver:  
    """  
    ChatReAct agent that uses ReAct reasoning pattern.
      
    Args:  
        env: Tau-bench environment instance  
        tools_info: List of tool information dictionaries  
        wiki: System prompt/wiki content for the agent  
        max_steps: Maximum number of steps to take  
        temperature: Temperature for generation
        use_reasoning: Whether to include reasoning steps
      
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
          
        # Create ReAct prompt
        react_prompt = create_react_prompt(wiki, tools_info, use_reasoning)
        
        # Set up initial messages (matching tau-bench format)  
        state.messages = [  
            ChatMessageSystem(content=react_prompt),  
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
                model=base_model, 
                temperature=temperature,
            )
            next_message = res.choices[0].message.model_dump()
            
            # Parse ReAct response to get action
            action = parse_react_response(next_message["content"])
              
            # Step environment (core tau-bench logic)  
            env_response = env.step(action)  
            reward = env_response.reward  
            info = {**info, **env_response.info.model_dump()}  
              
            # Handle responses (matching tau-bench message handling)  
            obs = env_response.observation
            if action.name != RESPOND_ACTION_NAME:
                obs = "API output: " + obs
                
            messages.extend([
                next_message,
                {"role": "user", "content": obs},
            ])
              
            # Store environment info in state  
            state.metadata["env_info"] = info  
            state.metadata["reward"] = reward  
              
            if env_response.done:  
                break  
          
        return state  
      
    return solve  


# ReAct instruction templates
REACT_INSTRUCTION = """
# Instruction
You need to act as an agent that use the above tools to help the user according to the above policy.

At each step, your generation should have exactly the following format:
Thought:
<A single line of reasoning to process the context and inform the decision making. Do not include extra lines.>
Action:
{"name": <The name of the action>, "arguments": <The arguments to the action in json format>}

The Action will be parsed, so it must be valid JSON.

You should not use made-up or placeholder arguments.

For example, if the user says "I want to know the current weather of San Francisco", and there is such a tool available
{
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the current weather",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "format": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The temperature unit to use. Infer this from the users location.",
                },
            },
            "required": ["location", "format"],
        },
    }
}

Your response can be like this:
Thought:
Since the user asks for the weather of San Francisco in USA, the unit should be in fahrenheit. I can query get_current_weather to get the weather.
Action:
{"name": "get_current_weather", "arguments": {"location": "San Francisco, CA", "format": "fahrenheit"}}

And if the tool returns "70F", your response can be:
Thought:
I can answer the user now.
Action:
{"name": "respond", "arguments": {"content": "The current weather of San Francisco is 70F."}}

Try to be helpful and always follow the policy.
"""


ACT_INSTRUCTION = """
# Instruction
You need to act as an agent that use the above tools to help the user according to the above policy.

At each step, your generation should have exactly the following format:

Action:
{"name": <The name of the action>, "arguments": <The arguments to the action in json format>}

You should not use made-up or placeholder arguments.

The Action will be parsed, so it must be valid JSON.

For example, if the user says "I want to know the current weather of San Francisco", and there is such a tool available
```json
{
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the current weather",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "format": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The temperature unit to use. Infer this from the users location.",
                },
            },
            "required": ["location", "format"],
        },
    }
}
```

Your response can be like this:
Action:
{"name": "get_current_weather", "arguments": {"location": "San Francisco, CA", "format": "fahrenheit"}}

And if the tool returns "70F", your response can be:
Action:
{"name": "respond", "arguments": {"content": "The current weather of San Francisco is 70F."}}

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only.
"""


