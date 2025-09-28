"""
Integration with actual tau-bench prompts.

This module shows how to properly integrate with tau-bench's actual prompts
instead of using generic placeholders.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional

# Path to tau-bench repository
TAU_BENCH_PATH = Path(__file__).parent.parent / "tau-bench" / "tau_bench"


def load_tau_bench_wiki(domain: str) -> str:
    """Load the actual wiki content from tau-bench.
    
    Args:
        domain: "retail" or "airline"
    
    Returns:
        Wiki content as string
    """
    wiki_path = TAU_BENCH_PATH / "envs" / domain / "wiki.md"
    if wiki_path.exists():
        with open(wiki_path, 'r') as f:
            return f.read()
    return ""


def load_tau_bench_tools_info(domain: str) -> List[Dict[str, Any]]:
    """Load the actual tools information from tau-bench.
    
    Args:
        domain: "retail" or "airline"
    
    Returns:
        List of tool information dictionaries
    """
    # This would need to be implemented by importing the actual tool classes
    # from tau-bench and extracting their info
    tools_path = TAU_BENCH_PATH / "envs" / domain / "tools"
    tools_info = []
    
    if tools_path.exists():
        for tool_file in tools_path.glob("*.py"):
            if tool_file.name != "__init__.py":
                # This would need to import and extract tool info
                # For now, return empty list
                pass
    
    return []


def get_tau_bench_react_instruction() -> str:
    """Get the actual REACT instruction from tau-bench."""
    return """
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


def get_tau_bench_act_instruction() -> str:
    """Get the actual ACT instruction from tau-bench."""
    return """
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


def get_tau_bench_user_system_prompt(instruction: Optional[str] = None) -> str:
    """Get the actual user simulation system prompt from tau-bench."""
    instruction_display = (
        ("\n\nInstruction: " + instruction + "\n")
        if instruction is not None
        else ""
    )
    return f"""You are a user interacting with an agent.{instruction_display}
Rules:
- Just generate one line at a time to simulate the user's message.
- Do not give away all the instruction at once. Only provide the information that is necessary for the current step.
- Do not hallucinate information that is not provided in the instruction. For example, if the agent asks for the order id but it is not mentioned in the instruction, do not make up an order id, just say you do not remember or have it.
- If the instruction goal is satisified, generate '###STOP###' as a standalone message without anything else to end the conversation.
- Do not repeat the exact instruction in the conversation. Instead, use your own words to convey the same information.
- Try to make the conversation as natural as possible, and stick to the personalities in the instruction."""


def get_tau_bench_react_user_system_prompt(instruction: Optional[str] = None) -> str:
    """Get the actual ReAct user simulation system prompt from tau-bench."""
    instruction_display = (
        ("\n\nInstruction: " + instruction + "\n")
        if instruction is not None
        else ""
    )
    return f"""You are a user interacting with an agent.{instruction_display}
Rules:
- First, generate a Thought about what to do next (this message will not be sent to the agent).
- Then, generate a one line User Response to simulate the user's message (this message will be sent to the agent).
- Do not give away all the instruction at once. Only provide the information that is necessary for the current step.
- Do not hallucinate information that is not provided in the instruction. For example, if the agent asks for the order id but it is not mentioned in the instruction, do not make up an order id, just say you do not remember or have it.
- If the instruction goal is satisified, generate '###STOP###' as the User Response without anything else to end the conversation.
- Do not repeat the exact instruction in the conversation. Instead, use your own words to convey the same information.
- Try to make the conversation as natural as possible, and stick to the personalities in the instruction.

Format:

Thought:
<the thought>

User Response:
<the user response (this will be parsed and sent to the agent)>"""


def create_tau_bench_agent_prompt(
    domain: str,
    strategy: str,
    tools_info: List[Dict[str, Any]],
    wiki: str
) -> str:
    """Create the actual agent prompt used in tau-bench.
    
    Args:
        domain: "retail" or "airline"
        strategy: "tool_calling", "react", "act", "few_shot"
        tools_info: List of tool information dictionaries
        wiki: Wiki content
    
    Returns:
        Complete agent prompt
    """
    if strategy == "react":
        instruction = get_tau_bench_react_instruction()
    elif strategy == "act":
        instruction = get_tau_bench_act_instruction()
    else:
        instruction = get_tau_bench_act_instruction()  # Default to ACT
    
    return f"{wiki}\n#Available tools\n{json.dumps(tools_info, indent=2)}{instruction}"


def create_tau_bench_few_shot_prompt(
    domain: str,
    tools_info: List[Dict[str, Any]],
    wiki: str,
    few_shot_displays: List[str],
    num_few_shots: int = 5
) -> str:
    """Create the actual few-shot prompt used in tau-bench.
    
    Args:
        domain: "retail" or "airline"
        tools_info: List of tool information dictionaries
        wiki: Wiki content
        few_shot_displays: List of few-shot example displays
        num_few_shots: Number of few-shot examples to include
    
    Returns:
        Complete few-shot prompt
    """
    import random
    
    # Sample few-shot examples
    sampled_few_shot_displays = random.sample(few_shot_displays, min(num_few_shots, len(few_shot_displays)))
    few_shots = "\n\n".join([f"Example {i+1}:\n{display}" for i, display in enumerate(sampled_few_shot_displays)])
    
    return f"{wiki}\n\n{few_shots}"


def load_tau_bench_few_shot_displays(domain: str) -> List[str]:
    """Load few-shot displays from tau-bench.
    
    Args:
        domain: "retail" or "airline"
    
    Returns:
        List of few-shot display strings
    """
    # This would need to load from the few_shot_data directory
    few_shot_path = Path(__file__).parent.parent / "tau-bench" / "few_shot_data"
    domain_file = f"MockRetailDomainEnv-few_shot.jsonl" if domain == "retail" else f"MockAirlineDomainEnv-few_shot.jsonl"
    
    displays = []
    file_path = few_shot_path / domain_file
    
    if file_path.exists():
        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                displays.append(data.get("display", ""))
    
    return displays


# Example usage of proper tau-bench integration
def create_proper_tau_bench_system_prompt(
    domain: str,
    strategy: str = "tool_calling"
) -> str:
    """Create a system prompt using actual tau-bench components.
    
    Args:
        domain: "retail" or "airline"
        strategy: Agent strategy
    
    Returns:
        Complete system prompt
    """
    # Load actual tau-bench components
    wiki = load_tau_bench_wiki(domain)
    tools_info = load_tau_bench_tools_info(domain)
    
    if strategy == "few_shot":
        few_shot_displays = load_tau_bench_few_shot_displays(domain)
        return create_tau_bench_few_shot_prompt(domain, tools_info, wiki, few_shot_displays)
    else:
        return create_tau_bench_agent_prompt(domain, strategy, tools_info, wiki)


def create_proper_tau_bench_user_prompt(
    instruction: str,
    user_strategy: str = "llm"
) -> str:
    """Create a user simulation prompt using actual tau-bench components.
    
    Args:
        instruction: User instruction
        user_strategy: User simulation strategy
    
    Returns:
        User simulation prompt
    """
    if user_strategy == "react":
        return get_tau_bench_react_user_system_prompt(instruction)
    else:
        return get_tau_bench_user_system_prompt(instruction)
