"""
Dataset loading and sample creation for tau-bench evaluations.

This module handles loading tau-bench data and converting it to inspect_evals format.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from inspect_ai.dataset import Dataset, Sample, MemoryDataset
from inspect_ai.model import ChatMessage, ChatMessageAssistant, ChatMessageSystem, ChatMessageUser
from inspect_ai.solver import Solver, TaskState, Generate, solver
from inspect_ai.tool import Tool, ToolInfo, ToolParam, ToolParams
from tau_bench_dataclasses import Action, RESPOND_ACTION_NAME
from envs.base import Env


def load_tau_bench_data(domain: str, split: str = "test") -> Dict[str, Any]:
    """Load tau-bench data for a specific domain and split.
    
    Args:
        domain: Either "retail" or "airline"
        split: Data split ("train", "dev", "test")
    
    Returns:
        Dictionary containing tasks and environment data
    """
    # Load domain-specific data
    if domain == "retail":
        data_path = Path(__file__).parent / "retail" / "data"
        tasks_path = Path(__file__).parent / "retail" / f"tasks_{split}.py"
    elif domain == "airline":
        data_path = Path(__file__).parent / "airline" / "data"
        if split == "test":
            tasks_path = Path(__file__).parent / "airline" / f"tasks_{split}.py"
        else:
            tasks_path = Path(__file__).parent / "airline" / "tasks.py"

    else:
        raise ValueError(f"Unknown domain: {domain}")
    
    # Load environment data
    data = {}
    if domain == "retail":
        for file_name in ["users.json", "orders.json", "products.json"]:
            file_path = data_path / file_name
            if file_path.exists():
                with open(file_path, 'r') as f:
                    data[file_name.replace('.json', '')] = json.load(f)

    elif domain == "airline":
        for file_name in ["users.json","reservations.json", "flights.json"]:
            file_path = data_path / file_name
            if file_path.exists():
                with open(file_path, 'r') as f:
                    data[file_name.replace('.json', '')] = json.load(f)

    print(len(data))
    
    tasks = load_tau_bench_tasks(domain, split)
    
    return {
        "data": data,
        "tasks": tasks,
        "domain": domain,
        "split": split
    }


def load_tau_bench_tasks(domain: str, split: str) -> List[Dict[str, Any]]:
    """Load tasks from tau-bench tasks files.
    
    Args:
        domain: Domain ("retail" or "airline")
        split: Data split ("train", "dev", "test")
    
    Returns:
        List of task dictionaries
    """
    tasks_path = Path(__file__).parent / domain / f"tasks_{split}.py"
    
    if not tasks_path.exists():
        # Fallback to tasks.py if split-specific file doesn't exist
        tasks_path = Path(__file__).parent / domain / "tasks.py"
    
    if not tasks_path.exists():
        return []
    
    try:
        # Import the tasks module directly
        import importlib.util
        spec = importlib.util.spec_from_file_location(f"{domain}_tasks_{split}", tasks_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Get the tasks list from the module
        if hasattr(module, 'TASKS'):
            return module.TASKS
        elif hasattr(module, 'TASKS_TRAIN'):
            return module.TASKS_TRAIN
        elif hasattr(module, 'TASKS_DEV'):
            return module.TASKS_DEV
        elif hasattr(module, 'TASKS_TEST'):
            return module.TASKS_TEST
        elif hasattr(module, 'tasks'):
            return module.tasks
        else:
            print(f"Warning: No task variable found in {tasks_path}")
            return []
            
    except Exception as e:
        print(f"Warning: Could not load tasks from {tasks_path}: {e}")
        return []


def load_tau_bench_tools(domain: str) -> List[Dict[str, Any]]:
    """Load tool information from tau-bench tools.
    
    Args:
        domain: Domain ("retail" or "airline")
    
    Returns:
        List of tool information dictionaries
    """
    tools_path = Path(__file__).parent / "envs" / domain / "tools"
    
    tools_info = []
    
    if not tools_path.exists():
        return tools_info
    
    import importlib.util
    
    for tool_file in tools_path.glob("*.py"):
        if tool_file.name == "__init__.py":
            continue
            
        try:
            # Load the tool module
            spec = importlib.util.spec_from_file_location(tool_file.stem, tool_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find the tool class and get its info
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (hasattr(attr, 'get_info') and 
                    hasattr(attr, 'invoke') and 
                    callable(getattr(attr, 'get_info')) and
                    not attr_name.startswith('_') and  # Skip private attributes
                    attr_name != 'Tool'):  # Skip the base Tool class
                    # This looks like a tool class
                    try:
                        tool_info = attr.get_info()
                        tools_info.append(tool_info)
                        break
                    except NotImplementedError:
                        # Skip classes that haven't implemented get_info
                        continue
                    
        except Exception as e:
            print(f"Warning: Could not load tool {tool_file.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return tools_info





def create_tool_calling_sample(
    task: Dict[str, Any],
    domain_data: Dict[str, Any],
    available_tools: List[Dict[str, Any]],
    task_index: int = 0
) -> Sample:
    """Create a sample that tests tool calling capabilities.
    
    Args:
        task: Task dictionary
        domain_data: Domain-specific data
        available_tools: List of available tools with their schemas
    
    Returns:
        Sample object for tool calling evaluation
    """
    messages = []
    
    # Add system prompt with tool information
    system_prompt = create_tool_calling_system_prompt(available_tools)
    messages.append(ChatMessageSystem(content=system_prompt))
    
    # Handle both Task objects and dictionaries
    if hasattr(task, 'instruction'):
        # Task object
        instruction = task.instruction
        user_id = task.user_id
        annotator = getattr(task, 'annotator', 0)  # Default to 0 if annotator not available
        actions = task.actions if hasattr(task, 'actions') else []
    else:
        # Dictionary
        instruction = task.get("instruction", "")
        user_id = task.get("user_id", "unknown")
        annotator = task.get("annotator", 0)
        actions = task.get("actions", [])
    
    # Add user instruction
    messages.append(ChatMessageUser(content=instruction))
    
    # Expected tool calls
    expected_tool_calls = []
    for action in actions:
        if hasattr(action, 'name'):
            # Action object
            expected_tool_calls.append({
                "name": action.name,
                "arguments": action.kwargs if hasattr(action, 'kwargs') else {}
            })
        else:
            # Dictionary
            expected_tool_calls.append({
                "name": action.get("name", ""),
                "arguments": action.get("arguments", {})
            })
    
    return Sample(
        id=f"tool_{user_id}_{annotator}_{task_index}",
        input="",
        metadata={
            "domain": "retail",  # Default domain, can be overridden by caller
            "user_id": user_id,
            "task_type": "tool_calling",
            "available_tools": available_tools,
            "expected_actions": actions
        }
    )


def create_system_prompt(domain: str) -> str:
    """Create a system prompt for the given domain."""
    if domain == "retail":
        return """You are a helpful customer service agent for an online retail store. 
You can help customers with orders, returns, exchanges, and account management. 
Always be polite and helpful."""
    elif domain == "airline":
        return """You are a helpful customer service agent for an airline. 
You can help customers with flight bookings, reservations, cancellations, and modifications. 
Always be polite and helpful."""
    else:
        return "You are a helpful customer service agent. Always be polite and helpful."


def create_tool_calling_system_prompt(available_tools: List[Dict[str, Any]]) -> str:
    """Create a system prompt that includes tool information."""
    base_prompt = """You are a helpful customer service agent. You have access to various tools to help customers.
Always use the appropriate tools when needed to fulfill customer requests.

Available tools:"""
    
    for tool in available_tools:
        base_prompt += f"\n- {tool['name']}: {tool.get('description', 'No description available')}"
    
    return base_prompt


def get_type(type_str: str | None) -> str:
    """Convert type string to appropriate type."""
    if type_str is None:
        return "string"  # Default type
    return type_str


def create_tool_param(param_dict: Dict[str, Any] | None) -> ToolParam | None:
    """Helper function to create ToolParam instances recursively"""
    if param_dict is None:
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
        type=get_type(param_dict.get("type")),
        description=param_dict.get("description"),
        default=param_dict.get("default"),
        enum=param_dict.get("enum"),
        items=items,
        properties=properties,  # type: ignore
        additionalProperties=param_dict.get("additionalProperties"),
        required=param_dict.get("required"),
    )



def tau_bench_dataset(
    domain: str = "retail",
    split: str = "test",
    task_type: str = "conversation",
    max_samples: Optional[int] = None,
    shuffle: bool = True,
    seed: Optional[int] = None
) -> Dataset:
    """Create a dataset from tau-bench data.
    
    Args:
        domain: Domain to use ("retail" or "airline")
        split: Data split ("train", "dev", "test")
        task_type: Type of task ("conversation", "simple", "tool_calling")
        max_samples: Maximum number of samples to include
        shuffle: Whether to shuffle the dataset
        seed: Random seed for shuffling
    
    Returns:
        Dataset object for inspect_evals
    """
    # Load tau-bench data
    data = load_tau_bench_data(domain, split)
    tasks = data["tasks"]
    
    if max_samples:
        tasks = tasks[:max_samples]
    
    if shuffle and seed is not None:
        import random
        random.seed(seed)
        random.shuffle(tasks)
    
    # Create samples based on task type
    samples = []
    for i, task in enumerate(tasks):
        try:
            available_tools = []  # Placeholder
            sample = create_tool_calling_sample(task, data["data"], available_tools, i)
        except Exception as e:
            print(f"Warning: Could not create sample for task {e}")
            continue
        samples.append(sample)
    
    return MemoryDataset(samples=samples)
