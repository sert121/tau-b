"""
Main evaluation functions for tau-bench in inspect_evals format.

This module provides the main evaluation tasks that follow the inspect_evals pattern
while preserving the core functionality of tau-bench.
"""

import json
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from inspect_ai import Task, task
from inspect_ai.solver import system_message, generate, basic_agent
from inspect_ai.tool import Tool
from inspect_ai.scorer import Scorer, Score, Target, scorer, accuracy
from inspect_ai.solver import TaskState

from dataset import tau_bench_dataset, load_tau_bench_tools
from agents import tool_calling_agent
from envs import get_env
from envs.user import UserStrategy

# Task directory for configuration files
TASK_DIR = Path(__file__).parent


def create_tau_bench_env(domain: str, user_strategy: str = "llm", user_model: str = "gpt-4o", task_split: str = "test", user_provider: str = "openai") -> Any:  
    """Create a tau-bench environment for the specified domain."""
    return get_env(  
        env_name=domain,  
        user_strategy=user_strategy,  
        user_model=user_model,  
        task_split=task_split,
        user_provider=user_provider
    )


def load_wiki_content(domain: str) -> str:  
    """Load wiki content for the specified domain."""
    if domain == "retail":  
        from envs.retail.wiki import WIKI  
        return WIKI  
    elif domain == "airline":  
        from envs.airline.wiki import WIKI  
        return WIKI  
    else:  
        raise ValueError(f"Unknown domain: {domain}")


@scorer(metrics=[accuracy()])
def create_tau_bench_scorer() -> Scorer:
    """
    Create a tau-bench compatible scorer that evaluates based on reward and actions.
    
    This scorer follows the original tau-bench evaluation logic:
    1. Check if database changes are correct (actions executed properly)
    2. Check if expected outputs are present in responses
    3. Return reward-based score (0.0 or 1.0)
    """
    async def score(state: TaskState, target: Target) -> Score:
        # Extract reward and info from state metadata (set by the agent)
        reward = state.metadata.get("reward", 0.0)
        env_info = state.metadata.get("env_info", {})
        
        # Determine if the task was completed successfully
        success = reward == 1.0
        
        # Extract additional scoring information
        r_actions = env_info.get("r_actions", False)
        r_outputs = env_info.get("r_outputs", 1.0)
        outputs = env_info.get("outputs", {})
        
        # Create detailed score information
        score_info = {
            "reward": reward,
            "r_actions": r_actions,
            "r_outputs": r_outputs,
            "outputs": outputs,
            "success": success
        }
        
        # Return score with success/failure and detailed info
        return Score(
            value=1.0 if success else 0.0,
            answer=f"Reward: {reward}, Actions: {r_actions}, Outputs: {r_outputs}",
            metadata=score_info
        )
    
    return score


'''
@task
def create_tau_bench_task_retail(
    task_split: str = "test",
    user_strategy: str = "llm", 
    user_model: str = "gpt-4o",
    max_steps: int = 30,
    temperature: float = 0.0
) -> Task:
    """
    Create a retail domain tau-bench task for inspect_ai evaluation.
    """
    dataset = tau_bench_dataset(domain="retail", split=task_split)
    env = create_tau_bench_env("retail", user_strategy, user_model, task_split, user_provider="openai")
    tools_info = load_tau_bench_tools("retail")
    wiki = load_wiki_content("retail")
    solver = tool_calling_agent(env=env, tools_info=tools_info, wiki=wiki, max_steps=max_steps, temperature=temperature)
    scorer = create_tau_bench_scorer()
    return Task(dataset=dataset, solver=solver, scorer=scorer)
'''


@task
def create_tau_bench_task_airline(
    task_split: str = "test",
    user_strategy: str = "llm", 
    user_model: str = "gpt-4o",
    max_steps: int = 30,
    temperature: float = 0.0
) -> Task:  
    """
    Create an airline domain tau-bench task for inspect_ai evaluation.
    
    Args:
        task_split: Data split to use ("train", "test", "dev")
        user_strategy: User simulation strategy
        user_model: Model for user simulation
        max_steps: Maximum number of agent steps
        temperature: Temperature for generation
    
    Returns:
        Task object for inspect_ai evaluation
    """
    # Load dataset  
    dataset = tau_bench_dataset(domain="airline", split=task_split)  
      
    # Create environment and tools
    env = create_tau_bench_env("airline", user_strategy, user_model, task_split, user_provider="openai")
    tools_info = load_tau_bench_tools("airline")  
    wiki = load_wiki_content("airline")
      
    # Create solver using tool calling agent
    solver = tool_calling_agent(
        env=env,
        tools_info=tools_info, 
        wiki=wiki,
        max_steps=max_steps,
        temperature=temperature
    )
      
    # Create scorer
    scorer = create_tau_bench_scorer()
      
    return Task(dataset=dataset, solver=solver, scorer=scorer)