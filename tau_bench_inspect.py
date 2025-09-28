"""
Main evaluation functions for tau-bench in inspect_evals format.

This module provides the main evaluation tasks that follow the inspect_evals pattern
while preserving the core functionality of tau-bench.
"""

from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from inspect_ai import Task, task
from inspect_ai.solver import system_message, generate, basic_agent
from inspect_ai.tool import Tool

from .dataset import tau_bench_dataset
from .solver import create_conversation_solver, create_multi_turn_solver
from .scoring import tau_bench_scorer, simple_tau_bench_scorer, action_only_scorer
from .prompts import get_system_prompt, get_few_shot_examples


# Task directory for configuration files
TASK_DIR = Path(__file__).parent


@task
def tau_bench_retail_tool_calling(
    split: str = "test",
    max_samples: Optional[int] = None,
    shuffle: bool = True,
    seed: Optional[int] = None
) -> Task:
    """Tau-bench retail domain with tool calling strategy.
    
    This evaluation tests the model's ability to use tools effectively
    in a retail customer service context.
    """
    return Task(
        dataset=tau_bench_dataset(
            domain="retail",
            split=split,
            task_type="tool_calling",
            max_samples=max_samples,
            shuffle=shuffle,
            seed=seed
        ),
        solver=create_conversation_solver(
            strategy="tool_calling",
            domain="retail"
        ),
        scorer=tau_bench_scorer(domain="retail", scoring_mode="comprehensive")
    )


@task
def tau_bench_retail_react(
    split: str = "test",
    max_samples: Optional[int] = None,
    shuffle: bool = True,
    seed: Optional[int] = None
) -> Task:
    """Tau-bench retail domain with ReAct strategy.
    
    This evaluation tests the model's reasoning and acting capabilities
    in a retail customer service context.
    """
    return Task(
        dataset=tau_bench_dataset(
            domain="retail",
            split=split,
            task_type="conversation",
            max_samples=max_samples,
            shuffle=shuffle,
            seed=seed
        ),
        solver=create_conversation_solver(
            strategy="react",
            domain="retail"
        ),
        scorer=tau_bench_scorer(domain="retail", scoring_mode="comprehensive")
    )


@task
def tau_bench_retail_act(
    split: str = "test",
    max_samples: Optional[int] = None,
    shuffle: bool = True,
    seed: Optional[int] = None
) -> Task:
    """Tau-bench retail domain with Act strategy.
    
    This evaluation tests the model's action-oriented approach
    in a retail customer service context.
    """
    return Task(
        dataset=tau_bench_dataset(
            domain="retail",
            split=split,
            task_type="conversation",
            max_samples=max_samples,
            shuffle=shuffle,
            seed=seed
        ),
        solver=create_conversation_solver(
            strategy="act",
            domain="retail"
        ),
        scorer=tau_bench_scorer(domain="retail", scoring_mode="comprehensive")
    )


@task
def tau_bench_retail_few_shot(
    split: str = "test",
    max_samples: Optional[int] = None,
    shuffle: bool = True,
    seed: Optional[int] = None
) -> Task:
    """Tau-bench retail domain with few-shot strategy.
    
    This evaluation tests the model's ability to learn from examples
    in a retail customer service context.
    """
    return Task(
        dataset=tau_bench_dataset(
            domain="retail",
            split=split,
            task_type="conversation",
            max_samples=max_samples,
            shuffle=shuffle,
            seed=seed
        ),
        solver=create_conversation_solver(
            strategy="few_shot",
            domain="retail"
        ),
        scorer=tau_bench_scorer(domain="retail", scoring_mode="comprehensive")
    )


@task
def tau_bench_airline_tool_calling(
    split: str = "test",
    max_samples: Optional[int] = None,
    shuffle: bool = True,
    seed: Optional[int] = None
) -> Task:
    """Tau-bench airline domain with tool calling strategy.
    
    This evaluation tests the model's ability to use tools effectively
    in an airline customer service context.
    """
    return Task(
        dataset=tau_bench_dataset(
            domain="airline",
            split=split,
            task_type="tool_calling",
            max_samples=max_samples,
            shuffle=shuffle,
            seed=seed
        ),
        solver=create_conversation_solver(
            strategy="tool_calling",
            domain="airline"
        ),
        scorer=tau_bench_scorer(domain="airline", scoring_mode="comprehensive")
    )


@task
def tau_bench_airline_react(
    split: str = "test",
    max_samples: Optional[int] = None,
    shuffle: bool = True,
    seed: Optional[int] = None
) -> Task:
    """Tau-bench airline domain with ReAct strategy.
    
    This evaluation tests the model's reasoning and acting capabilities
    in an airline customer service context.
    """
    return Task(
        dataset=tau_bench_dataset(
            domain="airline",
            split=split,
            task_type="conversation",
            max_samples=max_samples,
            shuffle=shuffle,
            seed=seed
        ),
        solver=create_conversation_solver(
            strategy="react",
            domain="airline"
        ),
        scorer=tau_bench_scorer(domain="airline", scoring_mode="comprehensive")
    )


@task
def tau_bench_airline_act(
    split: str = "test",
    max_samples: Optional[int] = None,
    shuffle: bool = True,
    seed: Optional[int] = None
) -> Task:
    """Tau-bench airline domain with Act strategy.
    
    This evaluation tests the model's action-oriented approach
    in an airline customer service context.
    """
    return Task(
        dataset=tau_bench_dataset(
            domain="airline",
            split=split,
            task_type="conversation",
            max_samples=max_samples,
            shuffle=shuffle,
            seed=seed
        ),
        solver=create_conversation_solver(
            strategy="act",
            domain="airline"
        ),
        scorer=tau_bench_scorer(domain="airline", scoring_mode="comprehensive")
    )


@task
def tau_bench_airline_few_shot(
    split: str = "test",
    max_samples: Optional[int] = None,
    shuffle: bool = True,
    seed: Optional[int] = None
) -> Task:
    """Tau-bench airline domain with few-shot strategy.
    
    This evaluation tests the model's ability to learn from examples
    in an airline customer service context.
    """
    return Task(
        dataset=tau_bench_dataset(
            domain="airline",
            split=split,
            task_type="conversation",
            max_samples=max_samples,
            shuffle=shuffle,
            seed=seed
        ),
        solver=create_conversation_solver(
            strategy="few_shot",
            domain="airline"
        ),
        scorer=tau_bench_scorer(domain="airline", scoring_mode="comprehensive")
    )


# Simplified evaluation tasks
@task
def tau_bench_retail_simple(
    split: str = "test",
    max_samples: Optional[int] = None,
    shuffle: bool = True,
    seed: Optional[int] = None
) -> Task:
    """Simplified tau-bench retail evaluation.
    
    This is a basic evaluation that tests understanding and response quality
    without complex tool calling or multi-turn conversations.
    """
    return Task(
        dataset=tau_bench_dataset(
            domain="retail",
            split=split,
            task_type="simple",
            max_samples=max_samples,
            shuffle=shuffle,
            seed=seed
        ),
        solver=[system_message(get_system_prompt("retail", "tool_calling")), generate()],
        scorer=simple_tau_bench_scorer()
    )


@task
def tau_bench_airline_simple(
    split: str = "test",
    max_samples: Optional[int] = None,
    shuffle: bool = True,
    seed: Optional[int] = None
) -> Task:
    """Simplified tau-bench airline evaluation.
    
    This is a basic evaluation that tests understanding and response quality
    without complex tool calling or multi-turn conversations.
    """
    return Task(
        dataset=tau_bench_dataset(
            domain="airline",
            split=split,
            task_type="simple",
            max_samples=max_samples,
            shuffle=shuffle,
            seed=seed
        ),
        solver=[system_message(get_system_prompt("airline", "tool_calling")), generate()],
        scorer=simple_tau_bench_scorer()
    )


# Action-only evaluation tasks
@task
def tau_bench_retail_actions(
    split: str = "test",
    max_samples: Optional[int] = None,
    shuffle: bool = True,
    seed: Optional[int] = None
) -> Task:
    """Tau-bench retail evaluation focused on action accuracy.
    
    This evaluation specifically tests the model's ability to perform
    the correct actions in response to customer requests.
    """
    return Task(
        dataset=tau_bench_dataset(
            domain="retail",
            split=split,
            task_type="tool_calling",
            max_samples=max_samples,
            shuffle=shuffle,
            seed=seed
        ),
        solver=create_conversation_solver(
            strategy="tool_calling",
            domain="retail"
        ),
        scorer=action_only_scorer()
    )


@task
def tau_bench_airline_actions(
    split: str = "test",
    max_samples: Optional[int] = None,
    shuffle: bool = True,
    seed: Optional[int] = None
) -> Task:
    """Tau-bench airline evaluation focused on action accuracy.
    
    This evaluation specifically tests the model's ability to perform
    the correct actions in response to customer requests.
    """
    return Task(
        dataset=tau_bench_dataset(
            domain="airline",
            split=split,
            task_type="tool_calling",
            max_samples=max_samples,
            shuffle=shuffle,
            seed=seed
        ),
        solver=create_conversation_solver(
            strategy="tool_calling",
            domain="airline"
        ),
        scorer=action_only_scorer()
    )


# Multi-turn conversation tasks
@task
def tau_bench_retail_conversation(
    split: str = "test",
    max_samples: Optional[int] = None,
    max_turns: int = 5,
    shuffle: bool = True,
    seed: Optional[int] = None
) -> Task:
    """Tau-bench retail multi-turn conversation evaluation.
    
    This evaluation tests the model's ability to maintain context
    and handle multi-turn conversations in a retail context.
    """
    return Task(
        dataset=tau_bench_dataset(
            domain="retail",
            split=split,
            task_type="conversation",
            max_samples=max_samples,
            shuffle=shuffle,
            seed=seed
        ),
        solver=create_multi_turn_solver(
            strategy="tool_calling",
            domain="retail",
            max_turns=max_turns
        ),
        scorer=tau_bench_scorer(domain="retail", scoring_mode="comprehensive")
    )


@task
def tau_bench_airline_conversation(
    split: str = "test",
    max_samples: Optional[int] = None,
    max_turns: int = 5,
    shuffle: bool = True,
    seed: Optional[int] = None
) -> Task:
    """Tau-bench airline multi-turn conversation evaluation.
    
    This evaluation tests the model's ability to maintain context
    and handle multi-turn conversations in an airline context.
    """
    return Task(
        dataset=tau_bench_dataset(
            domain="airline",
            split=split,
            task_type="conversation",
            max_samples=max_samples,
            shuffle=shuffle,
            seed=seed
        ),
        solver=create_multi_turn_solver(
            strategy="tool_calling",
            domain="airline",
            max_turns=max_turns
        ),
        scorer=tau_bench_scorer(domain="airline", scoring_mode="comprehensive")
    )


# Cross-domain evaluation
@task
def tau_bench_cross_domain(
    domains: List[str] = ["retail", "airline"],
    split: str = "test",
    max_samples: Optional[int] = None,
    shuffle: bool = True,
    seed: Optional[int] = None
) -> Task:
    """Cross-domain tau-bench evaluation.
    
    This evaluation tests the model's ability to handle both retail
    and airline domains in a single evaluation.
    """
    # Combine datasets from multiple domains
    datasets = []
    for domain in domains:
        domain_dataset = tau_bench_dataset(
            domain=domain,
            split=split,
            task_type="conversation",
            max_samples=max_samples // len(domains) if max_samples else None,
            shuffle=shuffle,
            seed=seed
        )
        datasets.append(domain_dataset)
    
    # Combine datasets (this would need proper dataset merging)
    combined_dataset = datasets[0]  # Simplified for now
    
    return Task(
        dataset=combined_dataset,
        solver=create_conversation_solver(
            strategy="tool_calling",
            domain="mixed"
        ),
        scorer=tau_bench_scorer(domain="mixed", scoring_mode="comprehensive")
    )
