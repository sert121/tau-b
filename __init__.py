"""
Tau-bench Inspect Evals Integration

This module provides tau-bench evaluations in the inspect_evals format,
allowing tau-bench tasks to be run using the inspect_ai framework.

The module includes:
- Dataset loading and sample creation
- Multiple solver strategies (tool-calling, ReAct, Act, few-shot)
- Comprehensive scoring metrics
- Domain-specific prompts and templates
- Various evaluation task configurations

Main evaluation tasks:
- tau_bench_retail_tool_calling: Retail domain with tool calling
- tau_bench_retail_react: Retail domain with ReAct strategy
- tau_bench_retail_act: Retail domain with Act strategy
- tau_bench_retail_few_shot: Retail domain with few-shot learning
- tau_bench_airline_tool_calling: Airline domain with tool calling
- tau_bench_airline_react: Airline domain with ReAct strategy
- tau_bench_airline_act: Airline domain with Act strategy
- tau_bench_airline_few_shot: Airline domain with few-shot learning

Simplified evaluations:
- tau_bench_retail_simple: Basic retail evaluation
- tau_bench_airline_simple: Basic airline evaluation

Action-focused evaluations:
- tau_bench_retail_actions: Retail action accuracy
- tau_bench_airline_actions: Airline action accuracy

Multi-turn conversation evaluations:
- tau_bench_retail_conversation: Retail multi-turn conversations
- tau_bench_airline_conversation: Airline multi-turn conversations

Cross-domain evaluation:
- tau_bench_cross_domain: Mixed retail and airline domains
"""

from .tau_bench_inspect import (
    # Retail domain evaluations
    tau_bench_retail_tool_calling,
    tau_bench_retail_react,
    tau_bench_retail_act,
    tau_bench_retail_few_shot,
    tau_bench_retail_simple,
    tau_bench_retail_actions,
    tau_bench_retail_conversation,
    
    # Airline domain evaluations
    tau_bench_airline_tool_calling,
    tau_bench_airline_react,
    tau_bench_airline_act,
    tau_bench_airline_few_shot,
    tau_bench_airline_simple,
    tau_bench_airline_actions,
    tau_bench_airline_conversation,
    
    # Cross-domain evaluation
    tau_bench_cross_domain,
)

from .dataset import (
    tau_bench_dataset,
    load_tau_bench_data,
    create_conversation_sample,
    create_simple_task_sample,
    create_tool_calling_sample,
)

from .solver import (
    create_conversation_solver,
    create_multi_turn_solver,
    tool_calling_solver,
    react_solver,
    act_solver,
    few_shot_solver,
)

from .scoring import (
    tau_bench_scorer,
    simple_tau_bench_scorer,
    action_only_scorer,
    create_domain_specific_scorer,
)

from .prompts import (
    get_system_prompt,
    get_few_shot_examples,
    get_user_simulation_prompt,
    get_task_prompt,
    get_domain_knowledge,
    create_conversation_template,
    create_evaluation_prompt,
)

# Version information
__version__ = "0.1.0"
__author__ = "Tau-bench Inspect Integration"
__description__ = "Tau-bench evaluations for inspect_evals framework"

# Export all task functions
__all__ = [
    # Main evaluation tasks
    "tau_bench_retail_tool_calling",
    "tau_bench_retail_react", 
    "tau_bench_retail_act",
    "tau_bench_retail_few_shot",
    "tau_bench_retail_simple",
    "tau_bench_retail_actions",
    "tau_bench_retail_conversation",
    
    "tau_bench_airline_tool_calling",
    "tau_bench_airline_react",
    "tau_bench_airline_act", 
    "tau_bench_airline_few_shot",
    "tau_bench_airline_simple",
    "tau_bench_airline_actions",
    "tau_bench_airline_conversation",
    
    "tau_bench_cross_domain",
    
    # Dataset functions
    "tau_bench_dataset",
    "load_tau_bench_data",
    "create_conversation_sample",
    "create_simple_task_sample", 
    "create_tool_calling_sample",
    
    # Solver functions
    "create_conversation_solver",
    "create_multi_turn_solver",
    "tool_calling_solver",
    "react_solver",
    "act_solver",
    "few_shot_solver",
    
    # Scoring functions
    "tau_bench_scorer",
    "simple_tau_bench_scorer",
    "action_only_scorer",
    "create_domain_specific_scorer",
    
    # Prompt functions
    "get_system_prompt",
    "get_few_shot_examples",
    "get_user_simulation_prompt",
    "get_task_prompt",
    "get_domain_knowledge",
    "create_conversation_template",
    "create_evaluation_prompt",
]
