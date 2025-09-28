#!/usr/bin/env python3
"""
Example usage of tau-bench inspect evaluations.

This script demonstrates how to run various tau-bench evaluations
using the inspect_ai framework.
"""

import inspect_ai
from tau_bench_inspect import (
    # Retail evaluations
    tau_bench_retail_tool_calling,
    tau_bench_retail_react,
    tau_bench_retail_simple,
    tau_bench_retail_conversation,
    
    # Airline evaluations
    tau_bench_airline_tool_calling,
    tau_bench_airline_react,
    tau_bench_airline_simple,
    
    # Cross-domain
    tau_bench_cross_domain,
    
    # Custom components
    tau_bench_dataset,
    create_conversation_solver,
    tau_bench_scorer
)


def run_basic_evaluation():
    """Run a basic retail evaluation."""
    print("Running basic retail evaluation...")
    
    result = inspect_ai.eval(
        task=tau_bench_retail_simple(
            split="test",
            max_samples=3,
            shuffle=True,
            seed=42
        ),
        model="openai/gpt-4o"
    )
    
    print(f"Basic evaluation results: {result}")
    return result


def run_tool_calling_evaluation():
    """Run a tool-calling evaluation."""
    print("Running retail tool-calling evaluation...")
    
    result = inspect_ai.eval(
        task=tau_bench_retail_tool_calling(
            split="test",
            max_samples=2,
            shuffle=True,
            seed=42
        ),
        model="openai/gpt-4o"
    )
    
    print(f"Tool-calling evaluation results: {result}")
    return result


def run_react_evaluation():
    """Run a ReAct evaluation."""
    print("Running retail ReAct evaluation...")
    
    result = inspect_ai.eval(
        task=tau_bench_retail_react(
            split="test",
            max_samples=2,
            shuffle=True,
            seed=42
        ),
        model="openai/gpt-4o"
    )
    
    print(f"ReAct evaluation results: {result}")
    return result


def run_conversation_evaluation():
    """Run a multi-turn conversation evaluation."""
    print("Running retail conversation evaluation...")
    
    result = inspect_ai.eval(
        task=tau_bench_retail_conversation(
            split="test",
            max_samples=2,
            max_turns=3,
            shuffle=True,
            seed=42
        ),
        model="openai/gpt-4o"
    )
    
    print(f"Conversation evaluation results: {result}")
    return result


def run_airline_evaluation():
    """Run an airline evaluation."""
    print("Running airline tool-calling evaluation...")
    
    result = inspect_ai.eval(
        task=tau_bench_airline_tool_calling(
            split="test",
            max_samples=2,
            shuffle=True,
            seed=42
        ),
        model="openai/gpt-4o"
    )
    
    print(f"Airline evaluation results: {result}")
    return result


def run_cross_domain_evaluation():
    """Run a cross-domain evaluation."""
    print("Running cross-domain evaluation...")
    
    result = inspect_ai.eval(
        task=tau_bench_cross_domain(
            domains=["retail", "airline"],
            split="test",
            max_samples=2,
            shuffle=True,
            seed=42
        ),
        model="openai/gpt-4o"
    )
    
    print(f"Cross-domain evaluation results: {result}")
    return result


def run_custom_evaluation():
    """Run a custom evaluation with custom components."""
    print("Running custom evaluation...")
    
    # Create custom dataset
    dataset = tau_bench_dataset(
        domain="retail",
        split="test",
        task_type="conversation",
        max_samples=2,
        shuffle=True,
        seed=42
    )
    
    # Create custom solver
    solver = create_conversation_solver(
        strategy="react",
        domain="retail",
        max_iterations=5
    )
    
    # Create custom scorer
    scorer = tau_bench_scorer(
        domain="retail",
        scoring_mode="comprehensive"
    )
    
    # Create custom task
    task = inspect_ai.Task(
        dataset=dataset,
        solver=solver,
        scorer=scorer
    )
    
    # Run evaluation
    result = inspect_ai.eval(
        task=task,
        model="openai/gpt-4o"
    )
    
    print(f"Custom evaluation results: {result}")
    return result


def run_comparison_evaluation():
    """Run multiple evaluations and compare results."""
    print("Running comparison evaluation...")
    
    evaluations = [
        ("retail_simple", tau_bench_retail_simple(split="test", max_samples=2)),
        ("retail_react", tau_bench_retail_react(split="test", max_samples=2)),
        ("airline_simple", tau_bench_airline_simple(split="test", max_samples=2)),
    ]
    
    results = {}
    
    for name, task in evaluations:
        print(f"Running {name}...")
        result = inspect_ai.eval(
            task=task,
            model="openai/gpt-4o"
        )
        results[name] = result
        print(f"{name} results: {result}")
    
    return results


def main():
    """Main function to run all evaluations."""
    print("Starting tau-bench inspect evaluations...")
    
    try:
        # Run basic evaluation
        basic_result = run_basic_evaluation()
        
        # Run tool-calling evaluation
        tool_result = run_tool_calling_evaluation()
        
        # Run ReAct evaluation
        react_result = run_react_evaluation()
        
        # Run conversation evaluation
        conv_result = run_conversation_evaluation()
        
        # Run airline evaluation
        airline_result = run_airline_evaluation()
        
        # Run cross-domain evaluation
        cross_result = run_cross_domain_evaluation()
        
        # Run custom evaluation
        custom_result = run_custom_evaluation()
        
        # Run comparison evaluation
        comparison_results = run_comparison_evaluation()
        
        print("\nAll evaluations completed successfully!")
        
        # Print summary
        print("\n=== EVALUATION SUMMARY ===")
        print(f"Basic evaluation: {basic_result}")
        print(f"Tool-calling evaluation: {tool_result}")
        print(f"ReAct evaluation: {react_result}")
        print(f"Conversation evaluation: {conv_result}")
        print(f"Airline evaluation: {airline_result}")
        print(f"Cross-domain evaluation: {cross_result}")
        print(f"Custom evaluation: {custom_result}")
        print(f"Comparison results: {comparison_results}")
        
    except Exception as e:
        print(f"Error running evaluations: {e}")
        raise


if __name__ == "__main__":
    main()
