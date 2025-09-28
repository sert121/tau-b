"""
Scoring functions for tau-bench evaluations.

This module provides various scoring methods that align with tau-bench's
reward-based evaluation system while adapting to inspect_evals patterns.
"""

import json
import re
from typing import Any, Dict, List, Optional, Union
from inspect_ai.scorer import Metric, SampleScore, Score, Scorer, Target, accuracy, metric, scorer
from inspect_ai.solver import TaskState


def extract_actions_from_response(response: str) -> List[Dict[str, Any]]:
    """Extract actions from a model response.
    
    This function looks for tool calls or action patterns in the response.
    """
    actions = []
    
    # Look for JSON-like action patterns
    json_pattern = r'\{[^}]*"name"[^}]*\}'
    matches = re.findall(json_pattern, response)
    
    for match in matches:
        try:
            action = json.loads(match)
            if "name" in action:
                actions.append(action)
        except json.JSONDecodeError:
            continue
    
    # Look for simple action patterns
    action_patterns = [
        r"call\s+(\w+)\s*\(([^)]*)\)",
        r"use\s+(\w+)\s+with\s+([^.]*)",
        r"execute\s+(\w+)"
    ]
    
    for pattern in action_patterns:
        matches = re.findall(pattern, response, re.IGNORECASE)
        for match in matches:
            if len(match) >= 1:
                actions.append({
                    "name": match[0],
                    "arguments": match[1] if len(match) > 1 else {}
                })
    
    return actions


def calculate_action_similarity(
    predicted_actions: List[Dict[str, Any]], 
    expected_actions: List[Dict[str, Any]]
) -> float:
    """Calculate similarity between predicted and expected actions.
    
    Args:
        predicted_actions: Actions predicted by the model
        expected_actions: Ground truth actions
    
    Returns:
        Similarity score between 0 and 1
    """
    if not expected_actions:
        return 1.0 if not predicted_actions else 0.0
    
    if not predicted_actions:
        return 0.0
    
    # Simple name-based matching
    predicted_names = {action.get("name", "") for action in predicted_actions}
    expected_names = {action.get("name", "") for action in expected_actions}
    
    if not expected_names:
        return 1.0
    
    # Calculate Jaccard similarity
    intersection = len(predicted_names & expected_names)
    union = len(predicted_names | expected_names)
    
    return intersection / union if union > 0 else 0.0


def calculate_response_quality(response: str, expected_outputs: List[str]) -> float:
    """Calculate quality of the response based on expected outputs.
    
    Args:
        response: Model response
        expected_outputs: Expected output values
    
    Returns:
        Quality score between 0 and 1
    """
    if not expected_outputs:
        return 1.0  # No specific outputs expected
    
    # Check if response contains expected outputs
    response_lower = response.lower()
    matches = 0
    
    for expected in expected_outputs:
        if str(expected).lower() in response_lower:
            matches += 1
    
    return matches / len(expected_outputs) if expected_outputs else 1.0


def calculate_conversation_quality(
    response: str,
    user_request: str,
    domain: str = "retail"
) -> float:
    """Calculate quality of the conversation response.
    
    Args:
        response: Model response
        user_request: Original user request
        domain: Domain context
    
    Returns:
        Quality score between 0 and 1
    """
    score = 0.0
    
    # Check for politeness indicators
    politeness_indicators = [
        "please", "thank you", "i'd be happy to", "let me help",
        "i understand", "i'll help you", "of course"
    ]
    
    if any(indicator in response.lower() for indicator in politeness_indicators):
        score += 0.3
    
    # Check for domain-specific helpfulness
    if domain == "retail":
        helpful_indicators = [
            "order", "return", "exchange", "refund", "shipping",
            "product", "account", "payment"
        ]
    elif domain == "airline":
        helpful_indicators = [
            "flight", "reservation", "booking", "cancellation",
            "seat", "baggage", "check-in", "boarding"
        ]
    else:
        helpful_indicators = []
    
    if any(indicator in response.lower() for indicator in helpful_indicators):
        score += 0.3
    
    # Check for action-oriented language
    action_indicators = [
        "let me check", "i'll look into", "i can help",
        "i'll process", "i'll handle"
    ]
    
    if any(indicator in response.lower() for indicator in action_indicators):
        score += 0.4
    
    return min(score, 1.0)


@metric
def action_accuracy() -> Metric:
    """Metric for action accuracy."""
    def metric_func(scores: List[SampleScore]) -> float:
        if not scores:
            return 0.0
        
        total_score = sum(score.score.value for score in scores)
        return total_score / len(scores)
    
    return metric_func


@metric
def response_quality() -> Metric:
    """Metric for response quality."""
    def metric_func(scores: List[SampleScore]) -> float:
        if not scores:
            return 0.0
        
        total_score = sum(score.score.value for score in scores)
        return total_score / len(scores)
    
    return metric_func


@metric
def conversation_quality() -> Metric:
    """Metric for conversation quality."""
    def metric_func(scores: List[SampleScore]) -> float:
        if not scores:
            return 0.0
        
        total_score = sum(score.score.value for score in scores)
        return total_score / len(scores)
    
    return metric_func


@scorer(metrics=[accuracy(), action_accuracy(), response_quality()])
def tau_bench_scorer(
    domain: str = "retail",
    scoring_mode: str = "comprehensive"
) -> Scorer:
    """Main scorer for tau-bench evaluations.
    
    Args:
        domain: Domain being evaluated
        scoring_mode: Scoring mode ("comprehensive", "simple", "action_only")
    
    Returns:
        Scorer object
    """
    
    async def score(state: TaskState, target: Target) -> Score:
        # Extract model response
        response = state.output.completion if hasattr(state.output, 'completion') else ""
        
        # Extract expected actions from target
        expected_actions = []
        if hasattr(target, 'target') and target.target:
            if isinstance(target.target, list):
                expected_actions = target.target
            elif isinstance(target.target, dict) and "actions" in target.target:
                expected_actions = target.target["actions"]
        
        # Extract expected outputs
        expected_outputs = []
        if hasattr(target, 'target') and target.target:
            if isinstance(target.target, dict) and "outputs" in target.target:
                expected_outputs = target.target["outputs"]
        
        # Calculate different types of scores
        scores = {}
        
        if scoring_mode in ["comprehensive", "action_only"]:
            # Action-based scoring
            predicted_actions = extract_actions_from_response(response)
            action_score = calculate_action_similarity(predicted_actions, expected_actions)
            scores["action_accuracy"] = action_score
        
        if scoring_mode in ["comprehensive", "simple"]:
            # Response quality scoring
            response_score = calculate_response_quality(response, expected_outputs)
            scores["response_quality"] = response_score
            
            # Conversation quality scoring
            user_request = state.input[-1].text if hasattr(state.input[-1], 'text') else str(state.input[-1])
            conversation_score = calculate_conversation_quality(response, user_request, domain)
            scores["conversation_quality"] = conversation_score
        
        # Calculate overall score
        if scoring_mode == "comprehensive":
            overall_score = (
                scores.get("action_accuracy", 0) * 0.4 +
                scores.get("response_quality", 0) * 0.3 +
                scores.get("conversation_quality", 0) * 0.3
            )
        elif scoring_mode == "action_only":
            overall_score = scores.get("action_accuracy", 0)
        else:  # simple
            overall_score = (
                scores.get("response_quality", 0) * 0.6 +
                scores.get("conversation_quality", 0) * 0.4
            )
        
        return Score(
            value=overall_score,
            answer=response,
            metadata={
                "scores": scores,
                "domain": domain,
                "scoring_mode": scoring_mode,
                "predicted_actions": extract_actions_from_response(response),
                "expected_actions": expected_actions
            }
        )
    
    return score


@scorer(metrics=[accuracy()])
def simple_tau_bench_scorer() -> Scorer:
    """Simple scorer for basic tau-bench evaluations."""
    
    async def score(state: TaskState, target: Target) -> Score:
        response = state.output.completion if hasattr(state.output, 'completion') else ""
        
        # Simple binary scoring: did the model provide a reasonable response?
        if len(response.strip()) > 10:  # Basic length check
            score_value = 1.0
        else:
            score_value = 0.0
        
        return Score(
            value=score_value,
            answer=response,
            metadata={"simple_scoring": True}
        )
    
    return score


@scorer(metrics=[action_accuracy()])
def action_only_scorer() -> Scorer:
    """Scorer that only evaluates action accuracy."""
    
    async def score(state: TaskState, target: Target) -> Score:
        response = state.output.completion if hasattr(state.output, 'completion') else ""
        
        # Extract expected actions
        expected_actions = []
        if hasattr(target, 'target') and target.target:
            if isinstance(target.target, list):
                expected_actions = target.target
            elif isinstance(target.target, dict) and "actions" in target.target:
                expected_actions = target.target["actions"]
        
        # Calculate action similarity
        predicted_actions = extract_actions_from_response(response)
        action_score = calculate_action_similarity(predicted_actions, expected_actions)
        
        return Score(
            value=action_score,
            answer=response,
            metadata={
                "predicted_actions": predicted_actions,
                "expected_actions": expected_actions,
                "action_only": True
            }
        )
    
    return score


def create_domain_specific_scorer(domain: str) -> Scorer:
    """Create a domain-specific scorer.
    
    Args:
        domain: Domain for the scorer
    
    Returns:
        Domain-specific Scorer object
    """
    if domain == "retail":
        return tau_bench_scorer(domain="retail", scoring_mode="comprehensive")
    elif domain == "airline":
        return tau_bench_scorer(domain="airline", scoring_mode="comprehensive")
    else:
        return tau_bench_scorer(domain=domain, scoring_mode="simple")
