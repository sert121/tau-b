# Tau-bench Inspect Evals Integration

This module provides a scaffolding for integrating tau-bench evaluations into the inspect_evals framework, allowing tau-bench tasks to be run using the inspect_ai evaluation system.

## Overview

The tau-bench evaluation framework has been adapted to work with inspect_evals while preserving its core functionality. This integration provides:

- **Multiple evaluation strategies**: Tool-calling, ReAct, Act, and few-shot learning
- **Domain support**: Both retail and airline customer service domains
- **Flexible scoring**: Comprehensive, simple, and action-focused scoring modes
- **Multi-turn conversations**: Support for complex conversation flows
- **Cross-domain evaluation**: Mixed domain testing capabilities

## Architecture

### Key Components

1. **Dataset Module** (`dataset.py`): Handles loading tau-bench data and converting it to inspect_evals format
2. **Solver Module** (`solver.py`): Implements different agent strategies (tool-calling, ReAct, Act, few-shot)
3. **Scoring Module** (`scoring.py`): Provides comprehensive scoring metrics aligned with tau-bench's reward system
4. **Prompts Module** (`prompts.py`): Contains system prompts, templates, and domain-specific knowledge
5. **Main Evaluation** (`tau_bench_inspect.py`): Core evaluation tasks following inspect_evals patterns

### Design Philosophy

The scaffolding maintains the core tau-bench functionality while adapting to inspect_evals patterns:

- **Preserves tau-bench's simulation-based approach** through conversation samples
- **Maintains tool-calling capabilities** with proper tool integration
- **Supports multiple agent strategies** as in the original tau-bench
- **Provides flexible evaluation modes** from simple to comprehensive

## Available Evaluations

### Retail Domain Evaluations

- `tau_bench_retail_tool_calling`: Tool-calling strategy for retail customer service
- `tau_bench_retail_react`: ReAct strategy for retail customer service  
- `tau_bench_retail_act`: Act strategy for retail customer service
- `tau_bench_retail_few_shot`: Few-shot learning for retail customer service
- `tau_bench_retail_simple`: Simplified retail evaluation
- `tau_bench_retail_actions`: Action accuracy focused evaluation
- `tau_bench_retail_conversation`: Multi-turn conversation evaluation

### Airline Domain Evaluations

- `tau_bench_airline_tool_calling`: Tool-calling strategy for airline customer service
- `tau_bench_airline_react`: ReAct strategy for airline customer service
- `tau_bench_airline_act`: Act strategy for airline customer service  
- `tau_bench_airline_few_shot`: Few-shot learning for airline customer service
- `tau_bench_airline_simple`: Simplified airline evaluation
- `tau_bench_airline_actions`: Action accuracy focused evaluation
- `tau_bench_airline_conversation`: Multi-turn conversation evaluation

### Cross-Domain Evaluations

- `tau_bench_cross_domain`: Mixed retail and airline domain evaluation

## Usage Examples

### Basic Evaluation

```python
from tau_bench_inspect import tau_bench_retail_tool_calling

# Run retail tool-calling evaluation
task = tau_bench_retail_tool_calling(
    split="test",
    max_samples=10,
    shuffle=True,
    seed=42
)
```

### Custom Configuration

```python
from tau_bench_inspect import (
    tau_bench_dataset,
    create_conversation_solver,
    tau_bench_scorer
)

# Create custom dataset
dataset = tau_bench_dataset(
    domain="retail",
    split="test", 
    task_type="conversation",
    max_samples=5
)

# Create custom solver
solver = create_conversation_solver(
    strategy="react",
    domain="retail"
)

# Create custom scorer
scorer = tau_bench_scorer(
    domain="retail",
    scoring_mode="comprehensive"
)
```

### Multi-turn Conversations

```python
from tau_bench_inspect import tau_bench_retail_conversation

# Run multi-turn conversation evaluation
task = tau_bench_retail_conversation(
    split="test",
    max_samples=5,
    max_turns=3,
    shuffle=True
)
```

## Scoring Modes

### Comprehensive Scoring

- **Action Accuracy**: Measures correctness of tool calls and actions
- **Response Quality**: Evaluates response relevance and helpfulness  
- **Conversation Quality**: Assesses politeness and domain knowledge

### Simple Scoring

- **Basic Response Quality**: Simple length and content checks
- **Understanding**: Tests basic comprehension of requests

### Action-Only Scoring

- **Action Accuracy**: Focuses solely on correct action execution
- **Tool Usage**: Evaluates appropriate tool selection and usage

## Configuration Options

### Dataset Options

- `domain`: "retail" or "airline"
- `split`: "train", "dev", or "test"
- `task_type`: "conversation", "simple", or "tool_calling"
- `max_samples`: Maximum number of samples to include
- `shuffle`: Whether to shuffle the dataset
- `seed`: Random seed for reproducibility

### Solver Options

- `strategy`: "tool_calling", "react", "act", or "few_shot"
- `domain`: Domain context for the solver
- `max_iterations`: Maximum number of reasoning/action iterations
- `max_turns`: Maximum conversation turns (for multi-turn solvers)

### Scoring Options

- `scoring_mode`: "comprehensive", "simple", or "action_only"
- `domain`: Domain-specific scoring adjustments

## Implementation Notes

### Data Loading
The dataset module handles loading tau-bench data and converting it to inspect_evals format. It supports:
- Loading domain-specific data (retail/airline)
- Converting tasks to conversation samples
- Creating different sample types (conversation, simple, tool_calling)
- Handling data splits and filtering

### Solver Strategies
The solver module implements the four main tau-bench agent strategies:
- **Tool-calling**: Direct tool usage for task completion
- **ReAct**: Reasoning followed by action
- **Act**: Action-oriented approach
- **Few-shot**: Learning from example conversations

### Scoring System
The scoring module provides comprehensive evaluation metrics:
- **Action similarity**: Compares predicted vs expected actions
- **Response quality**: Evaluates response relevance and helpfulness
- **Conversation quality**: Assesses politeness and domain knowledge
- **Flexible scoring modes**: Comprehensive, simple, and action-focused

## Future Enhancements

This scaffolding provides a foundation for further development:

1. **Tool Integration**: Complete integration with tau-bench's tool system
2. **User Simulation**: More sophisticated user simulation for realistic evaluation
3. **Multi-modal Support**: Support for visual and other modalities
4. **Advanced Metrics**: Additional evaluation metrics and analysis
5. **Benchmark Integration**: Integration with standard evaluation benchmarks

## Dependencies

- `inspect_ai`: Core evaluation framework
- `inspect_evals`: Evaluation task definitions
- `tau-bench`: Original tau-bench implementation (for data and tools)

## Contributing

This scaffolding is designed to be extensible. Key areas for contribution:

1. **Tool Integration**: Complete the tool calling implementation
2. **User Simulation**: Enhance user simulation capabilities
3. **Scoring Metrics**: Add more sophisticated evaluation metrics
4. **Domain Support**: Add support for additional domains
5. **Performance**: Optimize for large-scale evaluation

## License

This integration follows the same license as the original tau-bench project.
