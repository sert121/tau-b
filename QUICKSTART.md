# Tau-bench Inspect Quick Start Guide

## Installation

```bash
# Install dependencies
pip install inspect_ai inspect_evals

# Install tau-bench (if not already installed)
pip install tau-bench
```

## Quick Run

### 1. Basic Evaluation
```python
from tau_bench_inspect import tau_bench_retail_simple
import inspect_ai

result = inspect_ai.eval(
    task=tau_bench_retail_simple(max_samples=5),
    model="openai/gpt-4o"
)
```

### 2. Command Line
```bash
inspect eval tau_bench_retail_simple --model openai/gpt-4o --max-samples 5
```

### 3. Run Example Script
```bash
python example_usage.py
```

## Available Evaluations

### Retail Domain
- `tau_bench_retail_simple` - Basic retail evaluation
- `tau_bench_retail_tool_calling` - Tool-calling strategy
- `tau_bench_retail_react` - ReAct strategy
- `tau_bench_retail_act` - Act strategy
- `tau_bench_retail_few_shot` - Few-shot learning
- `tau_bench_retail_actions` - Action accuracy focused
- `tau_bench_retail_conversation` - Multi-turn conversations

### Airline Domain
- `tau_bench_airline_simple` - Basic airline evaluation
- `tau_bench_airline_tool_calling` - Tool-calling strategy
- `tau_bench_airline_react` - ReAct strategy
- `tau_bench_airline_act` - Act strategy
- `tau_bench_airline_few_shot` - Few-shot learning
- `tau_bench_airline_actions` - Action accuracy focused
- `tau_bench_airline_conversation` - Multi-turn conversations

### Cross-Domain
- `tau_bench_cross_domain` - Mixed retail and airline

## Configuration Options

### Dataset Options
- `domain`: "retail" or "airline"
- `split`: "train", "dev", or "test"
- `task_type`: "conversation", "simple", or "tool_calling"
- `max_samples`: Maximum number of samples
- `shuffle`: Whether to shuffle the dataset
- `seed`: Random seed for reproducibility

### Solver Options
- `strategy`: "tool_calling", "react", "act", or "few_shot"
- `domain`: Domain context
- `max_iterations`: Maximum reasoning iterations
- `max_turns`: Maximum conversation turns

### Scoring Options
- `scoring_mode`: "comprehensive", "simple", or "action_only"
- `domain`: Domain-specific scoring

## Examples

### Simple Evaluation
```python
from tau_bench_inspect import tau_bench_retail_simple
import inspect_ai

result = inspect_ai.eval(
    task=tau_bench_retail_simple(
        split="test",
        max_samples=10,
        shuffle=True,
        seed=42
    ),
    model="openai/gpt-4o"
)
```

### Tool-Calling Evaluation
```python
from tau_bench_inspect import tau_bench_retail_tool_calling
import inspect_ai

result = inspect_ai.eval(
    task=tau_bench_retail_tool_calling(
        split="test",
        max_samples=5,
        shuffle=True,
        seed=42
    ),
    model="openai/gpt-4o"
)
```

### Multi-turn Conversation
```python
from tau_bench_inspect import tau_bench_retail_conversation
import inspect_ai

result = inspect_ai.eval(
    task=tau_bench_retail_conversation(
        split="test",
        max_samples=3,
        max_turns=5,
        shuffle=True,
        seed=42
    ),
    model="openai/gpt-4o"
)
```

### Custom Configuration
```python
from tau_bench_inspect import (
    tau_bench_dataset,
    create_conversation_solver,
    tau_bench_scorer
)
import inspect_ai

# Create custom task
dataset = tau_bench_dataset(
    domain="retail",
    split="test",
    task_type="conversation",
    max_samples=5
)

solver = create_conversation_solver(
    strategy="react",
    domain="retail"
)

scorer = tau_bench_scorer(
    domain="retail",
    scoring_mode="comprehensive"
)

task = inspect_ai.Task(
    dataset=dataset,
    solver=solver,
    scorer=scorer
)

result = inspect_ai.eval(task=task, model="openai/gpt-4o")
```

## Troubleshooting

### Common Issues

1. **Import Error**: Make sure tau-bench is installed and accessible
2. **Model Error**: Ensure you have valid API keys for the model provider
3. **Data Error**: Check that tau-bench data files are accessible
4. **Memory Error**: Reduce max_samples for large evaluations

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run evaluation with debug logging
result = inspect_ai.eval(task=task, model="openai/gpt-4o")
```

## Next Steps

1. **Custom Evaluations**: Create your own evaluation tasks
2. **Tool Integration**: Implement actual tau-bench tool calling
3. **Scoring Metrics**: Add custom scoring metrics
4. **Batch Evaluation**: Run multiple evaluations in parallel
5. **Results Analysis**: Analyze and visualize results
