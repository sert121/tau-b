

## Usage

### Basic Evaluation

```bash
# Run airline domain evaluation (react style)
inspect eval tau_bench_inspect.py -T base_model=openai/gpt-4o -T domain=airline -T agent_mode=react

# Run retail domain evaluation  (react style)
inspect eval tau_bench_inspect.py -T base_model=openai/gpt-4o -T domain=retail -T agent_mode=react
```

### With Different Base and user model (simualted user)

```bash
# Using OpenAI
inspect eval tau_bench_inspect.py -T base_model=openai/gpt-4o -T domain=airline -T agent_mode=react -T user_model=openai/gpt-4o-mini

```

## Configuration

### Environment Variables
```bash
export GROQ_API_KEY=your_groq_api_key
export OPENAI_API_KEY=your_openai_api_key
export ANTHROPIC_API_KEY=your_anthropic_api_key
```

### Model Configuration
The system supports various LLM providers through inspect_ai:
- Groq models: `groq/llama-3.1-8b-instant`, `groq/llama-3.1-70b-versatile`
- OpenAI models: `openai/gpt-4o-mini`, `openai/gpt-4o`
- Anthropic models: `anthropic/claude-3-5-sonnet-20241022`

## Dependencies

- `inspect_ai`: Core evaluation framework
- `litellm`: LLM provider integration
- `pydantic`: Data validation and serialization
- `groq`: Groq API client (optional)
- `openai`: OpenAI API client (optional)
- `anthropic`: Anthropic API client (optional)

## License

This integration follows the same license as the original tau-bench project.

## Results

# Tau-Bench Evaluation Results

(user model is also gpt4o)

| Domain  | Agent Mode | Base Model    | Samples | Accuracy |
|---------|------------|---------------|---------|----------|
| retail  | react      | openai/gpt-4o | 115     | 0.548    |
| airline | react      | openai/gpt-4o | 50      | 0.460    |

![Image_1](https://ibb.co/0RW97B6V)  
![Image_2](https://ibb.co/39g7K3G4)
