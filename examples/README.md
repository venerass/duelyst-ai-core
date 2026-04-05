# Examples

Working examples for `duelyst-ai-core`. Each script is self-contained — set your API keys and run.

## Prerequisites

```bash
pip install duelyst-ai-core
```

For web search support:

```bash
pip install "duelyst-ai-core[search]"
```

## API Keys

Set environment variables for the providers you want to use:

```bash
export ANTHROPIC_API_KEY=your-key-here
export OPENAI_API_KEY=your-key-here
export GOOGLE_API_KEY=your-key-here       # optional
export TAVILY_API_KEY=your-key-here       # optional, for web search
```

## Examples

| Script | Description | Models Used |
|--------|-------------|-------------|
| [`basic_debate.py`](basic_debate.py) | Minimal debate with Markdown output | Claude Haiku + GPT mini |
| [`streaming_debate.py`](streaming_debate.py) | Real-time events via `arun_with_events()` | Claude Haiku + GPT mini |
| [`custom_debate.py`](custom_debate.py) | Custom models, instructions, web search | Claude Haiku + GPT mini + Gemini Flash judge |
| [`json_output.py`](json_output.py) | Machine-consumable JSON output for pipelines | Claude Haiku + GPT mini |

### basic_debate.py

The simplest possible debate. Two cheap models argue a topic, then the result is formatted as Markdown.

```bash
python examples/basic_debate.py
```

### streaming_debate.py

Uses `arun_with_events()` to receive events as the debate progresses. This is the same pattern the FastAPI backend uses for SSE streaming.

```bash
python examples/streaming_debate.py
```

### custom_debate.py

Full configuration surface: specific models for each side and judge, custom per-side instructions, web search tools, and convergence tuning.

```bash
python examples/custom_debate.py
```

### json_output.py

Outputs the complete debate result as JSON. Useful for automation, piping into `jq`, or feeding into other tools.

```bash
python examples/json_output.py > debate.json
```
