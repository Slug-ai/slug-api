# slug-api

OpenAI-compatible API in front of Slug's inference backends, so the `openai` package, OpenRouter and the like can use them.

/slug_api
server.py: the API (chat completions, streaming, files and batches) and the `Backend` it drives
queue.py: requests waiting for a backend, and the ones it's generating

Backends live with their engines:
- slugai: `controller/serve.py`, the Mac pipeline
- qwen38-27b-aws: `scripts/serve.py`, Qwen3.8-27B on Inferentia2

# Setup

```
uv sync
uv run pytest
```

Install it in an engine's environment with `uv add git+https://github.com/Slug-ai/slug-api` (or `pip install git+https://github.com/Slug-ai/slug-api`).
