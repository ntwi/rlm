# Recursive Language Models (minimal version) 

[Link to the original blogpost 📝](https://alexzhang13.github.io/blog/2025/rlm/)

I received a lot of requests to put out a notebook or gist version of the codebase I've been using. Sadly it's a bit entangled with a bunch of random state, cost, and code execution tracking logic that I want to clean up while I run other experiments. In the meantime, I've re-written a simpler version of what I'm using so people can get started building on top and writing their own RLM implementations. Happy hacking!

![teaser](media/rlm.png)

I've provided a basic, minimal implementation of a recursive language model (RLM) with a REPL environment for OpenAI clients. Like the blogpost, we only implement recursive sub-calls with `depth=1` inside the RLM environment. Enabling further depths is as simple as replacing the `Sub_RLM` class with the `RLM_REPL` class, but you may need to finagle the `exec`-based REPL environments to work better here (because now your sub-RLMs have their own REPL environments!).

In this stripped implementation, we exclude a lot of the logging, cost tracking, prompting, and REPL execution details of the experiments run in the blogpost. It's relatively easy to modify and build on top of this code to reproduce those results, but it's currently harder to go from my full codebase to supporting any new functionality.

## Basic Example
We have all the basic dependencies in `requirements.txt`, although none are really necessary if you change your implementation (`openai` for LM API calls, `dotenv` for .env loading, and `rich` for logging).

In `main.py`, we have a basic needle-in-the-haystack (NIAH) example that embeds a random number inside ~1M lines of random words, and asks the model to go find it. It's a silly Hello World type example to emphasize that `RLM.completion()` calls are meant to replace `LM.completion()` calls.

## Code Structure
In the `rlm/` folder, the two main files are `rlm_repl.py` and `repl.py`. 
* `rlm_repl.py` offers a basic implementation of an RLM using a REPL environment in the `RLM_REPL` class. The `completion()` function gets called when we query an RLM.
* `repl.py` is a simple `exec`-based implementation of a REPL environment that adds an LM sub-call function. To make the system truly recursive beyond `depth=1`, you can replace the `Sub_RLM` class with `RLM_REPL` (they all inherit from the `RLM` base class).

The functionality for parsing and handling base LLM clients are all in `rlm/utils/`. We also add example prompts here.

> The `rlm/logger/` folder mainly contains optional logging utilities used by the RLM REPL implementation. If you want to enable colorful or enhanced logging outputs, you may need to install the [`rich`](https://github.com/Textualize/rich) library as a dependency.
```
pip install rich
```

When you run your code, you'll see something like this:

![Example logging output using `rich`](media/rich.png)

## Provider Support and Models
The unified client now supports multiple providers with performance‑oriented defaults (lazy initialization, short payloads, HTTP keep‑alive via SDKs):

- OpenAI: e.g., gpt-5, gpt-4o, etc. Set OPENAI_API_KEY.
- Google Gemini: gemini-2.5-pro. Set GOOGLE_API_KEY.
- xAI Grok: grok4-latest (OpenAI‑compatible). Set XAI_API_KEY and optional XAI_BASE_URL (defaults to https://api.x.ai/v1).
- Anthropic Claude: claude-sonnet-4.5 (and other Claude models). Set ANTHROPIC_API_KEY.
- Ollama (local): e.g., gpt-oss:120b, gpt-oss:20b, deepseek-coder-v2. Requires a running Ollama server. Optional OLLAMA_HOST (defaults to http://localhost:11434).

Usage example (choose your model):

```python
from rlm.rlm_repl import RLM_REPL
# Cloud example
rlm = RLM_REPL(model="gemini-2.5-pro", recursive_model="claude-sonnet-4.5", enable_logging=True)
answer = rlm.completion(context="...big context...", query="Your question")

# Local (Ollama) example
# Make sure Ollama is running and the model exists locally: e.g., `ollama pull gpt-oss:20b`
rlm_local = RLM_REPL(model="gpt-oss:20b", recursive_model="gpt-oss:20b", enable_logging=True)
answer_local = rlm_local.completion(context="...big context...", query="Your question")
```

Environment configuration (.env):

```
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GOOGLE_API_KEY=...
XAI_API_KEY=...
# Optional
XAI_BASE_URL=https://api.x.ai/v1
OLLAMA_HOST=http://localhost:11434
```

Performance tips:
- Prefer concise prompts and low temperature (default 0.2) for determinism and speed.
- Reuse the same RLM_REPL instance to keep provider clients warm.
- Use the REPL’s llm_query to chunk large contexts and summarize incrementally.
- Set max_iterations appropriately; default is 20 in RLM_REPL and 10 in main’s example.
