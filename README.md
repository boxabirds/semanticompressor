# Semanticompressor

Command-line tool for iteratively producing lossless semantic compressions of text files with OpenAI models.

## Setup

1. Ensure Python 3.9+ and [uv](https://github.com/astral-sh/uv) are available.
2. Create (or reuse) a project environment: `uv venv`
3. Install project dependencies (none currently, but keeps the lockfile in sync): `uv sync`
4. Export your OpenAI key: `export OPENA_API_KEY=sk-...`
5. (Optional) Set a default model via `export OPENAI_MODEL=model-id`. The tool defaults to `gpt-5-mini`.

## Usage

Run directly inside the uv environment (console script entry point):

```
uv run semanticompressor <path-to-file> [--output OUTPUT] [--model MODEL] [--max-iterations N]
```

- The compressor writes results to `<stem>.sc.md` in the source directory unless `--output` is given.
- Iterations stop when fewer than five gaps remain and all are minor, or when the ten-attempt cap is reached.
- Progress summaries emit as `Iteration N: X major, Y moderate, Z minor`.

List available models (no file required):

```
uv run semanticompressor --list-models
```

You can also execute the thin wrapper script directly if you prefer shorter commands:

```
uv run sc.py <path-to-file>
```

### Managing Dependencies

When you need another Python package:

```
uv add <package-name>
uv sync
```

`uv` will update `pyproject.toml` and `uv.lock`, then make it available inside the virtual environment.

## How It Works

1. Generates an initial semantic compression of the original file.
2. Iteratively compares the latest draft to the original, requesting a redraft plus structured gap report.
3. Stops early when remaining gaps are all minor (fewer than five) or after ten attempts.
4. Writes the latest draft to disk and prints a log of iteration summaries.
