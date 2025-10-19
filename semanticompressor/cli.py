import argparse
import json
import os
import socket
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


DEFAULT_MAX_ITERATIONS = 10
DEFAULT_MODEL = "gpt-5-mini"


class OpenAIError(RuntimeError):
    """Domain-specific error to bubble up API issues."""


def log(message: str, *, verbose: bool) -> None:
    if verbose:
        print(f"[semanticompressor] {message}")


def _api_base() -> str:
    return os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")


def list_models(api_key: str, *, verbose: bool = False) -> List[str]:
    url = f"{_api_base()}/models"
    log(f"Fetching model list from {url}", verbose=verbose)
    try:
        request = urllib.request.Request(
            url,
            headers={"Authorization": f"Bearer {api_key}"},
            method="GET",
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read().decode("utf-8")
            status = response.getcode()
    except urllib.error.HTTPError as exc:
        raise OpenAIError(
            f"Failed to list models ({exc.code}): "
            f"{exc.read().decode('utf-8', errors='ignore')}"
        ) from exc
    except urllib.error.URLError as exc:
        raise OpenAIError(f"Failed to list models: {exc}") from exc

    if status != 200:
        raise OpenAIError(f"Failed to list models ({status}): {body}")

    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise OpenAIError(f"Failed to parse model list: {body}") from exc
    data = payload.get("data", [])
    log(f"Retrieved {len(data)} models", verbose=verbose)
    return sorted(model["id"] for model in data if "id" in model)


def chat_completion(
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    *,
    response_format: Optional[Dict[str, object]] = None,
    temperature: Optional[float] = None,
    verbose: bool = False,
) -> str:
    url = f"{_api_base()}/chat/completions"
    summary: Sequence[Tuple[str, int]] = [
        (msg.get("role", "?"), len(msg.get("content", ""))) for msg in messages
    ]
    log(
        f"Requesting completion model={model} messages={summary} "
        f"response_format={response_format} temperature={temperature}",
        verbose=verbose,
    )
    request_body: Dict[str, object] = {
        "model": model,
        "messages": messages,
    }
    if temperature is not None:
        request_body["temperature"] = temperature
    if response_format:
        request_body["response_format"] = response_format

    data_bytes = json.dumps(request_body).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data_bytes,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            headers = dict(response.info())
            body_text = response.read().decode("utf-8")
            status = response.getcode()
            log(
                f"Received response status={status} request-id={headers.get('x-request-id', 'unknown')}",
                verbose=verbose,
            )
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="ignore")
        raise OpenAIError(f"OpenAI API error ({exc.code}): {error_body}") from exc
    except urllib.error.URLError as exc:
        raise OpenAIError(f"OpenAI API request failed: {exc}") from exc
    except socket.timeout as exc:
        raise OpenAIError(f"OpenAI API request timed out: {exc}") from exc
    except TimeoutError as exc:
        raise OpenAIError(f"OpenAI API request timed out: {exc}") from exc

    if status != 200:
        raise OpenAIError(f"OpenAI API error ({status}): {body_text}")

    try:
        body = json.loads(body_text)
    except json.JSONDecodeError as exc:
        raise OpenAIError(f"Failed to parse OpenAI response: {body_text}") from exc
    usage = body.get("usage")
    if usage:
        log(
            f"Usage prompt_tokens={usage.get('prompt_tokens')} "
            f"completion_tokens={usage.get('completion_tokens')} "
            f"total_tokens={usage.get('total_tokens')}",
            verbose=verbose,
        )
    try:
        content = body["choices"][0]["message"]["content"]
        log(f"Completion received {len(content)} characters", verbose=verbose)
        return content
    except (KeyError, IndexError) as exc:
        raise OpenAIError(f"Unexpected response shape: {json.dumps(body)}") from exc


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise OpenAIError(f"File {path} is not valid UTF-8 text.")


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def initial_compression(
    api_key: str, model: str, original_text: str, *, verbose: bool
) -> str:
    system_prompt = (
        "You create lossless semantic compressions. Ensure every detail and meaning is preserved."
    )
    user_prompt = (
        "The file below needs to be losslessly semantically compressed. "
        "Provide just the updated file and nothing else.\n\n"
        "---FILE START---\n"
        f"{original_text}\n"
        "---FILE END---"
    )
    return chat_completion(
        api_key,
        model,
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        verbose=verbose,
    ).strip()


@dataclass
class EvaluationResult:
    redraft: str
    gaps: List[Dict[str, str]]


def evaluate_attempt(
    api_key: str,
    model: str,
    original_text: str,
    current_attempt: str,
    history_attempts: List[str],
    *,
    verbose: bool,
) -> EvaluationResult:
    history_section = "\n".join(
        f"---PREVIOUS ATTEMPT {idx + 1} START---\n{attempt}\n---PREVIOUS ATTEMPT {idx + 1} END---"
        for idx, attempt in enumerate(history_attempts)
    )

    history_block = history_section if history_section else "None."

    system_prompt = (
        "You compare semantic compression attempts to ensure they are lossless. "
        "Identify any semantic gaps, risk, or ambiguity and produce an improved attempt."
    )

    user_prompt = (
        "The first file is an attempt at lossless, semantic compression of the second file.\n"
        "Identify gaps and redraft. Return the redraft, and structured assessment of each gap "
        "and its criticality (major, moderate, or minor semantic gap). Previous attempts are "
        "included in the history section for reference.\n"
        "Respond strictly as JSON with the following shape:\n"
        '{\n'
        '  "redraft": "<improved lossless semantic compression>",\n'
        '  "gaps": [\n'
        '    {"description": "<issue>", "criticality": "major|moderate|minor"}\n'
        "  ]\n"
        "}\n"
        "Do not include commentary outside this JSON.\n\n"
        "---CURRENT ATTEMPT START---\n"
        f"{current_attempt}\n"
        "---CURRENT ATTEMPT END---\n\n"
        "---ORIGINAL FILE START---\n"
        f"{original_text}\n"
        "---ORIGINAL FILE END---\n\n"
        "---HISTORY START---\n"
        f"{history_block}\n"
        "---HISTORY END---"
    )

    response = chat_completion(
        api_key,
        model,
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        verbose=verbose,
    )

    try:
        payload = json.loads(response)
    except json.JSONDecodeError as exc:
        raise OpenAIError(f"Failed to parse JSON from evaluator: {response}") from exc

    redraft = payload.get("redraft")
    gaps = payload.get("gaps")
    if not isinstance(redraft, str) or not isinstance(gaps, list):
        raise OpenAIError(f"Evaluator returned unexpected JSON payload: {payload}")

    normalised_gaps: List[Dict[str, str]] = []
    for entry in gaps:
        if not isinstance(entry, dict):
            continue
        description = entry.get("description", "")
        criticality = entry.get("criticality", "")
        if not isinstance(description, str) or not isinstance(criticality, str):
            continue
        normalised_gaps.append(
            {
                "description": description.strip(),
                "criticality": criticality.strip().lower(),
            }
        )

    return EvaluationResult(redraft=redraft.strip(), gaps=normalised_gaps)


def count_criticalities(gaps: List[Dict[str, str]]) -> Dict[str, int]:
    counts = {"major": 0, "moderate": 0, "minor": 0, "other": 0}
    for gap in gaps:
        label = gap.get("criticality", "").lower()
        if label in counts:
            counts[label] += 1
        else:
            counts["other"] += 1
    return counts


def derive_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}.sc.md")


def run_compression(
    input_path: Path,
    output_path: Path,
    api_key: str,
    model: str,
    max_iterations: int,
    *,
    verbose: bool,
) -> Dict[str, object]:
    original_text = read_text(input_path)
    attempts: List[str] = []
    progress_log: List[str] = []

    print("Starting initial semantic compression...")
    first_attempt = initial_compression(api_key, model, original_text, verbose=verbose)
    attempts.append(first_attempt)

    latest = first_attempt
    for iteration in range(1, max_iterations + 1):
        if len(attempts) >= DEFAULT_MAX_ITERATIONS:
            print("Reached hard limit of 10 attempts.")
            break

        history_attempts = attempts[:-1]
        evaluation = evaluate_attempt(
            api_key,
            model,
            original_text,
            latest,
            history_attempts,
            verbose=verbose,
        )

        attempts.append(evaluation.redraft)
        latest = evaluation.redraft

        counts = count_criticalities(evaluation.gaps)
        log_entry = (
            f"Iteration {iteration}: "
            f"{counts['major']} major, {counts['moderate']} moderate, {counts['minor']} minor"
        )
        print(log_entry)
        progress_log.append(log_entry)

        gaps_all_minor = all(
            gap.get("criticality", "").lower() == "minor" for gap in evaluation.gaps
        )
        if len(evaluation.gaps) < 5 and gaps_all_minor:
            print("Stopping criteria met (fewer than 5 minor gaps).")
            break

    write_text(output_path, latest)
    print(f"Final compressed output written to {output_path}")
    return {
        "output_path": str(output_path),
        "progress_log": progress_log,
        "iterations": len(progress_log),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Semantic lossless compression tool powered by OpenAI.",
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="Path to the source text file to compress.",
    )
    parser.add_argument(
        "--output",
        help="File path for the compressed result. Defaults to <stem>.sc.md in place.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", DEFAULT_MODEL),
        help=f"Model to use (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=DEFAULT_MAX_ITERATIONS,
        help="Maximum number of refinement iterations (default 10, max 10).",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging of OpenAI API requests and responses.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = os.environ.get("OPENA_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY environment variable is required.", file=sys.stderr)
        sys.exit(1)

    if args.list_models:
        try:
            models = list_models(api_key, verbose=args.verbose)
        except OpenAIError as exc:
            print(str(exc), file=sys.stderr)
            sys.exit(1)
        for model in models:
            print(model)
        return

    if not args.input:
        print("Input file path is required.", file=sys.stderr)
        sys.exit(1)

    max_iterations = min(args.max_iterations, DEFAULT_MAX_ITERATIONS)
    if max_iterations < 1:
        print("max-iterations must be at least 1.", file=sys.stderr)
        sys.exit(1)

    input_path = Path(args.input)
    if not input_path.is_file():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output) if args.output else derive_output_path(input_path)

    try:
        run_compression(
            input_path=input_path,
            output_path=output_path,
            api_key=api_key,
            model=args.model,
            max_iterations=max_iterations,
            verbose=args.verbose,
        )
    except OpenAIError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError as exc:
        print(f"Network error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
