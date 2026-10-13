"""Shared pieces of the Gemini pairwise judge (Vertex AI, run from the Mac).

The rubrics are the same text the rest of the evaluation uses (src/mfr_generation_eval.py); a check
at import time keeps the two copies identical. Each comparison is judged in both A/B orders; a
verdict counts only when both orders pick the same response (position-consistent).
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

HELPFULNESS_RUBRIC = """Choose the response that is more helpful to the user. A better response is correct, relevant, clear, self-contained, appropriately detailed, and directly addresses the instruction. Do not reward verbosity by itself. Penalize unsupported claims, omissions, and unnecessary refusal."""

QUALITY_RUBRIC = """Choose the response with better overall answer quality. Consider correctness, relevance, clarity, coherence, specificity, and whether the answer fully satisfies the request. Do not reward verbosity by itself. Penalize factual errors, unsupported claims, and confusing presentation."""

_source = (REPO / "src" / "mfr_generation_eval.py").read_text(encoding="utf-8")
if HELPFULNESS_RUBRIC not in _source or QUALITY_RUBRIC not in _source:
    raise RuntimeError("rubrics differ from src/mfr_generation_eval.py; keep them identical")

PROMPT = """You are an impartial judge comparing two AI assistant responses to the same user request.

{rubric}

Judge only the responses. Ignore their order and their length except where length affects quality.

[User request]
{instruction}

[Response A]
{a}

[Response B]
{b}

Which response is better? Answer with "A" or "B"."""

SCHEMA = {"type": "OBJECT", "properties": {"choice": {"type": "STRING", "enum": ["A", "B"]}},
          "required": ["choice"]}


def rubric(behavior):
    return HELPFULNESS_RUBRIC if behavior == "helpful" else QUALITY_RUBRIC


def make_client(project, location):
    from google import genai
    return genai.Client(vertexai=True, project=project, location=location)


def judge_once(client, model, behavior, instruction, a, b, seed=544, retries=8):
    """One judgment: 'A' or 'B'; None if the model returns no usable choice. Retries on API errors."""
    from google.genai import types

    config = types.GenerateContentConfig(
        temperature=0.0, seed=seed, max_output_tokens=4096,
        response_mime_type="application/json", response_schema=SCHEMA,
        # no tools are used; turning off automatic function calling also silences its warning
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    prompt = PROMPT.format(rubric=rubric(behavior), instruction=instruction, a=a, b=b)
    for attempt in range(retries):
        try:
            response = client.models.generate_content(model=model, contents=prompt, config=config)
            usage = response.usage_metadata
            tokens = {"in": usage.prompt_token_count or 0,
                      "out": (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)}
            try:
                choice = json.loads(response.text or "{}").get("choice")
            except json.JSONDecodeError:
                choice = None
            return (choice if choice in ("A", "B") else None), tokens
        except Exception as error:  # rate limits, timeouts, transient server errors
            if attempt == retries - 1:
                raise
            wait = min(120, 2 ** attempt) + random.random()
            print(f"  retry {attempt + 1} after {type(error).__name__}: {str(error)[:80]} "
                  f"(waiting {wait:.0f}s)", flush=True)
            time.sleep(wait)


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path, record):
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
