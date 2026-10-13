#!/usr/bin/env python3
"""Billing test (< $1): a few Gemini calls through Vertex AI, to check that the Education credits pay.

Run it once, then check Google Cloud Billing -> Reports about 24 hours later. Usage for this project
should show a matching negative "credits" amount. Each call is a small pairwise-judging prompt, so
the token counts printed here also tell us what the real judge will cost.

  python billing_test.py --project PROJECT_ID --list                 # show available Gemini models
  python billing_test.py --project PROJECT_ID --model MODEL_ID       # run the test
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

PROMPT = """You are judging two answers to the same user request. Pick the answer that is more
helpful, correct and clearly written. Reply with exactly one letter: A or B.

[User request]
How do I keep cut basil fresh for a week?

[Answer A]
Trim the stems and stand the basil in a glass of water on the counter, loosely covered with a
plastic bag. Change the water every day or two. Do not refrigerate it; cold blackens the leaves.

[Answer B]
Put it in the fridge in a sealed container. It should last a while.

Your choice (A or B):"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, help="Google Cloud project on the Education billing account")
    parser.add_argument("--location", default="global", help="Vertex AI location (try us-central1 if global fails)")
    parser.add_argument("--model", default="gemini-2.5-pro")
    parser.add_argument("--calls", type=int, default=10)
    parser.add_argument("--list", action="store_true", help="Only list the Gemini models this project can use")
    parser.add_argument("--price-in", type=float, default=2.50, help="USD per 1M input tokens (estimate only)")
    parser.add_argument("--price-out", type=float, default=15.00, help="USD per 1M output+thinking tokens (estimate only)")
    parser.add_argument("--max-usd", type=float, default=1.00, help="Stop once the estimate passes this")
    args = parser.parse_args()

    from google import genai
    from google.genai import types

    client = genai.Client(vertexai=True, project=args.project, location=args.location)
    if args.list:
        for model in client.models.list(config={"query_base": True}):
            if "gemini" in model.name:
                print(model.name)
        return

    log_path = Path(__file__).with_name("billing_test_log.jsonl")
    total_in = total_out = 0
    estimate = 0.0
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    for call in range(1, args.calls + 1):
        response = client.models.generate_content(
            model=args.model, contents=PROMPT,
            config=types.GenerateContentConfig(
                temperature=0.0, max_output_tokens=2048,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)),
        )
        usage = response.usage_metadata
        tokens_in = usage.prompt_token_count or 0
        tokens_out = (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)
        total_in += tokens_in
        total_out += tokens_out
        estimate = total_in / 1e6 * args.price_in + total_out / 1e6 * args.price_out
        answer = (response.text or "").strip().replace("\n", " ")[:40]
        print(f"call {call:2d}: {tokens_in:5d} in, {tokens_out:5d} out (incl. thinking) -> {answer!r}  "
              f"running estimate ${estimate:.3f}", flush=True)
        if estimate > args.max_usd:
            print(f"Stopping: estimate passed ${args.max_usd:.2f}")
            break

    record = {"started_utc": started, "project": args.project, "location": args.location,
              "model": args.model, "calls": call, "input_tokens": total_in,
              "output_tokens": total_out, "estimated_usd": round(estimate, 4)}
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    per_call = (total_in + total_out) / max(1, call)
    print(f"\nDone: {call} calls, {total_in} input + {total_out} output tokens, about ${estimate:.2f}.")
    print(f"About {per_call:.0f} tokens per judgment. Logged to {log_path.name}.")
    print("Now wait ~24 h and check Billing -> Reports for this project: the cost should be "
          "offset by a credit line.")


if __name__ == "__main__":
    main()
