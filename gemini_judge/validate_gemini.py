#!/usr/bin/env python3
"""Validate the Gemini judge candidates against human labels and pick one by the predeclared rule.

Data: the 400 validation pairs (Helpful 200, Quality 200) in data/v2, never the locked test. Each
pair is judged in both A/B orders (chosen as A, then chosen as B). A pair agrees with the humans when
both orders pick the chosen response. Results are appended to results/validation_<model>.jsonl
after every call, so an interrupted run resumes where it stopped.

Rule (fixed 2026-10-12, before any result): the candidate with the highest overall agreement wins
and must reach 70%. If a stable model is within 2 points of the winner, the stable model wins.

  python validate_gemini.py --project mfr-dpo-judge            # all three candidates
  python validate_gemini.py --project mfr-dpo-judge --limit 5  # quick check (separate files)
"""

from __future__ import annotations

import argparse
import csv
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from gemini_judge import REPO, append_jsonl, judge_once, make_client, read_jsonl

CANDIDATES = {                      # model id -> stable (GA)?
    "gemini-2.5-pro": True,
    "gemini-3.1-pro-preview": False,
    "gemini-3.8-flash": True,
}
BEHAVIORS = ("helpful", "quality")
THRESHOLD, STABLE_MARGIN = 70.0, 2.0


def load_pairs(limit=None):
    pairs = []
    for behavior in BEHAVIORS:
        rows = sorted(read_jsonl(REPO / "data" / "v2" / f"{behavior}_val.jsonl"), key=lambda r: r["id"])
        pairs += [{**row, "behavior": behavior} for row in rows[:limit]]
    return pairs


def run_model(client, model, pairs, out_path, workers):
    done = {(r["id"], r["order"]) for r in read_jsonl(out_path)} if out_path.exists() else set()
    tasks = [(pair, order) for pair in pairs for order in ("chosen_first", "chosen_second")
             if (pair["id"], order) not in done]
    print(f"{model}: {len(done)} judgments already saved, {len(tasks)} to go", flush=True)

    def work(pair, order):
        a, b = ((pair["chosen"], pair["rejected"]) if order == "chosen_first"
                else (pair["rejected"], pair["chosen"]))
        choice, tokens = judge_once(client, model, pair["behavior"], pair["prompt"], a, b)
        picked_chosen = None if choice is None else (choice == "A") == (order == "chosen_first")
        return {"id": pair["id"], "behavior": pair["behavior"], "order": order, "choice": choice,
                "picked_chosen": picked_chosen, "tokens_in": tokens["in"], "tokens_out": tokens["out"]}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work, pair, order) for pair, order in tasks]
        for count, future in enumerate(as_completed(futures), 1):
            append_jsonl(out_path, future.result())
            if count % 50 == 0 or count == len(futures):
                print(f"  {model}: {count}/{len(futures)}", flush=True)


def summarize(model, out_path, pairs):
    by_key = {(r["id"], r["order"]): r for r in read_jsonl(out_path)}
    rows, totals = [], {"in": 0, "out": 0}
    for record in by_key.values():
        totals["in"] += record["tokens_in"]
        totals["out"] += record["tokens_out"]
    for behavior in (*BEHAVIORS, "overall"):
        subset = [p for p in pairs if behavior == "overall" or p["behavior"] == behavior]
        agree = consistent = 0
        outcomes = []
        for pair in subset:
            first = by_key.get((pair["id"], "chosen_first"), {}).get("picked_chosen")
            second = by_key.get((pair["id"], "chosen_second"), {}).get("picked_chosen")
            is_consistent = first is not None and first == second
            hit = is_consistent and first is True
            consistent += is_consistent
            agree += hit
            outcomes.append(1.0 if hit else 0.0)
        n = len(subset)
        rng = random.Random(544)
        boot = sorted(sum(rng.choices(outcomes, k=n)) / n for _ in range(2000)) if n else [0.0]
        rows.append({
            "model": model, "stable": CANDIDATES[model], "behavior": behavior, "pairs": n,
            "position_consistent_pct": round(100 * consistent / n, 2) if n else None,
            "agreement_pct": round(100 * agree / n, 2) if n else None,
            "agreement_ci_low_pct": round(100 * boot[int(0.025 * len(boot))], 2),
            "agreement_ci_high_pct": round(100 * boot[int(0.975 * len(boot)) - 1], 2),
            "agreement_among_consistent_pct": round(100 * agree / consistent, 2) if consistent else None,
            "tokens_in": totals["in"] if behavior == "overall" else None,
            "tokens_out": totals["out"] if behavior == "overall" else None,
        })
    return rows


def choose(overall):
    """Apply the predeclared rule to {model: overall agreement}."""
    passing = {m: a for m, a in overall.items() if a is not None and a >= THRESHOLD}
    if not passing:
        return None, "no candidate reached 70%: Gemini is not used"
    best = max(passing, key=passing.get)
    stable = [m for m in passing if CANDIDATES[m] and passing[m] >= passing[best] - STABLE_MARGIN]
    if CANDIDATES[best]:
        return best, f"{best} has the highest agreement and is stable"
    if stable:
        pick = max(stable, key=passing.get)
        return pick, f"{best} is highest but preview; stable {pick} is within {STABLE_MARGIN} points"
    return best, f"{best} (preview) is highest and no stable model is within {STABLE_MARGIN} points"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--location", default="global")
    parser.add_argument("--models", default=",".join(CANDIDATES))
    parser.add_argument("--limit", type=int, help="Quick check: first N pairs per behavior")
    parser.add_argument("--workers", type=int, default=8, help="Parallel API calls")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if unknown := [m for m in models if m not in CANDIDATES]:
        raise ValueError(f"not a predeclared candidate: {unknown}")
    pairs = load_pairs(args.limit)
    results = Path(__file__).with_name("results")
    results.mkdir(exist_ok=True)
    suffix = f"_limit{args.limit}" if args.limit else ""
    client = make_client(args.project, args.location)

    all_rows = []
    for model in models:
        out_path = results / f"validation_{model}{suffix}.jsonl"
        run_model(client, model, pairs, out_path, args.workers)
        all_rows += summarize(model, out_path, pairs)

    summary_path = results / f"validation_summary{suffix}.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    print()
    for row in all_rows:
        print(f"{row['model']:24s} {row['behavior']:8s} agreement {row['agreement_pct']:6.2f}% "
              f"[{row['agreement_ci_low_pct']:.1f}, {row['agreement_ci_high_pct']:.1f}]  "
              f"position-consistent {row['position_consistent_pct']:6.2f}%")
    overall = {r["model"]: r["agreement_pct"] for r in all_rows if r["behavior"] == "overall"}
    pick, reason = choose(overall)
    print(f"\nDECISION: {pick or 'none'} ({reason})" + ("  [quick check only]" if args.limit else ""))
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
