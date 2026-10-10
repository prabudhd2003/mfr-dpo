"""Pure CPU tests for automatic and blinded generation evaluation."""

import pandas as pd

from mfr_generation_eval import (
    parse_prometheus_choice,
    parse_wildguard_output,
    reconcile_pairwise_choices,
    summarize_pairwise,
    summarize_wildguard,
)
from mfr_review import make_pairwise_blind_review, score_pairwise_reviews


def test_wildguard_parser_and_xstest_summary():
    parsed = parse_wildguard_output(
        "Harmful request: no\nResponse refusal: yes\nHarmful response: no"
    )
    assert parsed == {
        "harmful_request": False,
        "response_refusal": True,
        "harmful_response": False,
        "parse_ok": True,
    }
    scores = pd.DataFrame([
        {**parsed, "xstest_label": "safe"},
        {**parse_wildguard_output(
            "Harmful request: yes\nResponse refusal: yes\nHarmful response: no"
        ), "xstest_label": "unsafe"},
    ])
    summary = summarize_wildguard(scores, "xstest")
    assert summary["over_refusal_rate_pct"] == 100.0
    assert summary["unsafe_refusal_rate_pct"] == 100.0
    assert summary["xstest_balanced_accuracy_pct"] == 50.0


def test_position_control_requires_consistent_winner():
    assert parse_prometheus_choice("Feedback... [RESULT] B") == "B"
    assert reconcile_pairwise_choices("A", "B") == ("A", True)
    assert reconcile_pairwise_choices("A", "A") == ("TIE", False)
    frame = pd.DataFrame({
        "final_choice": ["A", "B", "TIE"],
        "position_consistent": [True, True, False],
    })
    summary = summarize_pairwise(frame, "dapr", "lowest_margin")
    assert summary["candidate_score_pct"] == 50.0


def test_pairwise_human_review_is_blinded_and_scores_candidate(tmp_path):
    rows = []
    for behavior in ("helpful", "safe"):
        for index in range(3):
            for method in ("candidate", "baseline"):
                rows.append({
                    "id": f"{behavior}-{index}", "behavior": behavior,
                    "prompt": f"prompt {index}", "method": method,
                    "response": f"{method} response {index}",
                })
    sheet, key = make_pairwise_blind_review(
        pd.DataFrame(rows), "candidate", "baseline", prompts_per_behavior=2, seed=7
    )
    assert "method" not in sheet.columns
    assert len(sheet) == 4
    completed = sheet.copy()
    completed["preferred_response"] = completed["review_id"].map(
        key.set_index("review_id").apply(
            lambda row: "A" if row["method_A"] == "candidate" else "B", axis=1
        )
    )
    _, summary = score_pairwise_reviews([completed], key, seed=7, bootstrap_samples=100)
    overall = summary[summary["behavior"] == "overall"].iloc[0]
    assert overall["candidate_score_pct"] == 100.0
