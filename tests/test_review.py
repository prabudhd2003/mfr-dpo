"""Blinded-review construction tests."""

import pandas as pd

import mfr_review


def test_blind_sheet_has_every_method_without_revealing_names():
    rows = []
    for prompt_id in ("p1", "p2"):
        for method in ("none", "random", "mfr"):
            rows.append({"id": prompt_id, "prompt": prompt_id, "method": method,
                         "response": f"{method}-{prompt_id}"})
    sheet, key = mfr_review.make_blind_review(pd.DataFrame(rows), ["none", "random", "mfr"], 2, seed=1)
    assert len(sheet) == 6 and len(key) == 6
    assert "method" not in sheet and set(sheet["system"]) == {"A", "B", "C"}
