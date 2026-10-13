# Gemini judge (second, independent judge for Helpful/Quality)

Runs from your Mac through **Vertex AI** (not AI Studio), billed to the **Billing Account for
Education**. Step 1 is a billing test that spends under $1, to check that the credits cover Gemini
calls. Only if it does do we add Gemini as a second judge (code comes after the test).

## One-time setup (Mac)

1. **Project on the Education billing account.** In the Google Cloud console: create a project
   (ours: `mfr-dpo-judge`), then Billing -> link it to **Billing Account for Education**.
   Then APIs & Services -> enable **Vertex AI API**.
2. **Google Cloud CLI and login.**
   ```bash
   brew install --cask google-cloud-sdk            # skip if `gcloud --version` already works
   gcloud auth application-default login
   gcloud auth application-default set-quota-project YOUR_PROJECT_ID
   ```
   The login is stored in `~/.config/gcloud`, never in this repo.
3. **Python environment** (needs Python 3.10+; check with `python3 --version`).
   ```bash
   cd ~/Desktop/CSCI544/code/mfr-dpo/gemini_judge
   python3 -m venv gemini-judge-venv --prompt gemini-judge
   source gemini-judge-venv/bin/activate
   pip install -r requirements.txt
   ```

## Billing test (< $1)

```bash
source gemini-judge-venv/bin/activate
python billing_test.py --project YOUR_PROJECT_ID --list                       # which Gemini models exist
python billing_test.py --project YOUR_PROJECT_ID --model THE_PRO_MODEL_ID     # 10 small calls
```

It stops by itself if the estimate passes $1. About 24 hours later, open Billing -> Reports, filter
to the project, and check that the cost is matched by a negative **credits** amount (and that the
remaining credit on the Credits page went down). If the cost shows as payable, the credits do not
cover Gemini and we do not use it.

`billing_test_log.jsonl` (git-ignored) records what was run.

## Judge validation (3 candidates, 400 validation pairs)

```bash
source gemini-judge-venv/bin/activate
python validate_gemini.py --project mfr-dpo-judge --limit 5     # quick check, ~1 min, cents
python validate_gemini.py --project mfr-dpo-judge               # full run, ~2,400 calls, < $20
```

Each pair is judged in both A/B orders with the same rubric as the rest of the evaluation; it agrees
with the humans only if both orders pick the human-chosen response. Results are saved after every
call in `results/validation_<model>.jsonl` (git-ignored), so re-running the same command resumes.
The summary is `results/validation_summary.csv`, and the last line prints the decision.

Rule (fixed 2026-10-12, before any result): highest overall agreement wins and must reach 70%; if a
stable model (2.5 Pro, 3.8 Flash) is within 2 points of the winner, the stable model wins.
