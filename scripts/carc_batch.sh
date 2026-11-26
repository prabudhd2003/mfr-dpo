#!/bin/bash
# Run several training groups or locked tests one after another inside ONE Slurm job.
#   carc_batch.sh group  PROTOCOL OUTPUT_DIR LIST   (LIST lines: "order seed method1,method2")
#   carc_batch.sh locked PROTOCOL OUTPUT_DIR LIST   (LIST lines: one run directory each)
# A failed step is reported as "FAILED ..." and the job moves on. Finished work is skipped, so
# resubmitting the same list after a failure is safe. Locked lists must put each cell's `none`
# run before its other runs (the others reuse its Stage-1 scores).
set -uo pipefail
MODE="$1"; PROTOCOL="$2"; OUTPUT_DIR="$3"; LIST="$4"
# sbatch runs a temporary copy of this script from its spool folder, so find the repo from the
# protocol path (always REPO/configs/...), never from the script's own location.
REPO="$(cd "$(dirname "$PROTOCOL")/.." && pwd)"
if [ ! -f "$REPO/scripts/carc_eval_job.sh" ]; then echo "repo not found from protocol path: $REPO" >&2; exit 2; fi
ENV="$REPO/.conda/envs/mfr-dpo"
while read -r first second third; do
  [ -z "${first:-}" ] && continue
  case "$MODE" in
    group)
      echo "=== order $first seed $second: $third"
      bash "$REPO/scripts/carc_second_model_job.sh" group "$REPO" "$OUTPUT_DIR" "$ENV" l40s \
        "$PROTOCOL" "$first" "$second" "$third" || echo "FAILED order $first seed $second"
      ;;
    locked)
      if [ -f "$first/locked_test/results_test.csv" ]; then echo "skip $first"; continue; fi
      echo "=== $first"
      bash "$REPO/scripts/carc_eval_job.sh" locked-test "$REPO" "$OUTPUT_DIR" "$ENV" l40s \
        --run-dir "$first" --protocol "$PROTOCOL" || echo "FAILED $first"
      ;;
    *) echo "unknown mode $MODE (use group or locked)" >&2; exit 2 ;;
  esac
done < "$LIST"
