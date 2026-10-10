#!/usr/bin/env python3
"""Run or resume one frozen-protocol continual DPO experiment."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import mfr_cache
import mfr_data
import mfr_dpo
from mfr_replay import METHODS as IMPLEMENTED_METHODS, ReplayBuffer
from mfr_utils import (allowed_seeds, file_sha256, load_protocol, mark_run_complete, method_anchor_strength,
                       method_ewc_coefficient, method_old_per_step, run_info, save_json_atomic,
                       seed_everything, stage_seed,
                       stage1_compatibility_sha256,
                       validate_resume_settings, validate_stage1_source)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", required=True,
        help="CARC artifact root; run folders are written below OUTPUT_DIR/runs",
    )
    parser.add_argument("--order", type=int, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--start-stage", type=int, default=1)
    parser.add_argument("--stage1-from", help="Compatible completed run directory whose stage 1 should be reused")
    parser.add_argument("--reference-cache", help="Optional CSV made by build_reference_cache.py")
    parser.add_argument(
        "--protocol", default=str(ROOT / "configs" / "experiment_protocol.json"),
        help="Authoritative protocol JSON (default: primary Qwen protocol)",
    )
    return parser.parse_args()


def resolve_source_run(recorded, run_dir):
    """Use the recorded source path or the equivalent sibling under this artifact root."""
    source = Path(recorded)
    if source.exists():
        return source
    sibling = run_dir.parent / source.name
    return sibling if sibling.exists() else source


def score_sets(model, tokenizer, splits, stage, trained_on, run_meta, run_dir, protocol, reference_cache):
    rows = []
    for dataset in ("safe", "helpful", "quality"):
        scores = mfr_dpo.score_pairs(
            model, tokenizer, splits[dataset]["val"], beta=protocol["beta"],
            max_tokens=protocol["max_tokens"], reference_cache=reference_cache,
            desc=f"stage {stage}: {dataset} validation",
        )
        scores.insert(0, "id", splits[dataset]["val"]["id"].values)
        folder = run_dir / ("stage0_base" if stage == 0 else f"stage{stage}_{trained_on}")
        folder.mkdir(parents=True, exist_ok=True)
        scores.to_csv(folder / f"margins_{dataset}_val.csv", index=False)
        rows.append({**run_meta, "stage": stage, "trained_on": trained_on, "eval_set": dataset,
                     **mfr_dpo.summarize(scores)})
    return pd.DataFrame(rows)


def update_buffer(model, tokenizer, buffer, stage, dataset, train_df, stage_dir, protocol,
                  reference_cache, method, anchors=None):
    candidates = buffer.candidates(dataset, train_df)
    peak_scores = mfr_dpo.score_pairs(
        model, tokenizer, candidates, beta=protocol["beta"], max_tokens=protocol["max_tokens"],
        reference_cache=reference_cache, desc=f"stage {stage}: buffer peak"
    )
    buffer.add_stage(
        dataset, candidates, peak_scores["margin"], policy_margin=peak_scores["policy_margin"]
    )
    if method in mfr_dpo.ANCHOR_METHODS:
        if anchors is None:
            raise ValueError(f"{method} requires a mutable preference-anchor store")
        anchors.update(mfr_dpo.score_preference_anchors(
            model, tokenizer, candidates, max_tokens=protocol["max_tokens"],
            desc=f"stage {stage}: peak token anchors",
        ))
    older_rows = buffer.rows(exclude_dataset=dataset)
    if len(older_rows):
        current_scores = mfr_dpo.score_pairs(
            model, tokenizer, older_rows, beta=protocol["beta"], max_tokens=protocol["max_tokens"],
            reference_cache=reference_cache, desc=f"stage {stage}: older buffer state"
        )
        if method == "fmcr":
            # Together these frames cover every row that survived buffer rebalancing.
            all_scores = pd.concat([current_scores, peak_scores])
            all_scores = all_scores[~all_scores.index.duplicated(keep="last")]
            buffer.set_forecast_scores(
                all_scores, velocity_decay=protocol["fmcr_velocity_decay"], initialize=False
            )
        else:
            buffer.set_current(current_scores["margin"], policy_margin=(
                current_scores["policy_margin"] if method == "mfr_at_risk" else None))
    buffer.to_csv(stage_dir / "buffer.csv")
    if method in mfr_dpo.ANCHOR_METHODS:
        kept = set(buffer.rows()["id"])
        for pair_id in list(anchors):
            if pair_id not in kept:
                del anchors[pair_id]
        mfr_dpo.save_preference_anchors(anchors, stage_dir / "preference_anchors.npz")
    return buffer


def main():
    args = parse_args()
    protocol = load_protocol(args.protocol)
    order_key = str(args.order)
    if order_key not in protocol["orders"]:
        choices = ", ".join(sorted(protocol["orders"], key=int))
        raise ValueError(f"order {args.order} is not in the protocol; choose one of {choices}")
    configured_methods = protocol.get("methods", []) + protocol.get("secondary_methods", [])
    if args.method not in configured_methods:
        raise ValueError(
            f"method {args.method!r} is not in the protocol; choose one of {configured_methods}"
        )
    if args.method not in IMPLEMENTED_METHODS:
        raise ValueError(
            f"method {args.method!r} is configured but not implemented in src/mfr_replay.py"
        )
    if args.seed not in allowed_seeds(protocol, args.method):
        raise ValueError(
            f"seed {args.seed} is not allowed for {args.method}; "
            f"choose one of {allowed_seeds(protocol, args.method)}"
        )
    order = protocol["orders"][order_key]
    n_stages = len(order)
    if not 1 <= args.start_stage <= n_stages:
        raise ValueError(f"start stage must be between 1 and {n_stages}")
    old_per_step = method_old_per_step(protocol, args.method)
    anchor_strength = method_anchor_strength(protocol, args.method)
    ewc_coefficient = method_ewc_coefficient(protocol, args.method)
    run_name = f"{protocol['data_version']}_o{args.order}_{args.method}_s{args.seed}"
    print(
        f"Starting {run_name}: order={' -> '.join(order)}, "
        f"batch={protocol['new_per_step']} new + {old_per_step} replay",
        flush=True,
    )
    artifact_root = Path(args.output_dir).expanduser().resolve()
    run_dir = artifact_root / "runs" / run_name
    data_dir = ROOT / protocol["data_dir"]
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Run scripts/prepare_data.py first; missing {manifest_path}")
    manifest_hash = file_sha256(manifest_path)
    print("Loading and verifying the frozen data...", flush=True)
    splits = mfr_data.load_splits(data_dir)  # also verifies every JSONL against the manifest
    reference_cache = None
    if args.reference_cache:
        reference_cache = mfr_cache.load_reference_cache(args.reference_cache, {
            "protocol_version": protocol["protocol_version"], "data_version": protocol["data_version"],
            "model_revision": protocol["model_revision"], "data_manifest_sha256": manifest_hash,
        })
        reference_cache = reference_cache.set_index("id", drop=False)
    settings = {
        "run_name": run_name, "order_id": args.order, "order": order, "method": args.method,
        "seed": args.seed, "protocol_version": protocol["protocol_version"],
        "data_version": protocol["data_version"], "data_manifest_sha256": manifest_hash,
        "model_name": protocol["model_name"], "model_revision": protocol["model_revision"],
        "lr": protocol["learning_rate"], "beta": protocol["beta"],
        "new_per_step": protocol["new_per_step"], "old_per_step": old_per_step,
        "max_tokens": protocol["max_tokens"], "buffer_size": protocol["buffer_size"],
        "refreshes": protocol["refreshes"], "stage1_from": args.stage1_from,
        "stage1_run_name": Path(args.stage1_from).name if args.stage1_from else None,
        "lora_r": protocol["lora_r"], "lora_alpha": protocol["lora_alpha"],
        "lora_dropout": protocol["lora_dropout"], "epochs": protocol["epochs"],
        "micro_batch": protocol["micro_batch"],
        "fmcr_velocity_decay": protocol["fmcr_velocity_decay"],
        "fmcr_forecast_horizon": protocol["fmcr_forecast_horizon"],
        "anchor_strength": anchor_strength,
        "dapr_huber_delta": protocol["dapr_huber_delta"],
        "mir_lookahead_steps": protocol["mir_lookahead_steps"],
        "ewc_coefficient": (ewc_coefficient if args.method in mfr_dpo.EWC_METHODS else None),
        "ewc_fisher_pairs": (protocol["ewc_fisher_pairs"]
                             if args.method in mfr_dpo.EWC_METHODS else None),
        "ewc_fisher_batch_size": (protocol["ewc_fisher_batch_size"]
                                  if args.method in mfr_dpo.EWC_METHODS else None),
        "cpmr_rule": ("min_current_projected_one_refresh_interval_v1"
                      if args.method == "cpmr" else None),
        "dapr_rule": ("lowest_margin_directional_peak_token_anchor_v1"
                      if args.method == "dapr" else
                      "lowest_margin_weak_directional_peak_token_anchor_v1"
                      if args.method == "dapr_weak" else
                      "lowest_margin_live_margin_gated_directional_peak_token_anchor_v1"
                      if args.method == "dapr_gated" else
                      "lowest_margin_centered_directional_peak_token_anchor_v1"
                      if args.method == "dapr_c" else None),
        "mir_dpo_rule": ("one_incoming_step_dpo_loss_increase_v1"
                         if args.method == "mir_dpo" else None),
        "copr_adapted_rule": ("lowest_margin_peak_pair_distribution_mse_v1"
                              if args.method == "copr_adapted" else None),
        "ewc_rule": ("multi_anchor_diagonal_empirical_fisher_lora_only_v1"
                     if args.method in mfr_dpo.EWC_METHODS else None),
    }
    settings["stage1_compatibility_sha256"] = stage1_compatibility_sha256(settings)
    current_info = run_info(ROOT)
    if current_info.get("git_dirty"):
        raise RuntimeError("refusing to start/resume a scientific run from an uncommitted Git checkout")
    if (run_dir / "COMPLETE.json").exists():
        raise RuntimeError(f"run is already complete and will not be overwritten: {run_dir}")
    if args.start_stage == 1 and (run_dir / "results.csv").exists():
        raise RuntimeError("results already exist; set --start-stage to the first unfinished stage")
    run_dir.mkdir(parents=True, exist_ok=True)
    current_record = {**settings, **current_info}
    existing_settings = run_dir / "settings.json"
    if existing_settings.exists():
        with open(existing_settings, encoding="utf-8") as handle:
            old = json.load(handle)
        validate_resume_settings(current_record, old)
        resume_event = {
            "resumed": current_info.get("started"), "git_commit": current_info.get("git_commit"),
            "scientific_code_sha256": current_info.get("scientific_code_sha256"),
            "gpu": current_info.get("gpu"), "packages": current_info.get("packages"),
        }
        saved_settings = {**old, "resume_events": [*old.get("resume_events", []), resume_event]}
    else:
        saved_settings = current_record
    save_json_atomic(saved_settings, existing_settings)
    results = pd.DataFrame()
    buffer = ReplayBuffer(protocol["buffer_size"], args.seed)
    first_stage = args.start_stage
    previous_adapter = None
    if args.start_stage > 1:
        if args.start_stage == 2 and args.stage1_from:
            source_run = resolve_source_run(args.stage1_from, run_dir)
            if not (source_run / "COMPLETE.json").exists():
                raise RuntimeError(f"stage-1 source run is not complete: {source_run}")
            with open(source_run / "settings.json", encoding="utf-8") as handle:
                source_settings = json.load(handle)
            validate_stage1_source(current_record, source_settings)
            previous_adapter = source_run / f"stage1_{order[0]}"
            buffer = ReplayBuffer.from_csv(previous_adapter / "buffer.csv", protocol["buffer_size"], args.seed)
            source_results = pd.read_csv(source_run / "results.csv")
            results = source_results[source_results["stage"] <= 1].copy()
            results["run_name"], results["method"] = run_name, args.method
        else:
            previous_adapter = run_dir / f"stage{args.start_stage - 1}_{order[args.start_stage - 2]}"
            buffer = ReplayBuffer.from_csv(previous_adapter / "buffer.csv", protocol["buffer_size"], args.seed)
            results = pd.read_csv(run_dir / "results.csv")
            results = results[results["stage"] < args.start_stage]
    elif args.stage1_from:
        source_run = resolve_source_run(args.stage1_from, run_dir)
        if not (source_run / "COMPLETE.json").exists():
            raise RuntimeError(f"stage-1 source run is not complete: {source_run}")
        with open(source_run / "settings.json", encoding="utf-8") as handle:
            source_settings = json.load(handle)
        validate_stage1_source(current_record, source_settings)
        previous_adapter = source_run / f"stage1_{order[0]}"
        if not previous_adapter.exists():
            raise FileNotFoundError(previous_adapter)
        buffer = ReplayBuffer.from_csv(previous_adapter / "buffer.csv", protocol["buffer_size"], args.seed)
        source_results = pd.read_csv(source_run / "results.csv")
        results = source_results[source_results["stage"] <= 1].copy()
        results["run_name"], results["method"] = run_name, args.method
        first_stage = 2

    seed_everything(args.seed)
    print(f"Loading {protocol['model_name']} and the starting adapter...", flush=True)
    model, tokenizer = mfr_dpo.load_model(
        protocol["model_name"], protocol["lora_r"], adapter_path=previous_adapter,
        revision=protocol["model_revision"],
        lora_alpha=protocol["lora_alpha"], lora_dropout=protocol["lora_dropout"],
    )
    print("Model loaded.", flush=True)
    if reference_cache is not None:
        canary = mfr_cache.verify_reference_cache(
            model, tokenizer, splits["helpful"]["val"], reference_cache,
            beta=protocol["beta"], max_tokens=protocol["max_tokens"],
        )
        with open(existing_settings, encoding="utf-8") as handle:
            saved_settings = json.load(handle)
        saved_settings["reference_cache_canary"] = canary
        save_json_atomic(saved_settings, existing_settings)
        print("Reference cache canary passed:", canary)
    anchors = {} if args.method in mfr_dpo.ANCHOR_METHODS else None
    if args.method in mfr_dpo.ANCHOR_METHODS and len(buffer):
        anchor_path = Path(previous_adapter) / "preference_anchors.npz" if previous_adapter else None
        if anchor_path is not None and anchor_path.exists():
            anchors = mfr_dpo.load_preference_anchors(anchor_path)
            missing = set(buffer.rows()["id"]) - set(anchors)
            if missing:
                raise ValueError(f"saved peak anchors are missing buffer pairs: {sorted(missing)[:3]}")
            print(f"Loaded {len(anchors)} peak preference anchors.", flush=True)
        else:
            print("Capturing peak preference anchors from the starting checkpoint...", flush=True)
            anchors = mfr_dpo.score_preference_anchors(
                model, tokenizer, buffer.rows(), max_tokens=protocol["max_tokens"],
                desc="initial peak token anchors",
            )
    ewc_states = []
    if args.method in mfr_dpo.EWC_METHODS and len(buffer):
        state_path = Path(previous_adapter) / "ewc_states.pt" if previous_adapter else None
        if state_path is not None and state_path.exists():
            ewc_states = mfr_dpo.load_ewc_states(state_path, model)
            print(f"Loaded LoRA-EWC state for {len(ewc_states)} earlier behavior(s).", flush=True)
        else:
            if first_stage > 2:
                raise FileNotFoundError(
                    f"cannot resume EWC at stage {first_stage}; missing {state_path}"
                )
            fisher_rows = buffer.rows().head(protocol["ewc_fisher_pairs"])
            print("Estimating the Stage-1 LoRA Fisher from the frozen buffer sample...", flush=True)
            ewc_states = [mfr_dpo.estimate_lora_fisher(
                model, tokenizer, fisher_rows, beta=protocol["beta"],
                batch_size=protocol["ewc_fisher_batch_size"],
                max_tokens=protocol["max_tokens"], reference_cache=reference_cache,
                desc="stage 1: LoRA Fisher",
            )]
    run_meta = {"run_name": run_name, "order_id": args.order, "method": args.method, "seed": args.seed,
                "protocol_version": protocol["protocol_version"], "data_version": protocol["data_version"]}
    if first_stage == 1:
        results = score_sets(model, tokenizer, splits, 0, "base", run_meta, run_dir, protocol, reference_cache)

    for stage in range(first_stage, n_stages + 1):
        dataset = order[stage - 1]
        print(f"\nStage {stage}/{n_stages}: training on {dataset}", flush=True)
        stage_dir = run_dir / f"stage{stage}_{dataset}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        history, replay_log = mfr_dpo.train_stage_replay(
            model, tokenizer, splits[dataset]["train"], buffer=buffer, method=args.method,
            beta=protocol["beta"], lr=protocol["learning_rate"],
            new_per_step=protocol["new_per_step"], old_per_step=old_per_step,
            refreshes=protocol["refreshes"], micro_batch=protocol["micro_batch"],
            max_tokens=protocol["max_tokens"], seed=stage_seed(args.seed, stage),
            max_share_per_dataset=protocol["max_share_per_dataset"],
            desc=f"stage {stage}: {dataset}", progress_path=stage_dir / "history.csv",
            reference_cache=reference_cache,
            fmcr_velocity_decay=protocol["fmcr_velocity_decay"],
            fmcr_forecast_horizon=protocol["fmcr_forecast_horizon"],
            anchors=anchors, anchor_strength=anchor_strength,
            huber_delta=protocol["dapr_huber_delta"],
            mir_lookahead_steps=protocol["mir_lookahead_steps"],
            ewc_states=ewc_states, ewc_coefficient=ewc_coefficient,
        )
        model.save_pretrained(stage_dir)
        print(f"Stage {stage}/{n_stages} training saved; scoring validation sets...", flush=True)
        history.to_csv(stage_dir / "history.csv", index=False)
        replay_log.to_csv(stage_dir / "replay_log.csv", index=False)
        stage_results = score_sets(
            model, tokenizer, splits, stage, dataset, run_meta, run_dir, protocol, reference_cache
        )
        results = pd.concat([results, stage_results], ignore_index=True)
        if stage < len(order):
            if args.method in mfr_dpo.EWC_METHODS:
                fisher_rows = buffer.candidates(dataset, splits[dataset]["train"])
                fisher_rows = fisher_rows.head(protocol["ewc_fisher_pairs"])
                ewc_states.append(mfr_dpo.estimate_lora_fisher(
                    model, tokenizer, fisher_rows, beta=protocol["beta"],
                    batch_size=protocol["ewc_fisher_batch_size"],
                    max_tokens=protocol["max_tokens"], reference_cache=reference_cache,
                    desc=f"stage {stage}: LoRA Fisher",
                ))
                mfr_dpo.save_ewc_states(ewc_states, stage_dir / "ewc_states.pt")
            buffer = update_buffer(
                model, tokenizer, buffer, stage, dataset, splits[dataset]["train"], stage_dir,
                protocol, reference_cache, args.method, anchors=anchors,
            )
        # A stage becomes resumably complete only after all method-specific state and the next
        # stage's buffer have been saved. This prevents an interrupted Fisher estimate from making
        # the group runner skip directly to a stage whose EWC state does not exist.
        results.to_csv(run_dir / "results.csv", index=False)

    expected = [run_dir / "settings.json", run_dir / "results.csv"] + [
        run_dir / f"stage{stage}_{dataset}" / "history.csv" for stage, dataset in enumerate(order, 1)
    ]
    if args.stage1_from:
        expected = [path for path in expected if "stage1_" not in str(path)]
    if args.method in mfr_dpo.ANCHOR_METHODS and n_stages > 1:
        expected.append(
            run_dir / f"stage{n_stages - 1}_{order[n_stages - 2]}" / "preference_anchors.npz"
        )
    if args.method in mfr_dpo.EWC_METHODS and n_stages > 1:
        expected.append(
            run_dir / f"stage{n_stages - 1}_{order[n_stages - 2]}" / "ewc_states.pt"
        )
    mark_run_complete(run_dir, expected)
    print(results.tail(9).to_string(index=False))
    print(f"Complete: {run_dir}")


if __name__ == "__main__":
    main()
