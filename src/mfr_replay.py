"""Replay memory and auditable replay-selection rules.

The buffer holds `size` old preference pairs (500 by default), split evenly across the stages
already learned. For every stored pair we keep:

    peak_margin     its margin right after its own stage finished ("how well it was learned")
    current_margin  its margin the last time we re-scored the buffer

The methods differ in which stored pairs fill the replay slots of each training step:

    none            nothing is replayed
    random          uniform sample from the buffer
    random_high     uniform sample from the buffer with a larger budget set by the runner
    lowest_margin   the lowest current_margin (hard pairs, forgotten or not)
    mfr             the largest drop, peak_margin - current_margin (our method)
    fmcr            pairs forecast to cross a preference boundary before the next refresh
    cpmr            the lowest projected margin after a counterfactual new-task update interval
    dapr            lowest-margin replay plus directional peak-token anchoring
    dapr_weak       DAPR with a ten-times weaker directional anchor
    dapr_gated      DAPR whose anchor activates only after a live margin regression
    dapr_c          DAPR with common-shift-centred token anchoring
    mir_dpo         largest one-step increase in old-pair DPO loss (MIR adapted to DPO)
    copr_adapted    lowest-margin replay plus a peak pair-distribution constraint
    ewc_100         no replay; LoRA-EWC parameter regularization with coefficient 100
    ewc_1000        no replay; LoRA-EWC parameter regularization with coefficient 1,000
    ewc_10000       no replay; LoRA-EWC parameter regularization with coefficient 10,000

Everything here is CPU-only and fully seeded: which pairs enter the buffer depends on the run seed
and the dataset, never on the method, so every method stores exactly the same candidate pairs.
"""

import numpy as np
import pandas as pd

from mfr_utils import buffer_seed

METHODS = (
    "none", "random", "random_high", "lowest_margin", "mfr", "fmcr", "cpmr",
    "dapr", "dapr_weak", "dapr_gated", "dapr_c", "mir_dpo", "copr_adapted",
    "ewc_100", "ewc_1000", "ewc_10000", "mfr_balanced", "mfr_at_risk",
)
REGULARIZATION_METHODS = ("ewc_100", "ewc_1000", "ewc_10000")
REFRESH_METHODS = (
    "lowest_margin", "mfr", "fmcr", "cpmr", "dapr", "dapr_weak", "dapr_gated",
    "dapr_c", "mir_dpo", "copr_adapted", "mfr_balanced", "mfr_at_risk",
)
FORECAST_COLUMNS = [
    "previous_margin", "margin_velocity",
    "current_policy_margin", "previous_policy_margin", "policy_velocity",
]
COUNTERFACTUAL_COLUMNS = ["projected_margin", "predicted_drop"]
MIR_COLUMNS = [
    "current_margin_sum", "projected_margin_sum", "current_dpo_loss",
    "projected_dpo_loss", "interference_score",
]
EXTRA_COLUMNS = [
    "dataset", "peak_margin", "current_margin", *FORECAST_COLUMNS, *COUNTERFACTUAL_COLUMNS,
    *MIR_COLUMNS,
]
PLAN_COLUMNS = ["id", "dataset", "selection_score", "selection_rank", "method"]
FMCR_PLAN_COLUMNS = [
    *PLAN_COLUMNS, "risk_tier", "risk_label", "historical_drop",
    "current_margin", "current_policy_margin", "margin_velocity", "policy_velocity",
    "forecast_margin", "forecast_policy_margin", "time_to_crossing", "dataset_quota",
]
CPMR_PLAN_COLUMNS = [
    *PLAN_COLUMNS, "current_margin", "projected_margin", "worst_case_margin",
    "predicted_drop", "historical_drop",
]
BALANCED_PLAN_COLUMNS = [*PLAN_COLUMNS, "historical_drop", "current_margin", "dataset_quota"]
AT_RISK_PLAN_COLUMNS = [
    *PLAN_COLUMNS, "risk_tier", "historical_drop", "current_margin", "current_policy_margin",
    "dataset_quota",
]
MIR_PLAN_COLUMNS = [
    *PLAN_COLUMNS, "current_margin", "current_margin_sum", "projected_margin_sum",
    "current_dpo_loss", "projected_dpo_loss", "interference_score", "historical_drop",
]


def needs_refresh(method):
    """Whether a selection rule needs current margins rescored during a stage."""
    if method not in METHODS:
        raise ValueError(f"unknown replay method {method!r}; expected one of {METHODS}")
    return method in REFRESH_METHODS


def uses_replay(method):
    """Whether the method places stored pairs in later training batches."""
    if method not in METHODS:
        raise ValueError(f"unknown replay method {method!r}; expected one of {METHODS}")
    return method != "none" and method not in REGULARIZATION_METHODS


class ReplayBuffer:
    """Memory of old pairs. `rows()` returns a DataFrame with the split's columns plus EXTRA_COLUMNS."""

    def __init__(self, size=500, seed=0):
        self.size = int(size)
        self.seed = int(seed)
        self._rows = pd.DataFrame()

    # ---------------------------------------------------------------- filling

    def candidates(self, dataset, train_df):
        """The pairs of a finished stage that are eligible for the buffer (seeded, method-independent).

        Returns at most `size` rows of `train_df`; they still need to be scored (their peak margin)
        before add_stage().
        """
        n = min(self.size, len(train_df))
        return train_df.sample(n=n, random_state=buffer_seed(self.seed, dataset)).reset_index(drop=True)

    def add_stage(self, dataset, rows, peak_margin, policy_margin=None):
        """Store a finished stage's candidates with their peak margins, then rebalance the buffer.

        peak_margin: Series indexed by pair id (from mfr_dpo.score_pairs(...)["margin"]).
        After this call the full buffer has exactly ``size`` rows whenever enough candidates exist;
        any remainder is assigned deterministically to the earliest stored datasets.
        current_margin starts equal to peak_margin. Optional policy_margin is the policy's
        absolute chosen-vs-rejected margin and is needed by forecast-based replay.
        """
        new = rows.copy()
        new["dataset"] = dataset
        new["peak_margin"] = new["id"].map(peak_margin).astype(float)
        if new["peak_margin"].isna().any():
            missing = new.loc[new["peak_margin"].isna(), "id"].tolist()[:3]
            raise ValueError(f"peak margins missing for {missing} ...")
        new["current_margin"] = new["peak_margin"]
        new["previous_margin"] = new["peak_margin"]
        new["margin_velocity"] = 0.0
        if policy_margin is None:
            new["current_policy_margin"] = np.nan
        else:
            new["current_policy_margin"] = new["id"].map(policy_margin).astype(float)
            if new["current_policy_margin"].isna().any():
                missing = new.loc[new["current_policy_margin"].isna(), "id"].tolist()[:3]
                raise ValueError(f"policy margins missing for {missing} ...")
        new["previous_policy_margin"] = new["current_policy_margin"]
        new["policy_velocity"] = 0.0
        new["projected_margin"] = new["current_margin"]
        new["predicted_drop"] = 0.0
        new["current_margin_sum"] = np.nan
        new["projected_margin_sum"] = np.nan
        new["current_dpo_loss"] = np.nan
        new["projected_dpo_loss"] = np.nan
        new["interference_score"] = np.nan

        self._rows = pd.concat([self._rows, new], ignore_index=True) if len(self._rows) else new
        self._rebalance()
        return self

    def _rebalance(self):
        """Keep the buffer at `size` pairs, evenly split across the datasets in it (seeded)."""
        datasets = list(dict.fromkeys(self._rows["dataset"]))
        base, remainder = divmod(self.size, len(datasets))
        kept = []
        for index, dataset in enumerate(datasets):
            part = self._rows[self._rows["dataset"] == dataset]
            target = base + (1 if index < remainder else 0)
            if len(part) > target:
                part = part.sample(n=target, random_state=buffer_seed(self.seed, dataset) + len(datasets))
            kept.append(part.sort_values("id"))
        self._rows = pd.concat(kept, ignore_index=True)

    # ---------------------------------------------------------------- reading

    def rows(self, exclude_dataset=None):
        out = self._rows
        if exclude_dataset is not None:
            out = out[out["dataset"] != exclude_dataset]
        return out.reset_index(drop=True)

    def get(self, ids):
        """The stored rows for these ids, in the given order, as a list of dicts (ready for make_batch)."""
        by_id = self._rows.set_index("id")
        return [{"id": pair_id, **by_id.loc[pair_id].to_dict()} for pair_id in ids]

    def set_current(self, margin, policy_margin=None):
        """Update current margins from a fresh scoring pass (Series indexed by pair id).

        policy_margin (optional, At-Risk MFR) also updates the absolute policy margin. Pairs not in
        the scoring pass keep their stored values, exactly like the relative margin.
        """
        updated = self._rows["id"].map(margin)
        self._rows["previous_margin"] = self._rows["current_margin"].astype(float)
        self._rows["current_margin"] = updated.fillna(self._rows["current_margin"]).astype(float)
        if policy_margin is not None:
            policy = self._rows["id"].map(policy_margin)
            self._rows["current_policy_margin"] = (
                policy.fillna(self._rows["current_policy_margin"]).astype(float))

    def set_forecast_scores(self, scores, velocity_decay=0.5, initialize=False):
        """Update both margin trajectories used by FMCR.

        ``scores`` must be indexed by pair id and contain ``margin`` and ``policy_margin``.
        At the beginning of every new training stage, ``initialize=True`` resets velocities so
        slopes from the previous task are never extrapolated into the next task. Later updates use
        an exponential moving average of the per-refresh change.
        """
        required = {"margin", "policy_margin"}
        missing_columns = required - set(scores.columns)
        if missing_columns:
            raise ValueError(f"forecast scores are missing columns {sorted(missing_columns)}")
        if not 0 <= float(velocity_decay) < 1:
            raise ValueError("velocity_decay must be in [0, 1)")

        relative = self._rows["id"].map(scores["margin"])
        policy = self._rows["id"].map(scores["policy_margin"])
        if relative.isna().any() or policy.isna().any():
            missing = self._rows.loc[relative.isna() | policy.isna(), "id"].tolist()[:3]
            raise ValueError(f"forecast scores missing for {missing} ...")
        relative = relative.astype(float)
        policy = policy.astype(float)

        if initialize:
            self._rows["previous_margin"] = relative
            self._rows["current_margin"] = relative
            self._rows["margin_velocity"] = 0.0
            self._rows["previous_policy_margin"] = policy
            self._rows["current_policy_margin"] = policy
            self._rows["policy_velocity"] = 0.0
            return

        if self._rows["current_policy_margin"].isna().any():
            raise ValueError("initialize forecast scores before updating a buffer with no policy-margin history")
        old_relative = self._rows["current_margin"].astype(float)
        old_policy = self._rows["current_policy_margin"].astype(float)
        old_relative_velocity = self._rows["margin_velocity"].fillna(0.0).astype(float)
        old_policy_velocity = self._rows["policy_velocity"].fillna(0.0).astype(float)
        keep = float(velocity_decay)
        learn = 1.0 - keep

        self._rows["previous_margin"] = old_relative
        self._rows["current_margin"] = relative
        self._rows["margin_velocity"] = keep * old_relative_velocity + learn * (relative - old_relative)
        self._rows["previous_policy_margin"] = old_policy
        self._rows["current_policy_margin"] = policy
        self._rows["policy_velocity"] = keep * old_policy_velocity + learn * (policy - old_policy)

    def set_counterfactual_scores(self, current_margin, projected_margin):
        """Store CPMR's present and counterfactual future margins.

        Both inputs are Series indexed by pair id. ``predicted_drop`` is positive when training on
        the upcoming new-task interval without replay is forecast to damage an old preference.
        """
        current = self._rows["id"].map(current_margin)
        projected = self._rows["id"].map(projected_margin)
        if current.isna().any() or projected.isna().any():
            missing = self._rows.loc[current.isna() | projected.isna(), "id"].tolist()[:3]
            raise ValueError(f"counterfactual scores missing for {missing} ...")
        self._rows["previous_margin"] = self._rows["current_margin"].astype(float)
        self._rows["current_margin"] = current.astype(float)
        self._rows["projected_margin"] = projected.astype(float)
        self._rows["predicted_drop"] = current.astype(float) - projected.astype(float)

    def set_mir_scores(self, current_scores, projected_scores):
        """Store the one-step DPO-loss increase used by MIR-DPO.

        DPO loss for one pair is ``softplus(-margin_sum)``. A positive interference score means
        that a virtual update on the incoming task makes the old preference harder.
        """
        for frame, label in ((current_scores, "current"), (projected_scores, "projected")):
            if not {"margin", "margin_sum"} <= set(frame.columns):
                raise ValueError(f"MIR-DPO {label} scores require margin and margin_sum")
        ids = self._rows["id"]
        current_margin = ids.map(current_scores["margin"])
        current_sum = ids.map(current_scores["margin_sum"])
        projected_sum = ids.map(projected_scores["margin_sum"])
        if current_margin.isna().any() or current_sum.isna().any() or projected_sum.isna().any():
            missing = self._rows.loc[
                current_margin.isna() | current_sum.isna() | projected_sum.isna(), "id"
            ].tolist()[:3]
            raise ValueError(f"MIR-DPO scores missing for {missing} ...")
        current_loss = np.logaddexp(0.0, -current_sum.astype(float))
        projected_loss = np.logaddexp(0.0, -projected_sum.astype(float))
        self._rows["previous_margin"] = self._rows["current_margin"].astype(float)
        self._rows["current_margin"] = current_margin.astype(float)
        self._rows["current_margin_sum"] = current_sum.astype(float)
        self._rows["projected_margin_sum"] = projected_sum.astype(float)
        self._rows["current_dpo_loss"] = current_loss
        self._rows["projected_dpo_loss"] = projected_loss
        self._rows["interference_score"] = projected_loss - current_loss

    def forecast_ready(self):
        """Whether every buffered pair has the trajectory state required by FMCR."""
        return all(column in self._rows and not self._rows[column].isna().any()
                   for column in FORECAST_COLUMNS)

    def forgetting(self):
        """peak - current per pair (positive = forgotten)."""
        return (self._rows["peak_margin"] - self._rows["current_margin"]).set_axis(self._rows["id"])

    def __len__(self):
        return len(self._rows)

    # ---------------------------------------------------------------- saving

    def to_csv(self, path):
        self._rows.to_csv(path, index=False)

    @classmethod
    def from_csv(cls, path, size=500, seed=0):
        buffer = cls(size=size, seed=seed)
        buffer._rows = pd.read_csv(path)
        # Old Stage-1 buffers predate FMCR. Relative history can be initialized exactly from the
        # saved current margin. Absolute policy margins remain missing until a mandatory live
        # scoring pass at the beginning of the FMCR stage.
        if "previous_margin" not in buffer._rows:
            buffer._rows["previous_margin"] = buffer._rows["current_margin"]
        if "margin_velocity" not in buffer._rows:
            buffer._rows["margin_velocity"] = 0.0
        if "current_policy_margin" not in buffer._rows:
            buffer._rows["current_policy_margin"] = np.nan
        if "previous_policy_margin" not in buffer._rows:
            buffer._rows["previous_policy_margin"] = buffer._rows["current_policy_margin"]
        if "policy_velocity" not in buffer._rows:
            buffer._rows["policy_velocity"] = 0.0
        if "projected_margin" not in buffer._rows:
            buffer._rows["projected_margin"] = buffer._rows["current_margin"]
        if "predicted_drop" not in buffer._rows:
            buffer._rows["predicted_drop"] = 0.0
        for column in MIR_COLUMNS:
            if column not in buffer._rows:
                buffer._rows[column] = np.nan
        return buffer


# -------------------------------------------------------------------- choosing

def plan_interval(buffer, method, n_slots, rng, max_share_per_dataset=None,
                  plan_index=0, forecast_horizon=1.0):
    """Pair ids for the next interval's replay slots (one id per slot, in order).

    none: empty list. random/random_high: uniform without replacement. lowest_margin: lowest current margin.
    mfr: largest peak - current. Ties are broken randomly with `rng`. If the buffer has fewer pairs
    than slots, the ranking is cycled (each pair replayed more than once).
    max_share_per_dataset (e.g. 0.75) caps how much of one interval a single dataset may fill.
    """
    details = plan_interval_details(
        buffer, method, n_slots, rng, max_share_per_dataset,
        plan_index=plan_index, forecast_horizon=forecast_horizon,
    )
    return details["id"].tolist() if len(details) else []


def plan_interval_details(buffer, method, n_slots, rng, max_share_per_dataset=None,
                          plan_index=0, forecast_horizon=1.0):
    """Return the replay plan with auditable rank, score, dataset, and selection rule."""
    if method not in METHODS:
        raise ValueError(f"unknown replay method {method!r}; expected one of {METHODS}")
    if not uses_replay(method) or buffer is None or len(buffer) == 0 or n_slots <= 0:
        columns = (FMCR_PLAN_COLUMNS if method == "fmcr" else
                   CPMR_PLAN_COLUMNS if method == "cpmr" else
                   MIR_PLAN_COLUMNS if method == "mir_dpo" else
                   BALANCED_PLAN_COLUMNS if method == "mfr_balanced" else
                   AT_RISK_PLAN_COLUMNS if method == "mfr_at_risk" else PLAN_COLUMNS)
        return pd.DataFrame(columns=columns)

    rows = buffer.rows()
    tiebreak = rng.random(len(rows))

    if method == "fmcr":
        return _plan_fmcr(
            rows, n_slots, tiebreak, plan_index=int(plan_index),
            forecast_horizon=float(forecast_horizon),
        )
    if method == "cpmr":
        return _plan_cpmr(rows, n_slots, tiebreak, max_share_per_dataset)
    if method == "mir_dpo":
        return _plan_mir_dpo(rows, n_slots, tiebreak, max_share_per_dataset)
    if method == "mfr_balanced":
        return _plan_mfr_balanced(rows, n_slots, tiebreak, plan_index=int(plan_index))
    if method == "mfr_at_risk":
        return _plan_mfr_at_risk(rows, n_slots, tiebreak, plan_index=int(plan_index))

    if method in ("random", "random_high"):
        score = tiebreak
        order = np.argsort(tiebreak)                                     # a random permutation
    elif method in (
        "lowest_margin", "dapr", "dapr_weak", "dapr_gated", "dapr_c", "copr_adapted",
    ):
        score = -rows["current_margin"].to_numpy()
        order = np.lexsort((tiebreak, rows["current_margin"].to_numpy()))          # ascending
    else:  # mfr
        drop = (rows["peak_margin"] - rows["current_margin"]).to_numpy()
        score = drop
        order = np.lexsort((tiebreak, -drop))                                      # largest drop first

    ranked = rows.iloc[order].copy()
    ranked["selection_score"] = score[order]
    ranked["selection_rank"] = np.arange(1, len(ranked) + 1)
    if max_share_per_dataset is not None:
        ranked = _apply_cap(ranked, n_slots, max_share_per_dataset)

    repeats = [ranked.iloc[i % len(ranked)].copy() for i in range(n_slots)]
    selected = pd.DataFrame(repeats).reset_index(drop=True)
    selected["method"] = method
    return selected[PLAN_COLUMNS]


def _quota_plan(rows, ranked, n_slots, plan_index, method, columns):
    """Equal replay quota per old behavior (docs/BALANCED_MFR.md).

    ``ranked`` is the buffer in priority order. Each behavior takes its top pairs up to its quota and
    cycles only after all of its pairs are used. When the slots do not divide evenly, the extra
    slots go to the first behaviors of a list rotated by ``plan_index``, so no behavior is always
    favored. Selections are interleaved round-robin so every optimizer step stays balanced.
    """
    datasets = list(dict.fromkeys(rows["dataset"]))
    base, remainder = divmod(int(n_slots), len(datasets))
    shift = int(plan_index) % len(datasets)
    rotated = datasets[shift:] + datasets[:shift]
    quotas = {dataset: base + (index < remainder) for index, dataset in enumerate(rotated)}
    chosen = {}
    for dataset in datasets:
        part = ranked[ranked["dataset"] == dataset].copy().reset_index(drop=True)
        part["selection_rank"] = np.arange(1, len(part) + 1)
        part["dataset_quota"] = int(quotas[dataset])
        chosen[dataset] = [part.iloc[i % len(part)].copy() for i in range(quotas[dataset])]
    interleaved = [chosen[dataset][position]
                   for position in range(max(quotas.values()))
                   for dataset in rotated if position < len(chosen[dataset])]
    selected = pd.DataFrame(interleaved).reset_index(drop=True)
    selected["method"] = method
    return selected[columns]


def _plan_mfr_balanced(rows, n_slots, tiebreak, plan_index=0):
    """Balanced MFR: MFR's historical-drop ranking with equal quotas per old behavior."""
    work = rows.copy()
    work["historical_drop"] = work["peak_margin"] - work["current_margin"]
    order = np.lexsort((tiebreak, -work["historical_drop"].to_numpy()))
    ranked = work.iloc[order].copy()
    ranked["selection_score"] = ranked["historical_drop"]
    return _quota_plan(rows, ranked, n_slots, plan_index, "mfr_balanced", BALANCED_PLAN_COLUMNS)


def _plan_mfr_at_risk(rows, n_slots, tiebreak, plan_index=0):
    """At-Risk MFR: Balanced MFR, but pairs the policy currently gets wrong come first.

    Tier 0: current_policy_margin <= 0 (the tuned policy no longer prefers the chosen answer).
    Tier 1: everything else. Within a tier, largest historical relative-margin drop first. Zero is
    the only threshold, so nothing is tuned. A relative margin <= 0 alone does not put a pair in
    tier 0.
    """
    if "current_policy_margin" not in rows or rows["current_policy_margin"].isna().any():
        raise ValueError("At-Risk MFR needs a current policy margin for every buffered pair; "
                         "score the buffer before planning")
    work = rows.copy()
    work["historical_drop"] = work["peak_margin"] - work["current_margin"]
    work["risk_tier"] = np.where(work["current_policy_margin"] <= 0, 0, 1).astype(int)
    order = np.lexsort((tiebreak, -work["historical_drop"].to_numpy(), work["risk_tier"].to_numpy()))
    ranked = work.iloc[order].copy()
    ranked["selection_score"] = ranked["historical_drop"]
    return _quota_plan(rows, ranked, n_slots, plan_index, "mfr_at_risk", AT_RISK_PLAN_COLUMNS)


def _plan_mir_dpo(rows, n_slots, tiebreak, max_share_per_dataset=None):
    """Select pairs whose DPO loss rises most after one virtual incoming-task update."""
    missing = [column for column in MIR_COLUMNS if column not in rows]
    if missing or rows[MIR_COLUMNS].isna().any().any():
        raise ValueError(
            "MIR-DPO requires current and projected DPO-loss scores; "
            f"missing or incomplete columns: {missing or MIR_COLUMNS}"
        )
    work = rows.copy()
    work["historical_drop"] = work["peak_margin"] - work["current_margin"]
    order = np.lexsort((
        tiebreak,
        work["current_margin"].to_numpy(),
        -work["interference_score"].to_numpy(),
    ))
    ranked = work.iloc[order].copy()
    ranked["selection_score"] = ranked["interference_score"]
    ranked["selection_rank"] = np.arange(1, len(ranked) + 1)
    if max_share_per_dataset is not None:
        ranked = _apply_cap(ranked, n_slots, max_share_per_dataset)
    repeats = [ranked.iloc[i % len(ranked)].copy() for i in range(n_slots)]
    selected = pd.DataFrame(repeats).reset_index(drop=True)
    selected["method"] = "mir_dpo"
    return selected[MIR_PLAN_COLUMNS]


def _plan_cpmr(rows, n_slots, tiebreak, max_share_per_dataset=None):
    """Replay pairs with the weakest present-or-projected counterfactual margin.

    The worse of the current and projected margins is the primary key. Therefore a forecast can add
    future risk but cannot hide a pair that is already weak, and the rule reduces to Lowest Margin
    when the upcoming interval is predicted to have no effect. Predicted interference and
    historical forgetting are secondary keys. This avoids a weighted score and introduces no tuned
    mixing coefficient.
    """
    missing = [column for column in COUNTERFACTUAL_COLUMNS if column not in rows]
    if missing or rows[COUNTERFACTUAL_COLUMNS].isna().any().any():
        raise ValueError(
            "CPMR requires current and projected counterfactual margins; "
            f"missing or incomplete columns: {missing or COUNTERFACTUAL_COLUMNS}"
        )

    work = rows.copy()
    work["historical_drop"] = work["peak_margin"] - work["current_margin"]
    work["worst_case_margin"] = np.minimum(
        work["current_margin"], work["projected_margin"]
    )
    order = np.lexsort((
        tiebreak,
        -work["historical_drop"].to_numpy(),
        -work["predicted_drop"].to_numpy(),
        work["worst_case_margin"].to_numpy(),
    ))
    ranked = work.iloc[order].copy()
    ranked["selection_score"] = -ranked["worst_case_margin"]
    ranked["selection_rank"] = np.arange(1, len(ranked) + 1)
    if max_share_per_dataset is not None:
        ranked = _apply_cap(ranked, n_slots, max_share_per_dataset)

    repeats = [ranked.iloc[i % len(ranked)].copy() for i in range(n_slots)]
    selected = pd.DataFrame(repeats).reset_index(drop=True)
    selected["method"] = "cpmr"
    return selected[CPMR_PLAN_COLUMNS]


def _plan_fmcr(rows, n_slots, tiebreak, plan_index=0, forecast_horizon=1.0):
    """Balanced replay of pairs forecast to cross a preference boundary.

    Priority tiers are intentionally defined by meaningful zero boundaries rather than tuned
    thresholds: already failed absolute policy preference, forecast absolute failure, forecast loss
    of reference-relative DPO advantage, then historical-drop fallback.
    """
    if forecast_horizon <= 0:
        raise ValueError("forecast_horizon must be positive")
    missing = [column for column in FORECAST_COLUMNS if column not in rows]
    if missing or rows[FORECAST_COLUMNS].isna().any().any():
        raise ValueError(
            "FMCR requires initialized relative and policy-margin trajectories; "
            f"missing or incomplete columns: {missing or FORECAST_COLUMNS}"
        )

    work = rows.copy()
    work["_tiebreak"] = tiebreak
    work["historical_drop"] = work["peak_margin"] - work["current_margin"]
    work["forecast_margin"] = (
        work["current_margin"] + forecast_horizon * work["margin_velocity"]
    )
    work["forecast_policy_margin"] = (
        work["current_policy_margin"] + forecast_horizon * work["policy_velocity"]
    )

    already_failed = work["current_policy_margin"] <= 0
    forecast_policy_failure = (~already_failed) & (work["forecast_policy_margin"] <= 0)
    relative_failed = (
        (~already_failed) & (~forecast_policy_failure) & (work["current_margin"] <= 0)
    )
    forecast_relative_failure = (
        (~already_failed) & (~forecast_policy_failure) & (~relative_failed)
        & (work["current_margin"] > 0) & (work["forecast_margin"] <= 0)
    )
    work["risk_tier"] = np.select(
        [already_failed, forecast_policy_failure, relative_failed, forecast_relative_failure],
        [0, 1, 2, 3], default=4,
    ).astype(int)
    labels = {
        0: "policy_failed", 1: "forecast_policy_crossing",
        2: "relative_failed", 3: "forecast_relative_crossing",
        4: "historical_drop_fallback",
    }
    work["risk_label"] = work["risk_tier"].map(labels)

    policy_fall = -work["policy_velocity"]
    relative_fall = -work["margin_velocity"]
    work["time_to_crossing"] = np.inf
    work.loc[already_failed, "time_to_crossing"] = 0.0
    valid_policy = forecast_policy_failure & (policy_fall > 0)
    work.loc[valid_policy, "time_to_crossing"] = (
        work.loc[valid_policy, "current_policy_margin"] / policy_fall[valid_policy]
    )
    work.loc[relative_failed, "time_to_crossing"] = 0.0
    valid_relative = forecast_relative_failure & (relative_fall > 0)
    work.loc[valid_relative, "time_to_crossing"] = (
        work.loc[valid_relative, "current_margin"] / relative_fall[valid_relative]
    )

    # Within each tier, crossing time gives urgency and historical drop keeps the original MFR
    # signal as a deterministic secondary criterion.
    order = np.lexsort((
        work["_tiebreak"].to_numpy(),
        -work["historical_drop"].to_numpy(),
        work["time_to_crossing"].to_numpy(),
        work["risk_tier"].to_numpy(),
    ))
    work = work.iloc[order].copy()

    datasets = list(dict.fromkeys(rows["dataset"]))
    base, remainder = divmod(int(n_slots), len(datasets))
    rotated = datasets[plan_index % len(datasets):] + datasets[:plan_index % len(datasets)]
    quotas = {dataset: base + (index < remainder) for index, dataset in enumerate(rotated)}

    selected_by_dataset = {}
    for dataset in datasets:
        ranked = work[work["dataset"] == dataset].copy().reset_index(drop=True)
        ranked["selection_rank"] = np.arange(1, len(ranked) + 1)
        ranked["selection_score"] = ranked["historical_drop"]
        ranked["dataset_quota"] = int(quotas[dataset])
        quota = int(quotas[dataset])
        selected_by_dataset[dataset] = [ranked.iloc[i % len(ranked)].copy() for i in range(quota)]

    # Round-robin interleaving keeps each optimizer step balanced when two replay slots are used.
    interleaved = []
    for position in range(max(quotas.values())):
        for dataset in rotated:
            chosen = selected_by_dataset[dataset]
            if position < len(chosen):
                interleaved.append(chosen[position])
    selected = pd.DataFrame(interleaved).reset_index(drop=True)
    selected["method"] = "fmcr"
    return selected[FMCR_PLAN_COLUMNS]


def diagnostics(buffer):
    """Compact diagnostics for plotting and checking the state of the replay memory."""
    rows = buffer.rows().copy()
    if not len(rows):
        return pd.DataFrame(columns=["dataset", "pairs", "mean_peak", "mean_current", "mean_forgetting"])
    rows["forgetting"] = rows["peak_margin"] - rows["current_margin"]
    return rows.groupby("dataset", as_index=False).agg(
        pairs=("id", "size"),
        mean_peak=("peak_margin", "mean"),
        mean_current=("current_margin", "mean"),
        mean_forgetting=("forgetting", "mean"),
    )


def _apply_cap(ranked, n_slots, max_share):
    """Re-order so that no dataset fills more than max_share of the first n_slots picks."""
    cap = max(1, int(np.floor(max_share * n_slots)))
    picked, overflow, counts = [], [], {}
    for _, row in ranked.iterrows():
        dataset = row["dataset"]
        if len(picked) < n_slots and counts.get(dataset, 0) < cap:
            picked.append(row)
            counts[dataset] = counts.get(dataset, 0) + 1
        else:
            overflow.append(row)
    return pd.DataFrame(picked + overflow)
