"""The replay memory and the four ways to fill the replay slots.

The buffer holds `size` old preference pairs (500 by default), split evenly across the stages
already learned. For every stored pair we keep:

    peak_margin     its margin right after its own stage finished ("how well it was learned")
    current_margin  its margin the last time we re-scored the buffer

The four methods differ only in which stored pairs fill the 2 replay slots of each training step:

    none            nothing is replayed
    random          uniform sample from the buffer
    lowest_margin   the lowest current_margin (hard pairs, forgotten or not)
    mfr             the largest drop, peak_margin - current_margin (our method)

Everything here is CPU-only and fully seeded: which pairs enter the buffer depends on the run seed
and the dataset, never on the method, so all four methods store exactly the same pairs.
"""

import numpy as np
import pandas as pd

from mfr_utils import buffer_seed

METHODS = ("none", "random", "lowest_margin", "mfr")
EXTRA_COLUMNS = ["dataset", "peak_margin", "current_margin"]


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

    def add_stage(self, dataset, rows, peak_margin):
        """Store a finished stage's candidates with their peak margins, then rebalance the buffer.

        peak_margin: Series indexed by pair id (from mfr_dpo.score_pairs(...)["margin"]).
        After this call every stored dataset holds size // (number of datasets) pairs.
        current_margin starts equal to peak_margin.
        """
        new = rows.copy()
        new["dataset"] = dataset
        new["peak_margin"] = new["id"].map(peak_margin).astype(float)
        if new["peak_margin"].isna().any():
            missing = new.loc[new["peak_margin"].isna(), "id"].tolist()[:3]
            raise ValueError(f"peak margins missing for {missing} ...")
        new["current_margin"] = new["peak_margin"]

        self._rows = pd.concat([self._rows, new], ignore_index=True) if len(self._rows) else new
        self._rebalance()
        return self

    def _rebalance(self):
        """Keep the buffer at `size` pairs, evenly split across the datasets in it (seeded)."""
        datasets = list(dict.fromkeys(self._rows["dataset"]))
        per_dataset = max(1, self.size // len(datasets))
        kept = []
        for dataset in datasets:
            part = self._rows[self._rows["dataset"] == dataset]
            if len(part) > per_dataset:
                part = part.sample(n=per_dataset, random_state=buffer_seed(self.seed, dataset) + len(datasets))
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

    def set_current(self, margin):
        """Update current margins from a fresh scoring pass (Series indexed by pair id)."""
        updated = self._rows["id"].map(margin)
        self._rows["current_margin"] = updated.fillna(self._rows["current_margin"]).astype(float)

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
        return buffer


# -------------------------------------------------------------------- choosing

def plan_interval(buffer, method, n_slots, rng, max_share_per_dataset=None):
    """Pair ids for the next interval's replay slots (one id per slot, in order).

    none: empty list. random: uniform without replacement. lowest_margin: lowest current margin.
    mfr: largest peak - current. Ties are broken randomly with `rng`. If the buffer has fewer pairs
    than slots, the ranking is cycled (each pair replayed more than once).
    max_share_per_dataset (e.g. 0.75) caps how much of one interval a single dataset may fill.
    """
    if method not in METHODS:
        raise ValueError(f"unknown replay method {method!r}; expected one of {METHODS}")
    if method == "none" or buffer is None or len(buffer) == 0 or n_slots <= 0:
        return []

    rows = buffer.rows()
    tiebreak = rng.random(len(rows))

    if method == "random":
        order = np.argsort(tiebreak)                                     # a random permutation
    elif method == "lowest_margin":
        order = np.lexsort((tiebreak, rows["current_margin"].to_numpy()))          # ascending
    else:  # mfr
        drop = (rows["peak_margin"] - rows["current_margin"]).to_numpy()
        order = np.lexsort((tiebreak, -drop))                                      # largest drop first

    ranked = rows.iloc[order]
    if max_share_per_dataset is not None:
        ranked = _apply_cap(ranked, n_slots, max_share_per_dataset)

    ids = ranked["id"].tolist()
    return [ids[i % len(ids)] for i in range(n_slots)]


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
