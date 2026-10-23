# Softmax-Sampled Most-Forgotten Replay (MFR-Softmax)

## Motivation

Standard Most-Forgotten Replay (MFR) ranks stored preference pairs by their measured loss of
preference margin and always replays the largest drops. This gives strong priority to the most
forgotten examples, but repeatedly selecting only the top-ranked pairs can reduce coverage of the
memory buffer and make selection sensitive to small differences around the cutoff.

MFR-Softmax keeps the same forgetting signal and replay budget, but turns the scores into a
probability distribution. It therefore favors highly forgotten examples while retaining a chance
to revisit every stored pair.

## Forgetting score

For preference pair \(i\), let \(m_i^{peak}\) be its reference-relative DPO margin immediately
after learning its original task, and let \(m_i^{current}\) be its margin at the latest buffer
refresh. Its forgetting score is

\[
f_i = m_i^{peak} - m_i^{current}.
\]

A positive value means that the learned preference has weakened. A larger value indicates more
forgetting.

## Softmax sampling

At each buffer refresh, convert forgetting scores into probabilities:

\[
p_i = \frac{\exp(f_i / \tau)}{\sum_j \exp(f_j / \tau)},
\]

where \(\tau > 0\) is the temperature. Each replay slot is then sampled categorically from this
distribution using the run's seeded random-number generator. Sampling is with replacement, like
token sampling, so a severely forgotten pair can be selected multiple times in one refresh
interval.

Temperature controls the exploration--exploitation tradeoff:

- \(\tau < 1\): concentrates probability on the most-forgotten pairs and approaches deterministic
  top-score selection as \(\tau\) approaches zero.
- \(\tau = 1\): uses the forgetting scores at their natural scale and is the default.
- \(\tau > 1\): flattens the distribution and approaches uniform random replay as temperature
  increases.

For numerical stability, implementation uses \(\exp((f_i-\max_j f_j)/\tau)\). If a per-dataset
share cap is configured, candidates from a dataset that has reached the cap are temporarily masked
and the remaining probabilities are renormalized. The cap is relaxed only when every represented
dataset is masked, making the requested replay budget feasible even when the buffer contains only
one earlier task.

## Controlled comparison

Compare `mfr_softmax` against deterministic `mfr` and uniform `random` under exactly the same:

- memory-buffer contents and size;
- replay share and optimizer-step budget;
- margin refresh schedule;
- model initialization, task order, and random seeds; and
- evaluation sets and retention metrics.

The only changed factor is the replay selection rule. Log the selected pair, forgetting score,
global softmax probability, conditional probability used for the draw, rank, and temperature.

The main hypothesis is that MFR-Softmax will retain old preferences better than uniform replay and
may improve robustness over deterministic MFR by increasing buffer coverage, without materially
reducing learning on the current task. Temperature should be fixed before the main comparison; a
small development-only sweep such as \(\{0.25, 0.5, 1.0, 2.0\}\) can be used if tuning is allowed.

## Configuration

Add `"mfr_softmax"` to `secondary_methods` (or `methods`) in
`configs/experiment_protocol.json`. Optionally add:

```json
"mfr_softmax_temperature": 1.0
```

If omitted, the implementation uses `1.0`.
