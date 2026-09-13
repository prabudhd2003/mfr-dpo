# Preliminary results (pre-v2)

These results came from notebooks 03–06 and the existing Drive runs. They are evidence that the question is
worth studying, but they must not be mixed with protocol-v2 results.

## What the pilot established

For the safe → helpful → quality no-replay pilot, validation preference accuracy was:

| Validation behavior | After safe | After helpful | After quality |
|---|---:|---:|---:|
| safe | 87.5 | 70.5 | 69.5 |
| helpful | 52.5 | 61.5 | 61.5 |
| quality | 57.5 | 73.0 | 77.0 |

Safety fell 18 points from its post-learning value, showing substantial interference. Helpfulness did not fall
on the normalized metric, although it did fall on the summed DPO metric. Forgetting was concentrated: a minority
of validation pairs accounted for a large fraction of positive margin loss. This supports testing targeted replay.

The Drive also contains initial seed-0 runs for no replay, random replay, lowest-current-margin replay, and MFR
in one order. Four method runs in one order and one seed are not enough to establish that MFR wins.

## Why these are preliminary

- The old splits used exact-string prompt de-duplication and contain a normalized cross-split/cross-dataset prompt
  collision.
- The pilot seeded after adapter creation, so its initialization is not reproducible as a core baseline.
- Old replay runs used 16 new + 2 old examples, which is 11.1% replay rather than the proposed 10%.
- Dataset and model revisions were not pinned in run metadata.
- There was no pre-stage score for measuring new-stage gain, no paired uncertainty analysis, and no locked test,
  generation, automatic behavior, or blinded human evaluation.

Do not delete the old Drive folders. Label them `pre-v2` in the report and use them only as motivation and
debugging evidence. Final tables should read only run names beginning with `v2_`.
