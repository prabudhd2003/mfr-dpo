# Data card

The project uses three preference streams. Exact source revisions and generated-file hashes are written to
`data/v2/manifest.json` by notebook 01.

| Behavior | Source | Label used |
|---|---|---|
| helpful | NVIDIA HelpSteer2 preference subset | response preferred with strength at least 2 |
| safe | PKU-SafeRLHF | the response marked safe when exactly one response is safe |
| quality | UltraFeedback binarized | higher-scored response with score gap at least 1 |

“Quality” means general response quality; it is not a verified instruction-following-only subset. IFEval is
therefore a secondary generalization check, not the direct metric for this training stream.

Version 2 normalizes Unicode, case, and whitespace before prompt-level deduplication. It removes cross-dataset and
cross-split normalized prompt overlap, stores the original source row identifier, restricts the longer sequence to
1,024 tokens, and creates 2,000 / 200 / 300 train/validation/test pairs per behavior with seed 0.

Known limitations include noisy safety labels, possible factual errors in responses rated helpful, broad task
mixtures in UltraFeedback, and response-length differences. Notebook 02 displays fixed representative samples. Final
claims should refer to retention of these dataset preferences, not to universal safety or helpfulness.
