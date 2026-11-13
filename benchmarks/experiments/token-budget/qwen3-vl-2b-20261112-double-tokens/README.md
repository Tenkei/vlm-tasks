# Token budget followup (partial)

**Stopped at the user's request after 3 of 13 selected samples finished.** 1 in-flight request was interrupted; its token usage is unknown. Scores, token totals and latency below cover completed cases only.

Only samples reaching the baseline cap are included. Both conditions are scored on the same selected samples. [Paired raw responses and metrics](comparison.json) · [Selected sample identities](selection.json)

| Task | N | Token budget | Validation metrics | Failures | At cap | Input tokens | Output tokens | Mean s/sample |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| detect | 3 | 4,096 | P@.5 0.00%; R@.5 0.00%; F1@.5 0.00% | 3 | 3 | 4,106 | 12,288 | 180.245 |
| detect | 3 | 8,192 | P@.5 0.00%; R@.5 0.00%; F1@.5 0.00% | 3 | 3 | 4,106 | 24,576 | 381.196 |

'At cap' identifies possible truncation, not proof of truncation. This selected subset does not replace the 100-sample model comparison. Local monetary cost is unmeasured.

## Captured samples

| Task / baseline index | Image | Baseline response | Doubled-budget response | Output tokens before → after |
|---|---|---|---|---:|
| detect / 13 | [PNG](images/detect/013.png) | invalid | invalid; at cap | 4,096 → 8,192 |
| detect / 20 | [PNG](images/detect/020.png) | invalid | invalid; at cap | 4,096 → 8,192 |
| detect / 21 | [PNG](images/detect/021.png) | invalid | invalid; at cap | 4,096 → 8,192 |
