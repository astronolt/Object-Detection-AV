# Pre-registration amendment 1: schedule

Written 19 October 2026. This amends [PREREG.md](PREREG.md) as tagged `prereg-v1`
(`65fb41a58f0aecf65ff68d9f9c41dfc4046421d1`). Where the two differ, this file governs.

## Status when written

- On 15 October the split, the evaluator gate and the pipeline code were committed, and the
  pre-registration was tagged `prereg-v1`.
- No Colab step has run since: no data download, label round-trip check, smoke test, epoch decision
  or training.
- No model has been trained or scored on the new split.

## 1. A training cut-off replaces the fixed freeze date

Training did not start on 15 October, so the freeze time in section 3.4 (Friday 16 October, 10:00
WAT) passed before training began. It is replaced by a cut-off that is fixed when training starts:

- **Cut-off:** 24 hours after the earliest `QUEUE <queue> START` line in `runs/queue_*.log`, using
  the timestamps in those logs. If the halt rule in section 3.4 stops training, the restart is
  covered by a further amendment, which sets a new cut-off.
- A run counts as finished only if its `COMPLETED` file is time-stamped before the cut-off. Any
  other run is cut and logged, and is never used, even if it finishes later.
- Cuts follow the pre-registered order: seed 2, seed 1, 960, medium. The headline, the random split,
  E0, the single test opening, latency and the failure analysis are never cut.
- The freeze review starts when every queue has ended, or at the cut-off if that comes first. The
  test set is opened once, after the freeze review, and not overnight.

This supersedes the freeze date in section 3.4, the Friday-morning review in section 5.1 (item 3)
and the two-day schedule in section 8 (item 7).

## 2. Correction

Section 3.3 names the epoch-decision file `runs/EPOCH_DECISION.txt`; the code writes
`runs/EPOCH_DECISION.json`. Its contents are as section 3.3 describes.

## 3. Commit hashes

Before this amendment was committed, the messages of the four 15 October commits were edited.
Their authors, dates and file contents are unchanged, as the unchanged tree hashes show, but their
commit hashes changed, and so did the hash of the `prereg-v1` tag. PREREG.md (section 2.1) and the
message of the third commit cite the old hashes, which correspond as follows:

| 15 October | Now | Tree (unchanged) | Commit |
|---|---|---|---|
| `1ffa88e` | `81b8429` | `1a8408d` | Pre-register split procedure, recipe and evaluation protocol |
| `046815f` | `7a6e2de` | `10059d3` | Pin LF line endings so split hashes match on every machine |
| `f6ac8a3` | `c4459fc` | `586f667` | Split procedure revision 2: add deterministic local search |
| `6bd2aab` | `83c4662` | `bed602f` | Frozen split, evaluator gate, Colab pipeline |

The tag `prereg-v1` was `bdcc8f7` and is now `65fb41a`; it still points to the last of these
commits.

## Unchanged

Everything else in `prereg-v1` stands, including the split files and their hashes, the recipe, the
epoch rule, the run list, the evaluation and latency protocols, and the deviations in section 8
(items 1–6).
