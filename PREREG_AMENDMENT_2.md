# Pre-registration amendment 2: compute platform

Written 19 October 2026. This amends [PREREG.md](PREREG.md) as tagged `prereg-v1`
(`65fb41a58f0aecf65ff68d9f9c41dfc4046421d1`) and [PREREG_AMENDMENT_1.md](PREREG_AMENDMENT_1.md) as
tagged `prereg-v1.1` (`d870171706b4d2ae51cb624bdf95e9a8d910813a`). Where they differ, this file
governs.

## Status when written

- Nothing has run since `prereg-v1.1`: no data download, label round-trip check, smoke test, epoch
  decision or training.
- No model has been trained or scored on the new split.

## 1. Modal replaces Colab

Training moves from Google Colab to Modal (modal.com), a cloud service that bills GPU time by the
second. Its Starter plan includes US$30 of compute credit a month.

- **Hardware.** Every GPU step runs on Modal: the smoke test, every run in section 3.2, E0, the
  single test opening and latency. Each container has one NVIDIA L4, 4 physical CPU cores (8 vCPUs,
  also the limit) and a 20 GiB memory request (Modal bills the larger of the request and actual
  use). All training therefore stays on one GPU type, as section 3.3 already required. Before each
  GPU step, a check confirms that torch can use the L4.
- **Code.** [cloud/kitti_modal.py](cloud/kitti_modal.py) replaces `colab/kitti_queue.ipynb`, which is
  not used. As in the notebook, every container clones `code.bundle` at a git tag and runs the
  existing scripts unchanged, so every run folder records the commit. Modal runs the laptop's copy
  of `cloud/kitti_modal.py`, so every container that does work also refuses to run unless that copy
  is the one at the tag. Training uses `prereg-v1.2`, the tag on the commit that adds this file.
  Between `prereg-v1` and `prereg-v1.2` only the two amendment files and `cloud/kitti_modal.py`
  change; `scripts/`, `kitti/`, `configs/` and `splits/` do not, and every container refuses to run
  unless they are as they were at `prereg-v1`. The code for the test opening and latency is not
  written yet. It will come in a later tagged commit and run in the same kind of container.
- **Software.** The container image is Debian with Python 3.12 and `ultralytics==8.4.171`. Every
  other package is pinned in `cloud/kitti_modal.py` to the version pip resolves from the releases
  before 2 October 2026, the day after `ultralytics` 8.4.171 was released, except that OpenCV stays
  on its 4.x line (4.14.0.94) rather than the new 5.0 release. Among the pins are torch 2.14.1 and
  torchvision 0.29.1, built for CUDA 13.0. Every run folder also records the versions in
  `pip_freeze.txt` (section 7).
- **Storage.** Modal volumes take the place of Google Drive. A shared volume, `kitti-msc`, holds the
  zips, the Day-1 outputs, the smoke test and the launch records. Each run has a volume of its own,
  so that each volume has a single writer (Modal advises no more than five). It holds the run's
  queue log and folder, including `last.pt` after every epoch, a record of every container that
  trained the run, and a copy of each container's console output. After training, the runs are
  copied into one `runs/` folder in the original layout for the analysis, and every cut run is
  marked there with a `CUT` file.

## 2. Data

- The three zips of section 1 are downloaded once, from the same address, into the shared volume.
  Their SHA-256s are logged in `day1/zips_sha256.txt`. If the label or devkit hash differs from
  section 2.2, nothing more runs: the smoke test and the launch refuse to start until this data step
  has passed. The image zip has no pre-registered hash; its hash is logged.
- Every container unzips the training images and labels to its own disk, checks the label zip's
  hash again and reruns `scripts/convert_labels.py`. A container in which any label fails to
  round-trip within 1 px stops before training, so the round-trip part of the Day-1 gate (section 4)
  is passed in every container, not once.
- The 100-image overlay sheet is drawn once, by the data step, and reviewed before the smoke test.

## 3. Epochs, launch and budget (replaces section 3.3)

Section 3.3 rests on opening a second Colab runtime, which has no Modal equivalent. In its place:

- **The smoke test is unchanged** (`scripts/smoke_test.py`: YOLO11s for 3 epochs; YOLO11n, YOLO11m
  and YOLO11s at 960 for 1 epoch each) and runs in one container. If it is run more than once, the
  launch uses the latest run.
- **Rehearsal.** Before the launch, the launch is rehearsed once, for 1 epoch, in folders of its own,
  in two parts. Part 1 starts the headline run and a long queue (the other six training runs, one
  after another in one container), then stops watching after a few minutes and leaves them going.
  Part 2, started separately, re-attaches to them, stops the long queue as a test of stopping a
  run, and watches the headline to its end. If the headline's 1-epoch score triggers the halt rule,
  E0, the second phase, must not start; otherwise E0 must start once the headline has completed.
  Each rehearsal container stops after 2 hours and is not retried. Nothing from the rehearsal is
  used.
- **Every run trains for 100 epochs**, the plan doc's own setting.
- **The headline trains first.** The launch starts the headline run alone. When it has completed and
  passed the halt rule, every other run and E0 start together, each in its own container and its own
  queue, named by the run ID and logged in its queue log. If the headline ends without completing
  (for example, after failed retries), nothing else starts, and a relaunch resumes it. The two
  queues and the single-queue order of section 3.3 no longer apply.
- **Budget rule at launch.** The projected cost of all runs, E0 included, must not exceed US$25. The
  projection uses the smoke test and nothing else:
  - Seconds per epoch, for each model and image size: the `time` column of the smoke run's
    `results.csv` divided by its number of epochs. Ultralytics starts that clock after data set-up,
    so one-off start-up is left out. The smoke test's own wall-clock figure is used only if that
    file is missing.
  - Hours per run: seconds per epoch × 100 × 1.2 / 3,600, plus 0.15 h. The factor 1.2 allows for
    mosaic augmentation, which the smoke runs never use: Ultralytics switches mosaic off for the last
    10 epochs of a run, and for a 1–3 epoch run that is every epoch. The 0.15 h covers each
    container's unzipping, label conversion, image caching and final validation. E0 counts as
    0.25 h.
  - Cost: hours × US$1.1477, Modal's list price per hour for this container on 19 October 2026
    (L4 US$0.000222 per second; CPU US$0.0000131 per core per second; memory US$0.00000222 per GiB
    per second).
  - If the total is over US$25, runs are dropped in the cut order of section 3.4 (seed 2, seed 1,
    960, medium) until it is not. A dropped run is cut and logged, as amendment 1 describes. If the
    runs left after all four cuts are still over US$25, nothing starts and a further amendment is
    needed.
- **Budget rule during training.** Every minute, the launch adds up three things. The first is the
  container time every run has used so far, counting 2 minutes of start-up per container. The
  second is the cost of the launch's own container. The third is the time each unfinished or
  unstarted run still needs. For a run with at least two epoch-to-epoch gaps in `results.csv`, that
  is its remaining epochs times the median of its recent gaps, plus 0.1 h; for any other run, it is
  what is left of its projection (at least 0.1 h), and for E0 what is left of its 0.25 h (at least
  3 minutes). If the total, at the price above, goes over US$26, the next unfinished run in the cut
  order is cut, whether or not it has started. Runs outside the cut order are never cut for budget.
  The US$26 leaves about US$4 of the month's credit for the steps before and after training.
- **The decision**, its inputs and the time are written by `cloud/kitti_modal.py` to
  `runs/EPOCH_DECISION.json` before any run starts. It records the commit and the SHA-256 of
  `cloud/kitti_modal.py`. It is made once; a relaunch reuses it and must run the same commit.
  `scripts/decide_epochs.py` is not used. The file has the fields `cloud/kitti_modal.py` writes,
  including the epochs and queues that `scripts/train_queue.py` reads.

## 4. Halt, restarts, stalls and the cut-off (amends section 3.4)

- **Halt rule:** unchanged. Because the headline trains alone, a `HALT` file stops the launch before
  any other run starts, and only the headline's cost is lost. A relaunch after a halt starts nothing;
  the restart is covered by a further amendment, as amendment 1 says.
- **Restarts.** If a run fails inside its container, Modal retries it at most twice. Modal also
  restarts containers that crash or are preempted, and stops and retries a container that reaches
  12 hours. Every restart resumes from `last.pt`, as section 3.4 describes, and is logged in the
  run's queue log and container records. Each run is locked, so it is never trained by two
  containers at once. The launch keeps watching a run while Modal cannot be reached, and a relaunch
  never starts a second copy of a run whose state it cannot read.
- **Unreadable checkpoint.** If `last.pt` or `best.pt` cannot be read (for example, because the
  container stopped while writing it), the old files are kept under another name and the run starts
  again from epoch 1. The exception is a run whose training had finished: if only `last.pt` is
  unreadable, it is rebuilt from `best.pt`, and the run is validated rather than retrained. Both
  cases are logged in the run folder's `RESTARTS.log`.
- **Stalls.** A run that shows no progress for 1.5 hours (no new line in `results.csv`, no new E0
  prediction file, no final summary) is stopped, and this is logged in `runs/stops.json`.
- **Cut-off.** The cut-off of amendment 1 applies unchanged: 24 hours after the earliest
  `QUEUE <queue> START` line. It is written to `runs/CUTOFF` once known. Any run still training then
  is stopped and recorded as cut, and a run whose `COMPLETED` time is after the cut-off is also
  treated as cut. A container that started before the cut-off was written uses the earliest `START`
  line in its own run's log plus 24 hours, which can only be later than the cut-off, by the minutes
  between the first `START` lines.
- **Relaunch.** If the launch stops before every run has ended, it may be started again with the
  same command and tag. It reuses `EPOCH_DECISION.json`, watches runs that are still going, skips
  completed runs and never restarts a run that was cut. It resumes the others from `last.pt`,
  including a run that failed or was stopped for a stall. It does not start while another launch, a
  rehearsal or a rehearsal's runs are still going. After a halt or after the cut-off it starts
  nothing. After the cut-off it stops whatever is still running; after a halt it stops every run but
  the headline, which ends by itself.

## 5. Latency (amends section 6)

Latency is measured on a Modal NVIDIA L4 instead of the Colab L4, and the second session is a
second, separate container. Everything else in section 6 stands.

## 6. Deviations (amends section 8)

- Item 2 now reads: **Epochs follow amendment 2, section 3**: 100 epochs for every run, with runs cut
  in the pre-registered order when the projected cost is over the budget. The plan doc's rule was
  "50 if Day-1 timing projects more than 22 L4-hours".
- Item 4 now reads: **Latency is measured on a Modal L4**, not the lab machine.

## Unchanged

Everything else in `prereg-v1` and amendment 1 stands, including the split files and their hashes,
the recipe, the run list, the halt threshold, the cut order and the never-cut list, the cut-off, and
the evaluation, single test opening and environment logging protocols.
