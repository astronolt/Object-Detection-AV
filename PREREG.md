# Pre-registration: YOLO–KITTI MSc compute run

Written 15 October 2026, before any training or any model scoring on the new split.
Sources: the plan doc "YOLO–KITTI MSc: Audit & One-Week Plan" (3 Oct 2026, revision 24), quoted in
[docs/plan-extract.md](docs/plan-extract.md), and the build decisions in
[docs/decisions-15-oct.md](docs/decisions-15-oct.md). Where this file and the plan doc differ, this
file governs, and section 8 lists every difference.

## 1. Data

- Official KITTI 2D object files from `s3.eu-central-1.amazonaws.com/avg-kitti/`: `data_object_label_2.zip`,
  `devkit_object.zip`, `data_object_image_2.zip`. Ultralytics' `kitti.zip` is not used. Each zip's
  SHA-256 is logged where it is downloaded.
- 7,481 labelled images, 141 raw drives.

## 2. Split

### 2.1 Procedure (committed before it was run)

Implemented in [scripts/make_split.py](scripts/make_split.py); run once and never re-rolled.

1. Every image is mapped to its raw drive through `train_rand.txt` (1-based) → `train_mapping.txt`.
2. Seed `20261015`; 50 candidates drawn in order from one `random.Random(seed)` stream. Each
   candidate shuffles the sorted drive list, then assigns drives in that order, each to the partition
   whose image count is furthest below its 60/20/20 target. Ties go to train, then val, then test.
   - Each start is then improved by deterministic best-improvement local search. A step moves one
     drive or swaps two drives between partitions.
   - A step is taken only if it lowers the key (image-share excess over 0.5 points, balance error,
     sum of squared deviations). Ties go to the first step in sorted-drive order.
   - The search stops when no step helps.
3. Feasible means:
   - every partition's image share is within 0.5 percentage points of its target; and
   - in every partition, the Car, Pedestrian and Cyclist instance shares are within 1.2 percentage
     points of that partition's image share. This is the plan doc's "1.2 points" (the old split
     scores 1.15 on it).
4. The chosen candidate is the feasible one with the smallest balance error. Ties go to the smaller
   image-share error, then to the lower candidate index.
5. Random-split control: the chosen train + validation pool, shuffled with `random.Random(seed)` and
   cut into the same train and validation sizes. It uses the same `test.txt`.
6. Files hold one zero-padded image id per line, ascending, LF endings. Images 006500 and 007250 are
   not hand-placed; Chapter 3 states where they land, and that a stock COCO model scored them once
   before this protocol existed.

**Revision log.**

- Attempt 1 (commit `1ffa88e`) used the same seed with 10,000 greedy candidates and no local search.
  No candidate was feasible.
  - The best balance error was 1.47 points, the median 15.9.
  - The cause is cyclist clustering: the top 5 drives hold 49% of cyclists.
  - The script wrote no files.
- The procedure was revised to add the local search, and committed before attempt 2.
- Neither attempt looked at images, models or results; both used label counts only.

The old 3 October split cannot be rebuilt: the plan doc records no seed, no procedure and only a
truncated test hash, and the files themselves were lost. The Day-1 gate "split hashes match" is
therefore replaced by "split hashes recorded here and frozen".

### 2.2 Result

Attempt 2 (commit `f6ac8a3`):
- 34 of 50 candidates were feasible. The chosen one is candidate 37.
- Balance error 0.07 points; image-share error 0.45 points.
- Full report: [splits/split_report.json](splits/split_report.json), including every partition's
  drive list.

| Partition | Images | Drives | Car | Pedestrian | Cyclist | Moderate cyclists |
|---|---|---|---|---|---|---|
| Train | 4,488 (59.99%) | 79 | 17,225 | 2,693 | 977 | 573 |
| Validation | 1,463 (19.56%) | 41 | 5,639 | 876 | 317 | 255 |
| Test (sealed) | 1,530 (20.45%) | 21 | 5,878 | 918 | 333 | 270 |
| Random control: train | 4,488 | 118 | 17,266 | 2,602 | 975 | 632 |
| Random control: val | 1,463 | 95 | 5,598 | 967 | 319 | 196 |

- 93 of the random-control validation drives also appear in its training set. This is the intended
  leakage.
- Images 006500 and 007250 both fall in **train**, so neither is in the sealed test set.
- The test set has 21 drives, so drive-level bootstrap intervals on test will be correspondingly
  wide.

SHA-256 of the frozen files ([splits/SHA256SUMS](splits/SHA256SUMS)):

```
d1a38c0921ef8e4f183f18b82e65a839105f68b4dc376e45bcb55b9c74d59317  drive_disjoint/train.txt
1a675900dee877856ae237b34cc145c86ea911904ba8b68ce945e770f2c03dc4  drive_disjoint/val.txt
cd4cd7cb2f497211f21056433e95293504d68c215f63cbedbf8cb31e09ab4d7e  test.txt
b3e921fa21d817679032f4a14634313f2ad8923e07d27d1f2689333879d4e347  random/train.txt
95c3a81eb8661b0bd816fd0667d2051bec241710612057cf7fccc4cdea02eca8  random/val.txt
```

Labels: `data_object_label_2.zip` SHA-256 `4efc7622…aed880`; `devkit_object.zip` `ce0b76b6…4b4cf1`.
The Colab notebook checks both before converting.

### 2.3 Evaluator gate (passed locally, 15 Oct)

Run on the 5,951 development images (train + validation); test labels were not used.
[results/day1/evaluator_check.json](results/day1/evaluator_check.json):

- **Ground truth as detections:** AP|R40 = 100.00 for Car, Pedestrian and Cyclist at Easy,
  Moderate and Hard.
- **Every 10th detection removed:** 90.0–92.5 in every cell.
- **Boxes shifted by 25% of their width (IoU 0.6):**
  - Car 0.00–0.03, so IoU 0.7 is in effect.
  - Cyclist 100, and Pedestrian 97.8–98.7, so IoU 0.5 is in effect. Pedestrians fall short of 100
    because a shifted box can match a neighbour in a crowd.
- The first run set the Pedestrian/Cyclist tolerance at > 99 and was recorded as failed
  ([attempt 1](results/day1/evaluator_check_attempt1.json)). After that result was seen, the
  tolerance was relaxed to > 90; the numbers themselves did not change.

## 3. Training

### 3.1 Recipe

[configs/recipe.py](configs/recipe.py) `COMMON` is the complete argument set for every run, with
`ultralytics==8.4.171`. Every value is written out, including the 8.4.171 defaults that TAB 07 calls
"moderate" (`hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, translate=0.1, scale=0.5`). Key choices:

- AdamW, lr0 0.001, cosine schedule, 3 warm-up epochs, batch 16 (8 only if memory requires, and
  logged), nominal batch 64, `patience=0`.
- `cls_remap=False`, so no class starts with a COCO head row. In 8.4.171 the default `True` would give
  Car COCO's "car" row and no other class a head start.
- Mosaic on, switched off for the last 10 epochs; horizontal flip 0.5; no vertical flip or rotation.
- `deterministic=True`, `seed` as in the run table, AMP on.

### 3.2 Runs

| Run ID | Model | imgsz | Split | Seed |
|---|---|---|---|---|
| E1-Y11S-640-DD-S0 (headline) | YOLO11s | 640 | drive-disjoint | 0 |
| E0-COCO-640-VAL | COCO YOLO11n/s/m, no training | 640 | validation only | – |
| RS-Y11S-640-RND-S0 | YOLO11s | 640 | random control | 0 |
| E1-Y11N-640-DD-S0 | YOLO11n | 640 | drive-disjoint | 0 |
| E1-Y11M-640-DD-S0 | YOLO11m | 640 | drive-disjoint | 0 |
| E2-Y11S-960-DD-S0 | YOLO11s | 960 | drive-disjoint | 0 |
| E1-Y11S-640-DD-S1 | YOLO11s | 640 | drive-disjoint | 1 |
| E1-Y11S-640-DD-S2 | YOLO11s | 640 | drive-disjoint | 2 |

YOLO26s is not trained. It is timed untrained for the NMS comparison only (section 6).

### 3.3 Epochs: one rule, decided at the smoke test, applied to every run

At the smoke test a second Colab runtime is opened.

- If it is also an NVIDIA L4 and the compute-unit balance covers about 30 units, every run trains
  for **100 epochs**, the plan doc's own setting, split across two queues:
  - A: S0 → nano → 960 → seed 2
  - B: E0 → random split → medium → seed 1
- Otherwise the second runtime is closed and every run trains for **50 epochs** on one L4, in this
  order: S0, E0, random split, nano, medium, 960, seed 1, seed 2.

All training stays on one GPU type. The decision, its inputs and the time are logged in
`runs/EPOCH_DECISION.txt` before the first real run.

### 3.4 Stopping and cuts

- **Halt rule:** if the headline run's Car validation mAP@0.5 is below 0.60, every queue halts.
  Then the converter is fixed and training restarts from the headline run.
- **Freeze, Friday 16 Oct at 10:00 (WAT):** any run not finished is cut and logged. It is never
  used half-trained. Cuts follow the order seed 2, seed 1, 960, medium. The headline, the random
  split, E0, the single test opening, latency and the failure analysis are never cut.
- Each run saves `last.pt` every epoch to Drive. After a crash it resumes from `last.pt`, and the
  resume is logged.
- Checkpoint: Ultralytics `best.pt`, chosen by validation fitness.

## 4. Classes and labels

- YOLO classes: 0 Car (+Van), 1 Pedestrian (+Person_sitting), 2 Cyclist, 3 LargeVehicle (Truck+Tram).
  Misc and DontCare are dropped from training labels; the dropped DontCare boxes are a stated
  label-noise limitation.
- Each box is normalised by its own image's width and height. KITTI has four image sizes.
- **Day-1 gate:**
  - all 7,481 label files round-trip to within 1 px, with image 004696 checked explicitly;
  - the evaluator scores ground truth as 100 AP, with negative controls that must score below 100;
  - a 100-image label overlay sheet is reviewed.

## 5. Evaluation

### 5.1 Development and freeze

1. During development, validation only.
2. Freeze before testing: configs, `best.pt` weights, and one confidence threshold per model.
   - The threshold is the best all-class F1 point on that model's own validation predictions.
   - Safety recall is Pedestrian and Cyclist recall at that threshold, KITTI-style: Moderate
     difficulty, IoU 0.5, with Van, Person_sitting and DontCare ignored.
3. The freeze is reviewed with the student on Friday morning before the test set is opened.

### 5.2 Tables

- **(a) KITTI protocol.** The vendored mmdet3d evaluator
  ([third_party/mmdet3d_kitti_eval](third_party/mmdet3d_kitti_eval), commit `fe25f7a`), 2D boxes,
  AP over 40 recall points.
  - Car, Pedestrian and Cyclist at Easy/Moderate/Hard, IoU 0.7/0.5/0.5.
  - Scored against the original labels: Van and Person_sitting ignored, DontCare applied.
  - Difficulty is minimum height 40/25/25 px, maximum occlusion 0/1/2, maximum truncation
    0.15/0.3/0.5.
  - Predictions carry alpha = −10, so orientation is never scored.
  - LargeVehicle predictions are outside this table.
- **(b) Ultralytics COCO-style.** mAP@0.5 and mAP@0.5:0.95 on the four merged classes, labelled as
  such.
- The two tables are never mixed.
- Precision, recall and F1 at the frozen threshold are computed from saved predictions. Ultralytics'
  own test P/R/F1 are never quoted.
- Every saved prediction file uses conf 0.001, NMS IoU 0.7 and at most 300 detections per image.

### 5.3 E0 (contextual baseline)

- COCO-pretrained YOLO11n/s/m, no training, imgsz 640. Scored on validation now, and on test only
  at the single test opening.
- Class mapping: COCO car → Car, person → Pedestrian, bus/train/truck → LargeVehicle. Every other
  COCO class is discarded before NMS.
- After the mapping, a second class-wise NMS at IoU 0.7 removes duplicate LargeVehicle boxes (bus and
  truck predicted on the same object).
- COCO has no Cyclist class, so E0's Cyclist is reported as n/a.
- Every E0-versus-fine-tuned "overall" figure (RQ1, H1) averages the same classes for both models:
  Car and Pedestrian in the KITTI table.
- Riders detected as "person" count as Pedestrian false positives under KITTI's rules. This is part
  of the vocabulary gap; it is stated and not corrected for.

### 5.4 Single test opening

- One script scores E0 and every frozen model on `test.txt`.
- It logs the date, the git commit and the SHA-256 of `test.txt`, then writes a STARTED line and a
  COMPLETED line.
- If the runtime crashes before COMPLETED, the script may be re-run with the same commit and
  settings. Partial outputs are deleted unread, and every attempt is reported.

### 5.5 Uncertainty, leakage, failures

- **Uncertainty:**
  - Drive-level bootstrap 95% intervals for every per-class AP and every difference, using cached
    box matches.
  - Mean ± SD across the headline's three seeds, if seeds 1 and 2 finish.
- **Leakage:**
  - For the random-split and drive-split models: validation score minus sealed-test score, plus both
    scores on the common test set.
  - Validation drives and test drives are resampled independently.
- **Failure analysis:** test predictions at the frozen threshold, matched to the original labels.
  Misses, wrong classes and poor localisation are tabulated by box height × occlusion × truncation.

## 6. Latency

- On the Colab NVIDIA L4. This is a deviation: the plan doc names the lab machine, which is not
  available.
- Timed back to back in one session, then repeated in a second session:
  - YOLO11n, s and m at 640;
  - YOLO11s at 960;
  - stock YOLO26s with `nms=False` (NMS-free head) and `nms=None` (NMS head).
- Batch 1; the first 50 frames are discarded and at least 500 timed. Mean, median and p95 per stage
  are reported.
- conf 0.25; the input shape is stated; image decoding is excluded.
- The GPU model and the driver, CUDA and torch versions are recorded.

## 7. Environment logging

Every run folder gets `pip_freeze.txt`, `nvidia_smi.txt`, `run_config.json` (with the git commit) and
the Ultralytics `args.yaml`.

## 8. Deviations from the plan doc

1. **The split is new** (section 2). The old one could not be rebuilt.
2. **Epochs follow the section 3.3 rule.** The plan doc's rule was "50 if Day-1 timing projects more
   than 22 L4-hours".
3. **Dropped:** YOLO26 training, the Nigerian video and the live demo.
4. **Latency is measured on the Colab L4**, not the lab machine.
5. **Augmentation values are written out explicitly.** These are the 8.4.171 defaults standing in
   for TAB 07's "moderate".
6. **E0 adds** COCO train → LargeVehicle, a post-mapping NMS, and matched-class averaging.
7. **Schedule:** Day 1 is Thursday 15 Oct and the freeze is Friday 16 Oct at 10:00. The plan doc's
   week is compressed to two days.
