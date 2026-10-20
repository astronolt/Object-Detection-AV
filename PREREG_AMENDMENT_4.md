# Pre-registration amendment 4: evaluation code, freeze and single test opening

Written 20 October 2026. This amends [PREREG.md](PREREG.md) as tagged `prereg-v1` and amendments 1 to 3
as tagged `prereg-v1.1`, `prereg-v1.2` and `prereg-v1.3` (commit
`317332a15f0c9fd779eedcda027311f440eb16ef`). Where they differ, this file governs.

## Status when written

- All eight runs of section 3.2 had completed by 00:18:59 BST on 20 October 2026, before the amendment 1
  cut-off (19:05:40 BST on 20 October). Each of the seven training runs ran its 100 epochs, and the E0 item
  saved its validation predictions. No run was halted, cut or dropped.
- The Ultralytics validation scores that training prints (mAP@0.5 over the four merged classes, which the
  halt rule uses) have been seen. Nothing else has been scored: no KITTI-protocol score, no confidence
  threshold and no test prediction exists, and the test set has not been opened.
- This amendment adds the code that the pre-registration left to be written (the freeze, the single test
  opening and the latency measurements) and fixes details that PREREG.md section 5 left open. It changes no
  pre-registered rule.

## 1. Models

- Scored: the seven trained runs (each run's `best.pt`) and E0 (stock COCO YOLO11n, s and m at 640,
  section 5.3).
- Each trained run is scored at its training input size (640, or 960 for `E2-Y11S-960-DD-S0`). At the
  freeze it is scored on its own validation split: `random/val.txt` for the random-split run,
  `drive_disjoint/val.txt` for every other run and for E0. At the test opening every model is scored on
  `test.txt`.
- Every prediction file comes from `scripts/predict.py` as amended (confidence 0.001, NMS IoU 0.7, at most
  300 boxes per image, batches of 16).

## 2. The freeze (section 5.1, amendment 1)

- `evaluation/freeze.py` saves each model's validation predictions and fixes its confidence threshold: the
  best all-class F1 point, computed as Ultralytics 8.4.171 computes the "all classes" line of its
  F1-confidence curve. The predictions are matched to the four merged classes as
  `DetectionValidator.match_predictions` does, `ap_per_class` is run on the matches, and the threshold is
  the argmax of the smoothed mean F1 curve. E0 has no Cyclist predictions; its Cyclist F1 is zero at every
  threshold, which does not move the argmax.
- E0's validation predictions are the files that the E0 queue item saved during training (section 5.3),
  copied unchanged; the freeze makes them only if one is missing. Every other model's validation
  predictions are made by the freeze. Every prediction file must cover exactly the images of its split.
- It records each model's validation scores (both tables, precision, recall and F1 at the threshold, and
  safety recall) and the SHA-256 of each weights file in `eval/FREEZE.json`, which is never rewritten. It
  does not read `test.txt`.
- The freeze review of amendment 1 is the student's review of `FREEZE.json` before the test opening.

## 3. The single test opening (section 5.4)

- `evaluation/open_test.py`, run through `cloud/kitti_modal.py::open_test`. It refuses to start without
  `FREEZE.json`, if any weights file differs from its frozen SHA-256, if the SHA-256 of `test.txt` differs
  from the one in PREREG.md section 2, or once the opening has completed. The controlling code also
  refuses to start outside 07:00 to 20:00 WAT (amendment 1: not overnight).
- The STARTED and COMPLETED lines, with the time, the commit and the SHA-256 of `test.txt`, are written to
  `eval/test/TEST_OPENING.log`. If an attempt crashes, the next one, with the same commit and settings,
  first deletes the earlier attempt's outputs unread; every attempt stays in the log.

## 4. Tables, comparisons and uncertainty (sections 5.2, 5.3 and 5.5)

- Table (a): `kitti/evaluate.py`, unchanged: Car, Pedestrian and Cyclist at Easy, Moderate and Hard,
  AP|R40. E0's Cyclist is reported as n/a.
- Table (b): AP@0.5 and AP@0.5:0.95 per class from the matching of section 2 above, averaged over the classes a
  model predicts (the four classes for a trained run; Car, Pedestrian and LargeVehicle for E0) and
  labelled as such.
- Precision, recall and F1 at the frozen threshold: per class and macro-averaged, from the same matching
  at IoU 0.5. Safety recall: Pedestrian and Cyclist recall at the threshold, through the evaluator's own
  data cleaning and matching (Moderate, IoU 0.5).
- Bootstrap: 1,000 replicates with seed 20261020. Each replicate resamples the test drives with
  replacement and, independently, each validation split's drives; intervals are 95% percentile intervals.
  Each replicate's AP is computed from cached per-image matches (`evaluation/kitti_cache.py`), which
  reproduce the evaluator's arithmetic exactly. `evaluation/selfcheck.py` checks this on synthetic data
  and on resamples of it, checks the matching of section 2 above against Ultralytics' own, and checks that
  perfect predictions score as they must; the freeze runs it first and stops if any check fails. Before
  the bootstrap, the test opening checks for every model that the cached AP equals the evaluator's on the
  full test set, and stops if they differ by more than 1e-6.
- Pre-specified differences, on the test set at every difficulty: fine-tuned minus E0 at each size (n, s
  and m; Car, Pedestrian and their mean, as section 5.3 requires); YOLO11m minus YOLO11s; YOLO11s minus
  YOLO11n; YOLO11s at 960 minus YOLO11s at 640; and the random-split run minus the drive-split run on the
  common test set. "YOLO11s" is the headline run (seed 0).
- Leakage: for the random-split run and the headline run, validation minus test score (KITTI AP|R40), and
  the difference between the two gaps; validation and test drives are resampled independently.
- Seeds: the mean and standard deviation over seeds 0, 1 and 2 of the headline configuration.

## 5. Failure analysis (section 5.5)

- For each trained run, on the test set at its frozen threshold, each Car, Pedestrian and Cyclist object
  of the original labels is classed as found (a same-class box at the class's KITTI IoU of 0.7, 0.5 or
  0.5, matched one to one in order of score), poorly localised (otherwise, an unmatched same-class box
  with IoU of at least 0.3), given the wrong class (otherwise, another class's box with IoU of at least
  0.5) or missed.
- The outcomes are counted by box height (up to 25, 25 to 40, over 40 pixels), occlusion (0 to 3) and
  truncation (up to 0.15, 0.15 to 0.30, 0.30 to 0.50, over 0.50), separately and all three together.

## 6. Latency (section 6, amendment 2)

- `evaluation/latency.py`, run through `::latency`, in two sessions, each in a container of its own on a
  Modal L4. Models: the frozen YOLO11n, s and m at 640 (the seed 0 runs), the frozen YOLO11s at 960, and
  stock YOLO26s at 640 with `nms=False` and `nms=None`.
- Frames: the first 550 images of `drive_disjoint/val.txt`, decoded into memory before timing; 50
  discarded and 500 timed; batch 1; confidence 0.25. Ultralytics' per-image stage times (preprocess,
  inference, postprocess) are summarised as mean, median and 95th percentile, with the letterboxed input
  shape and the GPU, driver, CUDA, cuDNN and torch versions.

## 7. Software

- The evaluation image adds numba 0.68.0 and llvmlite 0.50.0, which the vendored evaluator needs and
  training never used. Every other package is that of amendment 2.

## 8. Outside the study

- Two demonstrations use the trained models but are not part of the evaluation, and nothing from them
  enters the results: `demo/nigeria_demo.py`, which annotates licensed Lagos road footage, and Anya
  (`anya/`), a web app that runs the 960-pixel YOLO11s and a COCO-pretrained YOLO11m on uploaded photos,
  videos or a live camera view. The Nigerian video and the live demonstration, both dropped in
  PREREG.md section 8, return in this way, without labels or scores.

## 9. Code and tag

- New: `evaluation/` (`freeze.py`, `open_test.py`, `latency.py`, `scoring.py`, `kitti_cache.py`,
  `models.py`, `selfcheck.py`), `demo/`, `anya/` and this file. Updated: `cloud/kitti_modal.py` (the three steps and the
  evaluation image).
- `scripts/`, `kitti/`, `configs/` and `splits/` are unchanged since `prereg-v1`, apart from
  `scripts/predict.py` (amendment 3), as every container checks.
- Training ran at `prereg-v1.3`. The evaluation runs at `prereg-v1.4`, the tag on the commit that adds this
  file.

## Unchanged

Everything else in `prereg-v1` and amendments 1 to 3 stands, including the split files and their hashes,
the recipe, the run list, the evaluation protocol, the single test opening and the environment logging.
