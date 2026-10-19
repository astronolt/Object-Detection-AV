# Pre-registration amendment 3: prediction batches

Written 19 October 2026. This amends [PREREG.md](PREREG.md) as tagged `prereg-v1`,
[PREREG_AMENDMENT_1.md](PREREG_AMENDMENT_1.md) as tagged `prereg-v1.1` and
[PREREG_AMENDMENT_2.md](PREREG_AMENDMENT_2.md) as tagged `prereg-v1.2`
(commit `96d26932a0e9bcc5cfb34e9f5491c165596ef8fc`). Where they differ, this file governs.

## Status when written

- Since `prereg-v1.2`, the data step has passed, the smoke test has run, and the launch has been
  rehearsed once (19 October 2026, Modal app `ap-RYJkWeN153ud2VSM9T1nEW`, folder
  `rehearsal/20261019-171829`).
- In that rehearsal the 1-epoch headline completed and passed the halt rule, part 2 re-attached to
  the runs and stopped the long queue, and E0 started once the headline had completed. E0 then
  stopped at its first model with a CUDA out-of-memory error, and the verdict was
  `REHEARSAL FAILED`. E0 saved no predictions.
- Nothing from the rehearsal is used. No run of section 3.2 has trained beyond the rehearsal's
  single epoch, no model has been scored on the new split, and the test set has not been opened.

## 1. The defect

`scripts/predict.py`, unchanged since `prereg-v1`, passes the whole list of image paths to
Ultralytics' `predict` in one call, with `batch=16`. Ultralytics 8.4.171 treats a list as images
held in memory: it opens every image and passes them all through the model as one batch
(`check_source`, `autocast_list`, `LoadPilAndNumpy`), so `batch` has no effect. For E0 on the 1,463
validation images this is one batch of 1,463 images, each letterboxed to 640 × 640, which needs far
more than the L4's 22 GiB. YOLO11n stopped when it asked for a further 6.70 GiB, the size of one of
its early feature maps for 1,463 images (48 channels × 160 × 160 × 4 bytes × 1,463). The
validation and test predictions of the trained models are to be saved through the same function,
so those steps would fail in the same way. No earlier step of the study ran this script: the
offline tests of amendment 2 simulated the training script that calls it, so the rehearsal was the
first step to run it.

## 2. The change

`scripts/predict.py` passes the paths 16 at a time (`PREDICT["batch"]` in `configs/recipe.py`,
unchanged), one `predict` call per group, and sets `rect=False`.

- With images of several sizes in one batch, Ultralytics letterboxes each of them to a 640 × 640
  square. The original single batch would therefore have used the square for every image, since the
  validation and test images come in four sizes. When all the images in a batch share one size,
  Ultralytics' predict mode letterboxes them to a smaller rectangle instead, so in groups of 16 an
  image's input would depend on which images share its group. `rect=False` keeps the square for
  every image, as before.
- Nothing else changes: the weights, `imgsz` 640, conf 0.001, NMS IoU 0.7, at most 300 detections
  per image, E0's COCO class filter and mapping, its second class-wise NMS, the output format and
  the line-count check.
- With the same package versions, on a CPU, the amended script was compared with the original on
  24 images in KITTI's four sizes, made from Ultralytics' sample photographs, for YOLO11n, s and m.
  The original passed all 24 images through the model at once; the amended script passed 16 and
  then 8. Both saved the same number of detections of each class for every image, with scores
  within 0.000002 and box corners within 0.002 pixels of each other, the size of floating-point
  differences between batch sizes.

## 3. Code and tag

- This is the only change to `scripts/`, `kitti/`, `configs/` and `splits/` since `prereg-v1`.
  `cloud/kitti_modal.py` allows exactly this one: every container refuses to run unless those
  folders are as they were at `prereg-v1`, apart from `scripts/predict.py`, whose SHA-256, computed
  on its text with LF line endings, must be
  `50807d48cd7fbe2c63e9ce531327acd3105a95f7ea1af5b47b23d1004b806da8`.
- Training uses `prereg-v1.3`, the tag on the commit that adds this file, the amended
  `scripts/predict.py` and the updated `cloud/kitti_modal.py`.

## 4. Rehearsal and earlier steps (amends amendment 2, section 3)

- The rehearsal is repeated at `prereg-v1.3`, in a new folder, and the launch starts only after a
  rehearsal at `prereg-v1.3` whose verdict begins with `REHEARSAL OK`. The first rehearsal's folder
  is kept, and nothing from it is used.
- The data step and the smoke test, run at `prereg-v1.2`, stand, since neither runs
  `scripts/predict.py`. The launch takes its budget projection from that smoke test, as amendment 2
  describes.

## Unchanged

Everything else in `prereg-v1` and amendments 1 and 2 stands, including the split files and their
hashes, the recipe, the run list, the halt rule, the cut order and the never-cut list, the budget
rules, the cut-off, and the evaluation, single test opening and environment logging protocols.
