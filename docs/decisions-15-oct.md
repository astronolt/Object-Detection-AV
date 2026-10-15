# Build decisions, 15 October 2026

Reply to your review of plan-extract.md. Go ahead with the fresh split and start building. The library points below were checked today against the ultralytics 8.4.171 wheel from PyPI and the mmdet3d KITTI evaluator source.

## 1. Fresh split: approved, on these conditions

- Write the procedure down and commit it before running it.
  - Objective: 60/20/20 of images, with drives disjoint, mapped through `train_rand.txt` → `train_mapping.txt`.
  - Balance: in every partition, the Car, Pedestrian and Cyclist instance shares are within 1.2 percentage points of that partition's image share. This definition gives 1.15 on the doc's old table, so it matches the doc's "1.2 points".
  - Also record the seed, the number of candidates searched and the tie-break.
  - Run it once and never re-roll it.
- Aim at those criteria, not at the old split's exact counts, and report whatever counts come out.
- The random-split control uses the new train + validation pool. Give it the same train and validation sizes as the drive-disjoint split, and the same test file.
- Build the split, hash it and run the evaluator's 100-AP check locally now. These need only the label and devkit zips, not the 12 GB images. Only the round-trip check and the smoke test have to wait for Colab.
- Put the procedure, the seed and the full SHA-256 of all three split files into the pre-registration. Commit and tag it, then give me the tag's hash so I can email it to my supervisor before training starts.
- Push the repo to a private GitHub remote (or a Drive-synced folder) after every milestone. The 3 October work was lost because it existed in one place only.
- 006500 and 007250: agreed, no hand-placing. Chapter 3 mentions them wherever they land.

## 2. `cls_remap` is real: keep `cls_remap=False`

It is a train argument in 8.4.171, with default `True` (`cfg/default.yaml`, line 26). When it is `True`, loading pretrained weights copies class-head rows wherever class names match, ignoring case (`nn/tasks.py`, `_remap_cls_by_names`). COCO's "car" matches "Car", so Car alone would start with a trained head row. `False` gives every class a fresh head, as the doc intends. There is no need to wait for the smoke test.

## 3. Values to write out explicitly (8.4.171 defaults)

- Colour: `hsv_h=0.015, hsv_s=0.7, hsv_v=0.4`
- Geometry: `translate=0.1, scale=0.5, shear=0.0, perspective=0.0`, plus the doc's `degrees=0.0, flipud=0.0, fliplr=0.5`
- Mixing: `mosaic=1.0, mixup=0.0, cutmix=0.0, copy_paste=0.0, bgr=0.0, multi_scale=0.0`, plus the doc's `close_mosaic=10`
- Optimiser and loss: `lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_momentum=0.8, nbs=64, box=7.5, cls=0.5, dfl=1.5, cls_pw=0.0`

`erasing` and `auto_augment` apply only to classification. The doc's other source claims hold in 8.4.171: `optimizer='auto'` ignores `lr0`, gradients accumulate to `nbs=64`, `cos_lr` defaults to off and `patience` defaults to 100.

## 4. Your proposed defaults

- Class order 0 Car, 1 Pedestrian, 2 Cyclist, 3 LargeVehicle: agreed.
- AP overlaps of Car 0.7, Pedestrian 0.5 and Cyclist 0.5: agreed. These are the 2D rows of `kitti_eval` in mmdet3d. Its difficulty rules for Easy/Moderate/Hard are a minimum height of 40/25/25 px, maximum occlusion of 0/1/2 and maximum truncation of 0.15/0.3/0.5.
- E0 at 640, with car → Car, person → Pedestrian, truck and bus → LargeVehicle, and Cyclist n/a: agreed, with three additions. Write all three into the pre-registration before E0 runs.
  - Also map COCO train → LargeVehicle, because LargeVehicle is Truck + Tram.
  - E0 has no Cyclist, so every E0-versus-fine-tuned "overall" figure for RQ1 and H1 must average the same classes for both models (Car and Pedestrian in the KITTI table). Otherwise a two-class mean is compared with a three-class one.
  - Riders detected as "person" count as Pedestrian false positives under KITTI's rules. That is part of the vocabulary gap: state it, and don't correct for it.
- torch: agreed. Also save `pip freeze` and `nvidia-smi` output in every run folder.

## 5. Epochs and runtimes

At the smoke test, open a second runtime.

- If it is also an L4 and my compute-unit balance covers about 30 units, split the queue across the two runtimes and keep 100 epochs, the doc's own setting. Training then ends around 04:00, within an hour of the 50-epoch single-runtime plan, and there is no epoch deviation to report.
- Otherwise, close the second runtime and run 50 epochs on one. All training stays on one GPU type.

Write this rule into the pre-registration before the first run, and apply it to every run.

Suggested lists, using the doc's L4 estimates at 100 epochs:

- Runtime A: S0 → nano → 960 → seed 2 (about 7.7 h)
- Runtime B: E0 → random split → medium → seed 1 (about 6.1 h)

The never-cut runs start first and the seeds run last, so anything unfinished at the freeze follows the doc's cut order.

## 6. Latency: the Colab L4

The lab machine can't be used tonight, so use the L4 and report it as a deviation.

- After training, time every model back to back in one session: n, s and m at 640, s at 960, and the stock YOLO26s with `nms=False` and `nms=None`. In 8.4.171 these two settings select the NMS-free head and the NMS head.
- Then repeat the timings in a second session.
- Record the GPU model and the driver, CUDA and torch versions.

The rest of the doc's latency protocol stays as written.

## 7. Order of work

- Tonight:
  1. Build the code.
  2. Run the split and evaluator checks locally.
  3. Commit, tag and email the pre-registration.
  4. On Colab, download the data, then run the round-trip check and the smoke test.
  5. Launch the queue.
- Friday morning:
  1. Freeze review with me.
  2. The single test opening.
  3. Latency, failure analysis and bootstrap.

The test opening is the one step that cannot be undone, so it happens after a rested freeze review, not overnight.
