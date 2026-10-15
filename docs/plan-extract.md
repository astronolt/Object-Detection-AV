# Plan extract: recipe, split, evaluation and E0

Extracted on 15 October 2026 from the Claude Docs doc "YOLO–KITTI MSc: Audit & One-Week Plan" (3 October 2026, revision 24): https://claude.ai/code/artifact/e94c5846-1b2c-4107-a64e-d1e8a3b40e59

Text in quotation marks and the code block are verbatim from the doc. Anything the doc does not record is marked "Not in the doc".

## 1. Training recipe

"Every run uses one explicit configuration; nothing is left to Ultralytics defaults. Only the model, image size, split and seed change between runs."

```python
# Shared by every training run. Pin: ultralytics==8.4.171 (and the same torch) on every machine.
common = dict(
    epochs=EPOCHS,          # one value for ALL runs: 100, or 50 if Day-1 timing forces it
    patience=0,             # 0 disables early stopping, so every run gets the same schedule
    batch=16,               # fixed, never -1; Ultralytics accumulates to a nominal batch of 64
    optimizer="AdamW", lr0=0.001, cos_lr=True,   # 'auto' would silently ignore lr0
    warmup_epochs=3, warmup_bias_lr=0.0,
    close_mosaic=10, freeze=None,                # no staged unfreezing (not supported)
    fliplr=0.5, flipud=0.0, degrees=0.0,         # TAB 07 values, stated explicitly
    cls_remap=False,        # fresh class head for every class, so Car gets no head start over Pedestrian/Cyclist
    amp=True, deterministic=True, seed=SEED,
    cache="disk", workers=8, imgsz=IMGSZ,
    project="/content/drive/MyDrive/msc/runs", name=RUN_ID,   # last.pt saved every epoch
)
```

These are the per-run values in queue priority order. The hours are the doc's L4 estimates at 100 epochs.

| Run ID | Model | imgsz | Split | Seed | L4 hours |
|---|---|---|---|---|---|
| E1-Y11S-640-DD-S0 (headline) | YOLO11s | 640 | drive-disjoint | 0 | 1.65 |
| RS-Y11S-640-RND-S0 | YOLO11s | 640 | random | 0 | 1.65 |
| E1-Y11N-640-DD-S0 | YOLO11n | 640 | drive-disjoint | 0 | 1.14 |
| E1-Y11M-640-DD-S0 | YOLO11m | 640 | drive-disjoint | 0 | 2.63 |
| E2-Y11S-960-DD-S0 | YOLO11s | 960 | drive-disjoint | 0 | 3.24 |
| E1-Y11S-640-DD-S1 | YOLO11s | 640 | drive-disjoint | 1 | 1.65 |
| E1-Y11S-640-DD-S2 | YOLO11s | 640 | drive-disjoint | 2 | 1.65 |
| E4-Y26S-640-DD-S0 | YOLO26s, `nms=False` | 640 | drive-disjoint | 0 | 1.87 |

- **Classes:** "Training classes stay as approved: Car (+Van), Pedestrian (+Person_sitting), Cyclist, LargeVehicle (Truck+Tram). Misc is dropped, and DontCare boxes are removed from training labels; that is a stated label-noise limitation."
- **Smoke test:** "YOLO11s 3 epochs; n, m, s@960, YOLO26s 1 epoch each", about 0.9 L4 hours.
- **Epochs:** "If that test projects more than 22 L4-hours of training, every run uses 50 epochs instead. That decision is made once, before the first real run, and never changed per run."
- **Batch:** "fixed at 16, or 8 only if memory requires, and logged". This comes from the supervisor message.
- **GPU cut order:** cut YOLO26s first, then seed 2, seed 1, YOLO11s at 960 and YOLO11m. Never cut the headline run, the random-split run, E0, the single test opening, latency or the failure analysis.

**Not in the doc:**
- No augmentation is set beyond `fliplr`, `flipud`, `degrees` and `close_mosaic`. There are no mosaic, HSV, scale or translate values, despite the "nothing is left to Ultralytics defaults" line. The comment calls the flip and rotation values "TAB 07 values", so check TAB 07 in the proposal .docx in the project folder for the rest.
- Class indices are not given. The four classes are listed in the order above.
- The torch version is not given.

## 2. Split

- **Rule:** drive-disjoint. Drive membership needs two devkit files: "image → `train_rand.txt` → `train_mapping.txt` → drive. Reading only `train_mapping.txt` gives a silently wrong split." There are 141 drives over 4 recording dates.
- **Balancing:** the split was "balanced by a seeded search". Elsewhere the doc says "the frozen split is balanced to within 1.2 points", in the context of cyclists clustering (the top 5 drives hold 49% of them). It never defines what the 1.2 points measures.
- **Seed: not recorded.** Neither is the search procedure. "42" appears nowhere in the doc.

| Partition | Images | Drives | Car | Pedestrian | Cyclist | Moderate cyclists |
|---|---|---|---|---|---|---|
| Train | 4,494 (60.1%) | 67 | 17,333 | 2,747 | 979 | 644 |
| Validation | 1,494 (20.0%) | 46 | 5,764 | 893 | 322 | 241 |
| Test (sealed) | 1,493 (20.0%) | 28 | 5,645 | 847 | 326 | 213 |

Counts use the original KITTI classes; Moderate is KITTI's Moderate difficulty.

- **Random-split control:** a random split of the 5,988-image development pool (train plus validation). It "shares the identical test file".
- **Hashes:** the doc records only one, and truncates it: the test file's SHA-256, `2b8736…74af8`. It has no train or validation hashes and no drive lists. It also says: "Freeze these files and never regenerate them."
  - A rebuild passes the hash gate only if it reproduces that test file byte for byte.
  - Any surviving copy of the 3 October test list can be checked against the 11 visible characters.
- **006500 and 007250:** the doc says these were sealed-test images found in the prototype folder and "quarantined". A stock COCO model scored them once before the protocol existed, and Chapter 3 must say so. The doc does not say they left the test set. The three partitions add up to all 7,481 images, so the 1,493 test images include them.

## 3. Evaluation

The evaluator is "A tested KITTI evaluator (the mmdet3d port, AP over 40 recall points)", vendored from mmdetection3d `kitti_utils/eval.py`: https://raw.githubusercontent.com/open-mmlab/mmdetection3d/main/mmdet3d/evaluation/functional/kitti_utils/eval.py

The protocol, verbatim:

1. **During development, use validation only.** Checkpoints are chosen by validation fitness, and E0 (the zero-shot COCO model) is scored on validation.
2. **Freeze before testing**: configs, weights, and one confidence threshold per model, taken from the best all-class F1 point on that model's validation run. "Safety recall" is fixed as KITTI-style Pedestrian and Cyclist recall at that threshold: Moderate difficulty, IoU 0.5, with Van, Person_sitting and DontCare ignored.
3. **Open the test set once.** One script scores E0 and every frozen model on test, and logs the date, git commit and test-file SHA-256. It writes a STARTED line, then a COMPLETED line. If Colab crashes before COMPLETED, it may re-run with the same commit and settings; partial outputs are deleted unread, and every attempt is reported.
4. **Report two tables, never mixed.** (a) The KITTI-protocol table: AP over 40 recall points for Car, Pedestrian and Cyclist at Easy/Moderate/Hard, scored against the original labels with Van and Person_sitting ignored and DontCare applied. (b) Ultralytics COCO-style mAP@0.5 and mAP@0.5:0.95 on the merged classes, labelled as such.
5. **Compute precision, recall and F1 at the frozen threshold** from saved predictions. Never quote Ultralytics' own test P/R/F1.
6. **Add uncertainty.** Drive-level bootstrap 95% intervals on every per-class AP and every difference; mean ± SD across the 3 seeds of the headline run. For the leakage gap, validation drives and test drives are resampled independently. The bootstrap reuses cached box matches so it runs in minutes, not the 30 hours a naive version would take.
7. **Measure latency on one named GPU** (the lab machine): batch 1, discard the first 50 frames, time at least 500, and report mean, median and p95 per stage. Use conf 0.25, state the input shape, and exclude image decoding. Time YOLO26 with `nms=False` and `nms=None` for a free NMS comparison.
8. **Leakage result**: for the random-split and drive-split models, report validation score minus sealed-test score, and both scores on the common test set.
9. **Failure analysis**: match test predictions at the frozen threshold to the original labels. Tabulate misses, wrong classes and poor localisation by box height × occlusion × truncation.

**Day-1 gate**, verbatim: "all 7,481 labels round-trip within 1 px; the split hashes match; the evaluator scores ground truth as 100 AP; the queue is running by 21:00. If it is not, fix it in the morning and launch at 50 epochs for every run."

The risks table adds an early warning for a silent label-conversion bug: "Round-trip error above 1 px; image 004696 fails; or the headline's Car validation mAP@0.5 is below 0.60, which halts the queue automatically." The response it gives is: "Stop training, fix the converter (check the narrower 1224/1238/1241-px images first), restart from the headline run."

The round-trip check depends on these converter points:
- Use the official KITTI files, not Ultralytics' kitti.zip. That download scales every label by 1,242 px, so image 004696 (1,224 px wide) ends up about 7 px off. It also uses a random split and stores JPEGs at about quality 50.
- Images come in four sizes: 1242×375 (81% of images), 1224×370, 1238×374 and 1241×376. "Each label must be normalised by its own image's size."
- The smoke test also produces a 100-image label overlay sheet for review.

**Not in the doc:** the AP IoU for Pedestrian and Cyclist. Only "IoU 0.7 for cars" is stated; KITTI's standard for the other two classes is 0.5. The IoU 0.5 in item 2 applies only to safety recall.

## 4. "E0 on validation"

E0 is the COCO-pretrained models scored zero-shot, with no training, at three scales rather than one. The doc defines it as "COCO models, no training; validation now, test only in the final run", and says "E0 runs at all three scales" (n, s and m). It is estimated at 0.15 L4 hours. The phrase comes from the queue order: "Saturday night runs S0, E0 on validation, the random split, nano, medium and 960. Sunday finishes 960, then seeds 1 and 2, then YOLO26." Because COCO has no Cyclist class, the doc says to call E0 a contextual baseline.

**Not in the doc:** the COCO-to-KITTI class mapping used to score E0, and E0's image size.
