"""Checks of the evaluation code against the code it reproduces (PREREG_AMENDMENT_4.md section 4).

1. The cached-match AP (kitti_cache.py) equals kitti/evaluate.py's kitti_ap40, on a synthetic data set and on
   bootstrap resamples of it.
2. coco_match marks every prediction exactly as Ultralytics' own DetectionValidator.match_predictions does.
3. Perfect predictions (the labels themselves) score as they must: KITTI AP|R40 of 100, Ultralytics AP of 0.995
   (the ceiling of its 101-point interpolation), precision, recall and safety recall of 1, and every object found.

Needs no KITTI data and no GPU. Exits non-zero if a check fails.
Usage: python evaluation/selfcheck.py
"""
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from evaluation.kitti_cache import CachedKitti  # noqa: E402
from evaluation.scoring import (coco_match, coco_table, failure_analysis, kitti_annos, kitti_recall_at,  # noqa: E402
                                kitti_table, merged_targets, prf_at)
from kitti.common import KITTI_TO_YOLO, YOLO_NAMES  # noqa: E402
from kitti.evaluate import dt_anno, gt_anno, kitti_ap40  # noqa: E402

NAMES = ["Car"] * 6 + ["Van", "Pedestrian", "Pedestrian", "Person_sitting", "Cyclist", "Truck", "DontCare", "Misc"]
FAILED = []


def check(ok, what):
    print(("PASS  " if ok else "FAIL  ") + what, flush=True)
    if not ok:
        FAILED.append(what)


def synth(rng, n_img, names=NAMES, max_objs=12, n_det_extra=40, noise=6.0):
    """KITTI-like labels, and detections near them (some given the wrong class) plus random false positives."""
    labels, dets = [], []
    for _ in range(n_img):
        objs = []
        for _ in range(rng.integers(0, max_objs)):
            x1, y1 = rng.uniform(0, 1100), rng.uniform(100, 300)
            w, h = rng.uniform(10, 200), rng.uniform(15, 150)
            objs.append({"name": str(rng.choice(names)), "truncated": float(rng.choice([0, 0, 0.1, 0.3, 0.6])),
                         "occluded": int(rng.integers(0, 4)), "alpha": -1.0, "bbox": [x1, y1, x1 + w, y1 + h]})
        d = []
        for o in objs:
            if o["name"] not in KITTI_TO_YOLO:
                continue
            name = YOLO_NAMES[KITTI_TO_YOLO[o["name"]]]
            for _ in range(rng.integers(0, 3)):
                b = np.array(o["bbox"]) + rng.normal(0, noise, 4)
                wrong = rng.random() < 0.1
                d.append([str(rng.choice(YOLO_NAMES)) if wrong else name, *b.tolist(), float(rng.uniform(0.001, 1))])
        for _ in range(rng.integers(0, n_det_extra)):
            x1, y1 = rng.uniform(0, 1100), rng.uniform(100, 300)
            d.append([str(rng.choice(YOLO_NAMES)), x1, y1, x1 + rng.uniform(10, 150), y1 + rng.uniform(15, 120),
                      float(rng.uniform(0.001, 0.6))])
        d.sort(key=lambda r: -r[5])  # NMS output order, as scripts/predict.py writes it
        labels.append(objs)
        dets.append(d)
    ids = [f"{k:06d}" for k in range(n_img)]
    return ids, dict(zip(ids, labels)), dict(zip(ids, dets))


def check_cache(rng):
    ids, labels, preds = synth(rng, 300)
    gts = [gt_anno(labels[i]) for i in ids]
    dts = [dt_anno([tuple(d) for d in preds[i]]) for i in ids]
    t0 = time.time()
    ref = kitti_ap40(gts, dts)
    cache = CachedKitti(gts, dts)
    got = cache.ap40()
    worst = max(abs(ref[c][d] - got[c][d]) for c in ref for d in ref[c])
    check(worst < 1e-9, f"cached AP equals kitti_ap40 on 300 synthetic images (largest difference {worst:.1e}; "
                        f"{time.time() - t0:.1f} s)")
    for trial in range(5):
        idx = rng.integers(0, len(gts), len(gts))
        ref_b = kitti_ap40([gts[i] for i in idx], [dts[i] for i in idx])
        got_b = cache.ap40(np.bincount(idx, minlength=len(gts)))
        worst = max(abs(ref_b[c][d] - got_b[c][d]) for c in ref_b for d in ref_b[c])
        check(worst < 1e-9, f"cached AP equals kitti_ap40 on bootstrap resample {trial + 1} (largest difference "
                            f"{worst:.1e}; Moderate {({c: round(ref_b[c]['moderate'], 2) for c in ref_b})})")


def check_matching(rng):
    import torch
    from ultralytics.models.yolo.detect import DetectionValidator
    from ultralytics.utils.metrics import box_iou
    v = DetectionValidator.__new__(DetectionValidator)  # match_predictions needs only the IoU thresholds
    v.iouv = torch.linspace(0.5, 0.95, 10)
    ids, labels, preds = synth(rng, 400, noise=12.0)
    ours = coco_match(preds, labels, ids)[0]
    theirs = []
    for i in ids:
        dets = preds[i]
        if not dets:
            continue
        gts = merged_targets(labels[i])
        if not gts:
            theirs.append(np.zeros((len(dets), 10), bool))
            continue
        iou = box_iou(torch.tensor([b for _, b in gts], dtype=torch.float64),
                      torch.tensor([d[1:5] for d in dets], dtype=torch.float64))
        theirs.append(v.match_predictions(torch.tensor([YOLO_NAMES.index(d[0]) for d in dets]),
                                          torch.tensor([c for c, _ in gts]), iou).numpy())
    theirs = np.concatenate(theirs)
    check(ours.shape == theirs.shape and bool((ours == theirs).all()),
          f"coco_match equals DetectionValidator.match_predictions on {len(ours)} predictions "
          f"({int(ours[:, 0].sum())} correct at IoU 0.5, {int(ours[:, -1].sum())} at IoU 0.95)")


def check_perfect(rng):
    names = ["Car"] * 3 + ["Van", "Pedestrian", "Pedestrian", "Person_sitting", "Cyclist", "Cyclist", "Truck",
                           "DontCare"]
    ids, labels, _ = synth(rng, 1500, names=names, max_objs=10)
    preds = {i: [[YOLO_NAMES[KITTI_TO_YOLO[o["name"]]], *o["bbox"], 0.9] for o in labels[i] if o["name"] in KITTI_TO_YOLO]
             for i in ids}
    match = coco_match(preds, labels, ids)
    coco = coco_table(match)
    check(all(abs(v["ap50"] - 0.995) < 1e-9 and abs(v["ap50_95"] - 0.995) < 1e-9 for v in coco["per_class"].values())
          and len(coco["per_class"]) == 4, f"perfect predictions: Ultralytics AP50 and AP50-95 of 0.995 for all four "
                                           f"classes; F1 threshold {coco['f1_threshold']:.3f}")
    prf = prf_at(match, coco["f1_threshold"])
    check(all(prf[c]["precision"] == 1 and prf[c]["recall"] == 1 for c in YOLO_NAMES),
          "perfect predictions: precision and recall of 1 at the F1 threshold")
    gts, dts = kitti_annos(preds, labels, ids)
    kt = kitti_table(gts, dts)
    check(all(abs(kt[c][d] - 100) < 1e-9 for c in kt for d in kt[c]),
          f"perfect predictions: KITTI AP|R40 of 100 for every class and difficulty ({kt['Cyclist']})")
    cached = CachedKitti(gts, dts).ap40()
    check(all(abs(cached[c][d] - 100) < 1e-9 for c in cached for d in cached[c]),
          "perfect predictions: cached AP of 100 as well")
    for c in ("Pedestrian", "Cyclist"):
        s = kitti_recall_at(gts, dts, c, 0.5)
        check(s["recall"] == 1 and s["fn"] == 0 and s["tp"] == s["valid_objects"],
              f"perfect predictions: {c} safety recall of 1 ({s['tp']} of {s['valid_objects']} Moderate objects)")
    rows = failure_analysis(preds, labels, ids, 0.5)
    check(len(rows) > 0 and all(r["outcome"] == "found" for r in rows),
          f"perfect predictions: all {len(rows)} Car, Pedestrian and Cyclist objects found")
    empty = {i: [] for i in ids}
    rows = failure_analysis(empty, labels, ids, 0.5)
    check(all(r["outcome"] == "missed" for r in rows), "no predictions: every object missed")


def main():
    rng = np.random.default_rng(20261020)
    check_cache(rng)
    check_matching(rng)
    check_perfect(rng)
    if FAILED:
        sys.exit(f"{len(FAILED)} check(s) failed")
    print("All checks passed.")


if __name__ == "__main__":
    main()
