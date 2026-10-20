"""Scoring for the freeze and the single test opening (PREREG.md section 5, PREREG_AMENDMENT_4.md).

Everything here reads saved prediction files (scripts/predict.py: conf 0.001, NMS IoU 0.7, at most 300 boxes
per image) and the original KITTI labels. Nothing here runs a model.

- Table (a), KITTI protocol: kitti/evaluate.py (the vendored mmdet3d evaluator), unchanged.
- Table (b), Ultralytics COCO-style: the predictions are matched to the four merged classes exactly as
  Ultralytics' DetectionValidator.match_predictions does, and scored with ultralytics.utils.metrics.ap_per_class.
- Confidence threshold: the best all-class F1 point of that matching on the model's own validation predictions,
  as ap_per_class picks it (the argmax of the smoothed mean F1-confidence curve).
- Safety recall: Pedestrian and Cyclist recall at the threshold, through the evaluator's own data cleaning and
  matching (Moderate, IoU 0.5, Van, Person_sitting and DontCare ignored).
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from kitti.common import DIFFICULTIES, KITTI_TO_YOLO, YOLO_NAMES, read_label  # noqa: E402
from kitti.evaluate import KITTI_CLASSES, dt_anno, gt_anno, kitti_ap40  # noqa: E402
from third_party.mmdet3d_kitti_eval import eval as mm  # noqa: E402

IOU_COCO = np.linspace(0.5, 0.95, 10)
KITTI_IOU = {"Car": 0.7, "Pedestrian": 0.5, "Cyclist": 0.5}
HEIGHT_BINS = [(0, 25, "<=25 px"), (25, 40, "25-40 px"), (40, 1e9, ">40 px")]  # (lo, hi]: up to 25, 25 to 40, over 40
TRUNC_BINS = [(-1, 0.15, "<=0.15"), (0.15, 0.3, "0.15-0.30"), (0.3, 0.5, "0.30-0.50"), (0.5, 2, ">0.50")]


# ---------------------------------------------------------------- inputs

def load_preds(path):
    """{image id: [[name, x1, y1, x2, y2, score], ...]} from a predict.py JSON-lines file."""
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            out[r["id"]] = r["dets"]
    return out


def load_labels(label_dir, ids):
    return {i: read_label(Path(label_dir) / f"{i}.txt") for i in ids}


def merged_targets(objs):
    """The four training classes, as the converted YOLO labels hold them (Misc and DontCare dropped)."""
    return [(KITTI_TO_YOLO[o["name"]], o["bbox"]) for o in objs if o["name"] in KITTI_TO_YOLO]


def iou_matrix(a, b):
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-7)


# ---------------------------------------------------------------- table (b): Ultralytics COCO-style

def coco_match(preds, labels, ids):
    """Correctness of every prediction at IoU 0.50:0.95, as DetectionValidator.match_predictions (greedy)."""
    tps, confs, pcls, tcls = [], [], [], []
    for i in ids:
        gts = merged_targets(labels[i])
        tcls.extend(c for c, _ in gts)
        dets = preds.get(i, [])
        if not dets:
            continue
        dcls = np.array([YOLO_NAMES.index(d[0]) for d in dets])
        dbox = np.array([d[1:5] for d in dets], float)
        correct = np.zeros((len(dets), len(IOU_COCO)), bool)
        if gts:
            gcls = np.array([c for c, _ in gts])
            iou = iou_matrix([b for _, b in gts], dbox) * (gcls[:, None] == dcls[None, :])
            for k, t in enumerate(IOU_COCO):
                m = np.array(np.nonzero(iou >= t)).T
                if m.shape[0]:
                    if m.shape[0] > 1:
                        m = m[iou[m[:, 0], m[:, 1]].argsort()[::-1]]
                        m = m[np.unique(m[:, 1], return_index=True)[1]]
                        m = m[np.unique(m[:, 0], return_index=True)[1]]
                    correct[m[:, 1].astype(int), k] = True
        tps.append(correct)
        confs.append(np.array([d[5] for d in dets], float))
        pcls.append(dcls)
    if tps:
        return np.concatenate(tps), np.concatenate(confs), np.concatenate(pcls), np.array(tcls)
    return np.zeros((0, len(IOU_COCO)), bool), np.zeros(0), np.zeros(0, int), np.array(tcls)


def coco_table(match, predicted_classes=YOLO_NAMES):
    """AP50 and AP50-95 per class, their means over the classes the model can predict, and the F1 threshold."""
    from ultralytics.utils.metrics import ap_per_class, smooth
    tp, conf, pcls, tcls = match
    _, _, p, r, f1, ap, classes, _, _, f1_curve, x, _ = ap_per_class(tp, conf, pcls, tcls)
    per_class = {YOLO_NAMES[c]: {"ap50": float(ap[k, 0]), "ap50_95": float(ap[k].mean())} for k, c in enumerate(classes)}
    use = [n for n in predicted_classes if n in per_class]
    best = int(smooth(f1_curve.mean(0), 0.1).argmax())
    return {
        "per_class": per_class,
        "classes_averaged": use,
        "map50": float(np.mean([per_class[n]["ap50"] for n in use])),
        "map50_95": float(np.mean([per_class[n]["ap50_95"] for n in use])),
        "f1_threshold": float(x[best]),
        "f1_at_threshold": float(f1_curve.mean(0)[best]),
    }


def prf_at(match, threshold, predicted_classes=YOLO_NAMES):
    """Precision, recall and F1 per class at a confidence threshold (IoU 0.5), and their macro means."""
    tp, conf, pcls, tcls = match
    out = {}
    for c, name in enumerate(YOLO_NAMES):
        if name not in predicted_classes:
            continue
        sel = (pcls == c) & (conf >= threshold)
        n_tp = int(tp[sel, 0].sum())
        n_fp = int(sel.sum()) - n_tp
        n_gt = int((tcls == c).sum())
        prec = n_tp / (n_tp + n_fp) if n_tp + n_fp else 0.0
        rec = n_tp / n_gt if n_gt else 0.0
        out[name] = {"tp": n_tp, "fp": n_fp, "fn": n_gt - n_tp, "precision": prec, "recall": rec,
                     "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0}
    out["macro"] = {k: float(np.mean([v[k] for v in out.values()])) for k in ("precision", "recall", "f1")}
    return out


# ---------------------------------------------------------------- table (a): KITTI protocol

def kitti_annos(preds, labels, ids):
    gts = [gt_anno(labels[i]) for i in ids]
    dts = [dt_anno([tuple(d) for d in preds.get(i, [])]) for i in ids]
    return gts, dts


def kitti_table(gts, dts, classes=KITTI_CLASSES):
    return kitti_ap40(gts, dts, classes)


def kitti_recall_at(gts, dts, cls_name, threshold, difficulty="moderate", min_overlap=None):
    """Recall at a confidence threshold through the evaluator's own cleaning and matching (safety recall)."""
    c = list(KITTI_CLASSES).index(cls_name)
    d = DIFFICULTIES.index(difficulty)
    min_overlap = KITTI_IOU[cls_name] if min_overlap is None else min_overlap
    overlaps, _, _, _ = mm.calculate_iou_partly(dts, gts, 0, num_parts=min(50, len(gts)))
    gdl, ddl, igs, ids_, dcs, _, n_valid = mm._prepare_data(gts, dts, c, d)
    n_tp = n_fn = 0
    for k in range(len(gts)):
        tp, fp, fn, _, _ = mm.compute_statistics_jit(overlaps[k], gdl[k], ddl[k], igs[k], ids_[k], dcs[k], 0,
                                                     min_overlap=min_overlap, thresh=threshold, compute_fp=True)
        n_tp += tp
        n_fn += fn
    return {"recall": n_tp / (n_tp + n_fn) if n_tp + n_fn else 0.0, "tp": int(n_tp), "fn": int(n_fn),
            "valid_objects": int(n_valid)}


# ---------------------------------------------------------------- failure analysis

def bin_of(value, bins):
    for lo, hi, name in bins:
        if lo < value <= hi or (lo == 0 and value == 0):
            return name
    return bins[-1][2]


def failure_analysis(preds, labels, ids, threshold):
    """Every Car, Pedestrian and Cyclist object of the original labels, at the frozen threshold, is found
    (same class, IoU at the class threshold, matched one-to-one by score), poorly localised (an unmatched
    same-class box with IoU >= 0.3), given the wrong class (another class's box with IoU >= 0.5) or missed."""
    rows = []
    for i in ids:
        objs = [o for o in labels[i] if o["name"] in KITTI_IOU]
        dets = [d for d in preds.get(i, []) if d[5] >= threshold]
        dets.sort(key=lambda d: -d[5])
        found = [False] * len(objs)
        used = [False] * len(dets)
        if objs and dets:
            iou = iou_matrix([o["bbox"] for o in objs], [d[1:5] for d in dets])
            for j, d in enumerate(dets):  # one-to-one, by descending score
                best, best_k = 0.0, -1
                for k, o in enumerate(objs):
                    if not found[k] and o["name"] == d[0] and iou[k, j] >= KITTI_IOU[o["name"]] and iou[k, j] > best:
                        best, best_k = iou[k, j], k
                if best_k >= 0:
                    found[best_k], used[j] = True, True
        else:
            iou = np.zeros((len(objs), len(dets)))
        for k, o in enumerate(objs):
            if found[k]:
                outcome = "found"
            elif any(not used[j] and dets[j][0] == o["name"] and iou[k, j] >= 0.3 for j in range(len(dets))):
                outcome = "poor localisation"
            elif any(dets[j][0] != o["name"] and iou[k, j] >= 0.5 for j in range(len(dets))):
                outcome = "wrong class"
            else:
                outcome = "missed"
            x1, y1, x2, y2 = o["bbox"]
            rows.append({"image": i, "class": o["name"], "outcome": outcome,
                         "height": bin_of(y2 - y1, HEIGHT_BINS), "occlusion": int(o["occluded"]),
                         "truncation": bin_of(o["truncated"], TRUNC_BINS)})
    return rows


def failure_tables(rows):
    """Counts of each outcome per class, by height, occlusion and truncation, and by all three together."""
    outcomes = ["found", "poor localisation", "wrong class", "missed"]

    def tab(keys):
        t = {}
        for r in rows:
            key = " | ".join([r["class"]] + [str(r[k]) for k in keys])
            t.setdefault(key, dict.fromkeys(outcomes, 0))[r["outcome"]] += 1
        return dict(sorted(t.items()))
    return {"by_class": tab([]), "by_height": tab(["height"]), "by_occlusion": tab(["occlusion"]),
            "by_truncation": tab(["truncation"]), "by_height_occlusion_truncation": tab(["height", "occlusion", "truncation"])}


# ---------------------------------------------------------------- drive-level bootstrap

def drive_groups(ids, drive_of):
    groups = {}
    for i in ids:
        groups.setdefault(drive_of[i], []).append(i)
    return [groups[d] for d in sorted(groups)]


def resample(groups, rng):
    picks = rng.integers(0, len(groups), len(groups))
    return [i for k in picks for i in groups[k]]


def ci(values):
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], float)
    if not len(v):
        return None
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
