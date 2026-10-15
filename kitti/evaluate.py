"""KITTI-protocol 2D AP|R40 through the vendored mmdet3d evaluator (PREREG.md section 5.2a)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from third_party.mmdet3d_kitti_eval import eval as mm  # noqa: E402

from kitti.common import DIFFICULTIES  # noqa: E402

KITTI_CLASSES = ("Car", "Pedestrian", "Cyclist")


def gt_anno(objs):
    """Original KITTI labels for one image (all classes, DontCare included)."""
    return {
        "name": np.array([o["name"] for o in objs], dtype=object),
        "truncated": np.array([o["truncated"] for o in objs], dtype=np.float64),
        "occluded": np.array([o["occluded"] for o in objs], dtype=np.int64),
        "alpha": np.array([o["alpha"] for o in objs], dtype=np.float64),
        "bbox": np.array([o["bbox"] for o in objs], dtype=np.float64).reshape(-1, 4),
    }


def dt_anno(dets):
    """Detections for one image: list of (kitti_name, x1, y1, x2, y2, score). alpha = -10, so no AOS."""
    return {
        "name": np.array([d[0] for d in dets], dtype=object),
        "bbox": np.array([d[1:5] for d in dets], dtype=np.float64).reshape(-1, 4),
        "score": np.array([d[5] for d in dets], dtype=np.float64),
        "alpha": np.full(len(dets), -10.0),
    }


def kitti_ap40(gt_annos, dt_annos, classes=KITTI_CLASSES):
    """Return {class: {difficulty: AP40}} at IoU 0.7/0.5/0.5 (the evaluator's strict 2D row)."""
    assert len(gt_annos) == len(dt_annos)
    # The evaluator reads gt_annos[k]['alpha'][0] on the first image; start at one with objects.
    first = next(k for k, a in enumerate(gt_annos) if len(a["name"]))
    gt = gt_annos[first:] + gt_annos[:first]
    dt = dt_annos[first:] + dt_annos[:first]
    _, res = mm.kitti_eval(gt, dt, list(classes), eval_types=["bbox"])  # fresh list: kitti_eval appends to it
    return {c: {d: float(res[f"KITTI/{c}_2D_AP40_{d}_strict"]) for d in DIFFICULTIES} for c in classes}
