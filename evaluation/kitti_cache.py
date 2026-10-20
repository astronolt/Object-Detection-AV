"""KITTI AP|R40 from cached box matches, for the drive-level bootstrap (PREREG.md section 5.5).

The vendored evaluator's AP for a set of images depends on each image only through (1) the scores of its
true-positive matches, which set the 41 score thresholds, and (2) its true-positive, false-positive and
false-negative counts at each threshold. Both come from the evaluator's own compute_statistics_jit, run once per
image here and cached: (2) is a step function of the threshold that changes only at the image's own detection
scores, so it is stored as steps. The AP of any multiset of images (a bootstrap sample) is then the evaluator's
arithmetic on weighted sums of the cached values, and equals kitti/evaluate.py's kitti_ap40 on the same images
(checked in evaluation/selfcheck.py and at every test opening).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kitti.common import DIFFICULTIES  # noqa: E402
from kitti.evaluate import KITTI_CLASSES  # noqa: E402
from third_party.mmdet3d_kitti_eval import eval as mm  # noqa: E402

STRICT_IOU = {"Car": 0.7, "Pedestrian": 0.5, "Cyclist": 0.5}  # the evaluator's strict 2D overlaps
N_SAMPLE_PTS = 41


class CachedKitti:
    def __init__(self, gts, dts, classes=KITTI_CLASSES):
        self.n = len(gts)
        self.classes = list(classes)
        overlaps, _, _, _ = mm.calculate_iou_partly(dts, gts, 0, num_parts=min(200, self.n))
        self.cache = {}
        for c in self.classes:
            ci = list(KITTI_CLASSES).index(c)
            for di, diff in enumerate(DIFFICULTIES):
                gdl, ddl, igs, idets, dcs, _, _ = mm._prepare_data(gts, dts, ci, di)
                ov = STRICT_IOU[c]
                steps_s, steps_d, steps_img, tps, tps_img = [], [], [], [], []
                base = np.zeros((self.n, 3), np.int64)
                for i in range(self.n):
                    def stats(thresh, compute_fp=True):
                        return mm.compute_statistics_jit(overlaps[i], gdl[i], ddl[i], igs[i], idets[i], dcs[i], 0,
                                                         min_overlap=ov, thresh=thresh, compute_fp=compute_fp)
                    thr = stats(0.0, compute_fp=False)[4]
                    tps.append(np.asarray(thr, np.float64))
                    tps_img.append(np.full(len(thr), i, np.int64))
                    tp, fp, fn, _, _ = stats(np.inf)
                    prev = np.array([tp, fp, fn], np.int64)
                    base[i] = prev
                    scores = ddl[i][:, -1][idets[i] != -1] if len(ddl[i]) else np.zeros(0)
                    for s in np.unique(scores)[::-1]:
                        tp, fp, fn, _, _ = stats(float(s))
                        cur = np.array([tp, fp, fn], np.int64)
                        if (cur != prev).any():
                            steps_s.append(float(s))
                            steps_d.append(cur - prev)
                            steps_img.append(i)
                        prev = cur
                order = np.argsort(-np.asarray(steps_s, np.float64), kind="stable")
                self.cache[(c, diff)] = {
                    "neg_s": -np.asarray(steps_s, np.float64)[order],
                    "d": np.asarray(steps_d, np.int64).reshape(-1, 3)[order],
                    "img": np.asarray(steps_img, np.int64)[order],
                    "base": base,
                    "tp_scores": np.concatenate(tps) if tps else np.zeros(0),
                    "tp_img": np.concatenate(tps_img) if tps_img else np.zeros(0, np.int64),
                }

    def ap40(self, weights=None):
        """{class: {difficulty: AP40}} for the multiset of images in which image i appears weights[i] times."""
        w = np.ones(self.n, np.int64) if weights is None else np.asarray(weights, np.int64)
        out = {}
        for c in self.classes:
            out[c] = {}
            for diff in DIFFICULTIES:
                k = self.cache[(c, diff)]
                base = (k["base"] * w[:, None]).sum(0)
                num_gt = int(base[2])  # every valid object is a false negative when no box is kept
                scores = np.repeat(k["tp_scores"], w[k["tp_img"]])
                if num_gt == 0 or not len(scores):
                    out[c][diff] = 0.0
                    continue
                thresholds = mm.get_thresholds(scores.copy(), num_gt)
                cum = np.cumsum(k["d"] * w[k["img"]][:, None], axis=0)
                prec = np.zeros(N_SAMPLE_PTS)
                for j, t in enumerate(thresholds):
                    m = np.searchsorted(k["neg_s"], -t, side="right")
                    tp, fp, _ = base + (cum[m - 1] if m else 0)
                    prec[j] = tp / (tp + fp)
                for j in range(len(thresholds)):
                    prec[j] = prec[j:].max()
                out[c][diff] = float(prec[1:].sum() / 40 * 100)
        return out
