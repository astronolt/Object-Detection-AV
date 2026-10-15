"""Day-1 gate: the evaluator scores ground truth as 100 AP, and two broken inputs score below 100.

Runs on the development pool only (drive-disjoint train + val); test labels are not touched.
Usage: python -I scripts/check_evaluator.py <label_2 dir> <splits dir> <out json>
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kitti.common import read_id_list, read_label  # noqa: E402
from kitti.evaluate import dt_anno, gt_anno, kitti_ap40  # noqa: E402


def main(label_dir, splits_dir, out_json):
    ids = sorted(read_id_list(Path(splits_dir) / "drive_disjoint/train.txt")
                 + read_id_list(Path(splits_dir) / "drive_disjoint/val.txt"))
    labels = [read_label(Path(label_dir) / f"{i}.txt") for i in ids]
    gts = [gt_anno(objs) for objs in labels]

    def as_dets(objs, shift=0.0):
        return [(o["name"], o["bbox"][0] + shift * (o["bbox"][2] - o["bbox"][0]), o["bbox"][1],
                 o["bbox"][2] + shift * (o["bbox"][2] - o["bbox"][0]), o["bbox"][3], 1.0)
                for o in objs if o["name"] != "DontCare"]

    perfect = kitti_ap40(gts, [dt_anno(as_dets(o)) for o in labels])

    # Control 1: every 10th detection (global order) removed -> recall < 1 -> AP < 100.
    k = 0
    dropped = []
    for objs in labels:
        keep = []
        for d in as_dets(objs):
            if k % 10:
                keep.append(d)
            k += 1
        dropped.append(dt_anno(keep))
    drop10 = kitti_ap40(gts, dropped)

    # Control 2: boxes shifted right by 25% of their width -> IoU 0.6:
    # below Car's 0.7 (AP 0), above Pedestrian/Cyclist's 0.5 (AP 100).
    shift = kitti_ap40(gts, [dt_anno(as_dets(o, 0.25)) for o in labels])

    checks = {
        "ground_truth_scores_100": all(abs(v - 100) < 1e-6 for c in perfect.values() for v in c.values()),
        "drop10_below_100": all(v < 100 for c in drop10.values() for v in c.values()),
        # Tolerances allow for shifted boxes matching a neighbouring object in a crowd. Attempt 1 used
        # > 99 for Pedestrian/Cyclist and got Pedestrian 97.8-98.7 (results/day1/evaluator_check_attempt1.json);
        # a 0.7 threshold would give ~0, so > 90 still separates 0.5 from 0.7.
        "shift25_car_near_zero": all(v < 1.0 for v in shift["Car"].values()),
        "shift25_ped_cyc_above_90": all(v > 90.0 for c in ("Pedestrian", "Cyclist")
                                        for v in shift[c].values()),
    }
    out = {"n_images": len(ids), "pool": "drive_disjoint train+val", "checks": checks,
           "ground_truth": perfect, "drop_every_10th": drop10, "shift_25pct_width": shift}
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    if not all(checks.values()):
        sys.exit("EVALUATOR GATE FAILED")
    print("EVALUATOR GATE PASSED")


if __name__ == "__main__":
    main(*sys.argv[1:4])
