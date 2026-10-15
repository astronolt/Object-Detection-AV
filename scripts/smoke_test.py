"""Smoke test: the full recipe for a few epochs, then queue-time projections (plan doc: s 3 epochs; n, m, s@960 1 each).

Seconds per epoch come from wall time, which includes start-up and the first disk-cache build, so the
projection errs long. Writes runs_root/smoke/smoke_projection.json.

Usage: python scripts/smoke_test.py --data-root /content/kitti --runs-root /content/drive/MyDrive/msc/runs
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configs.recipe import COMMON, QUEUES, RUNS  # noqa: E402
from kitti.runlog import log_env  # noqa: E402
from kitti.yolo_data import data_yaml  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data-root", required=True)
ap.add_argument("--runs-root", required=True)
a = ap.parse_args()

from ultralytics import YOLO  # noqa: E402

smoke_root = Path(a.runs_root) / "smoke"
data = data_yaml(a.data_root, "drive_disjoint")
plan = [("yolo11s.pt", 640, 3), ("yolo11n.pt", 640, 1), ("yolo11m.pt", 640, 1), ("yolo11s.pt", 960, 1)]
sec_per_epoch = {}
for model, imgsz, ep in plan:
    name = f"smoke-{Path(model).stem}-{imgsz}"
    log_env(smoke_root / name, {"smoke": True, "model": model, "imgsz": imgsz, "epochs": ep})
    t = time.time()
    YOLO(model).train(data=data, epochs=ep, imgsz=imgsz, seed=0, project=str(smoke_root), name=name,
                      **COMMON)
    sec_per_epoch[f"{model}@{imgsz}"] = (time.time() - t) / ep
    print(f"{name}: {sec_per_epoch[f'{model}@{imgsz}']:.0f} s/epoch", flush=True)


def hours(run_id, epochs):
    s = RUNS[run_id]
    if s.get("kind") == "e0":
        return 0.15
    return sec_per_epoch[f"{s['model']}@{s['imgsz']}"] * epochs / 3600


proj = {"sec_per_epoch": sec_per_epoch,
        "queue_hours": {f"{q}@{e}": round(sum(hours(r, e) for r in QUEUES[q]), 2)
                        for q, e in (("A", 100), ("B", 100), ("single", 50), ("single", 100))}}
(smoke_root / "smoke_projection.json").write_text(json.dumps(proj, indent=2) + "\n")
print(json.dumps(proj, indent=2))
