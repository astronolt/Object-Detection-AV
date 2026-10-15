"""Run one queue from configs/recipe.py with the epochs fixed in EPOCH_DECISION.json (PREREG.md section 3).

- Skips runs marked COMPLETED; resumes from last.pt after a crash (logged).
- Stops if runs_root/HALT exists. Writes HALT if the headline's Car val mAP@0.5 < 0.60.
- E0 item: COCO n/s/m predictions on validation, saved for scoring (no training).

Usage: python scripts/train_queue.py --queue A|B|single --data-root /content/kitti
           --runs-root /content/drive/MyDrive/msc/runs
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configs.recipe import (COMMON, E0_IMGSZ, E0_MODELS, HALT_MIN_CAR_VAL_MAP50, HEADLINE,  # noqa: E402
                            RUNS)
from kitti.runlog import log_env, log_line, now  # noqa: E402
from kitti.yolo_data import data_yaml, split_ids  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--queue", required=True)
ap.add_argument("--data-root", required=True)
ap.add_argument("--runs-root", required=True)
a = ap.parse_args()

runs_root = Path(a.runs_root)
decision_file = runs_root / "EPOCH_DECISION.json"
if not decision_file.exists():
    sys.exit("No EPOCH_DECISION.json: run scripts/decide_epochs.py after the smoke test.")
decision = json.loads(decision_file.read_text())
if a.queue not in decision["queues"]:
    sys.exit(f"Queue {a.queue} is not part of the decision {list(decision['queues'])}.")
EPOCHS = decision["epochs"]
QLOG = runs_root / f"queue_{a.queue}.log"


def halted():
    return (runs_root / "HALT").exists()


def run_e0(run_id):
    from predict import predict
    out_dir = runs_root / run_id
    log_env(out_dir, {"run_id": run_id, "kind": "e0", "imgsz": E0_IMGSZ, "split": "drive_disjoint val"})
    ids = split_ids("drive_disjoint", "val")
    for w in E0_MODELS:
        out = out_dir / f"{Path(w).stem}_val.jsonl"
        if out.exists():
            continue
        predict(w, ids, a.data_root, E0_IMGSZ, out, coco_e0=True)
        log_line(QLOG, f"{run_id}: saved {out.name}")


def run_train(run_id, spec):
    import torch
    from ultralytics import YOLO
    rundir = runs_root / run_id
    last = rundir / "weights" / "last.pt"
    data = data_yaml(a.data_root, spec["split"])
    rec = {"run_id": run_id, **spec, "epochs": EPOCHS, "common": COMMON, "data": data}
    finished = last.exists() and torch.load(last, map_location="cpu", weights_only=False).get("epoch", 0) == -1
    if finished:
        log_line(QLOG, f"{run_id}: training already finished; validating best.pt")
    elif last.exists():
        log_env(rundir, {**rec, "resume": True})
        log_line(QLOG, f"{run_id}: RESUME from {last}")
        YOLO(str(last)).train(resume=True)
    else:
        log_env(rundir, {**rec, "resume": False})
        log_line(QLOG, f"{run_id}: START {spec['model']} imgsz={spec['imgsz']} split={spec['split']} "
                       f"seed={spec['seed']} epochs={EPOCHS}")
        YOLO(spec["model"]).train(data=data, epochs=EPOCHS, imgsz=spec["imgsz"], seed=spec["seed"],
                                  project=str(runs_root), name=run_id, **COMMON)

    m = YOLO(str(rundir / "weights" / "best.pt")).val(
        data=data, split="val", imgsz=spec["imgsz"], batch=16, conf=0.001, iou=0.7,
        plots=False, project=str(rundir), name="val_best", exist_ok=True, verbose=False)
    cls_idx = list(m.box.ap_class_index)
    per_class = {m.names[c]: float(m.box.ap50[k]) for k, c in enumerate(cls_idx)}
    summ = {"time": now(), "val_map50": float(m.box.map50), "val_map50_95": float(m.box.map),
            "val_ap50_per_class": per_class}
    (rundir / "val_summary.json").write_text(json.dumps(summ, indent=2) + "\n")
    log_line(QLOG, f"{run_id}: val mAP50={summ['val_map50']:.4f} per-class AP50={per_class}")
    if run_id == HEADLINE and per_class.get("Car", 0.0) < HALT_MIN_CAR_VAL_MAP50:
        (runs_root / "HALT").write_text(f"{now()} headline Car val mAP50 {per_class.get('Car')} < "
                                        f"{HALT_MIN_CAR_VAL_MAP50}: check the label converter\n")
        log_line(QLOG, "HALT written: headline Car val mAP@0.5 below threshold")


sys.path.insert(0, str(Path(__file__).resolve().parent))
log_line(QLOG, f"QUEUE {a.queue} START epochs={EPOCHS} runs={decision['queues'][a.queue]}")
for run_id in decision["queues"][a.queue]:
    if halted():
        log_line(QLOG, f"HALT file present; stopping before {run_id}")
        break
    done = runs_root / run_id / "COMPLETED"
    if done.exists():
        log_line(QLOG, f"{run_id}: COMPLETED earlier; skipping")
        continue
    spec = RUNS[run_id]
    if spec.get("kind") == "e0":
        run_e0(run_id)
    else:
        run_train(run_id, {k: v for k, v in spec.items() if k != "l4_h100"})
    if halted():
        break
    done.write_text(now() + "\n")
    log_line(QLOG, f"{run_id}: COMPLETED")
log_line(QLOG, f"QUEUE {a.queue} END")
