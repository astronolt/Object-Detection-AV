"""The models scored at the freeze and at the single test opening (PREREG_AMENDMENT_4.md section 1)."""
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from configs.recipe import E0_IMGSZ, E0_MODELS, HEADLINE, RUNS  # noqa: E402
from kitti.common import YOLO_NAMES  # noqa: E402

TEST_LIST_SHA256 = "cd4cd7cb2f497211f21056433e95293504d68c215f63cbedbf8cb31e09ab4d7e"  # PREREG.md section 2
E0_CLASSES = ["Car", "Pedestrian", "LargeVehicle"]  # COCO has no cyclist (PREREG.md section 5.3)
SEEDS = ["E1-Y11S-640-DD-S0", "E1-Y11S-640-DD-S1", "E1-Y11S-640-DD-S2"]


def model_specs(runs_mnt):
    """key -> weights, input size, validation split and the classes the model can predict, in a fixed order."""
    specs = {}
    for rid, s in RUNS.items():
        if s.get("kind") == "e0":
            continue
        specs[rid] = {"kind": "trained", "weights": str(Path(runs_mnt) / rid / rid / "weights" / "best.pt"),
                      "imgsz": s["imgsz"], "val_split": s["split"], "coco_e0": False, "classes": list(YOLO_NAMES)}
    for w in E0_MODELS:
        key = "E0-" + Path(w).stem.upper().replace("YOLO", "Y")  # E0-Y11N, E0-Y11S, E0-Y11M
        specs[key] = {"kind": "e0", "weights": w, "imgsz": E0_IMGSZ, "val_split": "drive_disjoint",
                      "coco_e0": True, "classes": list(E0_CLASSES)}
    return specs


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    tmp.replace(path)


def ensure_weights(spec):
    """Trained runs use best.pt from their volume; E0 uses the stock COCO weights, downloaded by Ultralytics."""
    w = Path(spec["weights"])
    if spec["kind"] == "e0" and not w.exists():
        from ultralytics import YOLO
        YOLO(spec["weights"])  # downloads into the working directory
    if not w.exists():
        raise FileNotFoundError(f"weights not found: {w}")
    return w


__all__ = ["HEADLINE", "SEEDS", "TEST_LIST_SHA256", "E0_CLASSES", "model_specs", "sha256", "now", "write_json",
           "ensure_weights"]
