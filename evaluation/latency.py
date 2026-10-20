"""Latency (PREREG.md section 6, amendment 2, PREREG_AMENDMENT_4.md section 6).

One session: each model in turn, back to back, on an NVIDIA L4. Batch 1, confidence 0.25. Each model sees the
same 550 validation images, decoded into memory beforehand (decoding is excluded); the first 50 frames are
discarded and the next 500 timed. Ultralytics' own per-image stage times (preprocess, inference, postprocess,
with CUDA synchronised) are summarised as mean, median and 95th percentile.

Usage: python evaluation/latency.py --data-root /root/kitti --runs-mnt /runs --out <json> --session 1|2
"""
import argparse
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from evaluation.models import now, sha256, write_json  # noqa: E402
from kitti.yolo_data import image_paths, split_ids  # noqa: E402

WARMUP, TIMED = 50, 500
MODELS = [  # label, weights (relative to --runs-mnt for trained runs), input size, extra predict arguments
    ("YOLO11n, 640", "E1-Y11N-640-DD-S0", 640, {}),
    ("YOLO11s, 640", "E1-Y11S-640-DD-S0", 640, {}),
    ("YOLO11m, 640", "E1-Y11M-640-DD-S0", 640, {}),
    ("YOLO11s, 960", "E2-Y11S-960-DD-S0", 960, {}),
    ("YOLO26s (stock), 640, nms=False (NMS-free head)", "yolo26s.pt", 640, {"nms": False}),
    ("YOLO26s (stock), 640, nms=None (NMS head)", "yolo26s.pt", 640, {"nms": None}),
]


def stats(v):
    v = np.asarray(v, float)
    return {"mean": float(v.mean()), "median": float(np.median(v)), "p95": float(np.percentile(v, 95))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--runs-mnt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--session", type=int, required=True)
    a = ap.parse_args()
    import cv2
    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.data.augment import LetterBox

    ids = split_ids("drive_disjoint", "val")[:WARMUP + TIMED]
    frames = [cv2.imread(p) for p in image_paths(a.data_root, ids)]
    if any(f is None for f in frames):
        sys.exit("an image could not be read")
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    env = {"session": a.session, "time": now(), "gpu": torch.cuda.get_device_name(0), "nvidia_smi": smi,
           "torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
           "ultralytics": ultralytics.__version__, "python": platform.python_version(),
           "frames": {"warmup": WARMUP, "timed": TIMED, "source": "drive_disjoint val, first 550 ids",
                      "image_shapes": sorted({f"{f.shape[1]}x{f.shape[0]}" for f in frames})}}
    results = {}
    for label, ref, imgsz, extra in MODELS:
        w = Path(a.runs_mnt) / ref / ref / "weights" / "best.pt" if ref.startswith("E") else Path(ref)
        model = YOLO(str(w))
        stride = int(max(model.model.stride)) if hasattr(model.model, "stride") else 32
        shape = LetterBox(new_shape=(imgsz, imgsz), auto=True, stride=stride)(image=frames[0]).shape[:2]
        times = {"preprocess": [], "inference": [], "postprocess": []}
        for k, img in enumerate(frames):
            r = model.predict(img, imgsz=imgsz, conf=0.25, batch=1, verbose=False, **extra)[0]
            if k >= WARMUP:
                for s in times:
                    times[s].append(r.speed[s])
        total = np.sum([times[s] for s in times], axis=0)
        results[label] = {"weights": str(w), "weights_sha256": sha256(w), "imgsz": imgsz, "predict_args": extra,
                          "input_tensor_hw": list(shape), "ms": {**{s: stats(v) for s, v in times.items()},
                                                                 "total": stats(total)}}
        print(f"{label}: input {shape[1]}x{shape[0]}; inference mean {results[label]['ms']['inference']['mean']:.2f} ms, "
              f"total mean {results[label]['ms']['total']['mean']:.2f} ms, p95 {results[label]['ms']['total']['p95']:.2f} ms",
              flush=True)
        del model
        torch.cuda.empty_cache()
    write_json(a.out, {"environment": env, "models": results})
    print(f"written: {a.out}")


if __name__ == "__main__":
    main()
