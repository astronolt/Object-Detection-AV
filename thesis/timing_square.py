"""Supplementary timing on square inputs. This is not part of the pre-registered study.

The pre-registered timing (evaluation/latency.py) fed each KITTI frame to a detector on its own, so the software
padded it to the smallest shape the network accepts: 640 x 224 or 960 x 320 pixels. The test predictions were scored
on square inputs, 640 x 640 or 960 x 960. This script times the four frozen YOLO11 models both ways in one session,
so that accuracy and speed can be paired like for like.

Everything else follows the pre-registered protocol: one NVIDIA L4, PyTorch at 32-bit precision, batch size 1,
confidence 0.25, the first 550 drive-disjoint validation images decoded into memory beforehand, 50 discarded as
warm-up and 500 timed, using Ultralytics' own per-stage times (with CUDA synchronised). To even out drift within the
session, each model is timed in the order rectangular, square, square, rectangular, and each configuration's two
passes (1,000 timed images) are pooled. The shape of the tensor that reaches the network is recorded, to confirm
that each configuration really was rectangular or square.

It reads only the KITTI images and the frozen weights, and writes timing_square.json to the volume
kitti-msc-thesis-figs. It changes nothing on the study's volumes and never reads a test image.

From the project folder in Git Bash (about 15 minutes of L4 time, well under US$1):

  python -m modal run thesis/timing_square.py

Then copy the table it prints at the end into the chat. To check it on a computer without a GPU (a few images,
CPU, any weights):

  python thesis/timing_square.py --local IMAGE_DIR WEIGHTS_N WEIGHTS_S WEIGHTS_M WEIGHTS_960
"""
import json
import platform
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
# On the computer this file sits in <project>/thesis; inside the Modal container the project folders it needs are
# copied to /root/repo.
REPO = HERE.parent if (HERE.parent / "kitti").is_dir() else Path("/root/repo")

WARMUP, TIMED = 50, 500
ORDER = ("rectangular", "square", "square", "rectangular")  # ABBA, to balance drift within the session
MODELS = [  # label, run id, input size
    ("YOLO11n, 640", "E1-Y11N-640-DD-S0", 640),
    ("YOLO11s, 640", "E1-Y11S-640-DD-S0", 640),
    ("YOLO11m, 640", "E1-Y11M-640-DD-S0", 640),
    ("YOLO11s, 960", "E2-Y11S-960-DD-S0", 960),
]


def stats(v):
    import numpy as np
    v = np.asarray(v, float)
    return {"mean": float(v.mean()), "median": float(np.median(v)), "p95": float(np.percentile(v, 95)), "n": int(v.size)}


def time_models(frames, weights, device=None, warmup=WARMUP, timed=TIMED):
    """Time each model on the frames, rectangular and square, in ABBA order. weights: {label: path}."""
    import numpy as np
    import torch
    from ultralytics import YOLO

    results = {}
    for label, _, imgsz in MODELS:
        model = YOLO(str(weights[label]))
        seen = []
        hook = model.model.register_forward_pre_hook(lambda m, inp: seen.append(tuple(inp[0].shape)))
        pooled = {c: {"preprocess": [], "inference": [], "postprocess": []} for c in ("rectangular", "square")}
        shapes = {}
        for config in ORDER:
            rect = config == "rectangular"
            for k, img in enumerate(frames[:warmup + timed]):
                kw = {"device": device} if device else {}
                r = model.predict(img, imgsz=imgsz, conf=0.25, batch=1, rect=rect, verbose=False, **kw)[0]
                if k >= warmup:
                    for s in pooled[config]:
                        pooled[config][s].append(r.speed[s])
            shapes[config] = list(seen[-1][2:])  # height, width of the tensor that reached the network
        hook.remove()
        entry = {"weights": str(weights[label]), "imgsz": imgsz}
        for config, times in pooled.items():
            total = np.sum([times[s] for s in times], axis=0)
            entry[config] = {"input_hw": shapes[config],
                             "ms": {**{s: stats(v) for s, v in times.items()}, "total": stats(total)}}
        results[label] = entry
        r, q = entry["rectangular"]["ms"]["total"], entry["square"]["ms"]["total"]
        print(f"{label}: rectangular {shapes['rectangular']} median {r['median']:.2f} ms; "
              f"square {shapes['square']} median {q['median']:.2f} ms", flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results


def table(results):
    """The summary to paste into the chat: medians and 95th percentiles of the total time per image."""
    lines = ["| Model | Rectangular input | Median ms | p95 ms | Square input | Median ms | p95 ms | Square minus rectangular |",
             "|:--|:--|--:|--:|:--|--:|--:|--:|"]
    for label, e in results.items():
        r, q = e["rectangular"], e["square"]
        lines.append(f"| {label} | {r['input_hw'][1]} x {r['input_hw'][0]} | {r['ms']['total']['median']:.2f} | "
                     f"{r['ms']['total']['p95']:.2f} | {q['input_hw'][1]} x {q['input_hw'][0]} | "
                     f"{q['ms']['total']['median']:.2f} | {q['ms']['total']['p95']:.2f} | "
                     f"{q['ms']['total']['median'] - r['ms']['total']['median']:+.2f} |")
    lines.append("")
    lines.append("Stage medians, square (pre-processing / forward pass / post-processing, ms):")
    for label, e in results.items():
        m = e["square"]["ms"]
        lines.append(f"  {label}: {m['preprocess']['median']:.2f} / {m['inference']['median']:.2f} / "
                     f"{m['postprocess']['median']:.2f}")
    return "\n".join(lines)


try:
    import modal
except ImportError:  # local use only
    modal = None

if modal is not None:
    PINS = [  # the study's versions (cloud/kitti_modal.py), so the weights load and run as they were timed
        "Jinja2==3.1.6", "MarkupSafe==3.0.3", "PyYAML==6.0.3", "anyio==4.15.1", "certifi==2026.7.22",
        "charset-normalizer==3.5.2", "cloudpickle==3.1.2", "contourpy==1.4.0", "cuda-bindings==13.4.3",
        "cuda-pathfinder==1.8.2", "cuda-toolkit==13.0.3.0", "cycler==0.12.1", "filelock==4.0.9",
        "fonttools==4.66.1", "fsspec==2026.9.0", "h11==0.16.0", "httpcore==1.0.9", "httpx==0.28.1", "idna==3.20",
        "kiwisolver==1.5.1", "matplotlib==3.11.2", "mpmath==1.3.0", "networkx==3.7", "numpy==2.5.3",
        "nvidia-cublas==13.1.1.3", "nvidia-cuda-cupti==13.0.85", "nvidia-cuda-nvrtc==13.0.88",
        "nvidia-cuda-runtime==13.0.96", "nvidia-cudnn-cu13==9.24.0.43", "nvidia-cufft==12.0.0.61",
        "nvidia-cufile==1.15.1.6", "nvidia-curand==10.4.0.35", "nvidia-cusolver==12.0.4.66",
        "nvidia-cusparse==12.6.3.3", "nvidia-cusparselt-cu13==0.8.1", "nvidia-ml-py==13.615.71",
        "nvidia-nccl-cu13==2.30.7", "nvidia-nvjitlink==13.4.92", "nvidia-nvshmem-cu13==3.4.5",
        "nvidia-nvtx==13.0.85", "opencv-python==4.14.0.94", "packaging==26.3", "pillow==12.3.0",
        "polars-runtime-32==1.44.2", "polars==1.44.2", "psutil==7.2.2", "pyparsing==3.3.3",
        "python-dateutil==2.9.0.post0", "requests==2.34.2", "setuptools==84.0.0", "six==1.17.0", "sympy==1.14.0",
        "torch==2.14.1", "torchvision==0.29.1", "triton==3.8.0", "typing_extensions==4.16.0",
        "ultralytics-platform==0.1.75", "ultralytics-thop==2.2.2", "ultralytics==8.4.171", "urllib3==2.8.0",
    ]
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("libgl1", "libglib2.0-0")
        .pip_install(*PINS)
        .env({"PYTHONUNBUFFERED": "1"})
        .add_local_dir(REPO / "kitti", "/root/repo/kitti", ignore=["__pycache__"])
        .add_local_dir(REPO / "splits", "/root/repo/splits")
        .add_local_file(Path(__file__), "/root/repo/thesis/timing_square.py")
    )
    app = modal.App("kitti-timing-square")
    VOLS = {"/vol": modal.Volume.from_name("kitti-msc").read_only(),
            "/out": modal.Volume.from_name("kitti-msc-thesis-figs", create_if_missing=True)}
    for _, rid, _ in MODELS:
        VOLS[f"/runs/{rid}"] = modal.Volume.from_name(f"kitti-msc-{rid.lower()}").read_only()

    @app.function(image=image, gpu="L4", cpu=4.0, memory=20480, timeout=3600, volumes=VOLS)
    def timing():
        import subprocess
        import zipfile

        import cv2
        import torch
        import ultralytics

        sys.path.insert(0, "/root/repo")
        from kitti.yolo_data import split_ids

        name = torch.cuda.get_device_name(0)
        if "L4" not in name:
            raise RuntimeError(f"expected an NVIDIA L4, got {name}")
        weights = {label: Path(f"/runs/{rid}/{rid}/weights/best.pt") for label, rid, _ in MODELS}
        zpath = Path("/vol/raw/data_object_image_2.zip")
        missing = [str(p) for p in [zpath, *weights.values()] if not p.exists()]
        if missing:  # stop at once if an input is not where it should be
            raise RuntimeError("Inputs not found: " + ", ".join(missing))

        ids = split_ids("drive_disjoint", "val")[:WARMUP + TIMED]
        local = Path("/root/val_images")
        local.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zpath) as z:  # only the 550 images needed, not the whole archive
            for i in ids:
                (local / f"{i}.png").write_bytes(z.read(f"training/image_2/{i}.png"))
        frames = [cv2.imread(str(local / f"{i}.png")) for i in ids]
        if any(f is None for f in frames):
            raise RuntimeError("an image could not be read")

        smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True).stdout.strip()
        env = {"note": "Supplementary analysis after the test opening; not part of the pre-registered study.",
               "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "gpu": name, "nvidia_smi": smi,
               "torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
               "ultralytics": ultralytics.__version__, "python": platform.python_version(),
               "frames": {"warmup": WARMUP, "timed_per_pass": TIMED, "passes": list(ORDER),
                          "source": "drive_disjoint val, first 550 ids"}}
        results = time_models(frames, weights)
        out = {"environment": env, "models": results, "table": table(results)}
        Path("/out/timing_square.json").write_text(json.dumps(out, indent=1))
        VOLS["/out"].commit()
        return out["table"]

    @app.local_entrypoint()
    def main():
        print("\n" + timing.remote())
        print("\nSaved as timing_square.json on the volume kitti-msc-thesis-figs. Copy the table above into the chat.")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--local":
    import cv2
    img_dir, *w = sys.argv[2:7]
    paths = sorted(Path(img_dir).glob("*.png")) + sorted(Path(img_dir).glob("*.jpg"))
    frames = [cv2.imread(str(p)) for p in paths][:6]
    res = time_models(frames, {label: w[k] for k, (label, _, _) in enumerate(MODELS)}, device="cpu", warmup=1, timed=3)
    print(table(res))
