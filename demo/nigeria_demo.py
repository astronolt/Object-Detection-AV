"""Demonstration on Nigerian roads: run a trained detector over road videos and save annotated copies.

This is not part of the pre-registered study. It runs one finished run's best.pt on footage the study never
used, at confidence 0.25 (the setting of the latency protocol), to show what the detector finds on Nigerian
roads. No accuracy is measured: the footage has no labels.

From the project folder in Git Bash:

  python -m modal volume put -f kitti-msc "C:/Users/pi/Downloads/lagos-road.mp4" nigeria-demo-in/
  python -m modal run demo/nigeria_demo.py --video nigeria-demo-in/lagos-road.mp4 --credit "Video: Shedrack Salami, Pexels"
  python -m modal volume get --force kitti-msc nigeria-demo "G:/My Drive/msc"

--video is a file or a folder on the kitti-msc volume (every .mp4, .mov, .avi or .mkv in a folder is processed).
--run picks the model (default: YOLO11s at 960 pixels, the best of the runs on validation); --conf the
confidence threshold; --credit a line written on every frame, for the footage's creator.

Each video gives, in nigeria-demo on the volume: <name>_<run>.mp4 (annotated, H.264), three sample frames as JPG,
<name>_<run>.json (counts by class and timing) and <name>_<run>_frames.csv (detections per frame).
"""
import csv
import json
import shutil
import subprocess
import time
from pathlib import Path

import modal

# The trained runs and the input size each was trained at (configs/recipe.py).
RUNS = {
    "E2-Y11S-960-DD-S0": 960,
    "E1-Y11S-640-DD-S0": 640,
    "E1-Y11S-640-DD-S1": 640,
    "E1-Y11S-640-DD-S2": 640,
    "E1-Y11N-640-DD-S0": 640,
    "E1-Y11M-640-DD-S0": 640,
}
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv"}

# The same package versions as the training image (cloud/kitti_modal.py), so the model loads as it was saved.
PINS = [
    "Jinja2==3.1.6", "MarkupSafe==3.0.3", "PyYAML==6.0.3", "anyio==4.15.1", "certifi==2026.7.22",
    "charset-normalizer==3.5.2", "cloudpickle==3.1.2", "contourpy==1.4.0", "cuda-bindings==13.4.3",
    "cuda-pathfinder==1.8.2", "cuda-toolkit==13.0.3.0", "cycler==0.12.1", "filelock==4.0.9", "fonttools==4.66.1",
    "fsspec==2026.9.0", "h11==0.16.0", "httpcore==1.0.9", "httpx==0.28.1", "idna==3.20", "kiwisolver==1.5.1",
    "matplotlib==3.11.2", "mpmath==1.3.0", "networkx==3.7", "numpy==2.5.3", "nvidia-cublas==13.1.1.3",
    "nvidia-cuda-cupti==13.0.85", "nvidia-cuda-nvrtc==13.0.88", "nvidia-cuda-runtime==13.0.96",
    "nvidia-cudnn-cu13==9.24.0.43", "nvidia-cufft==12.0.0.61", "nvidia-cufile==1.15.1.6",
    "nvidia-curand==10.4.0.35", "nvidia-cusolver==12.0.4.66", "nvidia-cusparse==12.6.3.3",
    "nvidia-cusparselt-cu13==0.8.1", "nvidia-ml-py==13.615.71", "nvidia-nccl-cu13==2.30.7",
    "nvidia-nvjitlink==13.4.92", "nvidia-nvshmem-cu13==3.4.5", "nvidia-nvtx==13.0.85", "opencv-python==4.14.0.94",
    "packaging==26.3", "pillow==12.3.0", "polars-runtime-32==1.44.2", "polars==1.44.2", "psutil==7.2.2",
    "pyparsing==3.3.3", "python-dateutil==2.9.0.post0", "requests==2.34.2", "setuptools==84.0.0", "six==1.17.0",
    "sympy==1.14.0", "torch==2.14.1", "torchvision==0.29.1", "triton==3.8.0", "typing_extensions==4.16.0",
    "ultralytics-platform==0.1.75", "ultralytics-thop==2.2.2", "ultralytics==8.4.171", "urllib3==2.8.0",
]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0", "ffmpeg")
    .pip_install(*PINS)
    .env({"PYTHONUNBUFFERED": "1"})
)
app = modal.App("kitti-demo")
vol = modal.Volume.from_name("kitti-msc")
VOLUMES = {"/vol": vol, **{f"/runs/{rid}": modal.Volume.from_name(f"kitti-msc-{rid.lower()}").read_only()
                           for rid in RUNS}}


def put_text(img, text, org, scale):
    """White text on a dark box, readable on any background."""
    import cv2
    font, thick = cv2.FONT_HERSHEY_SIMPLEX, max(1, round(scale * 2))
    (w, h), base = cv2.getTextSize(text, font, scale, thick)
    x, y = org
    cv2.rectangle(img, (x - 6, y - h - 8), (x + w + 6, y + base + 6), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), font, scale, (255, 255, 255), thick, cv2.LINE_AA)


def annotate_video(src, weights, imgsz, conf, out_dir, stem, label, credit="", max_width=1920):
    """Run the detector over every frame of src; write the annotated video, sample frames and counts."""
    import cv2
    from ultralytics import YOLO

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    scale = min(1.0, max_width / w)
    ow, oh = 2 * round(w * scale / 2), 2 * round(h * scale / 2)  # even sizes, as H.264 requires
    text_scale = max(0.5, oh / 1080 * 0.9)

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / f"{stem}.avi"
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"MJPG"), fps, (ow, oh))
    model = YOLO(str(weights))
    samples = {max(n_frames // 4, 0), n_frames // 2, (3 * n_frames) // 4}
    totals, conf_sums, rows, saved = {}, {}, [], []
    t0, k = time.time(), 0
    for r in model.predict(source=str(src), stream=True, imgsz=imgsz, conf=conf, verbose=False):
        frame = r.plot()  # the frame with boxes and class labels drawn
        if scale < 1:
            frame = cv2.resize(frame, (ow, oh), interpolation=cv2.INTER_AREA)
        put_text(frame, label, (16, int(40 * text_scale) + 8), text_scale)
        if credit:
            put_text(frame, credit, (16, oh - 16), text_scale * 0.8)
        writer.write(frame)
        counts = {}
        for c, s in zip(r.boxes.cls.int().tolist(), r.boxes.conf.tolist()):
            name = r.names[c]
            counts[name] = counts.get(name, 0) + 1
            conf_sums[name] = conf_sums.get(name, 0.0) + s
        for name, n in counts.items():
            totals[name] = totals.get(name, 0) + n
        rows.append(counts)
        if k in samples:
            p = out_dir / f"{stem}_frame{k:05d}.jpg"
            cv2.imwrite(str(p), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            saved.append(p.name)
        k += 1
    writer.release()
    seconds = time.time() - t0
    if k == 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"no frames read from {src}")

    video_out = out_dir / f"{stem}.mp4"
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-crf", "23", "-movflags", "+faststart", str(video_out)], check=True)
        tmp.unlink()
    else:
        video_out = tmp  # no ffmpeg: keep the MJPG AVI

    names = sorted({n for row in rows for n in row} | set(totals))
    with open(out_dir / f"{stem}_frames.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["frame", *names])
        for i, row in enumerate(rows):
            wr.writerow([i, *[row.get(n, 0) for n in names]])
    summary = {
        "video": str(src.name), "model": label, "weights": str(weights), "imgsz": imgsz, "conf": conf,
        "frames": k, "fps_source": round(fps, 2), "size_source": [w, h], "size_output": [ow, oh],
        "seconds": round(seconds, 1), "frames_per_second_processed": round(k / seconds, 1) if seconds else None,
        "detections": {n: totals.get(n, 0) for n in names},
        "detections_per_frame": {n: round(totals.get(n, 0) / k, 2) for n in names},
        "mean_confidence": {n: round(conf_sums[n] / totals[n], 3) for n in names if totals.get(n)},
        "outputs": [video_out.name, *saved, f"{stem}.json", f"{stem}_frames.csv"],
    }
    (out_dir / f"{stem}.json").write_text(json.dumps(summary, indent=2))
    return summary


@app.function(image=image, gpu="L4", cpu=4.0, memory=16384, volumes=VOLUMES, timeout=3600)
def annotate(video: str, run: str, conf: float, credit: str, max_width: int):
    vol.reload()
    src = Path("/vol") / video.strip("/")
    if src.is_dir():
        videos = sorted(p for p in src.iterdir() if p.suffix.lower() in VIDEO_SUFFIXES)
    elif src.is_file():
        videos = [src]
    else:
        raise FileNotFoundError(f"{video} is not on the kitti-msc volume; upload it first with modal volume put")
    if not videos:
        raise FileNotFoundError(f"no .mp4, .mov, .avi or .mkv file in {video}")
    weights = Path("/runs") / run / run / "weights" / "best.pt"
    if not weights.is_file():
        raise FileNotFoundError(f"no best.pt for {run} at {weights}")
    imgsz = RUNS[run]
    model = {"Y11N": "YOLO11n", "Y11S": "YOLO11s", "Y11M": "YOLO11m"}[run.split("-")[1]]
    label = f"{model}, trained on KITTI ({imgsz} px), confidence {conf:g}"
    out_dir = Path("/vol/nigeria-demo")
    results = []
    for v in videos:
        print(f"{v.name}: running {run} ...")
        s = annotate_video(v, weights, imgsz, conf, out_dir, f"{v.stem}_{run}", label, credit, max_width)
        print(f"{v.name}: {s['frames']} frames in {s['seconds']} s; detections {s['detections']}")
        results.append(s)
        vol.commit()
    return results


@app.local_entrypoint()
def main(video: str, run: str = "E2-Y11S-960-DD-S0", conf: float = 0.25, credit: str = "", max_width: int = 1920):
    if run not in RUNS:
        raise SystemExit(f"--run must be one of: {', '.join(RUNS)}")
    results = annotate.remote(video, run, conf, credit, max_width)
    print(json.dumps(results, indent=2))
    print('Download the results: python -m modal volume get --force kitti-msc nigeria-demo "G:/My Drive/msc"')
