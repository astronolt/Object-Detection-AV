"""Anya (Igbo for "eyes"): a web app that finds road users in photos, videos and a live camera view.

Two detectors, chosen on the page:
- Study model: YOLO11s trained on KITTI at 960 pixels in the MSc study (run E2-Y11S-960-DD-S0, best.pt).
  It finds the study's four classes: Car (with vans), Pedestrian, Cyclist and Large vehicle (trucks, trams).
- General road model: YOLO11m pretrained on COCO and not trained on KITTI, like the study's E0 baseline.
  It also finds motorcycles, buses, bicycles, traffic lights and stop signs; a person whose box sits on a
  motorcycle or a bicycle is shown as Rider.

Anya is a demonstration. It is not part of the pre-registered evaluation, and nothing uploaded is kept: each
file is processed in a temporary folder that is deleted when the response has been sent.

Deploy from the project folder in Git Bash:   python -m modal deploy anya/app.py
Modal prints the address (https://...modal.run). Stop it when it is not needed:   python -m modal app stop anya
"""
import base64
import io
import json
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
KITTI_RUN = "E2-Y11S-960-DD-S0"  # the study's YOLO11s at 960 pixels, the best of the runs on validation
KITTI_WEIGHTS = f"/w/{KITTI_RUN}/weights/best.pt"
GENERAL_WEIGHTS = "/root/yolo11m.pt"
IMGSZ = 960
MAX_SIDE = 1920  # longer side of a photo or video frame after reduction, before detection and drawing
MAX_VIDEO_SECONDS = 60
MAX_UPLOAD_MB = 300
VIDEO_BATCH = 8

COCO_KEEP = {0: "Person", 1: "Bicycle", 2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck", 9: "Traffic light",
             11: "Stop sign"}
KITTI_NAMES = {"Car": "Car", "Pedestrian": "Pedestrian", "Cyclist": "Cyclist", "LargeVehicle": "Large vehicle"}
COLOURS = {
    "Car": "#2F80ED", "Pedestrian": "#27AE60", "Person": "#27AE60", "Cyclist": "#F2994A", "Rider": "#F2994A",
    "Large vehicle": "#9B51E0", "Bus": "#9B51E0", "Truck": "#A0522D", "Motorcycle": "#EB5757",
    "Bicycle": "#F2C94C", "Traffic light": "#00A3BF", "Stop sign": "#D81B60",
}
DARK_TEXT = {"Bicycle"}  # labels drawn in dark text on a light colour
MODELS = {
    "study": {"title": "Study model", "detail": "YOLO11s trained on KITTI in this study, 960 px",
              "classes": ["Car", "Pedestrian", "Cyclist", "Large vehicle"]},
    "general": {"title": "General road model", "detail": "YOLO11m pretrained on COCO, not trained on KITTI",
                "classes": ["Car", "Motorcycle", "Rider", "Person", "Bicycle", "Bus", "Truck", "Traffic light",
                            "Stop sign"]},
}


# ---------------------------------------------------------------- detection

def mark_riders(dets):
    """Relabel a Person as Rider when the person's box sits on a Motorcycle or Bicycle box."""
    vehicles = [d["box"] for d in dets if d["name"] in ("Motorcycle", "Bicycle")]
    for d in dets:
        if d["name"] != "Person" or not vehicles:
            continue
        px1, py1, px2, py2 = d["box"]
        area = max((px2 - px1) * (py2 - py1), 1e-6)
        cx = (px1 + px2) / 2
        for vx1, vy1, vx2, vy2 in vehicles:
            inter = max(0.0, min(px2, vx2) - max(px1, vx1)) * max(0.0, min(py2, vy2) - max(py1, vy1))
            if inter / area >= 0.2 and vx1 <= cx <= vx2 and py1 < vy1 and py2 <= vy2 + 0.15 * (py2 - py1):
                d["name"] = "Rider"
                break
    return dets


class Detector:
    """Both models, loaded once per container. One GPU, so inference runs one request at a time."""

    def __init__(self, study_weights, general_weights, imgsz=IMGSZ, device=None):
        import numpy as np
        import torch
        from ultralytics import YOLO

        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.imgsz = imgsz
        self.lock = threading.Lock()
        self.models = {"study": YOLO(study_weights), "general": YOLO(general_weights)}
        blank = np.zeros((544, 960, 3), np.uint8)
        for key in self.models:  # load the weights onto the GPU before the first visitor
            self.detect(key, [blank], 0.25)

    def detect(self, key, frames, conf):
        """For each BGR frame, a list of {name, conf, box: [x1, y1, x2, y2]} in that frame's pixels."""
        kw = dict(imgsz=self.imgsz, conf=conf, iou=0.7, device=self.device, verbose=False)
        if key == "general":
            kw["classes"] = sorted(COCO_KEEP)
        with self.lock:
            results = self.models[key].predict(frames, **kw)
        out = []
        for r in results:
            dets = []
            for xyxy, c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.int().tolist(), r.boxes.conf.tolist()):
                if key == "general":
                    name = COCO_KEEP.get(c)
                    if name is None:
                        continue
                else:
                    name = KITTI_NAMES.get(r.names[c], r.names[c])
                dets.append({"name": name, "conf": round(float(s), 3), "box": [round(float(v), 1) for v in xyxy]})
            out.append(mark_riders(dets) if key == "general" else dets)
        return out


def counts(dets):
    c = {}
    for d in dets:
        c[d["name"]] = c.get(d["name"], 0) + 1
    return dict(sorted(c.items(), key=lambda kv: (-kv[1], kv[0])))


# ---------------------------------------------------------------- drawing

def bgr(hex_colour):
    h = hex_colour.lstrip("#")
    return int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)


def put_label(img, text, x, y_top, colour, scale, thick, dark=False):
    import cv2
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), base = cv2.getTextSize(text, font, scale, thick)
    pad = max(2, thick + 1)
    box_h = th + base + 2 * pad
    top = y_top - box_h if y_top - box_h >= 0 else y_top
    x = max(0, min(x, img.shape[1] - tw - 2 * pad))
    cv2.rectangle(img, (x, top), (x + tw + 2 * pad, top + box_h), colour, -1)
    cv2.putText(img, text, (x + pad, top + pad + th), font, scale, (25, 25, 25) if dark else (255, 255, 255),
                thick, cv2.LINE_AA)


def draw(img, dets):
    """Boxes and labels on img (BGR, changed in place)."""
    import cv2
    short = min(img.shape[:2])
    line = max(2, round(short / 400))
    scale = max(0.45, short / 1100)
    thick = max(1, round(short / 700))
    for d in sorted(dets, key=lambda d: d["conf"]):
        x1, y1, x2, y2 = (int(round(v)) for v in d["box"])
        colour = bgr(COLOURS.get(d["name"], "#FFFFFF"))
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, line, cv2.LINE_AA)
        put_label(img, f'{d["name"]} {d["conf"]:.2f}', x1, y1, colour, scale, thick, d["name"] in DARK_TEXT)
    return img


def caption(img, text):
    """A small caption at the bottom left of a video frame."""
    import cv2
    short = min(img.shape[:2])
    scale = max(0.45, short / 1300)
    thick = max(1, round(short / 800))
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = 12, img.shape[0] - 12
    cv2.rectangle(img, (x - 6, y - th - 10), (x + tw + 6, y + base + 2), (20, 20, 20), -1)
    cv2.putText(img, text, (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick, cv2.LINE_AA)


def fit(img, max_side=MAX_SIDE):
    import cv2
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1:
        img = cv2.resize(img, (2 * round(w * s / 2), 2 * round(h * s / 2)), interpolation=cv2.INTER_AREA)
    return img


# ---------------------------------------------------------------- photos and videos

def read_photo(data):
    """Decode a photo upright (phone photos carry their rotation in EXIF), as BGR."""
    import numpy as np
    from PIL import Image, ImageOps
    im = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    return fit(np.ascontiguousarray(np.asarray(im)[:, :, ::-1]))


def jpeg_data_url(img, quality=88):
    import cv2
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def process_photo(det, data, model_keys, conf):
    img = read_photo(data)
    results = []
    for key in model_keys:
        t0 = time.time()
        dets = det.detect(key, [img], conf)[0]
        ms = (time.time() - t0) * 1000
        results.append({"model": key, **MODELS[key], "image": jpeg_data_url(draw(img.copy(), dets)),
                        "counts": counts(dets), "detections": len(dets), "milliseconds": round(ms, 1)})
    return {"width": img.shape[1], "height": img.shape[0], "results": results}


def process_video(det, key, src, dst, conf, max_seconds=MAX_VIDEO_SECONDS):
    """Annotate every frame of src (up to max_seconds) into dst as H.264 MP4; return a summary."""
    import cv2
    cap = cv2.VideoCapture(str(src))
    ok, first = cap.read()
    if not cap.isOpened() or not ok:
        raise ValueError("This video could not be read. MP4 (H.264) works best.")
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and 1 <= fps <= 120 else 25.0
    first = fit(first)
    oh, ow = first.shape[:2]
    limit = int(round(max_seconds * fps))
    raw = dst.with_suffix(".avi")
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"MJPG"), fps, (ow, oh))
    label = f"Anya - {MODELS[key]['title']} ({MODELS[key]['detail'].split(',')[0]})"
    stats, n, t0 = {}, 0, time.time()

    def flush(frames):
        for frame, dets in zip(frames, det.detect(key, frames, conf)):
            draw(frame, dets)
            caption(frame, label)
            writer.write(frame)
            for name, c in counts(dets).items():
                s = stats.setdefault(name, {"max": 0, "sum": 0})
                s["max"], s["sum"] = max(s["max"], c), s["sum"] + c

    batch, frame = [], first
    while frame is not None and n < limit:
        if frame.shape[:2] != (oh, ow):
            frame = cv2.resize(fit(frame), (ow, oh), interpolation=cv2.INTER_AREA)
        batch.append(frame)
        n += 1
        if len(batch) == VIDEO_BATCH:
            flush(batch)
            batch = []
        ok, frame = cap.read()
        frame = frame if ok else None
    if batch:
        flush(batch)
    truncated = frame is not None
    cap.release()
    writer.release()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264", "-preset", "veryfast",
                    "-pix_fmt", "yuv420p", "-crf", "23", "-movflags", "+faststart", str(dst)], check=True)
    raw.unlink(missing_ok=True)
    seconds = time.time() - t0
    return {
        "model": key, "title": MODELS[key]["title"], "detail": MODELS[key]["detail"],
        "frames": n, "video_seconds": round(n / fps, 1), "truncated": truncated, "size": [ow, oh],
        "processing_seconds": round(seconds, 1), "frames_per_second": round(n / seconds, 1) if seconds else None,
        "classes": {name: {"max_in_frame": s["max"], "per_frame": round(s["sum"] / n, 2)}
                    for name, s in sorted(stats.items(), key=lambda kv: (-kv[1]["sum"], kv[0]))},
    }


# ---------------------------------------------------------------- the web app

def build_api(det, page_html):
    import cv2
    import numpy as np
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import FileResponse, HTMLResponse
    from starlette.background import BackgroundTask

    api = FastAPI(title="Anya", docs_url=None, redoc_url=None)
    info = {"models": MODELS, "colours": COLOURS, "max_video_seconds": MAX_VIDEO_SECONDS,
            "max_upload_mb": MAX_UPLOAD_MB, "device": det.device}
    page = page_html.replace("/*ANYA_INFO*/null", json.dumps(info))

    def read_upload(file):
        data = file.file.read(MAX_UPLOAD_MB * 1024 * 1024 + 1)
        if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
            raise HTTPException(413, f"Files up to {MAX_UPLOAD_MB} MB, please.")
        if not data:
            raise HTTPException(400, "The file is empty.")
        return data

    def check(model, conf, allow_both=False):
        keys = ["study", "general"] if (allow_both and model == "both") else [model]
        if any(k not in MODELS for k in keys):
            raise HTTPException(400, "Unknown model.")
        if not 0.05 <= conf <= 0.95:
            raise HTTPException(400, "Confidence must be between 0.05 and 0.95.")
        return keys

    @api.get("/", response_class=HTMLResponse)
    def index():
        return page

    @api.get("/api/info")
    def get_info():
        return info

    @api.post("/api/photo")
    def photo(file: UploadFile = File(...), model: str = Form("study"), conf: float = Form(0.25)):
        keys = check(model, conf, allow_both=True)
        data = read_upload(file)
        try:
            return process_photo(det, data, keys, conf)
        except HTTPException:
            raise
        except Exception:  # an unreadable photo
            raise HTTPException(400, "This photo could not be read. JPG or PNG works best.")

    @api.post("/api/video")
    def video(file: UploadFile = File(...), model: str = Form("study"), conf: float = Form(0.25)):
        key = check(model, conf)[0]
        data = read_upload(file)
        work = Path(tempfile.mkdtemp(prefix="anya-"))
        try:
            suffix = Path(file.filename or "video.mp4").suffix.lower() or ".mp4"
            src = work / f"in{suffix}"
            src.write_bytes(data)
            dst = work / "anya.mp4"
            summary = process_video(det, key, src, dst, conf)
        except ValueError as e:
            shutil.rmtree(work, ignore_errors=True)
            raise HTTPException(400, str(e))
        except Exception as e:
            shutil.rmtree(work, ignore_errors=True)
            raise HTTPException(500, f"The video could not be processed ({type(e).__name__}).")
        return FileResponse(dst, media_type="video/mp4", filename="anya.mp4",
                            headers={"X-Anya-Summary": json.dumps(summary)},
                            background=BackgroundTask(shutil.rmtree, work, ignore_errors=True))

    @api.post("/api/frame")
    def frame(file: UploadFile = File(...), model: str = Form("study"), conf: float = Form(0.25)):
        key = check(model, conf)[0]
        data = read_upload(file)
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(400, "Unreadable frame.")
        t0 = time.time()
        dets = det.detect(key, [img], conf)[0]
        return {"width": img.shape[1], "height": img.shape[0], "detections": dets, "counts": counts(dets),
                "milliseconds": round((time.time() - t0) * 1000, 1)}

    return api


# ---------------------------------------------------------------- Modal

# The same package versions as the training image (cloud/kitti_modal.py), plus the web framework.
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
WEB_PINS = [
    "fastapi==0.150.0", "starlette==1.9.0", "pydantic==2.14.1", "pydantic_core==2.50.1", "annotated-types==0.8.0",
    "annotated-doc==0.0.5", "typing-inspection==0.4.4", "opentelemetry-api==1.45.1", "python-multipart==0.0.32",
]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0", "ffmpeg")
    .pip_install(*PINS, *WEB_PINS)
    .run_commands("cd /root && python -c \"from ultralytics import YOLO; YOLO('yolo11m.pt')\"")
    .env({"PYTHONUNBUFFERED": "1"})
    .add_local_file(HERE / "index.html", "/root/anya/index.html")
)
app = modal.App("anya")
weights = modal.Volume.from_name(f"kitti-msc-{KITTI_RUN.lower()}").read_only()


@app.function(image=image, gpu="L4", cpu=4.0, memory=16384, volumes={"/w": weights}, timeout=600,
              scaledown_window=600, max_containers=1)
@modal.concurrent(max_inputs=8)
@modal.asgi_app(label="anya")
def web():
    det = Detector(KITTI_WEIGHTS, GENERAL_WEIGHTS)
    return build_api(det, Path("/root/anya/index.html").read_text(encoding="utf-8"))
