"""Figures, numbers and demonstration videos for the dissertation that need the KITTI images, the scored test
predictions or the trained weights.

This is not part of the pre-registered study. It reads the frozen test predictions, the labels and the trained
weights, changes nothing on the study's volumes, and writes only to the volume kitti-msc-thesis-figs:

  fig_pr_curves.png       precision against recall on the sealed test set, KITTI Moderate (Figure 4.1)
  fig_neardup.png         a random-split validation image and its nearest training frame (Figure 4.4)
  fig_test_examples.png   the headline model's detections on four test images chosen by fixed rules (Figure 4.10)
  fig_karlsruhe.png       Anya's study model on three frames of a sealed test drive in Karlsruhe (Figure 4.12)
  fig_lagos.png           Anya's study model and general road model on three frames of Lagos footage (Figure 4.13)
  karlsruhe_clip.mp4      the test-drive stretch as a plain video, ready to upload to Anya
  karlsruhe_study.mp4     the same stretch annotated by the study model, as Anya draws it
  karlsruhe_general.mp4   the same stretch annotated by the general road model
  lagos_<n>_study.mp4     the first 12 seconds of each of the first two Lagos videos, annotated by the study model
  lagos_<n>_general.mp4   the same, annotated by the general road model
  stats.json              narrow boxes at 640 pixels (Section 1.2), objects found at any confidence (exploratory,
                          Section 4.2.6), the objects found on the Karlsruhe stretch (Section 4.2.8), the rules'
                          choices and the checks

From the project folder in Git Bash (about 20 minutes; the Lagos outputs use the videos already uploaded to
kitti-msc/nigeria-demo-in for demo/nigeria_demo.py, and are skipped if there are none):

  python -m modal run --detach thesis/thesis_figures.py
  mkdir -p "/g/My Drive/msc/thesis_figures"
  MSYS_NO_PATHCONV=1 python -m modal volume get --force kitti-msc-thesis-figs / "G:/My Drive/msc/thesis_figures"

(The folder must exist first, or Modal writes every file to the same path. MSYS_NO_PATHCONV=1 stops Git Bash from
turning the lone / into a Windows folder path.)

To test it on a computer without Modal:
  python thesis/thesis_figures.py --local LABELS IMAGES MAPPING SPLITS PRED_DIR RESULTS_JSON WEIGHTS_960 GENERAL_PT
      VIDEO_DIR OUT    (VIDEO_DIR may be '-')
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# On the computer this file sits in <project>/thesis. Inside the Modal container Modal runs it as
# /root/thesis_figures.py and the project folders it needs are copied to /root/repo.
REPO = HERE.parent if (HERE.parent / "evaluation").is_dir() else Path("/root/repo")

HEADLINE, RUN_M, RUN_960, E0_M = "E1-Y11S-640-DD-S0", "E1-Y11M-640-DD-S0", "E2-Y11S-960-DD-S0", "E0-Y11M"
TRAINED = ["E1-Y11S-640-DD-S0", "RS-Y11S-640-RND-S0", "E1-Y11N-640-DD-S0", "E1-Y11M-640-DD-S0", "E2-Y11S-960-DD-S0",
           "E1-Y11S-640-DD-S1", "E1-Y11S-640-DD-S2"]
NAMES = {E0_M: "Zero-shot YOLO11m", HEADLINE: "YOLO11s, 640 px (headline)", RUN_M: "YOLO11m, 640 px",
         RUN_960: "YOLO11s, 960 px"}
COLOUR = {"Car": "#2a78d6", "Pedestrian": "#eb6834", "Cyclist": "#1baf7a", "LargeVehicle": "#eda100",
          "Large vehicle": "#eda100", "Person": "#eb6834", "Rider": "#1baf7a", "Bicycle": "#8e5bd6",
          "Motorcycle": "#d64545", "Bus": "#eda100", "Truck": "#b07c0c", "Traffic light": "#5f6b7a",
          "Stop sign": "#5f6b7a"}
MODEL_COLOUR = {E0_M: "#8a8a8a", HEADLINE: "#1f1f1f", RUN_M: "#2a78d6", RUN_960: "#1baf7a"}
COCO_KEEP = {0: "Person", 1: "Bicycle", 2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck", 9: "Traffic light",
             11: "Stop sign"}
TEXT_W_IN = 14.6 / 2.54  # the thesis text width

# Anya's own colours, names and drawing (anya/app.py), so that the demonstration frames look as Anya shows them
ANYA_COLOURS = {
    "Car": "#2F80ED", "Pedestrian": "#27AE60", "Person": "#27AE60", "Cyclist": "#F2994A", "Rider": "#F2994A",
    "Large vehicle": "#9B51E0", "Bus": "#9B51E0", "Truck": "#A0522D", "Motorcycle": "#EB5757",
    "Bicycle": "#F2C94C", "Traffic light": "#00A3BF", "Stop sign": "#D81B60",
}
KITTI_NAMES = {"Car": "Car", "Pedestrian": "Pedestrian", "Cyclist": "Cyclist", "LargeVehicle": "Large vehicle"}
DARK_TEXT = {"Bicycle"}
DEMO_CONF, DEMO_IMGSZ = 0.25, 960  # Anya's default confidence and its input size


# ---------------------------------------------------------------- helpers

def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for f in Path("/usr/share/fonts/opentype/inter").glob("Inter-*.otf"):
        font_manager.fontManager.addfont(str(f))
    family = "Inter" if any("Inter" in f.name for f in font_manager.fontManager.ttflist) else "DejaVu Sans"
    plt.rcParams.update({"font.family": family, "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                         "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7, "savefig.dpi": 300,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#b9b9b0",
                         "axes.labelcolor": "#3a3a36", "xtick.color": "#5a5a55", "ytick.color": "#5a5a55",
                         "axes.grid": True, "grid.color": "#e4e4de", "grid.linewidth": 0.6})
    return plt


def read_png_size(path):
    with open(path, "rb") as f:
        head = f.read(24)
    return int.from_bytes(head[16:20], "big"), int.from_bytes(head[20:24], "big")


def frame_of(mapping_dir):
    """Image id -> (drive, frame index within the drive), through train_rand.txt -> train_mapping.txt."""
    d = Path(mapping_dir)
    rand = [int(v) for v in d.joinpath("train_rand.txt").read_text().strip().split(",")]
    mapping = d.joinpath("train_mapping.txt").read_text().splitlines()
    return {f"{i:06d}": (mapping[r - 1].split()[1], int(mapping[r - 1].split()[2])) for i, r in enumerate(rand)}


def match_any(preds, labels, ids, threshold):
    """Every Car/Pedestrian/Cyclist object with the score of the same-class box matched to it (one to one, by
    score, at the class's KITTI IoU) among boxes at or above threshold, or None. As failure_analysis matches."""
    from evaluation.scoring import KITTI_IOU, iou_matrix, bin_of, HEIGHT_BINS
    rows = []
    for i in ids:
        objs = [o for o in labels[i] if o["name"] in KITTI_IOU]
        dets = sorted([d for d in preds.get(i, []) if d[5] >= threshold], key=lambda d: -d[5])
        score = [None] * len(objs)
        if objs and dets:
            iou = iou_matrix([o["bbox"] for o in objs], [d[1:5] for d in dets])
            for j, d in enumerate(dets):
                best, bk = 0.0, -1
                for k, o in enumerate(objs):
                    if score[k] is None and o["name"] == d[0] and iou[k, j] >= KITTI_IOU[o["name"]] and iou[k, j] > best:
                        best, bk = iou[k, j], k
                if bk >= 0:
                    score[bk] = float(d[5])
        for k, o in enumerate(objs):
            x1, y1, x2, y2 = o["bbox"]
            rows.append({"image": i, "class": o["name"], "score": score[k], "height": bin_of(y2 - y1, HEIGHT_BINS),
                         "occlusion": int(o["occluded"]), "bbox": o["bbox"]})
    return rows


# ---------------------------------------------------------------- 1. numbers

def narrow_boxes(labels, sizes):
    """Share of boxes narrower than 16 px after the image is scaled to 640 (and 960) px wide."""
    out = {}
    for c in ("Car", "Pedestrian", "Cyclist"):
        widths = [((o["bbox"][2] - o["bbox"][0]), sizes[i][0]) for i, objs in labels.items() for o in objs
                  if o["name"] == c]
        n = len(widths)
        out[c] = {"objects": n}
        for size in (640, 960):
            k = sum(1 for w, W in widths if w * size / W < 16)
            out[c][f"narrower_than_16px_at_{size}"] = {"count": k, "share": round(k / n, 4) if n else None}
    return out


def any_confidence(preds_by_model, thresholds, labels, ids):
    """Exploratory: objects found at the frozen threshold and at any saved confidence (>= 0.001), by class and
    height; and the scores at which small objects were found."""
    import numpy as np
    out = {}
    for key, preds in preds_by_model.items():
        rows = match_any(preds, labels, ids, 0.0)
        res = {}
        for c in ("Car", "Pedestrian", "Cyclist"):
            for h in ("<=25 px", "25-40 px", ">40 px"):
                sel = [r for r in rows if r["class"] == c and r["height"] == h]
                any_ = [r for r in sel if r["score"] is not None]
                scores = [r["score"] for r in any_]
                res[f"{c} | {h}"] = {
                    "objects": len(sel), "found_any_confidence": len(any_),
                    "score_of_match_median": round(float(np.median(scores)), 4) if scores else None,
                    "score_of_match_p90": round(float(np.percentile(scores, 90)), 4) if scores else None}
        thr_rows = match_any(preds, labels, ids, thresholds[key])
        for c in ("Car", "Pedestrian", "Cyclist"):
            for h in ("<=25 px", "25-40 px", ">40 px"):
                res[f"{c} | {h}"]["found_at_frozen_threshold"] = sum(
                    1 for r in thr_rows if r["class"] == c and r["height"] == h and r["score"] is not None)
        out[key] = {"threshold": thresholds[key], "by_class_height": res}
    return out


# ---------------------------------------------------------------- 2. precision-recall curves

def pr_curves(caches, results, out_png):
    """Precision and recall at every score step, from the evaluator's own statistics (KITTI Moderate)."""
    import numpy as np
    plt = style()
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_W_IN, 2.45), sharey=True)
    checks = {}
    for ax, c in zip(axes, ("Car", "Pedestrian", "Cyclist")):
        handles = []
        for key, cache in caches.items():
            if key == E0_M and c == "Cyclist":
                continue
            k = cache.cache[(c, "moderate")]
            base = k["base"].sum(0)
            stats = base + np.cumsum(k["d"], axis=0) if len(k["d"]) else base[None, :]
            tp, fp, fn = stats[:, 0].astype(float), stats[:, 1].astype(float), stats[:, 2].astype(float)
            prec = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 1.0)
            rec = tp / np.maximum(tp + fn, 1)
            env = np.maximum.accumulate(prec[::-1])[::-1]  # interpolated precision, as AP uses
            ap = cache.ap40()[c]["moderate"]
            want = results["models"][key]["kitti_ap40"][c]["moderate"]
            checks[f"{key} {c}"] = {"ap40_from_cache": ap, "ap40_reported": want, "diff": abs(ap - want)}
            ls = "--" if key == E0_M else "-"
            ax.plot(np.r_[0, rec], np.r_[env[0] if len(env) else 1, env], color=MODEL_COLOUR[key], lw=1.2, ls=ls)
            handles.append(Line2D([], [], color=MODEL_COLOUR[key], lw=1.2, ls=ls, label=f"{ap:.1f}"))
        ax.set_title(c, loc="left", fontweight="semibold")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "0.25", "0.5", "0.75", "1"])
        ax.set_xlabel("Recall")
        ax.legend(handles=handles, title="AP|R40", loc="lower left", frameon=False, title_fontsize=7)
    axes[0].set_ylabel("Precision (interpolated)")
    names = [Line2D([], [], color=MODEL_COLOUR[k], lw=1.4, ls="--" if k == E0_M else "-", label=NAMES[k])
             for k in caches]
    fig.legend(handles=names, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.08), frameon=False)
    fig.subplots_adjust(wspace=0.14)
    fig.savefig(out_png, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return checks


# ---------------------------------------------------------------- 3. test examples

def choose_examples(rows, labels, ids):
    """Four test images by fixed rules, from the headline model's outcomes at its frozen threshold: the image whose
    car recall is closest to the overall car recall (among images with at least 5 cars), and the images with the most
    labelled pedestrians, the most labelled cyclists and the most labelled objects up to 40 pixels tall."""
    by_img = {}
    for r in rows:
        by_img.setdefault(r["image"], []).append(r)
    car_all = [r for r in rows if r["class"] == "Car"]
    car_rate = sum(r["score"] is not None for r in car_all) / max(len(car_all), 1)
    chosen, rules = [], []

    def pick(rule, key, cands, short):
        cands = [i for i in cands if i not in chosen]
        if not cands:
            return
        best = sorted(cands, key=lambda i: (key(i), i))[0]
        chosen.append(best)
        rules.append({"image": best, "rule": rule, "short": short})

    def cars(i):
        return [r for r in by_img.get(i, []) if r["class"] == "Car"]
    pick("car recall closest to the overall car recall, among images with at least 5 cars",
         lambda i: abs(sum(r["score"] is not None for r in cars(i)) / len(cars(i)) - car_rate),
         [i for i in ids if len(cars(i)) >= 5], "typical car recall")
    pick("most pedestrians", lambda i: -sum(r["class"] == "Pedestrian" for r in by_img.get(i, [])),
         [i for i in ids if any(r["class"] == "Pedestrian" for r in by_img.get(i, []))], "most pedestrians")
    pick("most cyclists", lambda i: -sum(r["class"] == "Cyclist" for r in by_img.get(i, [])),
         [i for i in ids if any(r["class"] == "Cyclist" for r in by_img.get(i, []))], "most cyclists")
    pick("most objects up to 40 px tall",
         lambda i: -sum(r["height"] in ("<=25 px", "25-40 px") for r in by_img.get(i, [])),
         [i for i in ids if any(r["height"] in ("<=25 px", "25-40 px") for r in by_img.get(i, []))],
         "most small objects")
    return rules, car_rate


def draw_examples(rules, rows, preds, threshold, image_dir, out_png, keep=0.7):
    """Each image is printed at the full text width without its upper 30%, which shows sky and buildings above the
    road. A box reaching into that strip is cut at the image's top edge. With four images the figure is always
    14.6 cm wide and 15.1 cm high."""
    import matplotlib.image as mpimg
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    plt = style()
    n = len(rules)
    imgs = {r["image"]: mpimg.imread(str(Path(image_dir) / f"{r['image']}.png")) for r in rules}
    panel = TEXT_W_IN * keep * 375 / 1242      # inches: a cropped KITTI image at the text width
    title, legend = 0.2, 0.28                  # inches above each image and above the first
    fig_h = legend + n * (title + panel)
    fig = plt.figure(figsize=(TEXT_W_IN, fig_h))
    letters = "abcdefgh"
    for k, rule in enumerate(rules):
        i = rule["image"]
        img = imgs[i]
        hi, wi = img.shape[:2]
        y0 = int(round((1 - keep) * hi))
        bottom = fig_h - legend - (k + 1) * (title + panel)
        ax = fig.add_axes([0, bottom / fig_h, 1, panel / fig_h])
        ax.imshow(img[y0:], extent=(0, wi, hi, y0), aspect="auto")
        ax.set_xlim(0, wi)
        ax.set_ylim(hi, y0)
        ax.set_axis_off()
        for d in preds.get(i, []):
            if d[5] < threshold:
                continue
            x1, y1, x2, y2 = d[1:5]
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, lw=1.0, ec=COLOUR[d[0]]))
        missed = 0
        for r in rows:
            if r["image"] == i and r["score"] is None:
                x1, y1, x2, y2 = r["bbox"]
                ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, lw=1.0, ls=(0, (2, 1.5)),
                                       ec=COLOUR[r["class"]]))
                missed += 1
        objs = sum(1 for r in rows if r["image"] == i)
        ax.set_title(f"({letters[k]}) image {i}, {rule['short']}: {objs - missed} of {objs} objects found",
                     loc="left", fontsize=7.5, pad=2.5)
    handles = [Line2D([], [], color=COLOUR[c], lw=1.4, label=c) for c in ("Car", "Pedestrian", "Cyclist", "LargeVehicle")]
    handles += [Line2D([], [], color="#3a3a36", lw=1.2, label="detection"),
                Line2D([], [], color="#3a3a36", lw=1.2, ls=(0, (2, 1.5)), label="missed object")]
    fig.legend(handles=handles, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.0), frameon=False,
               borderaxespad=0.1, handlelength=2.2)
    fig.savefig(out_png, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- 4. near-duplicate pair

def neardup(splits_dir, mapping_dir, labels, image_dir, out_png):
    import matplotlib.image as mpimg
    plt = style()
    fr = frame_of(mapping_dir)
    train = set(Path(splits_dir, "random", "train.txt").read_text().split())
    val = sorted(Path(splits_dir, "random", "val.txt").read_text().split())
    by_drive = {}
    for i in train:
        d, f = fr[i]
        by_drive.setdefault(d, []).append((f, i))
    pair = None
    for v in val:
        d, f = fr[v]
        if len([o for o in labels.get(v, []) if o["name"] != "DontCare"]) < 3:
            continue
        near = [(abs(tf - f), ti, tf) for tf, ti in by_drive.get(d, []) if abs(tf - f) == 1]
        if near:
            gap, ti, tf = sorted(near)[0]
            pair = {"validation": v, "training": ti, "drive": d, "frames": [tf, f], "rule":
                    "the lowest-numbered random-split validation image with at least three labelled objects and a "
                    "training image of the same drive one frame away"}
            break
    if pair is None:
        return None
    fig, axes = plt.subplots(2, 1, figsize=(TEXT_W_IN, 3.85))
    for ax, (lab, i, f) in zip(axes, [("(a) training image", pair["training"], pair["frames"][0]),
                                      ("(b) random-split validation image", pair["validation"], pair["frames"][1])]):
        ax.imshow(mpimg.imread(str(Path(image_dir) / f"{i}.png")))
        ax.set_axis_off()
        ax.set_title(f"{lab} {i}: drive {pair['drive']}, frame {f}", loc="left", fontsize=7.5, pad=3)
    fig.subplots_adjust(hspace=0.18, left=0.005, right=0.995, top=0.94, bottom=0.01)
    fig.savefig(out_png, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return pair


# ---------------------------------------------------------------- 5. demonstrations (as Anya shows them)

def mark_riders(dets):
    """As Anya does: a Person whose box sits on a Motorcycle or Bicycle box is a Rider."""
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


def detect(model, frame, key, conf=DEMO_CONF, imgsz=DEMO_IMGSZ):
    """Detections on one BGR frame, named as Anya names them."""
    kw = dict(imgsz=imgsz, conf=conf, iou=0.7, verbose=False)
    if key == "general":
        kw["classes"] = sorted(COCO_KEEP)
    res = model.predict(frame, **kw)[0]
    dets = []
    for xyxy, ci, s in zip(res.boxes.xyxy.tolist(), res.boxes.cls.int().tolist(), res.boxes.conf.tolist()):
        name = COCO_KEEP.get(ci) if key == "general" else KITTI_NAMES.get(res.names[ci], res.names[ci])
        if name:
            dets.append({"name": name, "conf": float(s), "box": [float(v) for v in xyxy]})
    return mark_riders(dets) if key == "general" else dets


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


def draw(img, dets, print_mm=None, label_mm=1.8):
    """Boxes and labels as Anya draws them (BGR image, changed in place). For a figure, print_mm (the printed width
    of the image) sizes the labels to about label_mm tall in print, so that they stay legible."""
    import cv2
    short = min(img.shape[:2])
    line = max(2, round(short / 400))
    scale = max(0.45, short / 1100)
    thick = max(1, round(short / 700))
    if print_mm:
        scale = label_mm * img.shape[1] / print_mm / 22.0
        thick = max(1, round(scale * 1.6))
        line = max(2, round(scale * 2.6))
    for d in sorted(dets, key=lambda d: d["conf"]):
        x1, y1, x2, y2 = (int(round(v)) for v in d["box"])
        colour = bgr(ANYA_COLOURS.get(d["name"], "#FFFFFF"))
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, line, cv2.LINE_AA)
        put_label(img, f'{d["name"]} {d["conf"]:.2f}', x1, y1, colour, scale, thick, d["name"] in DARK_TEXT)
    return img


def caption(img, text):
    """A small caption at the bottom left of a video frame, as Anya writes it."""
    import cv2
    short = min(img.shape[:2])
    scale = max(0.45, short / 1300)
    thick = max(1, round(short / 800))
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = 12, img.shape[0] - 12
    cv2.rectangle(img, (x - 6, y - th - 10), (x + tw + 6, y + base + 2), (20, 20, 20), -1)
    cv2.putText(img, text, (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick, cv2.LINE_AA)
    return img


def even(img, max_side=None):
    """Even width and height, as H.264 requires; optionally scaled so that the longer side is at most max_side."""
    import cv2
    h, w = img.shape[:2]
    if max_side and max(h, w) > max_side:
        s = max_side / max(h, w)
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
    return img[: h - h % 2, : w - w % 2]


def write_video(frames, path, fps):
    """Write BGR frames as an H.264 MP4 that plays in any browser (MJPG AVI if ffmpeg is missing)."""
    import shutil
    import subprocess
    import cv2
    frames = list(frames)
    if not frames:
        return None
    h, w = frames[0].shape[:2]
    tmp = Path(path).with_suffix(".avi")
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    for f in frames:
        vw.write(f if f.shape[:2] == (h, w) else cv2.resize(f, (w, h)))
    vw.release()
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "26", "-movflags", "+faststart", str(path)], check=True)
        tmp.unlink()
        return Path(path).name
    return tmp.name


def longest_stretch(mapping_dir, splits_dir):
    """The test drive with the longest run of labelled frames, each at most two video frames after the last."""
    fr = frame_of(mapping_dir)
    by = {}
    for i in Path(splits_dir, "test.txt").read_text().split():
        d, f = fr[i]
        by.setdefault(d, []).append((f, i))
    best = None
    for d in sorted(by):
        fl = sorted(by[d])
        s = 0
        for k in range(1, len(fl) + 1):
            if k == len(fl) or fl[k][0] - fl[k - 1][0] > 2:
                run = fl[s:k]
                if best is None or len(run) > len(best[1]):
                    best = (d, run)
                s = k
    return best


def karlsruhe(image_dir, labels, mapping_dir, splits_dir, weights_960, general_pt, out):
    """Anya's two models on the longest continuous stretch of any sealed test drive."""
    import cv2
    import matplotlib.image as mpimg  # noqa: F401
    from ultralytics import YOLO
    plt = style()
    drive, run = longest_stretch(mapping_dir, splits_dir)
    ids = [i for _, i in run]
    first, last = run[0][0], run[-1][0]
    seconds = (last - first + 1) / 10.0  # KITTI records at 10 frames a second
    fps = round(len(run) / seconds, 2)
    frames = [even(cv2.imread(str(Path(image_dir) / f"{i}.png"))) for i in ids]
    study, general = YOLO(str(weights_960)), YOLO(str(general_pt))
    dets = {"study": [], "general": []}
    for f in frames:
        dets["study"].append(detect(study, f, "study"))
        dets["general"].append(detect(general, f, "general"))
    credit = f"KITTI test drive {drive.replace('_sync', '')}, Karlsruhe, never used in training (CC BY-NC-SA 3.0)"
    videos = {"clip": write_video(frames, out / "karlsruhe_clip.mp4", fps)}
    for key, label in (("study", "Study model, YOLO11s trained on KITTI, 960 px"),
                       ("general", "General road model, YOLO11m trained on COCO")):
        drawn = [caption(draw(f.copy(), d), f"{label} | {credit}") for f, d in zip(frames, dets[key])]
        videos[key] = write_video(drawn, out / f"karlsruhe_{key}.mp4", fps)
    # objects of the stretch found by the study model at Anya's default confidence (all heights, and 25 px or taller)
    back = {v: k for k, v in KITTI_NAMES.items()}
    preds = {i: [[back.get(d["name"], d["name"]), *d["box"], d["conf"]] for d in ds]
             for i, ds in zip(ids, dets["study"])}
    rows = match_any(preds, labels, ids, 0.0)
    found = {}
    for c in ("Car", "Pedestrian", "Cyclist"):
        sel = [r for r in rows if r["class"] == c]
        tall = [r for r in sel if r["height"] != "<=25 px"]
        found[c] = {"objects": len(sel), "found": sum(r["score"] is not None for r in sel),
                    "objects_25px_or_taller": len(tall), "found_25px_or_taller": sum(r["score"] is not None for r in tall)}
    # the figure: three frames, at a quarter, a half and three quarters of the stretch, as Anya draws them, each
    # without its top 30% (sky and buildings above the road), so that the figure is always 14.6 cm wide and 10.8 cm high
    picks = [len(frames) // 4, len(frames) // 2, (3 * len(frames)) // 4]
    keep = 0.7
    panel = TEXT_W_IN * keep * 375 / 1242
    title = 0.2
    fig_h = len(picks) * (title + panel)
    fig = plt.figure(figsize=(TEXT_W_IN, fig_h))
    for k, j in enumerate(picks):
        img = draw(frames[j].copy(), dets["study"][j], print_mm=146, label_mm=2.0)
        y0 = int(round((1 - keep) * img.shape[0]))
        ax = fig.add_axes([0, (fig_h - (k + 1) * (title + panel)) / fig_h, 1, panel / fig_h])
        ax.imshow(cv2.cvtColor(img[y0:], cv2.COLOR_BGR2RGB), aspect="auto")
        ax.set_axis_off()
        ax.set_title(f"({'abc'[k]}) {(run[j][0] - first) / 10:.1f} s into the stretch, image {ids[j]}", loc="left",
                     fontsize=7.5, pad=2.5)
    fig.savefig(out / "fig_karlsruhe.png", facecolor="white")
    plt.close(fig)
    return {"drive": drive, "images": len(ids), "first_frame": first, "last_frame": last, "seconds": seconds,
            "fps": fps, "conf": DEMO_CONF, "figure_images": [ids[j] for j in picks], "found_by_study_model": found,
            "videos": videos,
            "rule": "the sealed test drive with the longest run of labelled frames, each at most two video frames "
                    "after the one before"}


def lagos(video_dir, weights_960, general_pt, out, clip_seconds=12, clip_side=1280):
    """Anya's two models on Lagos footage: three frames for the thesis and short annotated clips for the slides."""
    import cv2
    from ultralytics import YOLO
    plt = style()
    vids = sorted(p for p in Path(video_dir).glob("*") if p.suffix.lower() in (".mp4", ".mov", ".avi", ".mkv"))
    if not vids:
        return None
    study, general = YOLO(str(weights_960)), YOLO(str(general_pt))
    models = (("study", study, "Study model, YOLO11s trained on KITTI, 960 px"),
              ("general", general, "General road model, YOLO11m trained on COCO"))
    # the figure: one frame from the middle of each of the first three videos, or three frames of the first
    picks = [(v, 0.5) for v in vids[:3]] if len(vids) >= 3 else [(vids[0], q) for q in (0.25, 0.5, 0.75)]
    frames, used = [], []
    for v, q in picks:
        cap = cv2.VideoCapture(str(v))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(n * q)))
        ok, fr = cap.read()
        cap.release()
        if ok:
            frames.append(even(fr, 1280))
            used.append({"video": v.name, "frame": int(n * q), "of": n})
    rows = len(frames)
    fig, axes = plt.subplots(rows, 2, figsize=(TEXT_W_IN, rows * TEXT_W_IN / 2 * frames[0].shape[0] /
                                               frames[0].shape[1] + 0.35))
    axes = axes.reshape(rows, 2)
    for r, fr in enumerate(frames):
        for c, (key, model, label) in enumerate(models):
            ax = axes[r, c]
            ax.imshow(cv2.cvtColor(draw(fr.copy(), detect(model, fr, key), print_mm=72, label_mm=1.7),
                                   cv2.COLOR_BGR2RGB))
            ax.set_axis_off()
            if r == 0:
                ax.set_title("(a) Study model" if key == "study" else "(b) General road model", loc="left",
                             fontsize=7.5, pad=3)
    fig.subplots_adjust(wspace=0.02, hspace=0.04, left=0.005, right=0.995, top=0.95, bottom=0.01)
    fig.savefig(out / "fig_lagos.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    # clips for the slides: the first seconds of each of the first two videos
    clips = []
    for n, v in enumerate(vids[:2], 1):
        cap = cv2.VideoCapture(str(v))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        raw = []
        while len(raw) < int(clip_seconds * fps):
            ok, fr = cap.read()
            if not ok:
                break
            raw.append(even(fr, clip_side))
        cap.release()
        for key, model, label in models:
            drawn = [caption(draw(f.copy(), detect(model, f, key)), f"{label} | Video: {v.name}, Pexels") for f in raw]
            clips.append(write_video(drawn, out / f"lagos_{n}_{key}.mp4", fps))
    return {"figure_frames": used, "clips": clips, "conf": DEMO_CONF}


# ---------------------------------------------------------------- everything

def build(label_dir, image_dir, mapping_dir, splits_dir, pred_dir, results_json, weights_960, general_pt, video_dir,
          out_dir, strict=True):
    sys.path.insert(0, str(REPO))
    from evaluation.kitti_cache import CachedKitti
    from evaluation.scoring import kitti_annos, load_labels, load_preds
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = json.loads(Path(results_json).read_text())
    test_ids = Path(splits_dir, "test.txt").read_text().split()
    preds = {k: load_preds(Path(pred_dir) / f"{k}.jsonl") for k in TRAINED + [E0_M]}
    ids = [i for i in test_ids if i in preds[HEADLINE]]
    if strict and len(ids) != len(test_ids):
        raise RuntimeError(f"the headline predictions cover {len(ids)} of {len(test_ids)} test images")
    all_ids = sorted(p.stem for p in Path(label_dir).glob("*.txt"))
    labels = load_labels(label_dir, all_ids)
    sizes = {i: read_png_size(Path(image_dir) / f"{i}.png") for i in all_ids}
    stats = {"note": "Outside the pre-registered study; exploratory.", "test_images": len(ids)}
    stats["narrow_boxes_all_7481_images"] = narrow_boxes(labels, sizes)
    thresholds = {k: results["models"][k]["threshold"] for k in TRAINED}
    stats["found_at_any_confidence"] = any_confidence({k: preds[k] for k in TRAINED}, thresholds, labels, ids)
    caches = {}
    for k in (E0_M, HEADLINE, RUN_M, RUN_960):
        gts, dts = kitti_annos(preds[k], labels, ids)
        caches[k] = CachedKitti(gts, dts)
    stats["pr_check"] = pr_curves(caches, results, out / "fig_pr_curves.png")
    if strict:
        worst = max(v["diff"] for v in stats["pr_check"].values())
        if worst > 1e-6:
            raise RuntimeError(f"AP from the cached matches differs from the reported AP by {worst}")
    rows = match_any(preds[HEADLINE], labels, ids, thresholds[HEADLINE])
    rules, car_rate = choose_examples(rows, labels, ids)
    stats["test_examples"] = {"rules": rules, "overall_car_recall_at_threshold": car_rate}
    draw_examples(rules, rows, preds[HEADLINE], thresholds[HEADLINE], image_dir, out / "fig_test_examples.png")
    stats["neardup"] = neardup(splits_dir, mapping_dir, labels, image_dir, out / "fig_neardup.png")
    stats["karlsruhe"] = karlsruhe(image_dir, labels, mapping_dir, splits_dir, weights_960, general_pt, out)
    stats["lagos"] = lagos(video_dir, weights_960, general_pt, out) if video_dir else None
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    return stats


def summary(stats):
    nb = stats["narrow_boxes_all_7481_images"]
    lines = ["Narrow boxes (under 16 px wide at 640 px): " + ", ".join(
        f"{c} " + (f"{100 * v['narrower_than_16px_at_640']['share']:.1f}%" if v['narrower_than_16px_at_640']['share']
                   is not None else "n/a") for c, v in nb.items())]
    for k in (HEADLINE, RUN_960):
        r = stats["found_at_any_confidence"][k]["by_class_height"]
        for key in ("Pedestrian | <=25 px", "Pedestrian | 25-40 px"):
            v = r[key]
            lines.append(f"{k} {key}: {v['found_at_frozen_threshold']} at the threshold, {v['found_any_confidence']} "
                         f"at any confidence, of {v['objects']} (median score {v['score_of_match_median']})")
    k = stats["karlsruhe"]
    lines.append(f"Karlsruhe: {k['drive']}, {k['images']} images over {k['seconds']:.1f} s; found by the study model: "
                 + ", ".join(f"{c} {v['found']}/{v['objects']}" for c, v in k["found_by_study_model"].items()))
    lines.append(f"Worst AP check difference: {max(v['diff'] for v in stats['pr_check'].values()):.2e}")
    lines.append(f"Examples: {[r['image'] for r in stats['test_examples']['rules']]}; near-duplicate pair: "
                 f"{stats['neardup'] and (stats['neardup']['training'], stats['neardup']['validation'])}; "
                 f"Lagos frames: {stats['lagos']}")
    return "\n".join(lines)


# ---------------------------------------------------------------- Modal

try:
    import modal
except ImportError:  # local use only
    modal = None

if modal is not None:
    PINS = [  # the training image's versions (cloud/kitti_modal.py), so the weights load as they were saved
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
        "numba==0.68.0", "llvmlite==0.50.0",
    ]
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("libgl1", "libglib2.0-0", "unzip", "ffmpeg")
        .run_commands("apt-get install -y fonts-inter || true")  # the thesis font; DejaVu Sans if unavailable
        .pip_install(*PINS)
        .run_commands("cd /root && python -c \"from ultralytics import YOLO; YOLO('yolo11m.pt')\"")
        .env({"PYTHONUNBUFFERED": "1"})
        .add_local_dir(REPO / "evaluation", "/root/repo/evaluation", ignore=["__pycache__"])
        .add_local_dir(REPO / "kitti", "/root/repo/kitti", ignore=["__pycache__"])
        .add_local_dir(REPO / "third_party", "/root/repo/third_party", ignore=["__pycache__"])
        .add_local_dir(REPO / "splits", "/root/repo/splits")
        .add_local_file(Path(__file__), "/root/repo/thesis/thesis_figures.py")
    )
    app = modal.App("kitti-thesis-figures")
    VOLS = {"/vol": modal.Volume.from_name("kitti-msc").read_only(),
            "/w": modal.Volume.from_name(f"kitti-msc-{RUN_960.lower()}").read_only(),
            "/out": modal.Volume.from_name("kitti-msc-thesis-figs", create_if_missing=True)}

    @app.function(image=image, gpu="L4", cpu=4.0, memory=20480, timeout=3600, volumes=VOLS)
    def figures():
        import subprocess
        raw, local = Path("/vol/raw"), Path("/root/kitti_raw")
        need = [raw / "data_object_image_2.zip", raw / "data_object_label_2.zip", raw / "devkit_object.zip",
                Path("/vol/eval/test/pred"), Path("/vol/eval/test/results.json"),
                Path(f"/w/{RUN_960}/weights/best.pt"), Path("/root/yolo11m.pt"), REPO / "evaluation",
                REPO / "kitti", REPO / "third_party", REPO / "splits" / "test.txt"]
        missing = [str(x) for x in need if not x.exists()]
        if missing:  # stop at once, before an hour of GPU time, if an input is not where it should be
            raise RuntimeError("Inputs not found: " + ", ".join(missing))
        videos = Path("/vol/nigeria-demo-in")
        print("Lagos videos:", sorted(v.name for v in videos.iterdir()) if videos.is_dir() else "none (Figure 4.13 skipped)")
        local.mkdir(parents=True, exist_ok=True)
        subprocess.run(["unzip", "-qo", str(raw / "data_object_image_2.zip"), "training/image_2/*", "-d", str(local)],
                       check=True)
        subprocess.run(["unzip", "-qo", str(raw / "data_object_label_2.zip"), "-d", str(local)], check=True)
        subprocess.run(["unzip", "-qo", str(raw / "devkit_object.zip"), "-d", str(local / "devkit")], check=True)
        mapping = next((local / "devkit").rglob("train_rand.txt")).parent
        stats = build(local / "training" / "label_2", local / "training" / "image_2", mapping, "/root/repo/splits",
                      "/vol/eval/test/pred", "/vol/eval/test/results.json", f"/w/{RUN_960}/weights/best.pt",
                      "/root/yolo11m.pt", str(videos) if videos.is_dir() else None, "/out")
        VOLS["/out"].commit()
        return summary(stats)

    @app.local_entrypoint()
    def main():
        print(figures.remote())
        print('Download the files: mkdir -p "/g/My Drive/msc/thesis_figures" && MSYS_NO_PATHCONV=1 python -m modal volume get '
              '--force kitti-msc-thesis-figs / "G:/My Drive/msc/thesis_figures"')


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--local":
    a = sys.argv[2:]
    st = build(*a[:8], a[8] if a[8] != "-" else None, a[9], strict=False)
    print(summary(st))
