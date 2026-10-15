"""KITTI label_2 -> YOLO labels, each box normalised by its OWN image size, with the Day-1 round-trip gate.

The round trip re-reads the written text files (not in-memory floats), converts back to pixels and
compares with the original KITTI boxes. Gate: every kept box within 1 px; 004696 (1224 px wide) checked
by name; every coordinate inside [0, 1]; object counts preserved per image.

Usage (Colab): python scripts/convert_labels.py <label_2 dir> <image_2 dir> <out root> <report json>
               [--overlay <splits dir> <overlay out dir>]
Writes <out root>/labels/<id>.txt and symlinks <out root>/images/<id>.png -> image_2.
"""
import json
import os
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kitti.common import KITTI_TO_YOLO, N_IMAGES, YOLO_NAMES, image_id, read_id_list, read_label  # noqa: E402

TOL_PX = 1.0


def convert(label_dir, image_dir, out_root):
    out_root = Path(out_root)
    (out_root / "labels").mkdir(parents=True, exist_ok=True)
    (out_root / "images").mkdir(parents=True, exist_ok=True)
    sizes, per_class, max_err, worst = Counter(), Counter(), 0.0, None
    errors, special = [], {}
    for n in range(N_IMAGES):
        i = image_id(n)
        src = Path(image_dir) / f"{i}.png"
        dst = out_root / "images" / f"{i}.png"
        if not dst.exists():
            os.symlink(src.resolve(), dst)
        with Image.open(src) as im:
            W, H = im.size
        sizes[f"{W}x{H}"] += 1
        kept = [o for o in read_label(Path(label_dir) / f"{i}.txt") if o["name"] in KITTI_TO_YOLO]
        lines = []
        for o in kept:
            x1, y1, x2, y2 = o["bbox"]
            lines.append(f"{KITTI_TO_YOLO[o['name']]} {(x1 + x2) / 2 / W:.8f} {(y1 + y2) / 2 / H:.8f} "
                         f"{(x2 - x1) / W:.8f} {(y2 - y1) / H:.8f}\n")
            per_class[YOLO_NAMES[KITTI_TO_YOLO[o["name"]]]] += 1
        lab = out_root / "labels" / f"{i}.txt"
        with open(lab, "w", newline="\n") as f:
            f.write("".join(lines))

        # Round trip from the file just written.
        back = [list(map(float, ln.split())) for ln in lab.read_text().splitlines()]
        if len(back) != len(kept):
            errors.append(f"{i}: {len(kept)} kept objects but {len(back)} label lines")
        img_err = 0.0
        for o, (c, xc, yc, w, h) in zip(kept, back):
            if not all(0.0 <= v <= 1.0 for v in (xc, yc, w, h)) or xc - w / 2 < -1e-6 or xc + w / 2 > 1 + 1e-6:  # 8-decimal rounding at image edges
                errors.append(f"{i}: coordinate outside [0,1]: {xc} {yc} {w} {h}")
            if int(c) != KITTI_TO_YOLO[o["name"]]:
                errors.append(f"{i}: class {c} for {o['name']}")
            rec = ((xc - w / 2) * W, (yc - h / 2) * H, (xc + w / 2) * W, (yc + h / 2) * H)
            e = max(abs(a - b) for a, b in zip(rec, o["bbox"]))
            img_err = max(img_err, e)
            if e > max_err:
                max_err, worst = e, i
            if e > TOL_PX:
                errors.append(f"{i}: round-trip error {e:.4f} px")
        if i in ("004696",):
            special[i] = {"size": f"{W}x{H}", "max_err_px": img_err, "objects": len(kept)}

    report = {"images": N_IMAGES, "image_sizes": dict(sizes), "boxes_per_class": dict(per_class),
              "max_roundtrip_err_px": max_err, "worst_image": worst, "checked": special,
              "n_errors": len(errors), "errors_first_50": errors[:50],
              "gate_passed": not errors and special.get("004696", {}).get("max_err_px", 99) <= TOL_PX}
    return report


def overlay_sheet(out_root, splits_dir, sheet_dir, n=100, per_sheet=10):
    """First n drive-disjoint TRAIN images (ascending id), YOLO boxes drawn back from the label files."""
    from PIL import ImageDraw
    colours = {0: (255, 60, 60), 1: (60, 255, 60), 2: (60, 160, 255), 3: (255, 200, 0)}
    ids = read_id_list(Path(splits_dir) / "drive_disjoint/train.txt")[:n]
    Path(sheet_dir).mkdir(parents=True, exist_ok=True)
    for s in range(0, len(ids), per_sheet):
        tiles = []
        for i in ids[s:s + per_sheet]:
            im = Image.open(Path(out_root) / "images" / f"{i}.png").convert("RGB")
            W, H = im.size
            d = ImageDraw.Draw(im)
            for ln in (Path(out_root) / "labels" / f"{i}.txt").read_text().splitlines():
                c, xc, yc, w, h = map(float, ln.split())
                d.rectangle([(xc - w / 2) * W, (yc - h / 2) * H, (xc + w / 2) * W, (yc + h / 2) * H],
                            outline=colours[int(c)], width=2)
                d.text(((xc - w / 2) * W + 2, (yc - h / 2) * H + 1), YOLO_NAMES[int(c)], fill=colours[int(c)])
            d.text((5, 5), i, fill=(255, 255, 255))
            tiles.append(im)
        sheet = Image.new("RGB", (max(t.width for t in tiles), sum(t.height for t in tiles)))
        y = 0
        for t in tiles:
            sheet.paste(t, (0, y))
            y += t.height
        sheet.save(Path(sheet_dir) / f"overlay_{s // per_sheet:02d}.jpg", quality=90)


if __name__ == "__main__":
    label_dir, image_dir, out_root, report_json = sys.argv[1:5]
    rep = convert(label_dir, image_dir, out_root)
    Path(report_json).parent.mkdir(parents=True, exist_ok=True)
    Path(report_json).write_text(json.dumps(rep, indent=2) + "\n")
    print(json.dumps(rep, indent=2))
    if "--overlay" in sys.argv:
        k = sys.argv.index("--overlay")
        overlay_sheet(out_root, sys.argv[k + 1], sys.argv[k + 2])
        print("overlay sheets written to", sys.argv[k + 2])
    if not rep["gate_passed"]:
        sys.exit("ROUND-TRIP GATE FAILED")
    print("ROUND-TRIP GATE PASSED")
