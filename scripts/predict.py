"""Save predictions for a list of images as JSON lines: {"id": ..., "dets": [[name, x1, y1, x2, y2, score], ...]}.

Settings come from configs/recipe.py PREDICT (conf 0.001, NMS IoU 0.7, max 300 per image, batch 16).
For E0 (COCO weights) only car/person/bus/train/truck are kept, mapped to KITTI names, and a second
class-wise NMS at IoU 0.7 removes duplicates created by the merge (PREREG.md section 5.3).

Ultralytics treats a list of paths as one in-memory batch and ignores `batch`, so the paths go in
PREDICT["batch"] at a time, and rect=False letterboxes every image to the same imgsz x imgsz square
whatever else is in its batch (PREREG_AMENDMENT_3.md).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configs.recipe import CLASS_NAMES, E0_COCO_TO_KITTI, PREDICT  # noqa: E402
from kitti.yolo_data import image_paths  # noqa: E402


def predict(weights, ids, data_root, imgsz, out_jsonl, coco_e0=False):
    import torch
    from torchvision.ops import batched_nms
    from ultralytics import YOLO

    model = YOLO(weights)
    names_out = sorted(set(E0_COCO_TO_KITTI.values())) if coco_e0 else None
    kw = dict(imgsz=imgsz, conf=PREDICT["conf"], iou=PREDICT["iou"], max_det=PREDICT["max_det"],
              batch=PREDICT["batch"], rect=False, stream=True, verbose=False)
    if coco_e0:
        kw["classes"] = sorted(E0_COCO_TO_KITTI)
    out = Path(out_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".partial")
    paths = image_paths(data_root, ids)
    step = PREDICT["batch"]
    with open(tmp, "w") as f:
        for i in range(0, len(paths), step):
            for r in model.predict(source=paths[i:i + step], **kw):
                b = r.boxes
                xyxy, score, cls = b.xyxy.cpu(), b.conf.cpu(), b.cls.cpu().long()
                if coco_e0:
                    name = [E0_COCO_TO_KITTI[int(c)] for c in cls]
                    grp = torch.tensor([names_out.index(n) for n in name], dtype=torch.long)
                    keep = batched_nms(xyxy.float(), score.float(), grp, PREDICT["iou"]) if len(name) else torch.tensor([], dtype=torch.long)
                    dets = [[name[k], *map(float, xyxy[k]), float(score[k])] for k in keep.tolist()]
                else:
                    dets = [[CLASS_NAMES[int(c)], *map(float, x), float(s)] for x, s, c in zip(xyxy, score, cls)]
                f.write(json.dumps({"id": Path(r.path).stem, "dets": dets}) + "\n")
    n = sum(1 for _ in open(tmp))
    if n != len(ids):
        sys.exit(f"{weights}: {n} prediction lines for {len(ids)} images")
    tmp.replace(out)
    return out
