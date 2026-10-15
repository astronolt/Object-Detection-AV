"""Ultralytics data YAMLs built from the frozen split files (ids -> absolute image paths)."""
from pathlib import Path

from kitti.common import YOLO_NAMES, read_id_list

REPO = Path(__file__).resolve().parents[1]
SPLIT_DIRS = {"drive_disjoint": "drive_disjoint", "random": "random"}


def image_paths(data_root, ids):
    return [f"{Path(data_root) / 'images' / i}.png" for i in ids]


def split_ids(split, part):
    """part in train/val/test; test is shared by both splits."""
    if part == "test":
        return read_id_list(REPO / "splits" / "test.txt")
    return read_id_list(REPO / "splits" / SPLIT_DIRS[split] / f"{part}.txt")


def data_yaml(data_root, split):
    root = Path(data_root)
    lists = root / "lists"
    lists.mkdir(parents=True, exist_ok=True)
    for part in ("train", "val"):
        (lists / f"{split}_{part}.txt").write_text("\n".join(image_paths(root, split_ids(split, part))) + "\n")
    names = "\n".join(f"  {k}: {v}" for k, v in enumerate(YOLO_NAMES))
    y = root / f"kitti_{split}.yaml"
    y.write_text(f"path: {root}\ntrain: {lists / f'{split}_train.txt'}\nval: {lists / f'{split}_val.txt'}\n"
                 f"names:\n{names}\n")
    return str(y)
