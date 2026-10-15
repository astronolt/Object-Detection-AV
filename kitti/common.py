"""Shared KITTI definitions: label parsing, class merge, drive mapping, difficulty rules."""
from pathlib import Path

N_IMAGES = 7481

# Training classes (YOLO indices), fixed in PREREG.md section 4.
YOLO_NAMES = ["Car", "Pedestrian", "Cyclist", "LargeVehicle"]
KITTI_TO_YOLO = {
    "Car": 0, "Van": 0,
    "Pedestrian": 1, "Person_sitting": 1,
    "Cyclist": 2,
    "Truck": 3, "Tram": 3,
}  # Misc and DontCare are dropped from training labels.

# KITTI difficulty rules, identical to clean_data() in the vendored evaluator.
DIFFICULTIES = ("easy", "moderate", "hard")
MIN_HEIGHT = (40, 25, 25)
MAX_OCCLUSION = (0, 1, 2)
MAX_TRUNCATION = (0.15, 0.3, 0.5)


def image_id(i):
    return f"{i:06d}"


def read_label(path):
    """Parse one KITTI label_2 file into a list of dicts (2D fields only)."""
    objs = []
    for line in Path(path).read_text().splitlines():
        f = line.split()
        if not f:
            continue
        objs.append({
            "name": f[0],
            "truncated": float(f[1]),
            "occluded": int(f[2]),
            "alpha": float(f[3]),
            "bbox": [float(v) for v in f[4:8]],  # x1, y1, x2, y2 in pixels
        })
    return objs


def read_all_labels(label_dir):
    label_dir = Path(label_dir)
    return {image_id(i): read_label(label_dir / f"{image_id(i)}.txt") for i in range(N_IMAGES)}


def is_moderate(obj):
    """Counts toward KITTI Moderate (inclusive of Easy), as clean_data() decides for a valid-class box."""
    x1, y1, x2, y2 = obj["bbox"]
    return (obj["occluded"] <= MAX_OCCLUSION[1] and obj["truncated"] <= MAX_TRUNCATION[1]
            and (y2 - y1) > MIN_HEIGHT[1])


def image_to_drive(devkit_mapping_dir):
    """Map image id -> raw drive, via train_rand.txt (1-based) -> train_mapping.txt.

    Reading train_mapping.txt alone, line i for image i, gives a silently wrong split.
    """
    d = Path(devkit_mapping_dir)
    rand = [int(v) for v in d.joinpath("train_rand.txt").read_text().strip().split(",")]
    mapping = d.joinpath("train_mapping.txt").read_text().splitlines()
    assert len(rand) == N_IMAGES and len(mapping) == N_IMAGES
    assert sorted(rand) == list(range(1, N_IMAGES + 1))
    return {image_id(i): mapping[rand[i] - 1].split()[1] for i in range(N_IMAGES)}


def read_id_list(path):
    return Path(path).read_text().split()


def write_id_list(path, ids):
    """One zero-padded id per line, ascending, LF endings, trailing newline. Hashed byte for byte."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="\n") as f:
        f.write("".join(f"{i}\n" for i in sorted(ids)))
