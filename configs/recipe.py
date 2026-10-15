"""The one training recipe. Only model, imgsz, split, seed (and the run's name) change between runs.

Every value is written out; nothing is left to Ultralytics defaults (8.4.171 values, PREREG.md section 3).
"""
ULTRALYTICS_VERSION = "8.4.171"

COMMON = dict(
    # schedule
    patience=0,                       # no early stopping: every run gets the same schedule
    batch=16,                         # fixed; 8 only if memory requires, and logged
    nbs=64,                           # gradients accumulate to a nominal batch of 64
    optimizer="AdamW", lr0=0.001, lrf=0.01, cos_lr=True,   # 'auto' would ignore lr0
    momentum=0.937, weight_decay=0.0005,
    warmup_epochs=3, warmup_momentum=0.8, warmup_bias_lr=0.0,
    # loss
    box=7.5, cls=0.5, dfl=1.5, cls_pw=0.0,
    # colour and geometry (TAB 07 "moderate" = these 8.4.171 defaults)
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0,
    flipud=0.0, fliplr=0.5, bgr=0.0,
    # mixing
    mosaic=1.0, close_mosaic=10, mixup=0.0, cutmix=0.0, copy_paste=0.0, multi_scale=0.0,
    # head and freezing
    freeze=None,
    cls_remap=False,                  # fresh head rows for every class; True would give Car COCO's "car" row
    # run mechanics
    amp=True, deterministic=True, cache="disk", workers=8,
    val=True, plots=True, exist_ok=True,
)

CLASS_NAMES = {0: "Car", 1: "Pedestrian", 2: "Cyclist", 3: "LargeVehicle"}

# l4_h100: the plan doc's L4 estimate at 100 epochs, used only for queue planning.
RUNS = {
    "E1-Y11S-640-DD-S0": dict(model="yolo11s.pt", imgsz=640, split="drive_disjoint", seed=0, l4_h100=1.65),  # headline
    "E0-COCO-640-VAL":   dict(kind="e0", l4_h100=0.15),
    "RS-Y11S-640-RND-S0": dict(model="yolo11s.pt", imgsz=640, split="random", seed=0, l4_h100=1.65),
    "E1-Y11N-640-DD-S0": dict(model="yolo11n.pt", imgsz=640, split="drive_disjoint", seed=0, l4_h100=1.14),
    "E1-Y11M-640-DD-S0": dict(model="yolo11m.pt", imgsz=640, split="drive_disjoint", seed=0, l4_h100=2.63),
    "E2-Y11S-960-DD-S0": dict(model="yolo11s.pt", imgsz=960, split="drive_disjoint", seed=0, l4_h100=3.24),
    "E1-Y11S-640-DD-S1": dict(model="yolo11s.pt", imgsz=640, split="drive_disjoint", seed=1, l4_h100=1.65),
    "E1-Y11S-640-DD-S2": dict(model="yolo11s.pt", imgsz=640, split="drive_disjoint", seed=2, l4_h100=1.65),
}
HEADLINE = "E1-Y11S-640-DD-S0"

# Epoch rule (PREREG.md section 3.3): two L4 runtimes and enough compute units -> 100 epochs on
# queues A and B; otherwise 50 epochs on the single queue. One value for every run.
QUEUES = {
    "A": ["E1-Y11S-640-DD-S0", "E1-Y11N-640-DD-S0", "E2-Y11S-960-DD-S0", "E1-Y11S-640-DD-S2"],
    "B": ["E0-COCO-640-VAL", "RS-Y11S-640-RND-S0", "E1-Y11M-640-DD-S0", "E1-Y11S-640-DD-S1"],
    "single": ["E1-Y11S-640-DD-S0", "E0-COCO-640-VAL", "RS-Y11S-640-RND-S0", "E1-Y11N-640-DD-S0",
               "E1-Y11M-640-DD-S0", "E2-Y11S-960-DD-S0", "E1-Y11S-640-DD-S1", "E1-Y11S-640-DD-S2"],
}

NEVER_CUT = ["E1-Y11S-640-DD-S0", "RS-Y11S-640-RND-S0", "E0-COCO-640-VAL"]
CUT_ORDER = ["E1-Y11S-640-DD-S2", "E1-Y11S-640-DD-S1", "E2-Y11S-960-DD-S0", "E1-Y11M-640-DD-S0"]

# Early warning for a silent label-conversion bug: halts every queue.
HALT_MIN_CAR_VAL_MAP50 = 0.60

# E0: COCO-pretrained models, no training (PREREG.md section 5.3).
E0_MODELS = ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt"]
E0_IMGSZ = 640
E0_COCO_TO_KITTI = {2: "Car", 0: "Pedestrian", 5: "LargeVehicle", 6: "LargeVehicle", 7: "LargeVehicle"}  # car, person, bus, train, truck

# Prediction settings for every saved prediction file (AP needs low-confidence boxes).
PREDICT = dict(conf=0.001, iou=0.7, max_det=300, batch=16)
