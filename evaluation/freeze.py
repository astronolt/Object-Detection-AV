"""The freeze (PREREG.md section 5.1, amendment 1, PREREG_AMENDMENT_4.md section 2).

For every trained run and every E0 model: save its validation predictions, fix its confidence threshold (the
best all-class F1 point on those predictions) and score it on its own validation split. Everything is written
to <out>/FREEZE.json with the SHA-256 of each weights file, so the test opening can check that nothing changed.
E0's validation predictions are the ones its queue item saved during training (PREREG.md section 5.3), copied
unchanged; they are made here only if that file is missing. This script never reads splits/test.txt.

Usage: python evaluation/freeze.py --data-root /root/kitti --label-dir <KITTI label_2> --runs-mnt /runs --out <dir>
"""
import argparse
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from evaluation.models import ensure_weights, model_specs, now, sha256, write_json  # noqa: E402
from evaluation.scoring import (coco_match, coco_table, kitti_annos, kitti_recall_at, kitti_table,  # noqa: E402
                                load_labels, load_preds, prf_at)
from kitti.runlog import git_commit  # noqa: E402
from kitti.yolo_data import split_ids  # noqa: E402

E0_RUN = "E0-COCO-640-VAL"  # the queue item that saved E0's validation predictions (scripts/train_queue.py)


def fmt(x):
    return f"{x:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--label-dir", required=True)
    ap.add_argument("--runs-mnt", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from predict import predict

    out = Path(a.out)
    pred_dir = out / "val_pred"
    pred_dir.mkdir(parents=True, exist_ok=True)
    specs = model_specs(a.runs_mnt)
    ids = {s: split_ids(s, "val") for s in ("drive_disjoint", "random")}
    labels = load_labels(a.label_dir, sorted(set(ids["drive_disjoint"]) | set(ids["random"])))
    freeze = {"time_started": now(), "git_commit": git_commit(), "rule": "PREREG.md 5.1; PREREG_AMENDMENT_4.md 2",
              "models": {}}
    lines = []
    for key, s in specs.items():
        w = ensure_weights(s)
        vids = ids[s["val_split"]]
        path = pred_dir / f"{key}.jsonl"
        source = "made by the freeze (scripts/predict.py)"
        if s["kind"] == "e0":
            saved = Path(a.runs_mnt) / E0_RUN / E0_RUN / f"{Path(s['weights']).stem}_val.jsonl"
            if saved.exists():
                if not path.exists():
                    shutil.copyfile(saved, path)
                source = f"saved by the E0 queue item during training ({saved}, SHA-256 {sha256(saved)})"
        if not path.exists():
            predict(str(w), vids, a.data_root, s["imgsz"], path, coco_e0=s["coco_e0"])
        preds = load_preds(path)
        if set(preds) != set(vids):
            sys.exit(f"{key}: {path} covers {len(preds)} images, not the {len(vids)} of {s['val_split']}/val.txt")
        match = coco_match(preds, labels, vids)
        coco = coco_table(match, s["classes"])
        thr = coco["f1_threshold"]
        gts, dts = kitti_annos(preds, labels, vids)
        kitti = kitti_table(gts, dts)
        safety = {c: kitti_recall_at(gts, dts, c, thr) for c in ("Pedestrian", "Cyclist") if c in s["classes"]}
        freeze["models"][key] = {
            "kind": s["kind"], "weights": str(w), "weights_sha256": sha256(w), "imgsz": s["imgsz"],
            "val_split": s["val_split"], "classes": s["classes"], "threshold": thr,
            "val_predictions": str(path), "val_predictions_sha256": sha256(path), "val_predictions_source": source,
            "val": {"kitti_ap40": kitti, "coco": coco, "prf_at_threshold": prf_at(match, thr, s["classes"]),
                    "safety_recall": safety},
        }
        mod = {c: fmt(v["moderate"]) for c, v in kitti.items()}
        line = (f"{key}: threshold {thr:.3f}; val {s['val_split']} mAP@0.5 {fmt(coco['map50'])} "
                f"(classes {', '.join(coco['classes_averaged'])}); KITTI Moderate AP40 {mod}; "
                f"safety recall {({c: fmt(v['recall']) for c, v in safety.items()})}")
        print(line, flush=True)
        lines.append(line)
    freeze["time_completed"] = now()
    write_json(out / "FREEZE.json", freeze)
    (out / "FREEZE_SUMMARY.txt").write_text("\n".join(
        [f"Freeze written {freeze['time_completed']} at commit {freeze['git_commit']}", *lines]) + "\n")
    print(f"FREEZE written: {out / 'FREEZE.json'}")


if __name__ == "__main__":
    main()
