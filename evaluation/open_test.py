"""The single test opening (PREREG.md sections 5.2-5.5, amendment 1, PREREG_AMENDMENT_4.md sections 3-6).

One script scores E0 and every frozen model on splits/test.txt. It logs the date, the git commit and the
SHA-256 of test.txt, then a STARTED line, and a COMPLETED line at the end. After a crash it may be run again
with the same commit and settings: the earlier attempt's outputs are deleted unread and every attempt stays in
the log. Once COMPLETED exists it refuses to run.

Usage: python evaluation/open_test.py --data-root /root/kitti --label-dir <label_2> --mapping-dir <devkit mapping>
           --runs-mnt /runs --out <dir> [--boot 1000] [--workers 8]
"""
import argparse
import json
import multiprocessing as mp
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from evaluation.kitti_cache import CachedKitti  # noqa: E402
from evaluation.models import HEADLINE, SEEDS, TEST_LIST_SHA256, ensure_weights, now, sha256, write_json  # noqa: E402
from evaluation.scoring import (ci, coco_match, coco_table, drive_groups, failure_analysis,  # noqa: E402
                                failure_tables, kitti_annos, kitti_recall_at, kitti_table, load_labels,
                                load_preds, prf_at)
from kitti.common import DIFFICULTIES, image_to_drive, read_id_list  # noqa: E402
from kitti.runlog import git_commit  # noqa: E402
from kitti.yolo_data import split_ids  # noqa: E402

RANDOM_SPLIT_RUN = "RS-Y11S-640-RND-S0"
BOOT_SEED = 20261020
CLASSES = ["Car", "Pedestrian", "Cyclist"]
# Pre-specified differences (PREREG_AMENDMENT_4.md section 4): (name, minuend, subtrahend, classes averaged)
COMPARISONS = [
    ("fine-tuned YOLO11n minus E0 YOLO11n", "E1-Y11N-640-DD-S0", "E0-Y11N", ["Car", "Pedestrian"]),
    ("fine-tuned YOLO11s minus E0 YOLO11s", HEADLINE, "E0-Y11S", ["Car", "Pedestrian"]),
    ("fine-tuned YOLO11m minus E0 YOLO11m", "E1-Y11M-640-DD-S0", "E0-Y11M", ["Car", "Pedestrian"]),
    ("YOLO11m minus YOLO11s", "E1-Y11M-640-DD-S0", HEADLINE, CLASSES),
    ("YOLO11s minus YOLO11n", HEADLINE, "E1-Y11N-640-DD-S0", CLASSES),
    ("YOLO11s at 960 minus YOLO11s at 640", "E2-Y11S-960-DD-S0", HEADLINE, CLASSES),
    ("random split minus drive split, common test set", RANDOM_SPLIT_RUN, HEADLINE, CLASSES),
]

_CACHES, _GROUPS = {}, {}


def weights_for(group, rng):
    g = _GROUPS[group]
    counts = np.bincount(rng.integers(0, g["n_groups"], g["n_groups"]), minlength=g["n_groups"])
    return counts[g["group_of"]]


def one_replicate(b):
    rng = np.random.default_rng([BOOT_SEED, b])
    w = {grp: weights_for(grp, rng) for grp in _GROUPS}  # test and each validation split resampled independently
    return {f"{part}|{key}": cache.ap40(w[_GROUP_OF[(part, key)]]) for (part, key), cache in _CACHES.items()}


_GROUP_OF = {}


def mean_of(table, classes, diff):
    return float(np.mean([table[c][diff] for c in classes]))


def summarise(point, reps):
    """Percentile 95% intervals for every per-class AP, every comparison, the leakage gaps and their difference."""
    out = {"per_model": {}, "comparisons": {}, "leakage": {}}
    for k, tab in point.items():
        out["per_model"][k] = {c: {d: {"ap": tab[c][d], "ci95": ci([r[k][c][d] for r in reps])} for d in DIFFICULTIES}
                               for c in tab}
    for name, a, b, classes in COMPARISONS:
        ka, kb = f"test|{a}", f"test|{b}"
        res = {}
        for c in classes:
            res[c] = {d: {"diff": point[ka][c][d] - point[kb][c][d],
                          "ci95": ci([r[ka][c][d] - r[kb][c][d] for r in reps])} for d in DIFFICULTIES}
        res["mean of " + ", ".join(classes)] = {
            d: {"diff": mean_of(point[ka], classes, d) - mean_of(point[kb], classes, d),
                "ci95": ci([mean_of(r[ka], classes, d) - mean_of(r[kb], classes, d) for r in reps])}
            for d in DIFFICULTIES}
        out["comparisons"][name] = {"minuend": a, "subtrahend": b, "classes": classes, "results": res}
    gaps = {}
    for key in (RANDOM_SPLIT_RUN, HEADLINE):
        kv, kt = f"val|{key}", f"test|{key}"
        gaps[key] = {c: {d: {"val": point[kv][c][d], "test": point[kt][c][d],
                             "gap": point[kv][c][d] - point[kt][c][d],
                             "ci95": ci([r[kv][c][d] - r[kt][c][d] for r in reps])} for d in DIFFICULTIES}
                     for c in CLASSES}
    rs, dd = (f"val|{RANDOM_SPLIT_RUN}", f"test|{RANDOM_SPLIT_RUN}"), (f"val|{HEADLINE}", f"test|{HEADLINE}")
    gaps["difference of gaps (random minus drive)"] = {
        c: {d: {"diff": (point[rs[0]][c][d] - point[rs[1]][c][d]) - (point[dd[0]][c][d] - point[dd[1]][c][d]),
                "ci95": ci([(r[rs[0]][c][d] - r[rs[1]][c][d]) - (r[dd[0]][c][d] - r[dd[1]][c][d]) for r in reps])}
            for d in DIFFICULTIES} for c in CLASSES}
    out["leakage"] = gaps
    return out


def seeds_summary(models):
    out = {}
    for c in CLASSES:
        out[c] = {}
        for d in DIFFICULTIES:
            v = [models[k]["kitti_ap40"][c][d] for k in SEEDS]
            out[c][d] = {"values": v, "mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1))}
    m50 = [models[k]["coco"]["map50"] for k in SEEDS]
    out["coco_map50"] = {"values": m50, "mean": float(np.mean(m50)), "sd": float(np.std(m50, ddof=1))}
    return out


def table_md(results):
    """A readable summary of the main tables (results.json holds everything)."""
    L = ["# Single test opening: results", "", f"Completed {results['time_completed']} at commit "
         f"{results['git_commit']}; test.txt SHA-256 {results['test_sha256']}; attempt {results['attempt']}.", "",
         "## Table (a): KITTI protocol, AP|R40 on the sealed test set (95% drive-level bootstrap interval)", "",
         "| Model | Threshold | " + " | ".join(f"{c} {d[0].upper()}" for c in CLASSES for d in DIFFICULTIES) + " |",
         "|---|---|" + "---|" * 9]
    boot = results["bootstrap"]["per_model"]
    for k, m in results["models"].items():
        cells = []
        for c in CLASSES:
            for d in DIFFICULTIES:
                if c not in m["classes"]:
                    cells.append("n/a")
                    continue
                v = boot[f"test|{k}"][c][d]
                cells.append(f"{v['ap']:.1f} [{v['ci95'][0]:.1f}, {v['ci95'][1]:.1f}]" if v["ci95"] else f"{v['ap']:.1f}")
        L.append(f"| {k} | {m['threshold']:.3f} | " + " | ".join(cells) + " |")
    L += ["", "## Table (b): Ultralytics COCO-style, four merged classes (test)", "",
          "| Model | mAP@0.5 | mAP@0.5:0.95 | Car AP50 | Pedestrian AP50 | Cyclist AP50 | LargeVehicle AP50 | Classes averaged |",
          "|---|---|---|---|---|---|---|---|"]
    for k, m in results["models"].items():
        pc = m["coco"]["per_class"]
        cell = [f"{pc[c]['ap50']:.3f}" if c in pc and c in m["classes"] else "n/a"
                for c in ("Car", "Pedestrian", "Cyclist", "LargeVehicle")]
        L.append(f"| {k} | {m['coco']['map50']:.3f} | {m['coco']['map50_95']:.3f} | " + " | ".join(cell)
                 + f" | {', '.join(m['coco']['classes_averaged'])} |")
    L += ["", "## Precision, recall and F1 at the frozen threshold (IoU 0.5), and safety recall (KITTI Moderate)", "",
          "| Model | Precision | Recall | F1 | Pedestrian safety recall | Cyclist safety recall |", "|---|---|---|---|---|---|"]
    for k, m in results["models"].items():
        mac = m["prf_at_threshold"]["macro"]
        sr = m["safety_recall"]
        L.append(f"| {k} | {mac['precision']:.3f} | {mac['recall']:.3f} | {mac['f1']:.3f} | "
                 f"{sr['Pedestrian']['recall']:.3f} | " + (f"{sr['Cyclist']['recall']:.3f} |" if "Cyclist" in sr else "n/a |"))
    L += ["", "## Pre-specified comparisons, KITTI Moderate AP|R40 differences (95% interval)", "",
          "| Comparison | Car | Pedestrian | Cyclist | Mean of the classes compared |", "|---|---|---|---|---|"]
    for name, comp in results["bootstrap"]["comparisons"].items():
        r = comp["results"]
        def cell(c):
            if c not in r:
                return "n/a"
            v = r[c]["moderate"]
            return f"{v['diff']:+.1f} [{v['ci95'][0]:+.1f}, {v['ci95'][1]:+.1f}]" if v["ci95"] else f"{v['diff']:+.1f}"
        mean_key = [k for k in r if k.startswith("mean of")][0]
        L.append(f"| {name} | {cell('Car')} | {cell('Pedestrian')} | {cell('Cyclist')} | {cell(mean_key)} |")
    L += ["", "## Leakage: validation minus test, KITTI Moderate AP|R40 (95% interval)", "",
          "| Model | Car | Pedestrian | Cyclist |", "|---|---|---|---|"]
    lk = results["bootstrap"]["leakage"]
    for key in (RANDOM_SPLIT_RUN, HEADLINE):
        cells = [f"{lk[key][c]['moderate']['val']:.1f} - {lk[key][c]['moderate']['test']:.1f} = "
                 f"{lk[key][c]['moderate']['gap']:+.1f} [{lk[key][c]['moderate']['ci95'][0]:+.1f}, "
                 f"{lk[key][c]['moderate']['ci95'][1]:+.1f}]" for c in CLASSES]
        L.append(f"| {key} | " + " | ".join(cells) + " |")
    dg = lk["difference of gaps (random minus drive)"]
    L.append("| Difference of gaps | " + " | ".join(
        f"{dg[c]['moderate']['diff']:+.1f} [{dg[c]['moderate']['ci95'][0]:+.1f}, {dg[c]['moderate']['ci95'][1]:+.1f}]"
        for c in CLASSES) + " |")
    s = results["seeds"]
    L += ["", "## Seeds (YOLO11s, 640, drive split): mean and SD across seeds 0, 1 and 2", "",
          "| Class | Easy | Moderate | Hard |", "|---|---|---|---|"]
    for c in CLASSES:
        L.append(f"| {c} | " + " | ".join(f"{s[c][d]['mean']:.1f} +/- {s[c][d]['sd']:.1f}" for d in DIFFICULTIES) + " |")
    L.append(f"| COCO-style mAP@0.5 | {s['coco_map50']['mean']:.3f} +/- {s['coco_map50']['sd']:.3f} | | |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--label-dir", required=True)
    ap.add_argument("--mapping-dir", required=True)
    ap.add_argument("--runs-mnt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    from predict import predict

    out = Path(a.out)
    test_dir = out / "test"
    log = test_dir / "TEST_OPENING.log"
    freeze_path = out / "FREEZE.json"
    if not freeze_path.exists():
        sys.exit("No FREEZE.json: the freeze comes before the test opening (PREREG.md 5.1).")
    if (test_dir / "COMPLETED").exists():
        sys.exit("The test set has already been opened (test/COMPLETED exists). It is opened once.")
    freeze = json.loads(freeze_path.read_text())

    # An attempt that crashed before COMPLETED: its outputs are deleted unread; the log keeps every attempt.
    test_dir.mkdir(parents=True, exist_ok=True)
    for p in test_dir.iterdir():
        if p.name != log.name:
            shutil.rmtree(p) if p.is_dir() else p.unlink()
    attempt = 1 + (sum(" STARTED " in ln for ln in log.read_text().splitlines()) if log.exists() else 0)

    test_list = REPO / "splits" / "test.txt"
    test_sha = sha256(test_list)
    if test_sha != TEST_LIST_SHA256:
        sys.exit(f"splits/test.txt has SHA-256 {test_sha}, not the pre-registered {TEST_LIST_SHA256}")
    for key, m in freeze["models"].items():
        got = sha256(ensure_weights({"kind": m["kind"], "weights": m["weights"]}))
        if got != m["weights_sha256"]:
            sys.exit(f"{key}: weights changed since the freeze ({got[:12]} != {m['weights_sha256'][:12]})")
    commit = git_commit()
    with open(log, "a") as f:
        f.write(f"{now()} STARTED attempt {attempt}; git commit {commit}; splits/test.txt SHA-256 {test_sha}; "
                f"freeze {freeze.get('time_completed')}\n")

    ids_test = read_id_list(test_list)
    ids_val = {s: split_ids(s, "val") for s in ("drive_disjoint", "random")}
    labels = load_labels(a.label_dir, sorted(set(ids_test) | set(ids_val["drive_disjoint"]) | set(ids_val["random"])))
    drive_of = image_to_drive(a.mapping_dir)

    pred_dir = test_dir / "pred"
    results = {"time_started": now(), "attempt": attempt, "git_commit": commit, "test_sha256": test_sha,
               "n_test_images": len(ids_test), "freeze_time": freeze.get("time_completed"), "models": {}}
    point = {}
    for key, m in freeze["models"].items():
        path = pred_dir / f"{key}.jsonl"
        predict(m["weights"], ids_test, a.data_root, m["imgsz"], path, coco_e0=(m["kind"] == "e0"))
        preds = load_preds(path)
        thr = m["threshold"]
        match = coco_match(preds, labels, ids_test)
        gts, dts = kitti_annos(preds, labels, ids_test)
        kitti = kitti_table(gts, dts)
        cache = CachedKitti(gts, dts)
        cached = cache.ap40()
        worst = max(abs(cached[c][d] - kitti[c][d]) for c in kitti for d in DIFFICULTIES)
        if worst > 1e-6:
            sys.exit(f"{key}: cached AP differs from the evaluator by {worst}; the bootstrap would be wrong")
        _CACHES[("test", key)] = cache
        point[f"test|{key}"] = kitti
        res = {"classes": m["classes"], "threshold": thr, "kitti_ap40": kitti,
               "coco": coco_table(match, m["classes"]), "prf_at_threshold": prf_at(match, thr, m["classes"]),
               "safety_recall": {c: kitti_recall_at(gts, dts, c, thr) for c in ("Pedestrian", "Cyclist")
                                 if c in m["classes"]},
               "predictions_sha256": sha256(path)}
        if m["kind"] == "trained":
            res["failures"] = failure_tables(failure_analysis(preds, labels, ids_test, thr))
        results["models"][key] = res
        print(f"{key}: KITTI Moderate {({c: round(kitti[c]['moderate'], 2) for c in kitti})}; "
              f"COCO mAP@0.5 {res['coco']['map50']:.3f}", flush=True)

    for key in (RANDOM_SPLIT_RUN, HEADLINE):  # leakage: validation scores from the frozen validation predictions
        m = freeze["models"][key]
        vids = ids_val[m["val_split"]]
        gts, dts = kitti_annos(load_preds(m["val_predictions"]), labels, vids)
        _CACHES[("val", key)] = CachedKitti(gts, dts)
        point[f"val|{key}"] = _CACHES[("val", key)].ap40()

    for name, ids in (("test", ids_test), ("val_drive_disjoint", ids_val["drive_disjoint"]),
                      ("val_random", ids_val["random"])):
        groups = drive_groups(ids, drive_of)
        pos = {i: k for k, i in enumerate(ids)}
        group_of = np.zeros(len(ids), np.int64)
        for g, members in enumerate(groups):
            group_of[[pos[i] for i in members]] = g
        _GROUPS[name] = {"n_groups": len(groups), "group_of": group_of}
    for (part, key) in _CACHES:
        _GROUP_OF[(part, key)] = "test" if part == "test" else f"val_{freeze['models'][key]['val_split']}"

    with mp.get_context("fork").Pool(a.workers) as pool:
        reps = pool.map(one_replicate, range(a.boot), chunksize=max(1, a.boot // (4 * a.workers)))
    results["bootstrap"] = {"replicates": a.boot, "seed": BOOT_SEED, "method": "drive-level, percentile 95%",
                            "drives": {k: v["n_groups"] for k, v in _GROUPS.items()}, **summarise(point, reps)}
    results["seeds"] = seeds_summary(results["models"])
    results["time_completed"] = now()
    write_json(test_dir / "results.json", results)
    (test_dir / "results.md").write_text(table_md(results), encoding="utf-8")
    with open(log, "a") as f:
        f.write(f"{results['time_completed']} COMPLETED attempt {attempt}\n")
    (test_dir / "COMPLETED").write_text(results["time_completed"] + "\n")
    print((test_dir / "results.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
