"""Build the frozen drive-disjoint split and the random-split control. Run ONCE; never re-roll.

Procedure (PREREG.md section 2):
  1. Map every image to its raw drive through train_rand.txt -> train_mapping.txt.
  2. For candidate k = 0 .. N_CANDIDATES-1, shuffle the sorted drive list with one
     random.Random(SEED) stream, then assign drives in that order, each to the partition
     whose image count is furthest below its target (60/20/20 of 7,481). Ties go to
     train, then val, then test.
  3. Revision 2 (attempt 1 found no feasible candidate): improve each start by deterministic
     best-improvement local search. A step moves one drive to another partition or swaps two
     drives in different partitions; it is taken only if it lowers the key
     (image-share excess over IMAGE_TOL_PP, balance error, sum of squared deviations).
     Ties between steps go to the first in sorted-drive order. Stop when no step helps.
  4. A candidate is feasible if every partition's image share is within IMAGE_TOL_PP of
     its target, and every partition's Car, Pedestrian and Cyclist instance shares are
     within BALANCE_TOL_PP of that partition's image share.
  5. Choose the feasible candidate with the smallest balance error; ties go to the smaller
     image-share error, then the lower candidate index.
  6. Random-split control: the chosen train+val pool, shuffled by random.Random(SEED),
     cut into the same train and val sizes. It shares the drive-disjoint test file.

Usage: python -I scripts/make_split.py <kitti label_2 dir> <devkit mapping dir> <out dir>
"""
import hashlib
import json
import platform
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kitti.common import N_IMAGES, image_to_drive, is_moderate, read_all_labels, write_id_list  # noqa: E402

SEED = 20261015
N_CANDIDATES = 50  # revision 2; attempt 1 used 10,000 greedy-only candidates and none was feasible
PARTS = ("train", "val", "test")
TARGET = {"train": 0.60, "val": 0.20, "test": 0.20}
IMAGE_TOL_PP = 0.5
BALANCE_TOL_PP = 1.2
BALANCE_CLASSES = ("Car", "Pedestrian", "Cyclist")
WATCH_IDS = ("006500", "007250")  # scored once by a stock COCO model before the protocol; disclose in Ch3


def evaluate(assign, drive_imgs, drive_cls, totals):
    imgs = Counter()
    cls = {p: Counter() for p in PARTS}
    for d, p in assign.items():
        imgs[p] += drive_imgs[d]
        cls[p].update(drive_cls[d])
    img_err = max(abs(imgs[p] / N_IMAGES - TARGET[p]) for p in PARTS) * 100
    bal = max(abs(cls[p][c] / totals[c] - imgs[p] / N_IMAGES)
              for p in PARTS for c in BALANCE_CLASSES) * 100
    return img_err, bal


def local_search(assign, drives, drive_imgs, drive_cls, totals):
    """Best-improvement moves and swaps on (image excess, balance error, sum of squared deviations)."""
    pi = {p: j for j, p in enumerate(PARTS)}
    part = [pi[assign[d]] for d in drives]
    n = [drive_imgs[d] for d in drives]
    c = [[drive_cls[d][k] for k in BALANCE_CLASSES] for d in drives]
    tot = [totals[k] for k in BALANCE_CLASSES]
    tgt = [TARGET[p] for p in PARTS]
    img = [0, 0, 0]
    cls = [[0, 0, 0] for _ in PARTS]
    for i, p in enumerate(part):
        img[p] += n[i]
        for k in range(3):
            cls[p][k] += c[i][k]

    def key(img, cls):
        excess = max(0.0, max(abs(img[p] / N_IMAGES - tgt[p]) for p in range(3)) * 100 - IMAGE_TOL_PP)
        devs = [cls[p][k] / tot[k] - img[p] / N_IMAGES for p in range(3) for k in range(3)]
        return (excess, max(abs(v) for v in devs) * 100, sum(v * v for v in devs))

    def shifted(moves):  # moves: list of (drive index, from, to)
        im = img[:]
        cl = [row[:] for row in cls]
        for i, a, b in moves:
            im[a] -= n[i]; im[b] += n[i]
            for k in range(3):
                cl[a][k] -= c[i][k]; cl[b][k] += c[i][k]
        return key(im, cl)

    current = key(img, cls)
    while True:
        best, best_moves = current, None
        for i in range(len(drives)):
            for q in range(3):
                if q != part[i]:
                    mv = [(i, part[i], q)]
                    s = shifted(mv)
                    if s < best:
                        best, best_moves = s, mv
        for i in range(len(drives)):
            for j in range(i + 1, len(drives)):
                if part[i] != part[j]:
                    mv = [(i, part[i], part[j]), (j, part[j], part[i])]
                    s = shifted(mv)
                    if s < best:
                        best, best_moves = s, mv
        if best_moves is None:
            break
        for i, a, b in best_moves:
            part[i] = b
            img[a] -= n[i]; img[b] += n[i]
            for k in range(3):
                cls[a][k] -= c[i][k]; cls[b][k] += c[i][k]
        current = best
    return {d: PARTS[part[i]] for i, d in enumerate(drives)}


def main(label_dir, mapping_dir, out_dir):
    out = Path(out_dir)
    if (out / "SHA256SUMS").exists():
        sys.exit(f"{out}/SHA256SUMS exists: the split is frozen and must not be regenerated.")

    labels = read_all_labels(label_dir)
    drive_of = image_to_drive(mapping_dir)
    drives = sorted(set(drive_of.values()))
    drive_imgs = Counter(drive_of.values())
    drive_cls = {d: Counter() for d in drives}
    for i, objs in labels.items():
        drive_cls[drive_of[i]].update(o["name"] for o in objs)
    totals = Counter()
    for c in drive_cls.values():
        totals.update(c)

    rng = random.Random(SEED)
    target_n = {p: TARGET[p] * N_IMAGES for p in PARTS}
    candidates = []
    for k in range(N_CANDIDATES):
        order = drives[:]
        rng.shuffle(order)
        count = dict.fromkeys(PARTS, 0)
        assign = {}
        for d in order:
            p = max(PARTS, key=lambda q: (target_n[q] - count[q], -PARTS.index(q)))
            assign[d] = p
            count[p] += drive_imgs[d]
        assign = local_search(assign, drives, drive_imgs, drive_cls, totals)
        img_err, bal = evaluate(assign, drive_imgs, drive_cls, totals)
        candidates.append((bal, img_err, k, assign))

    feasible = [c for c in candidates if c[1] <= IMAGE_TOL_PP and c[0] <= BALANCE_TOL_PP]
    if not feasible:
        sys.exit("No feasible candidate. Do not change the seed; report and revise the procedure in PREREG.md.")
    bal, img_err, k, assign = min(feasible, key=lambda c: (c[0], c[1], c[2]))

    split = {p: sorted(i for i in labels if assign[drive_of[i]] == p) for p in PARTS}
    pool = split["train"] + split["val"]
    shuffled = pool[:]
    random.Random(SEED).shuffle(shuffled)
    rnd = {"train": shuffled[:len(split["train"])], "val": shuffled[len(split["train"]):]}

    files = {
        "drive_disjoint/train.txt": split["train"],
        "drive_disjoint/val.txt": split["val"],
        "test.txt": split["test"],
        "random/train.txt": rnd["train"],
        "random/val.txt": rnd["val"],
    }
    for rel, ids in files.items():
        write_id_list(out / rel, ids)
    sums = {rel: hashlib.sha256((out / rel).read_bytes()).hexdigest() for rel in files}

    def stats(ids):
        c = Counter(o["name"] for i in ids for o in labels[i])
        mod_cyc = sum(1 for i in ids for o in labels[i] if o["name"] == "Cyclist" and is_moderate(o))
        return {"images": len(ids), "drives": len({drive_of[i] for i in ids}),
                "image_share_pct": round(100 * len(ids) / N_IMAGES, 2),
                **{c_: c[c_] for c_ in BALANCE_CLASSES},
                **{f"{c_}_share_pct": round(100 * c[c_] / totals[c_], 2) for c_ in BALANCE_CLASSES},
                "moderate_cyclists": mod_cyc}

    report = {
        "seed": SEED, "n_candidates": N_CANDIDATES, "n_feasible": len(feasible),
        "chosen_candidate": k, "balance_error_pp": round(bal, 3), "image_share_error_pp": round(img_err, 3),
        "python": platform.python_version(), "n_drives": len(drives),
        "drive_disjoint": {p: stats(split[p]) for p in PARTS},
        "random_control": {p: stats(rnd[p]) for p in ("train", "val")},
        "random_val_drives_shared_with_random_train": len({drive_of[i] for i in rnd["val"]}
                                                          & {drive_of[i] for i in rnd["train"]}),
        "watch_ids": {i: next(p for p in PARTS if i in split[p]) for i in WATCH_IDS},
        "drives": {p: sorted({drive_of[i] for i in split[p]}) for p in PARTS},
        "sha256": sums,
    }
    (out / "split_report.json").write_text(json.dumps(report, indent=2) + "\n")
    with open(out / "SHA256SUMS", "w", newline="\n") as f:
        f.write("".join(f"{h}  {rel}\n" for rel, h in sums.items()))
    print(json.dumps({k_: v for k_, v in report.items() if k_ != "drives"}, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:4])
