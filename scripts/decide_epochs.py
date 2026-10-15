"""Record the one epoch decision (PREREG.md section 3.3). Writes runs_root/EPOCH_DECISION.json once; never overwrites.

Usage: python scripts/decide_epochs.py --runs-root R --mode two-runtime|single
           --gpu-a "NVIDIA L4" [--gpu-b "NVIDIA L4"] --cu-balance 123 [--note "..."]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configs.recipe import QUEUES  # noqa: E402
from kitti.runlog import git_commit, now  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--runs-root", required=True)
ap.add_argument("--mode", choices=["two-runtime", "single"], required=True)
ap.add_argument("--gpu-a", required=True)
ap.add_argument("--gpu-b", default=None)
ap.add_argument("--cu-balance", type=float, required=True)
ap.add_argument("--note", default="")
a = ap.parse_args()

out = Path(a.runs_root) / "EPOCH_DECISION.json"
if out.exists():
    sys.exit(f"{out} exists; the epoch decision is made once:\n{out.read_text()}")

if a.mode == "two-runtime":
    ok = a.gpu_b is not None and "L4" in a.gpu_a and "L4" in a.gpu_b and a.cu_balance >= 30
    if not ok:
        sys.exit("Rule not met for two runtimes (both L4 and >= ~30 CU). Use --mode single.")
    epochs, queues = 100, {"A": QUEUES["A"], "B": QUEUES["B"]}
else:
    epochs, queues = 50, {"single": QUEUES["single"]}

smoke = Path(a.runs_root) / "smoke" / "smoke_projection.json"
rec = {"time": now(), "git_commit": git_commit(), "epochs": epochs, "mode": a.mode, "queues": queues,
       "gpu_a": a.gpu_a, "gpu_b": a.gpu_b, "cu_balance": a.cu_balance, "note": a.note,
       "smoke_projection": json.loads(smoke.read_text()) if smoke.exists() else None}
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(rec, indent=2) + "\n")
print(json.dumps(rec, indent=2))
