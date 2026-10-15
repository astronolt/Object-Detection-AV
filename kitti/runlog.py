"""Environment capture for every run folder (PREREG.md section 7)."""
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=120).stdout
    except Exception as e:  # nvidia-smi missing, etc.
        return f"ERROR: {e}\n"


def git_commit():
    head = sh(["git", "-C", str(REPO), "rev-parse", "HEAD"]).strip()
    dirty = bool(sh(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"]).strip())
    return head + ("-dirty" if dirty else "")


def log_env(folder, record):
    """Write pip freeze, nvidia-smi and a config record into folder/env/<timestamp>/ (one per start or resume)."""
    import torch
    import ultralytics
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    d = Path(folder) / "env" / stamp
    d.mkdir(parents=True, exist_ok=True)
    (d / "pip_freeze.txt").write_text(sh([sys.executable, "-m", "pip", "freeze"]))
    (d / "nvidia_smi.txt").write_text(sh(["nvidia-smi"]))
    rec = {"time": now(), "git_commit": git_commit(), "python": sys.version.split()[0],
           "torch": torch.__version__, "cuda": torch.version.cuda,
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "ultralytics": ultralytics.__version__, **record}
    (d / "run_config.json").write_text(json.dumps(rec, indent=2, default=str) + "\n")
    return rec


def log_line(path, msg):
    line = f"{now()} {msg}"
    print(line, flush=True)
    with open(path, "a") as f:
        f.write(line + "\n")
