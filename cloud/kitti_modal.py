"""Run the pre-registered KITTI pipeline on Modal NVIDIA L4 GPUs (PREREG_AMENDMENT_2.md, PREREG_AMENDMENT_3.md).

Replaces colab/kitti_queue.ipynb. Every container clones the code from code.bundle at a git tag and runs
the existing scripts. It refuses to run unless this file is the copy at that tag and scripts/, kitti/,
configs/ and splits/ there are those of prereg-v1, apart from scripts/predict.py, which must be the copy
amendment 3 records. Storage:

    volume kitti-msc            raw/        the official KITTI zips, downloaded once
                                day1/       zip hashes, round-trip reports, overlay sheets
                                runs/       smoke test, EPOCH_DECISION.json, launch.log, CUTOFF, calls.json,
                                            stops.json, and, after ::collect, every run in PREREG's layout
                                rehearsal/  one folder per rehearsal, never used for results
    volume kitti-msc-<run id>   one per run, so each has a single writer: the run's queue log and folder

Steps, from the project folder in Git Bash. prepare, smoke, rehearse and launch refuse to run without
--detach: without it, Modal stops everything a step started as soon as the step returns or the window closes.

    python -m modal run --detach cloud/kitti_modal.py::prepare           download, hash check, round-trip gate
    python -m modal volume get --force kitti-msc day1 "G:/My Drive/msc"  copy day1 for review
    python -m modal run --detach cloud/kitti_modal.py::smoke             smoke test, timings, projected cost
    python -m modal run --detach cloud/kitti_modal.py::rehearse          rehearsal, part 1 (about 6 minutes)
    python -m modal run --detach cloud/kitti_modal.py::rehearse --resume rehearsal, part 2: run it straight after
    python -m modal run --detach cloud/kitti_modal.py::launch            the decision, then the runs
    python -m modal run cloud/kitti_modal.py::status                     progress at any time (--stage rehearsal)
    python -m modal run --detach cloud/kitti_modal.py::collect           gather the runs into kitti-msc/runs
    python -m modal volume get --force kitti-msc runs "G:/My Drive/msc"  download the results

The headline run trains first; the other runs and E0 start together once it has completed and passed the
halt rule. Running launch again is safe: it refuses while another launch, a rehearsal or a rehearsal's runs
are still going; otherwise it reuses EPOCH_DECISION.json, watches runs still going, skips completed and cut
runs, and resumes the rest.

The bundle is read from G:/My Drive/msc/code.bundle; set KITTI_BUNDLE to use another path.
"""
import hashlib
import json
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import modal

TAG = "prereg-v1.3"
BUNDLE = os.environ.get("KITTI_BUNDLE", "G:/My Drive/msc/code.bundle")

# The runs of configs/recipe.py, in its order; checked against the tagged recipe whenever a launch starts.
RUN_IDS = ["E1-Y11S-640-DD-S0", "E0-COCO-640-VAL", "RS-Y11S-640-RND-S0", "E1-Y11N-640-DD-S0",
           "E1-Y11M-640-DD-S0", "E2-Y11S-960-DD-S0", "E1-Y11S-640-DD-S1", "E1-Y11S-640-DD-S2"]
HEADLINE = "E1-Y11S-640-DD-S0"
E0 = "E0-COCO-640-VAL"
# Rehearsal: the headline and a long queue start together. Part 2, from a new app, re-attaches to both, stops the
# long queue (a test of stopping a run) and watches the headline to its end: after a HALT, E0 (the second phase)
# must not start; otherwise it must start once the headline has completed.
REHEARSAL_LONG = ["E1-Y11N-640-DD-S0", "E1-Y11M-640-DD-S0", "E2-Y11S-960-DD-S0", "RS-Y11S-640-RND-S0",
                  "E1-Y11S-640-DD-S1", "E1-Y11S-640-DD-S2"]
REHEARSAL_EPOCHS = 1
REHEARSAL_PART1_MINUTES = 4  # part 1 stops watching after this; part 2 (--resume) re-attaches from a new app
REHEARSAL_TIMEOUT_HOURS = 2  # each rehearsal container stops after this and is not retried
REHEARSAL_STOP = "cancelled: the rehearsal's test of stopping a run"
STAGE_RE = re.compile(r"^(launch|rehearsal-\d{8}-\d{6})$")
LAUNCH_LOCK = "launch"  # one lock for launches and rehearsals, so they never overlap
FROZEN_TAG = "prereg-v1"  # PREREG_AMENDMENT_2 section 1: these folders are unchanged since this tag
FROZEN_PATHS = ("scripts", "kitti", "configs", "splits")
# PREREG_AMENDMENT_3: the one file changed since FROZEN_TAG, pinned by the SHA-256 of its LF-normalised text
AMENDED = {"scripts/predict.py": "50807d48cd7fbe2c63e9ce531327acd3105a95f7ea1af5b47b23d1004b806da8"}

VOL = Path("/vol")  # the shared volume
RUN = Path("/runvol")  # a training container's own run volume
RUNS_MNT = Path("/runs")  # every run volume, in the launch, status and collect containers
CODE = Path("/root/code")
BUNDLE_IN_CONTAINER = "/root/code.bundle"
RAW_LOCAL = Path("/root/kitti_raw")
DATA = Path("/root/kitti")
CONSOLE = Path("/root/console.log")  # local while it grows; copied to a volume every minute

KITTI_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/"
ZIPS = ["data_object_image_2.zip", "data_object_label_2.zip", "devkit_object.zip"]
EXPECTED_SHA256 = {  # PREREG.md section 2.2: the zips the split was built from
    "data_object_label_2.zip": "4efc76220d867e1c31bb980bbf8cbc02599f02a9cb4350effa98dbb04aaed880",
    "devkit_object.zip": "ce0b76b69c0c5f89690a0d65b7302bbbdb962a0c7e8aba6efc7050d1b04b4cf1",
}

# One container type for every GPU step: the smoke test, the runs, E0, and later the test opening and latency.
GPU = "L4"
CPU = 4.0  # physical cores (8 vCPUs), requested and also the limit, so timings and the bill use the same 4 cores
MEMORY = 20480  # MiB requested; Modal bills the larger of this and actual use
EPOCHS = 100  # PREREG_AMENDMENT_2.md section 3

# Modal list prices on 19 Oct 2026 (modal.com/pricing), US$ per second.
RATE_GPU, RATE_CORE, RATE_GIB = 0.000222, 0.0000131, 0.00000222
RATE_PER_HOUR = 3600 * (RATE_GPU + CPU * RATE_CORE + MEMORY / 1024 * RATE_GIB)
ORCH_CPU, ORCH_MEMORY = 0.25, 512  # the launch container, billed at 3x as non-preemptible
ORCH_RATE_PER_HOUR = 3 * 3600 * (ORCH_CPU * RATE_CORE + ORCH_MEMORY / 1024 * RATE_GIB)
BUDGET_USD = 25.0  # section 3: cap on the projected cost of all runs when the decision is made
LIVE_CAP_USD = 26.0  # section 3: cap on spent plus still-needed cost while the runs train
MOSAIC_ALLOWANCE = 1.2  # the 1-3 epoch smoke runs train without mosaic; full runs use it for 90 epochs
SETUP_HOURS = 0.15  # per run: unzip, label conversion, image cache, final validation
E0_HOURS = 0.25
STARTUP_HOURS = 2 / 60  # per container: start-up before the run's own clock starts
CUTOFF_HOURS = 24  # amendment 1
CUTOFF_GRACE_MINUTES = 15  # until a first container's own START line appears, it stops at its start + 24 h + this
STALL_HOURS = 1.5  # section 4
WATCH_MAX_HOURS = 23.8  # Modal stops any function at 24 h
MAX_POLL_FAILURES = 10  # consecutive failed status reads before a run counts as failed
HEARTBEAT_SECONDS = 60
STALE_MINUTES = 10  # a heartbeat older than this belongs to a container that has stopped
CLAIM_WAIT_MINUTES = 12
POLL_SECONDS = 60

# Every package in the image, resolved by pip for Python 3.12 on Linux from the releases before 2 Oct 2026 (the
# day after ultralytics 8.4.171 came out), with OpenCV kept on its 4.x line, so a rebuild cannot change them.
PINS = [
    "Jinja2==3.1.6", "MarkupSafe==3.0.3", "PyYAML==6.0.3", "anyio==4.15.1", "certifi==2026.7.22",
    "charset-normalizer==3.5.2", "cloudpickle==3.1.2", "contourpy==1.4.0", "cuda-bindings==13.4.3",
    "cuda-pathfinder==1.8.2", "cuda-toolkit==13.0.3.0", "cycler==0.12.1", "filelock==4.0.9", "fonttools==4.66.1",
    "fsspec==2026.9.0", "h11==0.16.0", "httpcore==1.0.9", "httpx==0.28.1", "idna==3.20", "kiwisolver==1.5.1",
    "matplotlib==3.11.2", "mpmath==1.3.0", "networkx==3.7", "numpy==2.5.3", "nvidia-cublas==13.1.1.3",
    "nvidia-cuda-cupti==13.0.85", "nvidia-cuda-nvrtc==13.0.88", "nvidia-cuda-runtime==13.0.96",
    "nvidia-cudnn-cu13==9.24.0.43", "nvidia-cufft==12.0.0.61", "nvidia-cufile==1.15.1.6",
    "nvidia-curand==10.4.0.35", "nvidia-cusolver==12.0.4.66", "nvidia-cusparse==12.6.3.3",
    "nvidia-cusparselt-cu13==0.8.1", "nvidia-ml-py==13.615.71", "nvidia-nccl-cu13==2.30.7",
    "nvidia-nvjitlink==13.4.92", "nvidia-nvshmem-cu13==3.4.5", "nvidia-nvtx==13.0.85", "opencv-python==4.14.0.94",
    "packaging==26.3", "pillow==12.3.0", "polars-runtime-32==1.44.2", "polars==1.44.2", "psutil==7.2.2",
    "pyparsing==3.3.3", "python-dateutil==2.9.0.post0", "requests==2.34.2", "setuptools==84.0.0", "six==1.17.0",
    "sympy==1.14.0", "torch==2.14.1", "torchvision==0.29.1", "triton==3.8.0", "typing_extensions==4.16.0",
    "ultralytics-platform==0.1.75", "ultralytics-thop==2.2.2", "ultralytics==8.4.171", "urllib3==2.8.0",
]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "unzip", "wget", "libgl1", "libglib2.0-0")
    .pip_install(*PINS)
    .env({"PYTHONUNBUFFERED": "1"})
    .add_local_file(BUNDLE, BUNDLE_IN_CONTAINER)
)
app = modal.App("kitti-msc")
vol = modal.Volume.from_name("kitti-msc", create_if_missing=True)
run_vols = {rid: modal.Volume.from_name(f"kitti-msc-{rid.lower()}", create_if_missing=True) for rid in RUN_IDS}
locks = modal.Dict.from_name("kitti-msc-locks", create_if_missing=True)  # atomic launch and run claims
SHARED = {str(VOL): vol}
ALL_VOLUMES = {str(VOL): vol, **{str(RUNS_MNT / rid): v for rid, v in run_vols.items()}}
GPU_KW = dict(image=image, gpu=GPU, cpu=(CPU, CPU), memory=MEMORY)


class GateFailed(Exception):
    """The label hash or the 1 px round-trip gate failed; retrying cannot help."""


class CutOff(Exception):
    """The run was stopped at the amendment 1 cut-off."""


class WrapperMismatch(Exception):
    """This file is not the copy at the tag the code was checked out from."""


class GPUCheckFailed(Exception):
    """torch cannot use an NVIDIA L4 in this container."""


class RunFailed(RuntimeError):
    """A training container failed. Raised so that Modal retries the run; once its retries are spent, the launch
    reads this as the run's end, never as a network problem."""


class LostLock(RuntimeError):
    """Another launch has taken over the launch lock, so this one stops at once."""


# ---------------------------------------------------------------- time, shell, code, files

def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_time(text):
    return datetime.fromisoformat(text.strip())


def read_time(path):
    """The timestamp in a one-line file such as CUTOFF; None if it is missing, empty or half-written."""
    try:
        return parse_time(Path(path).read_text())
    except (OSError, ValueError):
        return None


def age_minutes(stamp):
    return (datetime.now().astimezone() - parse_time(stamp)).total_seconds() / 60


def sh(cmd, cwd=None):
    cmd = [str(c) for c in cmd]
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def run_logged(cmd, cwd, log_path, deadline=None, check_every=5.0, linger=60.0):
    """Run cmd, passing its output to the Modal log and appending a copy to log_path in which progress-bar
    redraws are collapsed to their last state. `deadline` is an aware datetime, or a function returning one
    (it may change while cmd runs): once it passes, the whole process group is stopped and CutOff is raised.
    Children still holding the output open `linger` seconds after cmd exits are killed. Raises
    CalledProcessError if cmd fails."""
    cmd = [str(c) for c in cmd]
    print("$", " ".join(cmd), flush=True)
    deadline_now = deadline if callable(deadline) else (lambda: deadline)

    def clean(raw):
        return raw.rstrip(b"\r").decode("utf-8", "replace").split("\r")[-1].replace("\x1b[K", "") + "\n"

    fired = threading.Event()
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"=== {now()}\n$ {' '.join(cmd)}\n")
        log.flush()
        proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                start_new_session=True)
        finished = threading.Event()

        def signal_group(sig):
            try:
                os.killpg(proc.pid, sig)
            except (ProcessLookupError, PermissionError):
                pass

        def guard():
            while not finished.wait(check_every):
                due = deadline_now()
                if due is not None and datetime.now().astimezone() >= due:
                    fired.set()
                    signal_group(signal.SIGTERM)
                    if not finished.wait(30):
                        signal_group(signal.SIGKILL)
                    return

        def reap():
            proc.wait()
            if not finished.wait(linger):  # cmd has exited, but a leftover child still holds its output open
                signal_group(signal.SIGKILL)

        if deadline is not None:
            threading.Thread(target=guard, daemon=True).start()
        threading.Thread(target=reap, daemon=True).start()
        buf = b""
        while chunk := proc.stdout.read1(65536):
            sys.stdout.buffer.write(chunk)
            sys.stdout.flush()
            *lines, buf = (buf + chunk).split(b"\n")
            cr = buf.rfind(b"\r")
            if 0 < cr < len(buf) - 1:
                buf = buf[cr:]  # keep only the latest redraw of an unfinished line
            log.writelines(clean(raw) for raw in lines)
            log.flush()
        if buf:
            log.write(clean(buf))
        rc = proc.wait()
        finished.set()
    if fired.is_set():
        raise CutOff(f"stopped at the amendment 1 cut-off {deadline_now().isoformat(timespec='seconds')}")
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def checkout(tag, strict=True):
    """Clone the bundle and check out the tag, exactly as cell 3 of the Colab notebook did. Unless strict is False,
    refuse (WrapperMismatch) if scripts/, kitti/, configs/ or splits/ at the tag differ from prereg-v1 other than
    by the amended files of AMENDED."""
    if not (CODE / ".git").exists():
        sh(["git", "clone", "-q", BUNDLE_IN_CONTAINER, CODE])
    sh(["git", "-C", CODE, "checkout", "-q", tag])
    head = subprocess.run(["git", "-C", str(CODE), "rev-parse", "HEAD"], capture_output=True, text=True,
                          check=True).stdout.strip()
    print(f"code at {tag}: {head}", flush=True)
    problem = frozen_problem(tag)
    if problem and strict:
        raise WrapperMismatch(problem)
    return head


def frozen_problem(tag):
    """None if scripts/, kitti/, configs/ and splits/ at the checked-out tag are those of prereg-v1, except that
    each file of AMENDED must be the amended copy (PREREG_AMENDMENT_3)."""
    skip = [f":(exclude){rel}" for rel in AMENDED]
    r = subprocess.run(["git", "-C", str(CODE), "diff", "--quiet", FROZEN_TAG, "HEAD", "--", *FROZEN_PATHS, *skip],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return (f"{', '.join(p + '/' for p in FROZEN_PATHS)} at {tag} are not those of {FROZEN_TAG} (git diff exit "
                f"code {r.returncode}{': ' + r.stderr.strip() if r.stderr.strip() else ''}); amendments 2 and 3 say "
                f"they do not change, apart from {', '.join(AMENDED)}")
    for rel, want in AMENDED.items():
        f = CODE / rel
        got = wrapper_sha256(f) if f.exists() else "missing"
        if got != want:
            return f"{rel} at {tag} ({got[:12]}) is not the copy amendment 3 records ({want[:12]})"
    return None


def wrapper_sha256(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def check_wrapper(code=None, here=None):
    """Modal runs the laptop's copy of this file; only scripts/, kitti/ and configs/ come from the tag. Refuse
    unless this file is the tagged copy, so everything that runs is the tagged code. Returns its SHA-256."""
    tagged = Path(code or CODE) / "cloud" / "kitti_modal.py"
    mine = wrapper_sha256(here or __file__)
    if not tagged.exists():
        raise WrapperMismatch("the tagged code has no cloud/kitti_modal.py")
    theirs = wrapper_sha256(tagged)
    if mine != theirs:
        raise WrapperMismatch(f"cloud/kitti_modal.py on this laptop ({mine[:12]}) is not the copy at the tag "
                              f"({theirs[:12]}): commit and tag the change first, or restore the tagged file")
    return mine


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json_atomic(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str) + "\n")
    tmp.replace(path)


def write_text_atomic(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def check_stage(stage):
    if not STAGE_RE.match(stage or ""):
        raise ValueError(f"stage must be 'launch' or 'rehearsal-YYYYMMDD-HHMMSS', not {stage!r}")
    return stage


def shared_dir(stage):
    check_stage(stage)
    return VOL / "runs" if stage == "launch" else VOL / "rehearsal" / stage.split("-", 1)[1]


def run_root(rid, stage, base=None):
    """Where a run's queue log and folder live: at the top of its own volume, or in its rehearsal folder."""
    check_stage(stage)
    root = Path(base) if base is not None else RUNS_MNT / rid
    return root if stage == "launch" else root / "rehearsal" / stage.split("-", 1)[1]


def latest_rehearsal():
    found = sorted(p.name for p in (VOL / "rehearsal").glob("*") if p.is_dir()) if (VOL / "rehearsal").exists() else []
    return f"rehearsal-{found[-1]}" if found else None


def data_step_passed():
    """Section 2: nothing runs after a failed data step (hashes or round-trip gate)."""
    s = read_json(VOL / "day1" / "prepare_summary.json")
    if not isinstance(s, dict) or not s.get("gate_passed"):
        raise RuntimeError("The data step has not passed (day1/prepare_summary.json): run ::prepare first")
    for z, want in EXPECTED_SHA256.items():
        if s.get("zip_sha256", {}).get(z) != want:
            raise RuntimeError(f"day1/prepare_summary.json records a different SHA-256 for {z}")
    return s


# ---------------------------------------------------------------- data, GPU, checkpoints

def unpack_and_convert(report_json, console, commit, overlay_dir=None):
    """Unzip the training images and labels to local disk, then run the pre-registered converter.

    convert_labels.py exits with an error unless every label round-trips within 1 px (PREREG section 4), so a
    container that fails the gate never trains. GateFailed only when the label hash differs or the report says
    the gate failed; any other failure of the converter is an ordinary error, which Modal may retry."""
    raw = VOL / "raw"
    got = sha256(raw / "data_object_label_2.zip")
    if got != EXPECTED_SHA256["data_object_label_2.zip"]:
        raise GateFailed(f"data_object_label_2.zip on the volume has SHA-256 {got}, not the pre-registered one")
    RAW_LOCAL.mkdir(parents=True, exist_ok=True)
    sh(["unzip", "-qo", raw / "data_object_image_2.zip", "training/image_2/*", "-d", RAW_LOCAL])
    sh(["unzip", "-qo", raw / "data_object_label_2.zip", "-d", RAW_LOCAL])
    cmd = [sys.executable, "scripts/convert_labels.py", RAW_LOCAL / "training" / "label_2",
           RAW_LOCAL / "training" / "image_2", DATA, report_json]
    if overlay_dir is not None:
        cmd += ["--overlay", "splits", overlay_dir]
    try:
        run_logged(cmd, cwd=CODE, log_path=console)
    except subprocess.CalledProcessError as e:
        rep = read_json(report_json)
        if isinstance(rep, dict) and rep.get("gate_passed") is False:
            raise GateFailed(f"ROUND-TRIP GATE FAILED: {rep.get('n_errors')} errors; see {report_json}") from None
        raise RuntimeError(f"convert_labels.py exited with {e.returncode} without a failed-gate report") from None
    finally:
        commit()


GPU_CHECK = (
    "import json, torch\n"
    "assert torch.cuda.is_available(), 'torch cannot see a CUDA GPU'\n"
    "name = torch.cuda.get_device_name(0)\n"
    "assert 'L4' in name, f'expected an NVIDIA L4, got {name}'\n"
    "x = torch.randn(512, 512, device='cuda')\n"
    "y = torch.nn.functional.conv2d(torch.randn(2, 3, 64, 64, device='cuda'), torch.randn(8, 3, 3, 3, device='cuda'))\n"
    "torch.cuda.synchronize()\n"
    "assert float((x @ x).abs().sum()) > 0 and float(y.abs().sum()) > 0\n"
    "print(json.dumps({'gpu': name, 'torch': torch.__version__, 'cuda': torch.version.cuda,\n"
    "                  'cudnn': torch.backends.cudnn.version()}))\n"
)


def gpu_check():
    """Fail fast unless torch can use an L4. A driver or wheel mismatch would otherwise let Ultralytics train on
    the CPU while the GPU is billed. Runs in its own process, so this one never holds GPU memory."""
    r = subprocess.run([sys.executable, "-c", GPU_CHECK], capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        lines = (r.stderr or r.stdout or "").strip().splitlines()
        raise GPUCheckFailed(lines[-1] if lines else f"exit code {r.returncode}")
    return json.loads(r.stdout.strip().splitlines()[-1])


def memory_used_gib():
    try:
        import psutil
        return round(psutil.virtual_memory().used / 2 ** 30, 2)
    except Exception:  # noqa: BLE001
        return None


def loadable(path, tries=3, wait=10):
    """True if torch can load the checkpoint; a read error is retried before the file counts as torn."""
    import torch
    for i in range(tries):
        try:
            torch.load(path, map_location="cpu", weights_only=False)
            return True
        except Exception:  # noqa: BLE001
            if i < tries - 1:
                time.sleep(wait)
    return False


def rebuild_last_from_best(best, last):
    import torch
    ckpt = torch.load(best, map_location="cpu", weights_only=False)
    ckpt["epoch"] = -1  # what Ultralytics writes once training has finished
    torch.save(ckpt, last)


def guard_checkpoints(rundir, epochs, load_ok=loadable, rebuild=rebuild_last_from_best):
    """A container stopped mid-write can leave last.pt or best.pt unreadable (PREREG_AMENDMENT_2 section 4).

    - Training had finished and only last.pt is unreadable: rebuild it from best.pt, so the run is validated
      rather than retrained.
    - Otherwise: keep last.pt, best.pt and results.csv under a .before-restart-<time> suffix, and the run starts
      again from epoch 1.
    Logged in RESTARTS.log. Returns the message, or None when both files can be read."""
    rundir = Path(rundir)
    last, best = rundir / "weights" / "last.pt", rundir / "weights" / "best.pt"
    bad = [p for p in (last, best) if p.exists() and not load_ok(p)]
    if not bad:
        return None
    stamp = f"{datetime.now():%Y%m%d-%H%M%S}"
    finished = len(unique_epochs(read_results(rundir / "results.csv"))) >= epochs
    if finished and bad == [last] and best.exists():
        last.rename(last.with_name(f"last.pt.unreadable-{stamp}"))
        rebuild(best, last)
        msg = ("training had finished but last.pt could not be read; rebuilt it from best.pt, so the run is "
               "validated, not retrained")
    else:
        kept = []
        for p in (last, best, rundir / "results.csv"):
            if p.exists():
                p.rename(p.with_name(f"{p.name}.before-restart-{stamp}"))
                kept.append(p.name)
        msg = (f"{' and '.join(p.name for p in bad)} could not be read; kept {', '.join(kept)} with the suffix "
               f".before-restart-{stamp}; the run starts again from epoch 1")
    with open(rundir / "RESTARTS.log", "a") as f:
        f.write(f"{now()} {msg}\n")
    return msg


# ---------------------------------------------------------------- results and the decision

def read_results(path):
    """Rows of an Ultralytics results.csv with keys and values stripped; [] if missing or unreadable."""
    import csv
    try:
        with open(path, newline="") as f:
            return [{k.strip(): (v or "").strip() for k, v in r.items() if k} for r in csv.DictReader(f)]
    except (OSError, ValueError, csv.Error):
        return []


def epoch_times(rows):
    """(epoch, time) for every complete row."""
    pts = []
    for r in rows:
        try:
            pts.append((int(float(r["epoch"])), float(r["time"])))
        except (KeyError, ValueError):
            continue
    return pts


def unique_epochs(rows):
    return sorted({e for e, _ in epoch_times(rows)})


def attempt_speed(rows, last_n=10):
    """Seconds per epoch now: the median time between consecutive epochs over the last rows. A resume restarts
    Ultralytics' time column, which gives one negative or tiny gap; the median ignores it. None until there
    are two gaps to go on."""
    pts = epoch_times(rows)[-(last_n + 1):]
    gaps = [t2 - t1 for (e1, t1), (e2, t2) in zip(pts, pts[1:]) if e2 == e1 + 1 and t2 > t1]
    return statistics.median(gaps) if len(gaps) >= 2 else None


def smoke_epoch_seconds(smoke_dir, smoke_projection):
    """Seconds per epoch for each model@imgsz of the smoke test, and where each figure came from: the time column
    of the smoke run's results.csv divided by its epochs (Ultralytics starts that clock after data set-up), or
    the smoke test's own wall-clock figure if the file is missing."""
    sec, source = dict(smoke_projection["sec_per_epoch"]), {}
    for key in sec:
        model, imgsz = key.split("@")
        pts = epoch_times(read_results(Path(smoke_dir) / f"smoke-{Path(model).stem}-{imgsz}" / "results.csv"))
        source[key] = "wall clock (smoke_projection.json)"
        if pts:
            sec[key], source[key] = pts[-1][1] / len(pts), f"results.csv time / {len(pts)} epochs"
    return sec, source


def run_hours(spec, sec_per_epoch, epochs=EPOCHS):
    if spec.get("kind") == "e0":
        return E0_HOURS
    return sec_per_epoch[f"{spec['model']}@{spec['imgsz']}"] * epochs * MOSAIC_ALLOWANCE / 3600 + SETUP_HOURS


def make_decision(head, runs_spec, cut_order, sec_per_epoch, sources=None, smoke_projection=None, wrapper=None):
    """The epoch and budget decision of PREREG_AMENDMENT_2 section 3, from the smoke-test timings only."""
    hours = {rid: run_hours(spec, sec_per_epoch) for rid, spec in runs_spec.items()}

    def cost(ids):
        return sum(hours[r] for r in ids) * RATE_PER_HOUR

    keep, dropped = list(runs_spec), []
    for rid in cut_order:
        if cost(keep) <= BUDGET_USD:
            break
        keep.remove(rid)
        dropped.append(rid)
    return {
        "time": now(), "git_commit": head, "wrapper_sha256": wrapper, "stage": "launch",
        "rule": "PREREG_AMENDMENT_2.md section 3", "platform": "Modal", "gpu": "NVIDIA L4",
        "cpu_cores": CPU, "memory_mib": MEMORY, "mode": "modal-headline-first", "epochs": EPOCHS,
        "queues": {rid: [rid] for rid in keep},  # every run is its own queue
        "phases": [[HEADLINE], [rid for rid in keep if rid != HEADLINE]],  # the rest start once the headline passes
        "dropped_for_budget": dropped,
        "projected_hours": {rid: round(h, 2) for rid, h in hours.items()},
        "projected_cost_usd": round(cost(keep), 2), "budget_usd": BUDGET_USD,
        "within_budget": cost(keep) <= BUDGET_USD, "live_cap_usd": LIVE_CAP_USD,
        "rate_usd_per_hour": round(RATE_PER_HOUR, 4),
        "epoch_seconds": {k: round(v, 2) for k, v in sec_per_epoch.items()}, "epoch_seconds_source": sources,
        "mosaic_allowance": MOSAIC_ALLOWANCE, "setup_hours": SETUP_HOURS, "e0_hours": E0_HOURS,
        "smoke_projection": smoke_projection, "image_pins": PINS,
    }


def make_rehearsal_decision(head, stage, wrapper=None):
    long_q = REHEARSAL_LONG[0]
    return {
        "time": now(), "git_commit": head, "wrapper_sha256": wrapper, "stage": stage,
        "rule": "PREREG_AMENDMENT_2.md section 3: rehearsal of the launch; nothing from it is used",
        "mode": "rehearsal", "epochs": REHEARSAL_EPOCHS,
        "queues": {HEADLINE: [HEADLINE], long_q: list(REHEARSAL_LONG), E0: [E0]},
        "phases": [[HEADLINE, long_q], [E0]],
        "projected_hours": {HEADLINE: 0.25, long_q: 0.5, E0: 0.25}, "image_pins": PINS,
    }


def queue_starts(logs):
    """Timestamps of every 'QUEUE <queue> START' line in the given queue logs (amendment 1 cut-off)."""
    starts = []
    for log in logs:
        try:
            text = Path(log).read_text()
        except OSError:
            continue
        for line in text.splitlines():
            parts = line.split()
            if len(parts) >= 4 and parts[1] == "QUEUE" and parts[3] == "START":
                try:
                    starts.append(parse_time(parts[0]))
                except ValueError:
                    continue
    return starts


def own_deadline(queue_log):
    """The earliest START line in this run's own queue log plus 24 h. It can only be later than the amendment 1
    cut-off (the earliest START line of all runs plus 24 h). None until the run has a START line."""
    starts = queue_starts([queue_log])
    return min(starts) + timedelta(hours=CUTOFF_HOURS) if starts else None


# ---------------------------------------------------------------- stops, attempts, locks, heartbeats

def read_stops(shared):
    s = read_json(Path(shared) / "stops.json")
    return s if isinstance(s, dict) else {}


def record_stop(shared, rid, reason, final, commit=None):
    """Note why the launch stopped a run. A final stop (cut for budget or at the cut-off) is never restarted."""
    stops = read_stops(shared)
    stops[rid] = {"time": now(), "reason": reason, "final": bool(final)}
    write_json_atomic(Path(shared) / "stops.json", stops)
    if commit is not None:
        commit()


def read_attempts(rundir):
    """One record per container that trained this run: call, start, beat, end, result."""
    out = []
    for p in sorted((Path(rundir) / "attempts").glob("*.json")):
        rec = read_json(p)
        if isinstance(rec, dict):
            out.append(rec)
    return out


def attempt_hours(rec):
    try:
        seconds = (parse_time(rec.get("end") or rec["beat"]) - parse_time(rec["start"])).total_seconds()
        return max(seconds, 0.0) / 3600
    except (KeyError, TypeError, ValueError, AttributeError):
        return 0.0


def spent_hours(rundir):
    """Container hours used by a run so far, with each container's start-up added."""
    return sum(attempt_hours(a) + STARTUP_HOURS for a in read_attempts(rundir))


def stale(entry):
    try:
        return age_minutes(entry["beat"]) >= STALE_MINUTES
    except (KeyError, TypeError, ValueError, AttributeError):
        return True


def claim(store, key, me, sleep=time.sleep, wait_minutes=CLAIM_WAIT_MINUTES):
    """Take the lock `key` in the modal.Dict `store` for call `me`. Put-if-absent is atomic; so is replacing a stale
    holder, because only one claimer can create the takeover marker for that holder's last heartbeat. A retry of
    the same call goes straight through; a live holder is waited for up to wait_minutes, then False."""
    waited = 0
    for _ in range(wait_minutes + 10):
        mine = {"call": me, "beat": now()}
        if store.put(key, mine, skip_if_exists=True):
            return True
        cur = store.get(key)
        if cur is None:
            continue  # released in between: try again at once
        if cur.get("call") == me:
            store[key] = mine
            return True
        if stale(cur) and store.put(f"{key}|takeover|{cur.get('call')}|{cur.get('beat')}", me, skip_if_exists=True):
            store[key] = mine
            return True
        if waited >= wait_minutes:
            return False
        sleep(60)
        waited += 1
    return False


def beat_lock(store, key, me):
    """Refresh our heartbeat on the lock. False if another holder has taken it over; never steals it back."""
    cur = store.get(key)
    if cur is None or cur.get("call") == me:
        store[key] = {"call": me, "beat": now()}
        return True
    return False


def lock_live(store, key):
    """True if a container holds the lock with a fresh heartbeat."""
    cur = store.get(key)
    return isinstance(cur, dict) and not stale(cur)


def release(store, key, me):
    try:
        cur = store.get(key)
        if cur is not None and cur.get("call") == me:
            store.pop(key)
    except Exception as e:  # noqa: BLE001
        print(f"could not release {key}: {e!r}", flush=True)


class Heartbeat:
    """Call fn every `every` seconds in a background thread; stop() waits for the thread to finish."""

    def __init__(self, fn, every=HEARTBEAT_SECONDS):
        self._fn, self._stop = fn, threading.Event()
        self._thread = threading.Thread(target=self._loop, args=(every,), daemon=True)
        self._thread.start()

    def _loop(self, every):
        while not self._stop.wait(every):
            try:
                self._fn()
            except Exception as e:  # noqa: BLE001
                print(f"heartbeat failed: {e!r}", flush=True)

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=120)


# ---------------------------------------------------------------- watching the runs

def run_state(rundir, spec, cutoff=None, stop=None):
    """One row of the status table. A run counts as completed only if COMPLETED is time-stamped before the
    cut-off (amendment 1)."""
    rundir = Path(rundir)
    row = {"state": "waiting", "completed": None, "epochs_done": 0, "val_map50": None, "attempts": 0,
           "spent_h": 0.0, "mem_peak_gib": None, "note": ""}
    notes = []
    try:
        if (rundir / "COMPLETED").exists():
            row["completed"] = (rundir / "COMPLETED").read_text().strip()
        atts = read_attempts(rundir)
        row["attempts"], row["spent_h"] = len(atts), round(sum(attempt_hours(a) for a in atts), 2)
        peaks = [a.get("mem_peak_gib") for a in atts if isinstance(a.get("mem_peak_gib"), (int, float))]
        row["mem_peak_gib"] = max(peaks) if peaks else None
        if spec.get("kind") == "e0":
            notes.append(f"{len(list(rundir.glob('*_val.jsonl')))}/3 prediction files")
        else:
            rows = read_results(rundir / "results.csv")
            row["epochs_done"] = len(unique_epochs(rows))
            for r in reversed(rows):
                try:
                    row["val_map50"] = float(r["metrics/mAP50(B)"])
                    break
                except (KeyError, ValueError):
                    continue
        summ = read_json(rundir / "val_summary.json")
        if isinstance(summ, dict) and "val_map50" in summ:
            row["val_map50"] = summ["val_map50"]
        if (rundir / "RESTARTS.log").exists():
            notes.append("restarted (see RESTARTS.log)")
        live = [a for a in atts if not a.get("end") and not stale(a)]
        late = False
        if row["completed"] and cutoff is not None:
            try:
                late = parse_time(row["completed"]) >= cutoff
            except ValueError:
                late = True
        if isinstance(stop, dict) and stop.get("final"):  # a cut run is never used, even if it finished later
            row["state"] = "cut"
            notes.append(stop.get("reason", ""))
        elif late:
            row["state"] = "cut"
            notes.append(f"COMPLETED {row['completed']} is after the cut-off, so it is cut (amendment 1)")
        elif row["completed"]:
            row["state"] = "completed"
        elif (rundir / "CUT").exists():
            row["state"] = "cut"
            notes.append((rundir / "CUT").read_text().strip())
        elif live:
            row["state"] = "training"
            notes.append(f"heartbeat {age_minutes(live[-1]['beat']):.0f} min ago")
        elif atts:
            row["state"] = "stopped"
            last = atts[-1].get("result") or {}
            why = (stop or {}).get("reason") or last.get("failed") or last.get("note") or atts[-1].get("error")
            if why:
                notes.append(str(why))
    except (OSError, ValueError, KeyError) as e:
        notes.append(f"could not read everything ({e!r})")
    row["note"] = "; ".join(n for n in notes if n)
    return row


def progress(rundir):
    """A count that changes whenever a run moves on: results.csv lines, E0 prediction files, the end files."""
    rundir = Path(rundir)
    n = 0
    try:
        res = rundir / "results.csv"
        if res.exists():
            with open(res, "rb") as f:
                n += sum(1 for _ in f)
        n += len(list(rundir.glob("*_val.jsonl")))
        n += (rundir / "val_summary.json").exists() + (rundir / "COMPLETED").exists()
    except OSError:
        pass
    return n


def remaining_hours(rundir, spec, epochs, projected_h, spent_h):
    """Container hours a run still needs: at its observed speed once it has two epoch gaps, plus 0.1 h for its
    final validation; before that, what is left of its projection (at least 0.1 h)."""
    rundir = Path(rundir)
    if (rundir / "COMPLETED").exists():
        return 0.0
    if spec.get("kind") == "e0":
        return max(E0_HOURS - spent_h, 0.05)
    rows = read_results(rundir / "results.csv")
    sec = attempt_speed(rows)
    if sec is None:
        return max(projected_h - spent_h, 0.1)
    return max(epochs - len(unique_epochs(rows)), 0) * sec / 3600 + 0.1


class Monitor:
    """What a launch reads on each poll: run folders, queue logs and HALT on the run volumes, and the cut-off."""

    def __init__(self, decision, specs, roots, shared, reload, commit, started_at=None):
        self.decision, self.specs, self.roots = decision, specs, {r: Path(p) for r, p in roots.items()}
        self.shared, self._reload, self._commit = Path(shared), reload, commit
        self.started_at = started_at

    def reload(self):
        self._reload()

    def rundir(self, rid):
        return self.roots[rid] / rid

    def runs_of(self, q):
        return self.decision["queues"].get(q, [q])

    def halt(self):
        p = self.roots[HEADLINE] / "HALT" if HEADLINE in self.roots else None
        return p.read_text().strip() if p is not None and p.exists() else None

    def done(self, q):
        """A queue is done when the last run in it has COMPLETED."""
        return (self.roots[q] / self.runs_of(q)[-1] / "COMPLETED").exists()

    def progress(self, q):
        return sum(progress(self.roots[q] / r) for r in self.runs_of(q))

    def cutoff(self):
        """24 h after the earliest QUEUE <queue> START line (amendment 1); written to CUTOFF, once, when known."""
        known = read_time(self.shared / "CUTOFF")
        if known is not None:
            return known
        starts = queue_starts(self.roots[r] / f"queue_{r}.log" for r in self.roots)
        if not starts:
            return None
        cut = min(starts) + timedelta(hours=CUTOFF_HOURS)
        write_text_atomic(self.shared / "CUTOFF", cut.isoformat() + "\n")
        self._commit()
        return cut

    def projection_usd(self, pending):
        """US$ spent by every run's containers and by this launch so far, plus what the pending runs still need."""
        spent = {r: spent_hours(self.rundir(r)) for r in self.roots}
        still = sum(remaining_hours(self.rundir(r), self.specs[r], self.decision["epochs"],
                                    self.decision["projected_hours"].get(r, 0.0), spent[r]) for r in pending)
        launch = 0.0
        if self.started_at is not None:
            launch = (datetime.now().astimezone() - self.started_at).total_seconds() / 3600 * ORCH_RATE_PER_HOUR
        return (sum(spent.values()) + still) * RATE_PER_HOUR + launch


CUT_AT_CUTOFF = "cut at the amendment 1 cut-off"


def call_state(call):
    """Ask Modal about a call: ("running", None), ("ended", its result or why it ended) or ("unknown", the error).

    While a call runs, get(timeout=0) raises the built-in TimeoutError. Once it has ended, get() returns its result
    or raises what ended it: the run's own RunFailed, RemoteError after a cancel, InternalFailure,
    FunctionTimeoutError or OutputExpiredError. Only Modal's network and server errors leave the state unknown."""
    import grpclib
    import modal.exception as mx
    try:
        return "ended", call.get(timeout=0)
    except mx.NotFoundError as e:  # a server error too, but a call Modal does not know cannot be running
        return "ended", f"failed: not found ({e!r})"
    except mx.FunctionTimeoutError:
        return "ended", "failed: timed out"
    except mx.OutputExpiredError:
        return "ended", "failed: output expired"
    except (mx.ConnectionError, mx.ClientClosed, mx.TimeoutError, mx.ServiceError, mx.InternalError,
            mx.ResourceExhaustedError, grpclib.GRPCError) as e:
        return "unknown", e
    except TimeoutError:
        return "running", None
    except Exception as e:  # noqa: BLE001
        return "ended", f"failed: {e!r}"


def watch(calls, mon, note, cut_order, sleep=time.sleep, clock=time.time, wall=None, poll_seconds=POLL_SECONDS,
          stall_seconds=STALL_HOURS * 3600, live_cap_usd=LIVE_CAP_USD, max_seconds=WATCH_MAX_HOURS * 3600,
          beat=None, on_stop=None, extra_pending=(), test_stops=None, cancelled=None):
    """Wait for the runs in `calls`.

    - At the cut-off (amendment 1), cut every run still going. On HALT (PREREG section 3.4), cancel every run but
      the headline, which ends by itself once it has written HALT.
    - Stop a run with no progress for stall_seconds.
    - While spent plus remaining cost is over live_cap_usd, cut the next run of the cut order, including runs in
      `extra_pending`, which have not started yet.
    - A run Modal cannot be asked about stays watched; a failed cancel is retried on the next poll.
    - test_stops {run: reason} are stopped, as final, at the first poll (the rehearsal's test of stopping a run).
    on_stop(rid, reason, final) records every stop; `cancelled` collects the cancels that went through.
    Returns {run: outcome}."""
    wall = wall or (lambda: datetime.now().astimezone())
    pending, outcome = dict(calls), {}
    waiting = [r for r in extra_pending if r not in pending]  # later-phase runs, cut here before they start
    t0 = clock()
    seen = {rid: (None, clock()) for rid in pending}  # last progress count and when it changed
    unread = {rid: 0 for rid in pending}  # polls in a row without an answer from Modal
    stopping = {}  # runs being cancelled: reason
    test_stops = dict(test_stops or {})
    over_cap_noted = halt_noted = False
    last_cut = None

    def try_cancel(rid):
        try:
            pending[rid].cancel(terminate_containers=True)
        except Exception as e:  # noqa: BLE001
            note(f"{rid}: cancel failed ({e!r}); trying again next minute")
            return
        outcome[rid] = stopping.pop(rid)
        pending.pop(rid)
        if cancelled is not None:
            cancelled[rid] = outcome[rid]
        note(f"{rid}: {outcome[rid]}")

    def stop(rid, why, final):
        if on_stop is not None:
            try:
                on_stop(rid, why, final)
            except Exception as e:  # noqa: BLE001
                note(f"{rid}: could not record the stop ({e!r})")
        if rid in pending:
            first = rid not in stopping
            stopping[rid] = why
            if first:
                try_cancel(rid)

    def rule(name, fn):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            note(f"checking {name} failed ({e!r}); trying again next minute")
            return None

    while pending:
        if clock() - t0 >= max_seconds:
            note("the launch stops watching (Modal ends any function at 24 h); each run still going stops at its "
                 "own cut-off deadline")
            break
        sleep(poll_seconds)
        if beat is not None:
            try:
                beat()
            except LostLock:
                raise
            except Exception as e:  # noqa: BLE001
                note(f"launch heartbeat failed ({e!r})")
        for rid in list(stopping):
            if rid in pending:
                try_cancel(rid)
        # Results come from Modal, not from the volumes, so they are read even when a reload fails.
        for rid, call in list(pending.items()):
            state, info = call_state(call)
            if state == "running":
                unread[rid] = 0
                continue
            if state == "unknown":
                unread[rid] += 1
                if unread[rid] == 1 or unread[rid] % MAX_POLL_FAILURES == 0:
                    note(f"{rid}: Modal could not be asked about it ({unread[rid]} poll(s) in a row: {info!r}); "
                         "still watching it")
                continue
            outcome[rid] = stopping.pop(rid) if rid in stopping else info
            pending.pop(rid)
            note(f"{rid}: {outcome[rid]}")
        if not pending:
            break
        for rid, why in list(test_stops.items()):
            test_stops.pop(rid)
            if rid in pending:
                note(f"{rid}: stopping it now: {why}")
                stop(rid, why, final=True)
        try:
            mon.reload()
            fresh = True
        except Exception as e:  # noqa: BLE001
            note(f"volume reload failed ({e!r}); only a known cut-off is applied this minute")
            fresh = False
        halt = None
        if fresh:
            halt = rule("HALT", mon.halt)
            if halt:
                others = sorted(r for r in pending if r != HEADLINE and r not in stopping)
                if others or not halt_noted:
                    note(f"HALT found ({halt}); cancelling {others}"
                         + ("; the headline ends by itself" if HEADLINE in pending else ""))
                    halt_noted = True
                for rid in others:
                    stop(rid, "cancelled: HALT", final=False)
            got = rule("the cut-off", mon.cutoff)
            if got is not None:
                last_cut = got
        if last_cut is not None and wall() >= last_cut:
            targets = sorted(r for r in pending if stopping.get(r) != CUT_AT_CUTOFF)
            if targets:
                note(f"amendment 1 cut-off {last_cut.isoformat()} reached; cutting {targets}")
            for rid in targets:
                stop(rid, CUT_AT_CUTOFF, final=True)
            continue
        if not fresh:
            continue

        def stalls():
            for rid in list(pending):
                if rid in stopping:
                    continue
                count = mon.progress(rid)
                if count != seen[rid][0]:
                    seen[rid] = (count, clock())
                elif clock() - seen[rid][1] > stall_seconds:
                    stop(rid, f"stopped: no progress for {stall_seconds / 3600:g} h", final=False)

        rule("stalls", stalls)

        def budget():
            nonlocal over_cap_noted
            active = [r for r in pending if r not in stopping] + waiting
            if live_cap_usd is None or not active:
                return
            projected = mon.projection_usd(sorted(active))
            if projected <= live_cap_usd:
                return
            victim = next((r for r in cut_order if r in active), None)
            why = f"cut for budget: spent plus remaining ${projected:.2f} > ${live_cap_usd:.2f}"
            if victim is None:
                if not over_cap_noted:
                    note(f"spent plus remaining ${projected:.2f} > ${live_cap_usd:.2f}, but only runs outside the "
                         "cut order are left")
                    over_cap_noted = True
            elif victim in waiting:
                waiting.remove(victim)
                outcome[victim] = why
                stop(victim, why, final=True)
                note(f"{victim}: {why} (before it started)")
            else:
                stop(victim, why, final=True)

        if not halt:  # after a HALT nothing more will start, so nothing is cut for budget
            rule("the budget", budget)
    for rid in pending:
        outcome.setdefault(rid, f"{stopping[rid]} (the cancel is not confirmed yet)" if rid in stopping
                           else "still running when the launch stopped watching")
    return outcome


def still_running(call, tries=5, wait=3.0, sleep=time.sleep, live=None):
    """True while Modal still has the call; False once it has ended (completed, failed or cancelled). If Modal
    cannot be reached, a fresh heartbeat on the run's lock (`live`) still counts as running; otherwise raise, so a
    relaunch never starts a second copy of a run it cannot see."""
    last = None
    for i in range(tries):
        state, info = call_state(call)
        if state != "unknown":
            return state == "running"
        last = info
        if i < tries - 1:
            sleep(wait)
    try:
        if live is not None and live():
            return True
    except Exception:  # noqa: BLE001
        pass
    raise RuntimeError(f"cannot tell whether call {getattr(call, 'object_id', '?')} is still running ({last!r}); "
                       "not starting another copy. Try again later.")


def collect_into(roots, dest, stops=None, cutoff=None, dropped=(), at=None):
    """Copy each run's queue log, folder and any HALT from its own volume into dest (the layout of PREREG's runs/
    folder), and mark the cut runs with a CUT file (amendment 1: a run counts as finished only if its COMPLETED is
    time-stamped before the cut-off). Cut: every run the launch cut, dropped at the decision, or completed after the
    cut-off, and, once the cut-off has passed, every run without COMPLETED, unless the halt rule stopped training
    (then a further amendment decides). Returns {run: {files, state}}."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    at = at or datetime.now().astimezone()
    past = cutoff is not None and at >= cutoff
    halted = any((Path(root) / "HALT").exists() for root in roots.values())
    copied = {}
    for rid, root in roots.items():
        root, files = Path(root), 0
        log = root / f"queue_{rid}.log"
        if log.exists():
            shutil.copy2(log, dest / log.name)
            files += 1
        if (root / rid).is_dir():
            shutil.copytree(root / rid, dest / rid, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("rehearsal"))
            files += sum(1 for p in (root / rid).rglob("*") if p.is_file())
        if (root / "HALT").exists():
            shutil.copy2(root / "HALT", dest / "HALT")
            files += 1
        stop = (stops or {}).get(rid) or {}
        done = read_time(dest / rid / "COMPLETED")
        cut = dest / rid / "CUT"
        if cut.exists():
            cut.unlink()  # decided afresh on every collect
        why = None
        if rid in dropped:
            why = "dropped for budget at the decision (amendment 2 section 3)"
        elif stop.get("final"):
            why = f"{stop.get('reason')} ({stop.get('time')})"
        elif done is not None and cutoff is not None and done >= cutoff:
            why = f"COMPLETED {done.isoformat()} is after the cut-off {cutoff.isoformat()} (amendment 1)"
        elif done is None and past and not halted:
            why = f"not completed by the cut-off {cutoff.isoformat()} (amendment 1)"
        if why:
            write_text_atomic(cut, f"{now()} {why}\n")
        if why:
            state = f"cut: {why}"
        elif done is not None:
            state = "completed"
        elif halted:
            state = "not completed: training was halted, so a further amendment decides"
        else:
            state = "not completed: it can still be resumed before the cut-off"
        copied[rid] = {"files": files, "state": state}
    return copied


def reload_all():
    vol.reload()
    for v in run_vols.values():
        v.reload()


# ---------------------------------------------------------------- Modal functions

@app.function(image=image, volumes=SHARED, cpu=4.0, memory=8192, timeout=2 * 3600)
def prepare_data(tag: str):
    """Download the official zips once, check their hashes, run the round-trip gate, draw the overlay sheets."""
    head = checkout(tag)
    check_wrapper()
    raw, day1 = VOL / "raw", VOL / "day1"
    raw.mkdir(parents=True, exist_ok=True)
    day1.mkdir(parents=True, exist_ok=True)
    tmp = Path("/root/download")
    tmp.mkdir(parents=True, exist_ok=True)
    for z in ZIPS:
        if not (raw / z).exists():
            sh(["wget", "-q", "-O", tmp / z, KITTI_URL + z])
            sh(["cp", tmp / z, raw / f"{z}.partial"])  # one sequential write to the volume
            (raw / f"{z}.partial").rename(raw / z)
            (tmp / z).unlink()
    sums = {z: sha256(raw / z) for z in ZIPS}
    (day1 / "zips_sha256.txt").write_text("".join(f"{s}  {z}\n" for z, s in sums.items()))
    out = {"time": now(), "git_commit": head, "zip_sha256": sums, "error": None, "gate_passed": False}
    wrong = {z: sums[z] for z, want in EXPECTED_SHA256.items() if sums[z] != want}
    if wrong:
        out["error"] = f"SHA-256 differs from PREREG.md section 2.2 for {sorted(wrong)}"
    else:
        stamp = f"{datetime.now():%Y%m%d-%H%M%S}"
        report = day1 / "roundtrip_prepare.json"
        try:
            unpack_and_convert(report, CONSOLE, vol.commit, overlay_dir=day1 / "overlay")
        except GateFailed as e:
            out["error"] = str(e)
        finally:
            if CONSOLE.exists():
                shutil.copyfile(CONSOLE, day1 / f"prepare_console_{stamp}.log")
        rep = read_json(report) or {}
        for k in ("gate_passed", "max_roundtrip_err_px", "worst_image", "checked", "image_sizes",
                  "boxes_per_class", "n_errors"):
            out[k] = rep.get(k)
        out["gate_passed"] = bool(out.get("gate_passed")) and out["error"] is None
    write_json_atomic(day1 / "prepare_summary.json", out)
    vol.commit()
    return out


@app.function(**GPU_KW, volumes=SHARED, timeout=4 * 3600)
def smoke_test(tag: str):
    """The pre-registered smoke test (s 3 epochs; n, m and s at 960 for 1 epoch each), on one L4."""
    head = checkout(tag)
    wrapper = check_wrapper()
    data_step_passed()
    stamp = f"{datetime.now():%Y%m%d-%H%M%S}"
    smoke_dir = VOL / "runs" / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    gpu = gpu_check()
    peak = {"gib": memory_used_gib()}

    def sample():
        m = memory_used_gib()
        if m is not None and (peak["gib"] is None or m > peak["gib"]):
            peak["gib"] = m

    hb = Heartbeat(sample, every=30)
    try:
        unpack_and_convert(VOL / "day1" / f"roundtrip_smoke_{stamp}.json", CONSOLE, vol.commit)
        run_logged([sys.executable, "scripts/smoke_test.py", "--data-root", DATA, "--runs-root", VOL / "runs"],
                   cwd=CODE, log_path=CONSOLE)
    finally:
        hb.stop()
        if CONSOLE.exists():
            shutil.copyfile(CONSOLE, smoke_dir / f"console_{stamp}.log")
        vol.commit()
    smoke = json.loads((smoke_dir / "smoke_projection.json").read_text())
    sys.path.insert(0, str(CODE))
    from configs.recipe import CUT_ORDER, RUNS as RECIPE
    sec, sources = smoke_epoch_seconds(smoke_dir, smoke)
    s640 = read_results(smoke_dir / "smoke-yolo11s-640" / "results.csv")
    out = {"time": now(), "git_commit": head, "gpu": gpu, "memory_peak_gib": peak["gib"],
           "smoke_projection": smoke,
           "decision_preview": make_decision("preview", RECIPE, CUT_ORDER, sec, sources, smoke, wrapper),
           "s640_last_epoch": s640[-1] if s640 else None}
    write_json_atomic(smoke_dir / "preview.json", out)
    vol.commit()
    return out


@app.function(**GPU_KW, volumes=SHARED, timeout=12 * 3600, single_use_containers=True,
              retries=modal.Retries(max_retries=2, initial_delay=60.0))
def train_run(tag: str, run_id: str, stage: str = "launch"):
    """One queue (a run, or E0) through the unchanged scripts/train_queue.py. Spawned with the shared volume
    read-only at /vol and the run's own volume at /runvol."""
    try:
        return _train_run(tag, run_id, stage)
    except Exception as e:  # noqa: BLE001  one type, so the launch never mistakes a failure for a network problem
        raise RunFailed(f"{run_id}: {type(e).__name__}: {e}") from None


def _train_run(tag, run_id, stage):
    started = datetime.now().astimezone()
    result = {"run_id": run_id, "completed": False}
    try:
        head = checkout(tag)
        check_wrapper()
    except WrapperMismatch as e:
        return {**result, "failed": str(e)}
    rvol = run_vols[run_id]
    runs_root = run_root(run_id, stage, base=RUN)
    rundir = runs_root / run_id
    shared = shared_dir(stage)
    qlog = runs_root / f"queue_{run_id}.log"
    decision_text = (shared / "EPOCH_DECISION.json").read_text()
    decision = json.loads(decision_text)
    queue = decision["queues"].get(run_id, [run_id])
    if (runs_root / queue[-1] / "COMPLETED").exists():
        return {**result, "completed": True, "note": "completed earlier"}
    if decision["git_commit"] != head:
        return {**result, "failed": f"EPOCH_DECISION.json was made at {decision['git_commit']}, not {head}"}
    # The cut-off: CUTOFF if the launch has written it; else the earliest START line of this run's earlier
    # containers + 24 h; else, until this container's own START line appears, its start + 24 h 15 min.
    known = read_time(shared / "CUTOFF") or own_deadline(qlog)
    deadline = {"at": known or started + timedelta(hours=CUTOFF_HOURS, minutes=CUTOFF_GRACE_MINUTES),
                "final": known is not None}
    if datetime.now().astimezone() >= deadline["at"]:
        return {**result, "note": f"past the cut-off {deadline['at'].isoformat(timespec='seconds')}; not started"}
    me = modal.current_function_call_id() or "local"
    key = f"{stage}:{run_id}"
    if not claim(locks, key, me):
        return {**result, "note": "another container is training this run; not started"}
    stamp = f"{started:%Y%m%d-%H%M%S}"
    attempt = rundir / "attempts" / f"{stamp}.json"
    rec = {"call": me, "start": started.isoformat(timespec="seconds"), "beat": now(), "end": None, "tag": tag,
           "git_commit": head, "deadline": deadline["at"].isoformat(timespec="seconds"),
           "mem_peak_gib": memory_used_gib()}
    try:
        runs_root.mkdir(parents=True, exist_ok=True)
        copy = runs_root / "EPOCH_DECISION.json"  # train_queue.py reads the decision from its runs root
        if not copy.exists() or copy.read_text() != decision_text:
            copy.write_text(decision_text)
        write_json_atomic(attempt, rec)
    except Exception:
        release(locks, key, me)
        raise
    kept = rundir / f"console_{stamp}.log"
    io = threading.Lock()  # no heartbeat write while this container commits its volume

    def tick():
        with io:
            if not deadline["final"]:
                own = own_deadline(qlog)
                if own is not None:
                    deadline["at"], deadline["final"] = own, True
                    rec["deadline"] = own.isoformat(timespec="seconds")
            mem = memory_used_gib()
            if mem is not None and (rec.get("mem_peak_gib") is None or mem > rec["mem_peak_gib"]):
                rec["mem_peak_gib"] = mem
            rec["beat"] = now()
            write_json_atomic(attempt, rec)
            beat_lock(locks, key, me)
            if CONSOLE.exists():
                shutil.copyfile(CONSOLE, kept)

    def commit():
        with io:
            rvol.commit()

    hb = Heartbeat(tick)
    try:
        rec["gpu"] = gpu_check()
        current = next((r for r in queue if not (runs_root / r / "COMPLETED").exists()), queue[-1])
        rec["restart"] = guard_checkpoints(runs_root / current, decision["epochs"])
        unpack_and_convert(rundir / f"roundtrip_{stamp}.json", CONSOLE, commit)
        run_logged([sys.executable, "scripts/train_queue.py", "--queue", run_id, "--data-root", DATA,
                    "--runs-root", runs_root], cwd=CODE, log_path=CONSOLE, deadline=lambda: deadline["at"])
        result["completed"] = (runs_root / queue[-1] / "COMPLETED").exists()
        if (runs_root / "HALT").exists():
            result["halt"] = (runs_root / "HALT").read_text().strip()
    except GateFailed as e:
        result["failed"] = str(e)
    except CutOff as e:
        write_text_atomic(rundir / "CUT", f"{now()} {e}\n")
        with open(qlog, "a") as f:
            f.write(f"{now()} {run_id}: CUT at the amendment 1 cut-off\n")
        result["note"] = str(e)
    except Exception as e:  # recorded, then raised so Modal retries the run
        rec["error"] = repr(e)
        raise
    finally:
        hb.stop()
        rec["end"], rec["result"] = now(), dict(result)
        write_json_atomic(attempt, rec)
        if CONSOLE.exists():
            shutil.copyfile(CONSOLE, kept)
        release(locks, key, me)
        rvol.commit()
    return result


@app.function(image=image, volumes=ALL_VOLUMES, cpu=ORCH_CPU, memory=ORCH_MEMORY, timeout=24 * 3600,
              nonpreemptible=True)
def orchestrate(tag: str, stage: str = "launch", watch_minutes: float = 0.0):
    """Make the decision once, then start (or re-attach to) the runs phase by phase and watch them. stage is
    "launch"; or "rehearsal-<time>", part 1 of a new rehearsal, which stops watching after watch_minutes and leaves
    the runs going; or "rehearsal", part 2 of the latest rehearsal."""
    part = 0
    if stage == "rehearsal":
        stage, part = latest_rehearsal(), 2
        if stage is None:
            raise RuntimeError("No rehearsal to resume: run ::rehearse first")
    elif stage != "launch":
        part = 1
    check_stage(stage)
    head = checkout(tag)
    wrapper = check_wrapper()
    me = modal.current_function_call_id() or "local"
    if not claim(locks, LAUNCH_LOCK, me, wait_minutes=0):
        cur = locks.get(LAUNCH_LOCK) or {}
        raise RuntimeError(f"Another launch or rehearsal is running (heartbeat {cur.get('beat')}); not starting a "
                           "second one. Check progress with ::status.")

    def beat():
        if not beat_lock(locks, LAUNCH_LOCK, me):
            raise LostLock("another launch has taken over the launch lock; this one stops here")

    try:
        reh = latest_rehearsal()
        if stage == "launch" and reh is not None:
            busy = [rid for rid in RUN_IDS if lock_live(locks, f"{reh}:{rid}")]
            if busy:
                raise RuntimeError(f"The rehearsal {reh} is still training {busy}: run its part 2 (::rehearse "
                                   f"--resume), or wait for them to stop (at most {REHEARSAL_TIMEOUT_HOURS} h)")
        ro = vol.with_mount_options(read_only=True)
        limits = {} if stage == "launch" else {"timeout": REHEARSAL_TIMEOUT_HOURS * 3600, "retries": 0}
        return _orchestrate(
            tag, stage, head, wrapper,
            spawn=lambda rid: train_run.with_options(volumes={str(VOL): ro, str(RUN): run_vols[rid]},
                                                     **limits).spawn(tag, rid, stage),
            attach=modal.FunctionCall.from_id, reload=reload_all, commit=vol.commit, beat=beat,
            max_seconds=watch_minutes * 60 if watch_minutes > 0 else WATCH_MAX_HOURS * 3600, part=part,
            run_live=lambda rid: lock_live(locks, f"{stage}:{rid}"))
    finally:
        release(locks, LAUNCH_LOCK, me)
        vol.commit()


def _orchestrate(tag, stage, head, wrapper, spawn, attach, reload, commit, beat, max_seconds=WATCH_MAX_HOURS * 3600,
                 clock=time.time, sleep=time.sleep, part=0, run_live=None, **watch_kw):
    """The launch (part 0), or part 1 or 2 of a rehearsal. Returns the decision, the outcome of every run watched,
    any HALT, and which runs were re-attached and which were started; part 2 adds the rehearsal's verdict."""
    t0 = clock()
    started_at = datetime.now().astimezone()
    shared = shared_dir(stage)
    shared.mkdir(parents=True, exist_ok=True)
    log = shared / "launch.log"
    verdict_file = shared / "REHEARSAL_VERDICT.txt"

    def note(msg):
        print(msg, flush=True)
        with open(log, "a") as f:
            f.write(f"{now()} {msg}\n")

    if part == 2 and verdict_file.exists():
        verdict = verdict_file.read_text().strip()
        note(f"{stage}: part 2 has run already: {verdict}")
        return {"decision": None, "outcome": {}, "halt": None, "attached": [], "spawned": [], "verdict": verdict}
    data_step_passed()
    sys.path.insert(0, str(CODE))
    from configs.recipe import CUT_ORDER, RUNS as RECIPE
    if list(RECIPE) != RUN_IDS:
        raise RuntimeError(f"configs/recipe.py lists {list(RECIPE)}, not this file's {RUN_IDS}")
    roots = {rid: run_root(rid, stage) for rid in RUN_IDS}
    decision_file = shared / "EPOCH_DECISION.json"
    if decision_file.exists():
        decision = json.loads(decision_file.read_text())
        if decision["git_commit"] != head:
            raise RuntimeError(f"EPOCH_DECISION.json was made at {decision['git_commit']}; this launch is at {head} "
                               f"({tag}). Launch again with the tag the decision was made at.")
        note(f"{stage}: EPOCH_DECISION.json exists and is never remade; using it")
    elif part == 2:
        raise RuntimeError(f"{stage} has no EPOCH_DECISION.json, so its part 1 did not get far: run ::rehearse again")
    else:
        if stage != "launch":
            decision = make_rehearsal_decision(head, stage, wrapper)
        else:
            smoke_file = VOL / "runs" / "smoke" / "smoke_projection.json"
            if not smoke_file.exists():
                raise RuntimeError("No smoke-test result on the volume: run ::smoke first")
            smoke = json.loads(smoke_file.read_text())
            sec, sources = smoke_epoch_seconds(smoke_file.parent, smoke)
            decision = make_decision(head, RECIPE, CUT_ORDER, sec, sources, smoke, wrapper)
            if not decision["within_budget"]:
                note(f"not launched: the runs left after every cut still project ${decision['projected_cost_usd']}")
                raise RuntimeError("The projected cost is above the cap even after every cut; nothing was started "
                                   "(PREREG_AMENDMENT_2 section 3). A further amendment is needed.")
        write_json_atomic(decision_file, decision)
        commit()  # the runs must see the decision when they start
        note(f"{stage}: decision written: {decision['epochs']} epochs, phases {decision['phases']}, dropped "
             f"{decision.get('dropped_for_budget', [])}, projected ${decision.get('projected_cost_usd', '-')}")
    ids = list(decision["queues"])
    calls_file = shared / "calls.json"
    recorded = read_json(calls_file) or {}
    live_cap = LIVE_CAP_USD if stage == "launch" else None
    mon = Monitor(decision, {r: RECIPE[r] for r in ids}, {r: roots[r] for r in ids}, shared, reload, commit,
                  started_at=started_at)
    outcome, attached, spawned, cancelled, gone = {}, [], [], {}, {}
    phases = [[r for r in phase if r in ids] for phase in decision["phases"]]
    long_q = REHEARSAL_LONG[0]

    def on_stop(rid, reason, final):
        record_stop(shared, rid, reason, final, commit)

    def is_live(rid):
        return None if run_live is None else (lambda: run_live(rid))

    def reported(rid):
        res = outcome.get(rid)
        return res if isinstance(res, dict) else {}

    def completed(rid):
        return mon.done(rid) or bool(reported(rid).get("completed"))

    def cancel_recorded(rid, why):
        """Stop a call an earlier launch started, if it is still running. True if it was running."""
        if rid not in recorded:
            return False
        call = attach(recorded[rid])
        if not still_running(call, sleep=sleep, live=is_live(rid)):
            return False
        attached.append(rid)
        for attempt in range(5):
            try:
                call.cancel(terminate_containers=True)
                cancelled[rid] = why
                note(f"{rid}: {why} (it was still running)")
                return True
            except Exception as e:  # noqa: BLE001
                note(f"{rid}: cancel failed ({e!r}); trying again")
                sleep(10)
        raise RuntimeError(f"{rid} is still running and could not be cancelled; try again or stop it in Modal")

    def start(rid):
        """Re-attach to a live call from an earlier launch, or spawn one, recorded at once in calls.json. Part 2 of a
        rehearsal never starts a first-phase run: it only re-attaches to the ones part 1 started."""
        if rid in recorded:
            old = attach(recorded[rid])
            if still_running(old, sleep=sleep, live=is_live(rid)):
                attached.append(rid)
                note(f"{rid}: still running from an earlier launch; watching it")
                return old
        if part == 2 and rid in phases[0]:
            gone[rid] = str(call_state(attach(recorded[rid]))[1]) if rid in recorded else "part 1 never started it"
            note(f"{rid}: not running when part 2 started ({gone[rid]}); a rehearsal does not restart it")
            return None
        for attempt in range(3):
            try:
                call = spawn(rid)
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 2:
                    raise
                note(f"{rid}: could not start it ({e!r}); trying again")
                sleep(10)
        recorded[rid] = call.object_id
        spawned.append(rid)
        write_json_atomic(calls_file, recorded)
        commit()
        note(f"{rid}: started")
        return call

    def fresh_view(tries=3):
        """Reload the volumes, retrying; False if they could not be read."""
        for i in range(tries):
            try:
                reload()
                return True
            except Exception as e:  # noqa: BLE001
                note(f"volume reload failed ({e!r})" + ("; trying again" if i < tries - 1 else ""))
                if i < tries - 1:
                    sleep(20)
        return False

    def rehearsal_end(halt_text):
        """Part 2's last check: read every call afresh until Modal says each has ended; then the verdict."""
        states = {}
        for i in range(21):  # up to 5 minutes
            for rid, cid in recorded.items():
                if states.get(rid, ("",))[0] != "ended":
                    states[rid] = call_state(attach(cid))
            if all(s[0] == "ended" for s in states.values()):
                break
            sleep(15)
        for rid, (state, _) in states.items():
            if state != "ended":
                try:
                    attach(recorded[rid]).cancel(terminate_containers=True)
                    note(f"{rid}: still {state} at the end of the rehearsal; cancelled")
                except Exception as e:  # noqa: BLE001
                    note(f"{rid}: still {state} at the end of the rehearsal and the cancel failed ({e!r}); stop it "
                         "in Modal")
        fresh_view(1)
        verdict = rehearsal_verdict({
            "halt": halt_text, "attached": attached, "spawned": spawned, "cancelled": cancelled, "gone": gone,
            "outcome": outcome, "states": {r: s[0] for r, s in states.items()},
            "headline_completed": completed(HEADLINE), "e0_completed": completed(E0)})
        note(verdict)
        write_text_atomic(verdict_file, verdict + "\n")
        commit()
        return verdict

    def finish(halt_text):
        note(f"{stage}: watching ended: {json.dumps(outcome, default=str)}; re-attached {attached}; started "
             f"{spawned}")
        res = {"decision": {k: decision.get(k) for k in ("time", "git_commit", "epochs", "phases")},
               "outcome": outcome, "halt": halt_text, "attached": attached, "spawned": spawned}
        if part == 2:
            res["verdict"] = rehearsal_end(halt_text)
        return res

    halt_text, cut = mon.halt(), mon.cutoff()
    if halt_text or (cut is not None and datetime.now().astimezone() >= cut):
        # A relaunch after a halt or after the cut-off starts nothing; it only stops what is still running. After a
        # halt, the headline is left to end by itself, and watched if it is still going.
        why, final = ("cancelled: HALT", False) if halt_text else (CUT_AT_CUTOFF, True)
        for rid in list(recorded):
            if halt_text and rid == HEADLINE:
                continue
            if not mon.done(rid) and cancel_recorded(rid, why):
                outcome[rid] = why
                on_stop(rid, why, final)
        if halt_text and HEADLINE in recorded:
            old = attach(recorded[HEADLINE])
            if still_running(old, sleep=sleep, live=is_live(HEADLINE)):
                attached.append(HEADLINE)
                note(f"{HEADLINE}: wrote HALT and is still finishing; watching it to its end")
                outcome.update(watch({HEADLINE: old}, mon, note, CUT_ORDER, live_cap_usd=None, beat=beat,
                                     on_stop=on_stop, max_seconds=max_seconds - (clock() - t0), sleep=sleep,
                                     cancelled=cancelled, **watch_kw))
        note(f"{stage}: {'HALT (' + halt_text + ')' if halt_text else 'the cut-off has passed'}; nothing is started"
             + ("; a restart after a halt needs a further amendment" if halt_text else ""))
        return finish(halt_text)

    halt_text = None
    for k, phase in enumerate(phases):
        later = [r for p in phases[k + 1:] for r in p]
        stops = read_stops(shared)
        todo = [r for r in phase if not mon.done(r) and not (stops.get(r) or {}).get("final")]
        for r in phase:
            if r not in todo:
                note(f"{r}: {'completed earlier' if mon.done(r) else 'cut earlier; not restarted'}")
        if k > 0 and live_cap is not None and todo:
            # Phase 2 starts only if spent plus projected stays under the live cap; otherwise cut in the cut order.
            while todo and mon.projection_usd(todo) > live_cap:
                victim = next((r for r in CUT_ORDER if r in todo), None)
                if victim is None:
                    break
                todo.remove(victim)
                why = f"cut for budget: projected over ${live_cap:.2f} before the second phase"
                on_stop(victim, why, True)
                cancel_recorded(victim, why)
                outcome[victim] = why
                note(f"{victim}: {why}")
        calls = {}
        for r in todo:
            call = start(r)
            if call is not None:
                calls[r] = call
        test = {}
        if part == 2 and k == 0:
            if long_q not in attached:
                for rid, call in calls.items():
                    try:
                        call.cancel(terminate_containers=True)
                        cancelled[rid] = "cancelled: the rehearsal cannot go on"
                    except Exception as e:  # noqa: BLE001
                        note(f"{rid}: cancel failed ({e!r}); stop it in Modal")
                note(f"{stage}: the long queue was not running when part 2 started, so the rehearsal stops here")
                return finish(None)
            test = {long_q: REHEARSAL_STOP}
        outcome.update(watch(calls, mon, note, CUT_ORDER, live_cap_usd=live_cap, beat=beat, on_stop=on_stop,
                             max_seconds=max_seconds - (clock() - t0), extra_pending=later, sleep=sleep,
                             test_stops=test, cancelled=cancelled, **watch_kw))
        # The rest start only once the headline has completed and passed the halt rule, judged on a fresh view.
        fresh = fresh_view()
        halt_text = (mon.halt() if fresh else None) or reported(HEADLINE).get("halt")
        if halt_text:
            note(f"{stage}: HALT ({halt_text}); nothing else is started")
            break
        if not later:
            break
        again = "run part 2 (::rehearse --resume)" if part == 1 else "launch again to carry on"
        if not fresh:
            note(f"{stage}: the run volumes could not be read, so the halt rule cannot be checked; nothing else is "
                 f"started; {again}")
            break
        if clock() - t0 >= max_seconds:
            note(f"{stage}: stopped watching; {again}")
            break
        cut = mon.cutoff()
        if cut is not None and datetime.now().astimezone() >= cut:
            note(f"{stage}: the cut-off has passed; nothing else is started")
            break
        if not completed(HEADLINE):
            note(f"{stage}: the headline has not completed ({outcome.get(HEADLINE, 'not run')}); nothing else is "
                 "started. Launch again to resume it.")
            break
        note(f"{stage}: the headline completed and passed the halt rule; starting {later}")
    return finish(halt_text)


def rehearsal_verdict(v):
    """Part 2's verdict, from what it saw: halt, attached, spawned, cancelled, gone, outcome, states (each call's
    last state), headline_completed and e0_completed."""
    long_q = REHEARSAL_LONG[0]
    branch = ("HALT branch: the headline's 1-epoch score triggered the halt rule" if v["halt"] else
              "no-HALT branch: the headline's 1-epoch score passed the halt rule")
    if long_q not in v["attached"]:
        why = v["gone"].get(long_q, "not started or cut earlier")
        if "RunFailed" in why:
            return f"REHEARSAL FAILED: the long queue failed before part 2 started ({why})"
        return (f"REHEARSAL INCONCLUSIVE: the long queue was not running when part 2 started ({why}); run ::rehearse "
                "again and start part 2 straight after part 1")
    problems = []
    if long_q not in v["cancelled"]:
        problems.append("the long queue was re-attached but not cancelled")
    if HEADLINE in v["gone"]:
        problems.append(f"the headline had ended before part 2 without completing or halting ({v['gone'][HEADLINE]})")
    if v["halt"]:
        if E0 in v["spawned"] or E0 in v["outcome"]:
            problems.append("E0 started after a HALT")
    elif not v["headline_completed"]:
        problems.append(f"the headline neither completed nor wrote HALT ({v['outcome'].get(HEADLINE, 'no result')})")
    elif E0 not in v["spawned"]:
        problems.append("E0 did not start after the headline completed")
    elif not v["e0_completed"]:
        problems.append(f"E0 did not complete ({v['outcome'].get(E0, 'no result')})")
    still = sorted(r for r, s in v["states"].items() if s != "ended")
    if still:
        problems.append(f"Modal did not report these calls as ended at the end: {still}")
    if problems:
        return f"REHEARSAL FAILED ({branch}): " + "; ".join(problems)
    tail = "E0 never started" if v["halt"] else "E0 started once the headline had completed, and completed"
    return (f"REHEARSAL OK ({branch}): part 2 re-attached to the long queue that part 1 had started and stopped it; "
            f"{tail}; every call then read as ended")


def status_report(stage, head=None, wrapper_note=None, store=None):
    store = locks if store is None else store
    sys.path.insert(0, str(CODE))
    from configs.recipe import RUNS as RECIPE
    shared = shared_dir(stage)
    decision = read_json(shared / "EPOCH_DECISION.json")
    ids = list(decision["queues"]) if isinstance(decision, dict) else []
    roots = {rid: run_root(rid, stage) for rid in RUN_IDS}
    starts = queue_starts(roots[r] / f"queue_{r}.log" for r in ids)
    cutoff = read_time(shared / "CUTOFF")
    if cutoff is None and starts:
        cutoff = min(starts) + timedelta(hours=CUTOFF_HOURS)
    stops = read_stops(shared)
    runs = {rid: run_state(roots[rid] / rid, RECIPE[rid], cutoff, stops.get(rid)) for rid in ids}
    try:
        lock = store.get(LAUNCH_LOCK)
    except Exception:  # noqa: BLE001
        lock = None
    halt = roots[HEADLINE] / "HALT"
    try:
        tail = (shared / "launch.log").read_text().splitlines()[-10:]
    except OSError:
        tail = []
    try:
        verdict = (shared / "REHEARSAL_VERDICT.txt").read_text().strip() or None
    except OSError:
        verdict = None
    keys = ("time", "git_commit", "epochs", "mode", "phases", "dropped_for_budget", "projected_cost_usd",
            "budget_usd")
    return {
        "time": now(), "stage": stage, "git_commit": head, "wrapper_note": wrapper_note,
        "prepare": read_json(VOL / "day1" / "prepare_summary.json"),
        "smoke": read_json(VOL / "runs" / "smoke" / "preview.json"),
        "decision": {k: decision.get(k) for k in keys} if isinstance(decision, dict) else None,
        "launch_heartbeat_min": round(age_minutes(lock["beat"]), 1) if isinstance(lock, dict) else None,
        "first_start": min(starts).isoformat() if starts else None,
        "cutoff": cutoff.isoformat() if cutoff else None,
        "halt": halt.read_text().strip() if halt.exists() else None, "runs": runs,
        "spent_usd": round(sum(r["spent_h"] for r in runs.values()) * RATE_PER_HOUR, 2), "launch_log_tail": tail,
        "verdict": verdict,
    }


@app.function(image=image, volumes=ALL_VOLUMES, cpu=1.0, memory=1024, timeout=600)
def read_status(tag: str, stage: str = "launch"):
    reload_all()
    head = checkout(tag, strict=False)
    notes = [frozen_problem(tag)]
    try:
        check_wrapper()
    except WrapperMismatch as e:
        notes.append(str(e))
    note = "; ".join(n for n in notes if n) or None
    if stage == "rehearsal":
        stage = latest_rehearsal()
        if stage is None:
            return {"time": now(), "stage": "rehearsal", "git_commit": head, "wrapper_note": note, "none": True}
    return status_report(stage, head, note)


@app.function(image=image, volumes=ALL_VOLUMES, cpu=2.0, memory=2048, timeout=3600)
def collect_runs(tag: str):
    """After training: copy every run from its own volume into kitti-msc/runs, ready for one download."""
    reload_all()
    head = checkout(tag)
    check_wrapper()
    lock = locks.get(LAUNCH_LOCK)
    if isinstance(lock, dict) and not stale(lock):
        raise RuntimeError("A launch is still running; collect after it has ended (see ::status)")
    roots = {rid: run_root(rid, "launch") for rid in RUN_IDS}
    live = [rid for rid in RUN_IDS if any(not a.get("end") and not stale(a) for a in read_attempts(roots[rid] / rid))]
    if live:
        raise RuntimeError(f"Still training: {live}; collect after they have ended")
    shared = shared_dir("launch")
    decision = read_json(shared / "EPOCH_DECISION.json") or {}
    cutoff = read_time(shared / "CUTOFF")
    if cutoff is None:
        starts = queue_starts(roots[r] / f"queue_{r}.log" for r in RUN_IDS)
        cutoff = min(starts) + timedelta(hours=CUTOFF_HOURS) if starts else None
    runs = collect_into(roots, shared, read_stops(shared), cutoff, dropped=decision.get("dropped_for_budget", []))
    vol.commit()
    return {"time": now(), "git_commit": head, "cutoff": cutoff.isoformat() if cutoff else None, "runs": runs}


# ---------------------------------------------------------------- what you run from the laptop

def detached():
    """True if this `modal run` was started with --detach (or -d)."""
    try:
        import click
        ctx = click.get_current_context(silent=True)
        while ctx is not None:
            if isinstance(ctx.obj, dict) and "detach" in ctx.obj:
                return bool(ctx.obj["detach"])
            ctx = ctx.parent
    except Exception:  # noqa: BLE001
        pass
    return any(a in ("--detach", "-d") for a in sys.argv[1:])


def need_detach(step):
    """prepare, smoke, the rehearsal and the launch must keep running when this window closes or the step returns."""
    if detached():
        return True
    print("Not started. Run this step with --detach, or Modal stops it, and everything it started, as soon as this "
          f"window closes or the step returns:\n  python -m modal run --detach cloud/kitti_modal.py::{step}")
    return False


def print_smoke(res):
    p = res["decision_preview"]
    print(f"GPU: {res.get('gpu')}   peak memory in use: {res.get('memory_peak_gib')} GiB (of {MEMORY // 1024} requested)")
    for key, sec in p["epoch_seconds"].items():
        print(f"  {key:<16}{sec:>8.1f} s/epoch   from {p['epoch_seconds_source'][key]}")
    print("Projected hours per run:", json.dumps(p["projected_hours"], indent=2))
    print(f"If launched now: {p['epochs']} epochs, {len(p['queues'])} runs, projected ${p['projected_cost_usd']} "
          f"(cap ${p['budget_usd']}), dropped for budget: {p['dropped_for_budget'] or 'none'}")
    if not p["within_budget"]:
        print("Even after every cut the projection is over the cap: launch would refuse to start.")
    last = res.get("s640_last_epoch") or {}
    if last:
        print(f"YOLO11s at 640 after {last.get('epoch')} epochs: val mAP50 {last.get('metrics/mAP50(B)')} "
              "(a sanity check only; the halt rule applies to the headline after 100 epochs)")


def print_status(s):
    if s.get("none"):
        print("No rehearsal has run yet.")
        return
    print(f"Stage: {s['stage']}   code: {s['git_commit']}   {s['wrapper_note'] or 'wrapper matches the tag'}")
    p = s.get("prepare")
    print("Data step: " + ("not run" if not p else
                           f"gate {'PASSED' if p.get('gate_passed') else 'FAILED'}, max round-trip error "
                           f"{p.get('max_roundtrip_err_px')} px{'; ' + p['error'] if p.get('error') else ''}"))
    sm = s.get("smoke")
    if sm:
        pv = sm["decision_preview"]
        print(f"Smoke test {sm['time']}: projected ${pv['projected_cost_usd']} for {len(pv['queues'])} runs, dropped "
              f"{pv['dropped_for_budget'] or 'none'}; peak memory {sm.get('memory_peak_gib')} GiB")
    else:
        print("Smoke test: not run")
    d = s["decision"]
    if d is None:
        print("No EPOCH_DECISION.json for this stage yet.")
        return
    print(f"Decision {d['time']}: {d['epochs']} epochs, {d['mode']}, projected ${d.get('projected_cost_usd')} "
          f"(cap ${d.get('budget_usd')}), dropped for budget: {d.get('dropped_for_budget') or 'none'}")
    hb = s["launch_heartbeat_min"]
    print("Launch: " + ("not running" if hb is None or hb >= STALE_MINUTES else f"running (heartbeat {hb:.0f} min ago)"))
    print(f"First start {s['first_start']}; amendment 1 cut-off {s['cutoff']}")
    print(f"HALT: {s['halt'] or 'none'}")
    print(f"Spent so far by the run containers: ${s['spent_usd']}\n")
    print(f"{'run':<22}{'state':<11}{'epochs':>7}{'val mAP50':>11}{'tries':>7}{'hours':>7}{'GiB':>6}  note")
    for rid, r in s["runs"].items():
        m = "" if r["val_map50"] is None else f"{r['val_map50']:.3f}"
        g = "" if r.get("mem_peak_gib") is None else f"{r['mem_peak_gib']:.1f}"
        print(f"{rid:<22}{r['state']:<11}{r['epochs_done']:>7}{m:>11}{r['attempts']:>7}{r['spent_h']:>7.2f}"
              f"{g:>6}  {r['note']}")
    if s["launch_log_tail"]:
        print("\nlaunch.log (last lines):\n  " + "\n  ".join(s["launch_log_tail"]))
    if s.get("verdict"):
        print(f"\n{s['verdict']}")


@app.local_entrypoint()
def prepare(tag: str = TAG):
    if not need_detach("prepare"):
        return
    res = prepare_data.remote(tag)
    print(json.dumps(res, indent=2))
    print("ROUND-TRIP GATE PASSED" if res.get("gate_passed") else "ROUND-TRIP GATE FAILED")


@app.local_entrypoint()
def smoke(tag: str = TAG):
    if not need_detach("smoke"):
        return
    print_smoke(smoke_test.remote(tag))


@app.local_entrypoint()
def rehearse(tag: str = TAG, resume: bool = False):
    """Part 1 starts a fresh rehearsal and stops watching after a few minutes, leaving the runs going. Part 2
    (--resume), run straight after from a new `modal run`, re-attaches to them, stops the long queue and watches the
    headline (and E0, unless the headline halts) to the end. The verdict is also kept in the rehearsal's folder:
    ::status --stage rehearsal shows it."""
    if not need_detach("rehearse --resume" if resume else "rehearse"):
        return
    if not resume:
        stage = f"rehearsal-{datetime.now():%Y%m%d-%H%M%S}"
        print(f"Rehearsal part 1 ({stage}): the headline and a long queue for {REHEARSAL_EPOCHS} epoch, in their own "
              f"folders. This window returns in about {REHEARSAL_PART1_MINUTES + 2} minutes; then run part 2 at once:\n"
              "  python -m modal run --detach cloud/kitti_modal.py::rehearse --resume")
        res = orchestrate.remote(tag, stage, float(REHEARSAL_PART1_MINUTES))
        print(json.dumps(res["outcome"], indent=2, default=str))
        print("Part 1 done. Now run part 2 (--resume) straight away.")
        return
    print("Rehearsal part 2: re-attaching to the latest rehearsal's runs and watching them to the end.")
    res = orchestrate.remote(tag, "rehearsal", 0.0)
    print(json.dumps(res["outcome"], indent=2, default=str))
    print(res["verdict"])


@app.local_entrypoint()
def launch(tag: str = TAG):
    if not need_detach("launch"):
        return
    print("Starting the launch: the headline first, then the rest once it passes the halt rule. You can close this "
          "window; check progress with ::status.")
    res = orchestrate.remote(tag, "launch", 0.0)
    print(json.dumps(res["outcome"], indent=2, default=str))


@app.local_entrypoint()
def status(tag: str = TAG, stage: str = "launch"):
    print_status(read_status.remote(tag, stage))


@app.local_entrypoint()
def collect(tag: str = TAG):
    res = collect_runs.remote(tag)
    print(f"Cut-off: {res['cutoff']}")
    for rid, r in res["runs"].items():
        print(f"  {rid:<22}{r['files']:>6} files   {r['state']}")
    print('Now download them: python -m modal volume get --force kitti-msc runs "G:/My Drive/msc"')
