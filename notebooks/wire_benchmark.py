"""Run the etl-2 experiment, with one model live at any time."""

import argparse
import fcntl
import os
from pathlib import Path

import certifi

from wire_data import build_pairs, read_manifest
from wire_masks import annotate, train_yolo


def project_root():
    for path in (Path.cwd(), Path.cwd().parent):
        if (path / "data" / "etl-2").is_dir():
            return path
    raise FileNotFoundError("Откройте ноутбук из корня проекта или каталога notebooks.")


def run(root, stages=("yolo", "sam", "pairs", "match")):
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    import torch

    from wire_matching import MODEL_NAMES, release_memory, run_matcher

    root = root.resolve()
    data = root / "data" / "etl-2"
    work = data / "wire_benchmark"
    work.mkdir(exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(work / "ultralytics"))
    torch.set_num_threads(3)
    with (work / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        frames = read_manifest(data)
        yolo_checkpoint = work / "train-wire" / "weights" / "best.pt"
        if "yolo" in stages:
            yolo_checkpoint = train_yolo(root / "data" / "etl" / "archive", work)
            release_memory()
            annotate(frames, work, "yolo", yolo_checkpoint)
            release_memory()
        if "sam" in stages:
            annotate(frames, work, "sam", yolo_checkpoint, work / "sam2.1_t.pt")
            release_memory()
        pairs = []
        if "pairs" in stages or "match" in stages:
            pairs = build_pairs(frames, work)
            paired = {key for pair in pairs for key in (pair.a, pair.b)}
            print(f"Пары: {len(pairs)}, кадров с партнёром: {len(paired)}/{len(frames)}.", flush=True)
        if "match" in stages:
            run_models = MODEL_NAMES
        else:
            run_models = [name for name in MODEL_NAMES if name in stages]
            if run_models and not pairs:
                pairs = build_pairs(frames, work)
        for name in run_models:
            run_matcher(name, frames, pairs, work)


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    from wire_matching import MODEL_NAMES

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stages", nargs="*")
    args = parser.parse_args()
    unknown = set(args.stages) - {"yolo", "sam", "pairs", "match", *MODEL_NAMES}
    if unknown:
        parser.error("Unrecognized stages: " + ", ".join(sorted(unknown)))
    if args.stages:
        run(project_root(), args.stages)
    else:
        run(project_root())
