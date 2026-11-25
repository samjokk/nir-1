"""Prepare TTPLA supervision and create wire masks for the matching benchmark."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml


def prepare_ttpla(source, destination):
    """Write YOLO polygons and hold out entire video clips for validation."""
    destination = destination.resolve()
    counts = defaultdict(int)
    for subset, annotation_file in (("trainingset", "train.json"), ("testset", "test.json")):
        directory = source / subset
        coco = json.loads((directory / annotation_file).read_text())
        cable = next(category["id"] for category in coco["categories"] if category["name"] == "cable")
        by_image = defaultdict(list)
        for annotation in coco["annotations"]:
            if annotation["category_id"] == cable:
                by_image[annotation["image_id"]].append(annotation["segmentation"])
        for image in coco["images"]:
            name = image["file_name"]
            clip = int(name.split("_", 1)[0])
            split = "val" if clip % 7 == 0 else "train"
            image_dir = destination / "images" / split
            label_dir = destination / "labels" / split
            image_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            original = (directory / name).resolve(strict=True)
            target = image_dir / f"{subset}_{name}"
            if not target.exists():
                target.symlink_to(original)
            polygons = []
            scale = np.array([image["width"], image["height"]])
            for components in by_image[image["id"]]:
                for polygon in components:
                    coordinates = np.asarray(polygon).reshape(-1, 2)
                    if len(coordinates) >= 3:
                        points = np.clip(coordinates / scale, 0, 1).ravel()
                        polygons.append("0 " + " ".join(f"{point:.6f}" for point in points))
            (label_dir / f"{target.stem}.txt").write_text("\n".join(dict.fromkeys(polygons)))
            counts[split] += 1
    print(f"TTPLA: {dict(counts)}; разбиение по роликам.", flush=True)
    spec = destination / "dataset.yaml"
    spec.write_text(yaml.safe_dump({"path": str(destination), "train": "images/train",
                                    "val": "images/val", "names": {0: "cable"}}))
    return spec


def train_yolo(source, work, epochs=6):
    from ultralytics import YOLO
    import torch

    run = work.resolve() / "train-wire"
    weights = run / "weights" / "best.pt"
    if not weights.is_file():
        data = prepare_ttpla(source, work / "ttpla-yolo")
        device = 0 if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        pretrained = work.resolve() / "yolo11n-seg.pt"
        arguments = dict(data=str(data), imgsz=640, batch=4, epochs=epochs, workers=0,
                         device=device, mask_ratio=1, project=str(run.parent), name=run.name,
                         exist_ok=True, seed=47, plots=False, amp=False, optimizer="AdamW",
                         lr0=0.001, close_mosaic=2, mosaic=0.4, overlap_mask=True)
        model = YOLO(str(pretrained))
        model.train(**arguments)
    if not weights.is_file():
        raise FileNotFoundError(weights)
    return weights


def mask_path(work, method, frame):
    return work / "masks" / method / frame.pack / f"{frame.path.name}.npz"


def signature(checkpoint):
    import hashlib

    with checkpoint.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def masks_from_result(result, size):
    import cv2

    if result.masks is None:
        return np.empty((0, size, size), dtype=bool)
    masks = result.masks.data.cpu().numpy() > 0.5
    if masks.shape[1:] != (size, size):
        masks = np.stack([cv2.resize(mask.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST) > 0
                          for mask in masks])
    return masks


def save_masks(path, masks, stamp, prompt_kind):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, instances=np.packbits(masks, axis=-1),
                            side=masks.shape[-1], stamp=stamp, prompts=prompt_kind)
    temporary.replace(path)


def load_masks(path):
    with np.load(path, allow_pickle=False) as data:
        return np.unpackbits(data["instances"], axis=-1, count=int(data["side"])).astype(bool)


def is_current_mask(path, stamp):
    if path.is_file():
        with np.load(path, allow_pickle=False) as data:
            return str(data["stamp"]) == stamp
    return False


def line_prompts(image, limit=6):
    """Propose thin elongated regions when YOLO provides no usable prompts."""
    import cv2

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    raw_lines = cv2.createLineSegmentDetector().detect(gray)[0]
    if raw_lines is None:
        return []
    lines = raw_lines.reshape(-1, 2, 2)
    lengths = np.linalg.norm(lines[:, 1] - lines[:, 0], axis=1)
    proposals = []
    for index in np.argsort(-lengths):
        if lengths[index] < 55 or len(proposals) >= limit:
            break
        a, b = lines[index]
        midpoint = (a + b) / 2
        if any(np.min(np.linalg.norm(points - midpoint, axis=1)) < 18 for _, points, _ in proposals):
            continue
        points = a + np.array([0.18, 0.50, 0.82])[:, None] * (b - a)
        box = np.r_[np.minimum(a, b) - 9, np.maximum(a, b) + 9].clip(0, image.shape[0] - 1)
        proposals.append((box, points, None))
    return proposals


def yolo_prompts(instances, limit=10):
    from scipy.spatial import cKDTree

    prompts = []
    for mask in instances[:limit]:
        y, x = np.nonzero(mask)
        if len(x) < 12:
            continue
        xy = np.column_stack((x, y))
        center = xy.mean(axis=0)
        _, vectors = np.linalg.eigh(np.cov(xy.T))
        axis = vectors[:, -1]
        projection = (xy - center) @ axis
        samples = center + np.quantile(projection, [0.2, 0.5, 0.8])[:, None] * axis
        indices = cKDTree(xy).query(samples)[1]
        box = np.r_[xy.min(axis=0) - 5, xy.max(axis=0) + 5].clip(0, mask.shape[0] - 1)
        prompts.append((box, xy[indices], mask))
    return prompts


def plausible_sam_mask(mask, points, proposal):
    import cv2

    y, x = np.nonzero(mask)
    if len(x) < 12 or len(x) > mask.size * 0.05:
        return False
    covariance = np.cov(np.column_stack((x, y)).T)
    small, large = np.linalg.eigvalsh(covariance)
    if large < 100 or large < 16 * max(small, 1):
        return False
    neighborhood = cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    px, py = np.rint(points).astype(int).T
    inside = neighborhood[py.clip(0, mask.shape[0] - 1), px.clip(0, mask.shape[1] - 1)]
    if int(inside.sum()) < 2:
        return False
    return proposal is None or (neighborhood & proposal).sum() >= 0.18 * proposal.sum()


def annotate(frames, work, method, yolo_checkpoint, sam_checkpoint=None):
    """Run a single segmenter through the complete manifest and checkpoint every image."""
    import torch
    from ultralytics import SAM, YOLO
    from wire_data import SIDE, read_rgb

    yolo_stamp = "yolo-wire-v1:" + signature(yolo_checkpoint)
    revision = yolo_stamp if method == "yolo" else "sam-wire-v1:" + signature(sam_checkpoint) + ":" + yolo_stamp
    pending = [(frame, revision + ":" + frame.digest) for frame in frames.values()]
    pending = [(frame, stamp) for frame, stamp in pending if not is_current_mask(mask_path(work, method, frame), stamp)]
    if not pending:
        return
    device = 0 if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = YOLO(str(yolo_checkpoint)) if method == "yolo" else SAM(str(sam_checkpoint))
    for index, (frame, stamp) in enumerate(pending, 1):
        rgb = read_rgb(frame.path)
        image = rgb[..., ::-1].copy()
        if method == "yolo":
            result, = model.predict(image, imgsz=SIDE, conf=0.15, retina_masks=True, max_det=20,
                                    device=device, verbose=False)
            masks = masks_from_result(result, SIDE)
            prompt_kind = "image"
        else:
            source_mask = mask_path(work, "yolo", frame)
            if not is_current_mask(source_mask, yolo_stamp + ":" + frame.digest):
                raise RuntimeError(f"Сначала завершите YOLO-seg для {frame.key}")
            prompts = yolo_prompts(load_masks(source_mask))
            prompt_kind = "yolo" if prompts else "line"
            if not prompts:
                prompts = line_prompts(rgb)
            masks = np.empty((0, SIDE, SIDE), dtype=bool)
            if prompts:
                boxes, points, proposals = zip(*prompts)
                result, = model.predict(
                    image, imgsz=1024, bboxes=np.asarray(boxes), points=np.asarray(points),
                    labels=np.ones((len(points), 3), dtype=np.int32), conf=0.01,
                    device=device, verbose=False,
                )
                predicted = masks_from_result(result, SIDE)
                # Ultralytics stores the prompt index in this field, including after score filtering.
                prompt_indices = result.boxes.cls.cpu().numpy().astype(int)
                keep = [plausible_sam_mask(mask, points[i], proposals[i])
                        for mask, i in zip(predicted, prompt_indices, strict=True)]
                masks = predicted[np.asarray(keep, dtype=bool)]
            else:
                prompt_kind = "none"
        save_masks(mask_path(work, method, frame), masks, stamp, prompt_kind)
        if index % 50 == 0 or index == len(pending):
            print(f"{method}: {index}/{len(pending)}", flush=True)
