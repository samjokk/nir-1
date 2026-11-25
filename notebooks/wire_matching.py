"""Evaluate dense and sparse matchers at identical source anchors on wire masks."""

import gc
import hashlib
import json
import sys
import time
from functools import lru_cache

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.spatial import cKDTree

import matchers
from wire_data import SIDE, read_rgb, write_json
from wire_masks import mask_path


MODEL_NAMES = ("RoMa v1", "RoMa v2", "LoFTR", "UFM", "DISK + LightGlue")
MASK_NAMES = ("yolo", "sam")
TOLERANCE = 6.0
MAX_ANCHORS = 256
ANCHOR_CELL = 8


def release_memory():
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def sample_map(field, coordinates):
    x, y = np.asarray(coordinates, np.float32).T
    if not len(x):
        return np.empty((0,) + field.shape[2:], dtype=np.float32)
    values = cv2.remap(np.ascontiguousarray(field, dtype=np.float32), x[None], y[None],
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return values[0]


def sample_warp(payload, anchors):
    warp, confidence = payload
    h, w = confidence.shape
    positions = (anchors + 0.5) * (w / SIDE, h / SIDE) - 0.5
    return (sample_map(warp, positions) + 1) * SIDE / 2 - 0.5, sample_map(confidence, positions)


def to_numpy(x):
    return x.detach().float().cpu().numpy()


class UFM:
    def __init__(self, work, device):
        source = work / "ufm-source"
        weights = work / "ufm-weights"
        if not (weights / "model.safetensors").is_file() or not (source / "UniCeption" / "uniception").is_dir():
            raise FileNotFoundError("Нужны локальные исходники UFM, подмодуль UniCeption и веса UFM.")
        for directory in (source, source / "UniCeption"):
            absolute = str(directory.resolve())
            if absolute not in sys.path:
                sys.path.insert(0, absolute)
        from uniflowmatch.models.ufm import UniFlowMatchConfidence

        config = json.loads((weights / "config.json").read_text())
        sharing = config["info_sharing_kwargs"]
        # The released full weights replace the training-only initialization checkpoint.
        if sharing.get("pretrained_checkpoint_path"):
            sharing["pretrained_checkpoint_path"] = None
        self.model = UniFlowMatchConfidence.from_pretrained(
            str(weights), info_sharing_kwargs=sharing, local_files_only=True, strict=True,
        ).eval().to(device)
        self.device = device

    def __call__(self, a, b):
        answer = self.model.predict_correspondences_batched(
            source_image=torch.from_numpy(a).to(self.device), target_image=torch.from_numpy(b).to(self.device),
        )
        flow = to_numpy(answer.flow.flow_output[0]).transpose(1, 2, 0)
        overlap = to_numpy(answer.covisibility.mask[0]).reshape(SIDE, SIDE)
        y, x = np.indices(overlap.shape, dtype=np.float32)
        warp = (flow + np.stack((x, y), axis=-1) + 0.5) * 2 / SIDE - 1
        return warp, overlap


def make_matcher(name, work):
    if name == "UFM":
        return UFM(work, matchers.device)
    if name == "LoFTR":
        return matchers.LoFTRMatcher(size=640)
    if name == "DISK + LightGlue":
        return matchers.DiskLightGlueMatcher(size=640, n=2048)
    if name == "RoMa v1":
        matcher = matchers.RoMaV1Matcher(coarse_res=448, upsample_res=640)
        matcher.model.symmetric = False
        return matcher
    if name == "RoMa v2":
        return matchers.RoMaV2Matcher("fast")
    raise ValueError(f"Неизвестная модель: {name}")


@torch.inference_mode()
def raw_match(name, matcher, a, b):
    if name == "RoMa v2":
        result = matcher.model.match(a, b)
        return to_numpy(result["warp_AB"][0]), to_numpy(result["overlap_AB"][0, ..., 0])
    if name == "RoMa v1":
        warp, support = matcher.model.match(Image.fromarray(a), Image.fromarray(b), device=matchers.device)
        return to_numpy(warp[0, :, :, 2:]), to_numpy(support[0])
    return matcher(a, b)


def source_anchors(mask):
    interior = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    indices = np.flatnonzero(mask)
    if not indices.size:
        return np.empty((0, 2), dtype=np.float32)
    y, x = np.unravel_index(indices, mask.shape)
    columns = (mask.shape[1] + ANCHOR_CELL - 1) // ANCHOR_CELL
    cells = (y // ANCHOR_CELL) * columns + x // ANCHOR_CELL
    order = np.lexsort((-interior.ravel()[indices], cells))
    _, first = np.unique(cells[order], return_index=True)
    selected = order[first]
    if len(selected) > MAX_ANCHORS:
        selected = selected[np.linspace(0, len(selected) - 1, MAX_ANCHORS, dtype=int)]
    return np.column_stack((x[selected], y[selected])).astype(np.float32)


@lru_cache(maxsize=32)
def mask_geometry(path, mtime):
    # Cache only the union and anchor list, not the per-instance masks.
    from wire_masks import load_masks

    union = load_masks(path).any(axis=0)
    distance = cv2.distanceTransform((~union).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return source_anchors(union), distance


def query_sparse(prediction, anchors, source_distance):
    locations = np.asarray(prediction["kA"]).reshape(-1, 2)
    targets = np.asarray(prediction["kB"]).reshape(-1, 2)
    guess = np.full_like(anchors, np.nan)
    if not len(locations):
        return guess, np.zeros(len(anchors), dtype=bool)
    valid = (np.isfinite(locations).all(axis=1) & np.isfinite(targets).all(axis=1)
             & ((locations >= 0) & (locations < SIDE)).all(axis=1))
    locations, targets = locations[valid], targets[valid]
    on_wire = sample_map(source_distance, locations) <= TOLERANCE
    if not on_wire.any():
        return guess, np.zeros(len(anchors), dtype=bool)
    distance, nearest = cKDTree(locations[on_wire]).query(anchors, distance_upper_bound=TOLERANCE)
    accepted = np.isfinite(distance)
    guess[accepted] = targets[on_wire][nearest[accepted]]
    return guess, accepted


def score_match(payload, source_geometry, target_geometry):
    anchors, source_distance = source_geometry
    target_anchors, target_distance = target_geometry
    if not len(anchors) or not len(target_anchors):
        return {"anchors": len(anchors), "accepted": 0, "hits": 0, "eligible": False}, []
    if isinstance(payload, tuple):
        predicted, confidence = sample_warp(payload, anchors)
        accepted = (confidence > 0.5) & np.isfinite(predicted).all(axis=1)
    else:
        predicted, accepted = query_sparse(payload, anchors, source_distance)
    in_bounds = ((predicted >= -0.5) & (predicted < SIDE - 0.5)).all(axis=1)
    hits = np.zeros(len(anchors), dtype=bool)
    valid = accepted & in_bounds
    hits[valid] = sample_map(target_distance, predicted[valid]) <= TOLERANCE
    # Store accepted endpoints for notebook plots, including misses and outside predictions.
    points = np.column_stack((anchors[accepted], predicted[accepted], hits[accepted])).round(2).tolist()
    return {"anchors": len(anchors), "accepted": int(accepted.sum()), "hits": int(hits.sum()),
            "eligible": True}, points


def result_path(work, name, pair):
    return work / "matches" / name / f"{pair.key}.json"


def result_signature(work, name, images):
    revision = f"wires-v1:{name}:{SIDE}:{TOLERANCE}:{MAX_ANCHORS}:{ANCHOR_CELL}:{matchers.device.type}"
    for frame in images:
        revision += frame.digest
        for method in MASK_NAMES:
            path = mask_path(work, method, frame)
            stat = path.stat()
            revision += f":{stat.st_mtime_ns}:{stat.st_size}"
    return hashlib.sha256(revision.encode()).hexdigest()


def run_matcher(name, frames, pairs, work):
    pending = []
    for pair in pairs:
        images = tuple(frames[key] for key in (pair.a, pair.b))
        path = result_path(work, name, pair)
        revision = result_signature(work, name, images)
        if path.is_file() and json.loads(path.read_text())["signature"] == revision:
            continue
        pending.append((pair, images, path, revision))
    if not pending:
        print(f"{name}: все {len(pairs)} пары уже рассчитаны.", flush=True)
        return
    matcher = make_matcher(name, work)
    try:
        for index, (pair, images, path, revision) in enumerate(pending, 1):
            a, b = [read_rgb(frame.path) for frame in images]
            matchers._sync()
            started = time.perf_counter()
            payload = raw_match(name, matcher, a, b)
            matchers._sync()
            elapsed = time.perf_counter() - started
            records, points = {}, {}
            for method in MASK_NAMES:
                geometry = []
                for frame in images:
                    path_mask = mask_path(work, method, frame)
                    geometry.append(mask_geometry(path_mask, path_mask.stat().st_mtime_ns))
                records[method], points[method] = score_match(payload, *geometry)
            write_json(path, {"signature": revision, "model": name, "a": pair.a, "b": pair.b,
                              "pack": images[0].pack, "seconds": elapsed, "metrics": records, "points": points})
            if index % 25 == 0 or index == len(pending):
                print(f"{name}: {index}/{len(pending)} · {elapsed:.2f} с на последнюю пару", flush=True)
    finally:
        del matcher
        release_memory()


def read_results(work, pairs, names=MODEL_NAMES):
    rows = []
    for name in names:
        for pair in pairs:
            path = result_path(work, name, pair)
            if not path.is_file():
                continue
            data = json.loads(path.read_text())
            for method, numbers in data["metrics"].items():
                rows.append({key: data[key] for key in ("model", "a", "b", "pack", "seconds")}
                            | {"pair": pair.key, "mask": method} | numbers)
    return pd.DataFrame(rows)
