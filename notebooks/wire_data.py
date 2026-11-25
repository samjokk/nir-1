"""Build a reproducible manifest and plausible same-scene pairs for etl-2."""

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


SIDE = 640
PACKS = ("defect", "normal")


@dataclass(frozen=True)
class Frame:
    pack: str
    path: Path
    group: str
    position: int
    digest: str

    @property
    def key(self):
        return f"{self.pack}/{self.path.name}"


@dataclass(frozen=True)
class Pair:
    a: str
    b: str
    inliers: int

    @property
    def key(self):
        return hashlib.sha256(f"{self.a}\n{self.b}".encode()).hexdigest()[:18]


def capture_group(name):
    if match := re.match(r"DJI_(\d{8})(\d{6})_\d+_([A-Z])_", name):
        date, time, camera = match.groups()
        return f"DJI/{date}/{camera}", timestamp(date, time)
    if match := re.match(r"(\d{8}_\d{6})-(\d+)", name):
        return f"clip/{match[1]}", int(match[2])
    if match := re.match(r"(\d{8})_(\d{6})(?:\(\d+\))?\.", name):
        return f"photos/{match[1]}", timestamp(*match.groups())
    if match := re.match(r"(\d+)\s*\((\d+)\)", name):
        return f"numbered/{match[1]}", int(match[2])
    if match := re.match(r"image-(\d+)-", name):
        return "numbered/image", int(match[1])
    return f"single/{name}", 0


def timestamp(date, time):
    point = datetime.strptime(date + time, "%Y%m%d%H%M%S")
    return point.hour * 3600 + point.minute * 60 + point.second


def read_manifest(root):
    directories = {path.name.casefold(): path for path in root.iterdir() if path.is_dir()}
    frames = {}
    for pack in PACKS:
        for path in sorted(directories[pack].iterdir()):
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png"} or not path.is_file():
                continue
            group, position = capture_group(path.name)
            with path.open("rb") as image:
                digest = hashlib.file_digest(image, "sha256").hexdigest()
            frame = Frame(pack, path.resolve(), group, position, digest)
            frames[frame.key] = frame
    return frames


@lru_cache(maxsize=12)
def read_rgb(path):
    with Image.open(path) as raw:
        return np.asarray(ImageOps.exif_transpose(raw).convert("RGB").resize(
            (SIDE, SIDE), Image.Resampling.LANCZOS)).copy()


def write_json(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(document, ensure_ascii=False) + "\n")
    temporary.replace(path)


def pair_signature(frames):
    fingerprint = "\n".join(f"{key}:{frame.digest}" for key, frame in frames.items())
    return hashlib.sha256(("same-scene-sift-v1\n" + fingerprint).encode()).hexdigest()


def same_scene(a, b, matcher):
    xy_a, descriptors_a = a
    xy_b, descriptors_b = b
    if descriptors_a is None or descriptors_b is None or min(len(xy_a), len(xy_b)) < 2:
        return 0
    good = [best for best, second in matcher.knnMatch(descriptors_a, descriptors_b, k=2)
            if best.distance < 0.73 * second.distance]
    if len(good) < 16:
        return 0
    source = xy_a[[m.queryIdx for m in good]]
    target = xy_b[[m.trainIdx for m in good]]
    _, kept = cv2.findHomography(source, target, cv2.RANSAC, 3.0)
    if kept is None:
        return 0
    inliers = kept.ravel().astype(bool)
    count = int(inliers.sum())
    if count < 16 or count / len(good) < 0.35:
        return 0
    if min(cv2.contourArea(cv2.convexHull(points[inliers])) for points in (source, target)) < 520:
        return 0
    return count


def build_pairs(frames, work):
    manifest = work / "pairs.json"
    signature = pair_signature(frames)
    if manifest.is_file():
        previous = json.loads(manifest.read_text())
        if previous["signature"] == signature:
            return [Pair(**pair) for pair in previous["pairs"]]
    cv2.setRNGSeed(47)
    sift = cv2.SIFT_create(nfeatures=480)
    matcher = cv2.BFMatcher()

    @lru_cache(maxsize=16)
    def keypoints(frame):
        reduced = cv2.resize(read_rgb(frame.path), (384, 384), interpolation=cv2.INTER_AREA)
        points, descriptors = sift.detectAndCompute(cv2.cvtColor(reduced, cv2.COLOR_RGB2GRAY), None)
        return np.asarray([point.pt for point in points], np.float32), descriptors

    groups = defaultdict(list)
    for frame in frames.values():
        groups[frame.pack, frame.group].append(frame)
    candidates = []
    for (pack, group), entries in groups.items():
        entries.sort(key=lambda frame: (frame.position, frame.key))
        max_gap = 45 if group.startswith("DJI/") else 15 if group.startswith("photos/") else 24
        for index, a in enumerate(entries):
            for b in entries[index + 1:index + 5]:
                gap = b.position - a.position
                if gap > max_gap or a.digest == b.digest:
                    continue
                count = same_scene(keypoints(a), keypoints(b), matcher)
                if count:
                    candidates.append((gap, -count, Pair(a.key, b.key, count)))
    candidates.sort(key=lambda item: (item[0], item[1], item[2].a, item[2].b))

    # First use disjoint pairs, then give unpaired photos one verified partner where possible.
    uses = defaultdict(int)
    pairs = []
    for maximum in (1, 2):
        for _, _, pair in candidates:
            a, b = uses[pair.a], uses[pair.b]
            if min(a, b) == 0 and max(a, b) < maximum:
                pairs.append(pair)
                uses[pair.a] += 1
                uses[pair.b] += 1
    pairs.sort(key=lambda pair: (pair.a, pair.b))
    write_json(manifest, {"signature": signature, "pairs": [asdict(pair) for pair in pairs]})
    return pairs
