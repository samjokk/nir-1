"""Единый интерфейс для сравнения матчеров.

Каждый матчер — объект с полями name, dense и методом __call__(A, B) -> dict:
    kA   (N, 2) — точки на A, пиксели (x, y) исходного изображения A
    kB   (N, 2) — соответствующие точки на B, пиксели B
    conf (N,)   — уверенность (чем больше, тем лучше; шкалы у моделей разные)
A и B — uint8 RGB массивы (H, W, 3).

Плотные модели отдают поле на сетке с шагом GRID пикселей,
разреженные — то, что нашли сами.
"""
import time

import numpy as np
import torch
import torch.nn.functional as F
import cv2

from romav2.device import device

GRID = 8  # шаг сетки, на которой берём плотное поле, px исходного изображения


def _sync():
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def _to_tensor(a):
    return torch.from_numpy(a).permute(2, 0, 1)[None].float().div(255).to(device)


def _grid_pixels(H, W, step=GRID):
    ys, xs = np.mgrid[step / 2:H:step, step / 2:W:step]
    return np.stack([xs.ravel(), ys.ravel()], 1).astype(np.float32)


def _sample_warp(warp, conf, H_A, W_A, H_B, W_B):
    """warp: (h, w, 2) в нормированных координатах B [-1, 1]; conf: (h, w).
    Сэмплируем на сетке исходного A и переводим в пиксели B."""
    kA = _grid_pixels(H_A, W_A)
    g = torch.from_numpy(np.stack([kA[:, 0] / W_A * 2 - 1, kA[:, 1] / H_A * 2 - 1], 1)).to(warp.device)
    g = g[None, None].float()
    w = F.grid_sample(warp.permute(2, 0, 1)[None].float(), g, mode="bilinear", align_corners=False)[0, :, 0].T
    c = F.grid_sample(conf[None, None].float(), g, mode="bilinear", align_corners=False)[0, 0, 0]
    w, c = w.cpu().numpy(), c.cpu().numpy()
    kB = np.stack([(w[:, 0] + 1) / 2 * W_B, (w[:, 1] + 1) / 2 * H_B], 1)
    return kA, kB, c


# ---------------------------------------------------------------- плотные

class RoMaV2Matcher:
    dense = True

    def __init__(self, setting="precise", model=None):
        from romav2 import RoMaV2
        self.model = model if model is not None else RoMaV2()
        self.model.apply_setting(setting)
        self.name = f"RoMa v2 ({setting})"

    def __call__(self, A, B):
        p = self.model.match(A, B)
        return dict(zip(["kA", "kB", "conf"], _sample_warp(
            p["warp_AB"][0], p["overlap_AB"][0, ..., 0], *A.shape[:2], *B.shape[:2])))


class RoMaV1Matcher:
    dense = True
    name = "RoMa v1"

    def __init__(self, coarse_res=560, upsample_res=864):
        from romatch import roma_outdoor
        self.model = roma_outdoor(device=device, coarse_res=coarse_res, upsample_res=upsample_res,
                                  amp_dtype=torch.float16 if device.type == "cuda" else torch.float32)

    def __call__(self, A, B):
        from PIL import Image
        warp, cert = self.model.match(Image.fromarray(A), Image.fromarray(B), device=device)
        w2 = warp.shape[2] // 2  # симметричный режим: левая половина — A→B
        return dict(zip(["kA", "kB", "conf"], _sample_warp(
            warp[0, :, :w2, 2:], cert[0, :, :w2], *A.shape[:2], *B.shape[:2])))


class RAFTMatcher:
    """Оптический поток RAFT (torchvision). Вход приводим к long side = size, кратно 8."""
    dense = True

    def __init__(self, size=1024, iters=20):
        from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
        # на MPS часть операций RAFT не поддерживается, считаем на CPU если что
        self.dev = device if device.type == "cuda" else torch.device("cpu")
        self.model = raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2).eval().to(self.dev)
        self.size, self.iters = size, iters
        self.name = f"RAFT ({size})"

    @torch.inference_mode()
    def __call__(self, A, B):
        H, W = A.shape[:2]
        s = self.size / max(H, W)
        h, w = int(round(H * s / 8) * 8), int(round(W * s / 8) * 8)
        prep = lambda a: _to_tensor(cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA)).to(self.dev) * 2 - 1
        flow = self.model(prep(A), prep(B), num_flow_updates=self.iters)[-1][0]  # (2, h, w), px малого
        # поток (px малого изображения) -> warp в нормированных координатах B
        ys, xs = torch.meshgrid(torch.arange(h, device=flow.device), torch.arange(w, device=flow.device), indexing="ij")
        warp = torch.stack([(xs + 0.5 + flow[0]) / w * 2 - 1, (ys + 0.5 + flow[1]) / h * 2 - 1], -1)
        conf = torch.ones(h, w, device=flow.device)  # у RAFT нет уверенности
        return dict(zip(["kA", "kB", "conf"], _sample_warp(warp, conf, H, W, *B.shape[:2])))


# ---------------------------------------------------------------- разреженные

class DinoNNMatcher:
    """Взаимные ближайшие соседи по патчам DINOv3 (тот же бэкбон и слой, что у RoMa v2)."""
    dense = False

    def __init__(self, romav2_model, size=800, layer=-1):
        self.f, self.size, self.layer = romav2_model.f, size, layer
        self.name = "DINOv3 NN"

    @torch.inference_mode()
    def __call__(self, A, B):
        def feats(a):
            t = F.interpolate(_to_tensor(a), size=(self.size, self.size), mode="bicubic",
                              align_corners=False, antialias=True)
            return F.normalize(self.f(t)[self.layer][0].float(), dim=-1)  # (h, w, D)
        fA, fB = feats(A), feats(B)
        h, w, D = fA.shape
        S = fA.reshape(-1, D) @ fB.reshape(-1, D).T
        ab, ba = S.argmax(1), S.argmax(0)
        mutual = ba[ab] == torch.arange(len(ab), device=S.device)
        iA = torch.nonzero(mutual)[:, 0]
        iB = ab[iA]
        conf = S[iA, iB].cpu().numpy()
        iA, iB = iA.cpu().numpy(), iB.cpu().numpy()
        def px(i, a):
            H, W = a.shape[:2]
            return np.stack([(i % w + 0.5) / w * W, (i // w + 0.5) / h * H], 1)
        return dict(kA=px(iA, A), kB=px(iB, B), conf=conf)


class LoFTRMatcher:
    dense = False

    def __init__(self, size=840, pretrained="outdoor"):
        import kornia.feature as KF
        self.model = KF.LoFTR(pretrained=pretrained).eval().to(device)
        self.size = size
        self.name = f"LoFTR ({size})"

    @torch.inference_mode()
    def __call__(self, A, B):
        H, W = A.shape[:2]
        s = self.size / max(H, W)
        h, w = int(H * s) // 8 * 8, int(W * s) // 8 * 8
        g = lambda a: _to_tensor(cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA)).mean(1, keepdim=True)
        out = self.model({"image0": g(A), "image1": g(B)})
        sc = np.array([W / w, H / h], np.float32)
        return dict(kA=out["keypoints0"].cpu().numpy() * sc, kB=out["keypoints1"].cpu().numpy() * sc,
                    conf=out["confidence"].cpu().numpy())


class DiskLightGlueMatcher:
    dense = False

    def __init__(self, size=1024, n=4096):
        import kornia.feature as KF
        self.disk = KF.DISK.from_pretrained("depth").eval().to(device)
        self.lg = KF.LightGlueMatcher("disk").eval().to(device)
        self.size, self.n = size, n
        self.name = "DISK+LightGlue"

    @torch.inference_mode()
    def __call__(self, A, B):
        import kornia.feature as KF
        H, W = A.shape[:2]
        s = self.size / max(H, W)
        h, w = int(H * s) // 16 * 16, int(W * s) // 16 * 16
        t = lambda a: _to_tensor(cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA))
        fA, fB = self.disk(torch.cat([t(A), t(B)]), n=self.n, pad_if_not_divisible=True)
        lafA = KF.laf_from_center_scale_ori(fA.keypoints[None], torch.ones(1, len(fA.keypoints), 1, 1, device=device))
        lafB = KF.laf_from_center_scale_ori(fB.keypoints[None], torch.ones(1, len(fB.keypoints), 1, 1, device=device))
        dist, idx = self.lg(fA.descriptors, fB.descriptors, lafA, lafB, hw1=(h, w), hw2=(h, w))
        idx = idx.cpu().numpy()
        sc = np.array([W / w, H / h], np.float32)
        return dict(kA=fA.keypoints.cpu().numpy()[idx[:, 0]] * sc, kB=fB.keypoints.cpu().numpy()[idx[:, 1]] * sc,
                    conf=(1 - dist[:, 0]).cpu().numpy())


class TemplateMatcher:
    """Классика без нейросети: центр A (high-pass) ищем в B нормированной корреляцией.
    Даёт один глобальный сдвиг — возвращаем его как сетку одинаковых соответствий."""
    dense = False
    name = "Template (NCC)"

    def __init__(self, scale=4, blur=6):
        self.s, self.blur = scale, blur

    def __call__(self, A, B):
        def hp(a):
            g = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float32)
            g = cv2.resize(g, None, fx=1 / self.s, fy=1 / self.s, interpolation=cv2.INTER_AREA)
            g = g - cv2.GaussianBlur(g, (0, 0), self.blur)
            return (g - g.mean()) / (g.std() + 1e-6)
        a, b = hp(A), hp(B)
        h, w = a.shape
        r = cv2.matchTemplate(b, a[h // 4:3 * h // 4, w // 4:3 * w // 4], cv2.TM_CCOEFF_NORMED)
        y, x = np.unravel_index(r.argmax(), r.shape)
        d = np.array([(x - w // 4) * self.s, (y - h // 4) * self.s], np.float32)
        kA = _grid_pixels(*A.shape[:2], step=32)
        return dict(kA=kA, kB=kA + d, conf=np.full(len(kA), r.max(), np.float32))


# ---------------------------------------------------------------- запуск и метрики

def run(matcher, A, B):
    _sync(); t = time.time()
    out = matcher(A, B)
    _sync()
    out["time"] = time.time() - t
    out["kA"] = np.asarray(out["kA"], np.float32).reshape(-1, 2)
    out["kB"] = np.asarray(out["kB"], np.float32).reshape(-1, 2)
    out["conf"] = np.asarray(out["conf"], np.float32).reshape(-1)
    return out


def top_n(out, n):
    """Оставить n самых уверенных соответствий — одинаковый бюджет для всех моделей."""
    if len(out["conf"]) <= n:
        return out
    if np.ptp(out["conf"]) == 0:  # уверенности нет (RAFT, шаблон) — равномерная случайная выборка
        i = np.random.default_rng(0).choice(len(out["conf"]), n, replace=False)
    else:
        i = np.argsort(-out["conf"])[:n]
    return {**out, "kA": out["kA"][i], "kB": out["kB"][i], "conf": out["conf"][i]}


def cycle_rate(ab, ba, tol=5.0, radius=GRID):
    """Доля соответствий A→B, которые B→A возвращает на место (ошибка < tol px).
    Для точки kB берём ближайшую исходную точку прохода B→A в радиусе radius."""
    if len(ab["kA"]) == 0 or len(ba["kA"]) == 0:
        return np.nan
    from scipy.spatial import cKDTree
    d, j = cKDTree(ba["kA"]).query(ab["kB"], distance_upper_bound=radius)
    ok = np.isfinite(d)
    if not ok.any():
        return 0.0
    back = ba["kB"][j[ok]] + (ab["kB"][ok] - ba["kA"][j[ok]])  # поправка на смещение до узла
    err = np.linalg.norm(back - ab["kA"][ok], axis=1)
    return float((err < tol).sum() / len(ab["kA"]))


def pair_metrics(ab, ref, cycle=np.nan, tol_pt=30.0, tol_pair=60.0):
    d = ab["kB"] - ab["kA"]
    n = len(d)
    if n == 0:
        return dict(n=0, dx=np.nan, dy=np.nan, inlier=0.0, correct=False, zero=False,
                    cycle=cycle, time=ab["time"])
    med = np.median(d, 0)
    return dict(
        n=n, dx=med[0], dy=med[1],
        inlier=float((np.linalg.norm(d - ref, axis=1) < tol_pt).mean()),  # доля точек у истинного сдвига
        correct=bool(np.linalg.norm(med - ref) < tol_pair),               # пара целиком в верном режиме
        zero=bool(np.linalg.norm(med) < tol_pair),                        # ложный режим «почти ноль»
        cycle=cycle,
        time=ab["time"],
    )
