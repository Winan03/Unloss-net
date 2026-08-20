"""Ruta experimental del modelo v9b (best-effort). NUNCA define el estado final.

Degrada con gracia: sin torch o sin models/v9b_net.pt reporta attempted=False
y no bloquea. El motor de producción es el pipeline clásico (AGENTS.md).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_OK = True
except ImportError:  # pragma: no cover
    TORCH_OK = False

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "v9b_net.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") if TORCH_OK else None

FEAT = 384
POC_OUT = 128
POC_CACHE: dict[int, tuple] = {}

_net = None


# ---------- arquitectura V9Net (igual que notebooks/17) ----------

if TORCH_OK:  # pragma: no cover (sin torch en Render free nunca llega aquí)

    class ConvBN(nn.Module):
        def __init__(self, cin, cout, k=3, s=1, p=1):
            super().__init__()
            self.c = nn.Conv2d(cin, cout, k, s, p)
            self.bn = nn.BatchNorm2d(cout)

        def forward(self, x):
            return F.relu(self.bn(self.c(x)))


    class Down(nn.Module):
        def __init__(self, cin, cout):
            super().__init__()
            self.block = nn.Sequential(ConvBN(cin, cout), ConvBN(cout, cout))

        def forward(self, x):
            x = self.block(x)
            return F.max_pool2d(x, 2)


    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.c0 = ConvBN(3, 48)
            self.d1 = Down(48, 96)
            self.d2 = Down(96, 192)
            self.d3 = Down(192, FEAT)

        def forward(self, x):
            return self.d3(self.d2(self.d1(self.c0(x))))


    class GridHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.up = nn.Sequential(
                nn.ConvTranspose2d(FEAT, 96, 2, 2), nn.BatchNorm2d(96), nn.ReLU(inplace=True),
                nn.ConvTranspose2d(96, 32, 2, 2), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            )
            self.h = nn.Conv2d(32, 1, 1)

        def forward(self, f):
            return self.h(self.up(f))


    class V9Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = Encoder()
            self.grid = GridHead()

        def forward(self, x):
            return self.grid(self.enc(x))


# ---------- geometría del QR (v17) ----------

def _module_geometry(canvas, box, mc, border=4, out=POC_OUT):
    idx = np.floor((np.arange(out) + 0.5) * canvas / out).astype(np.int64)
    qrpx = (mc + 2 * border) * box
    x0 = (canvas - qrpx) // 2
    bounds = x0 + (border + np.arange(mc + 1)) * box
    dst = np.clip(np.searchsorted(idx, bounds), 0, out - 1)
    centers = 0.5 * (dst[:-1] + dst[1:] - 1)
    sx = idx[:, None]
    sy = idx[None, :]
    in_qr = (sx >= x0) & (sx < x0 + qrpx) & (sy >= x0) & (sy < x0 + qrpx)
    mx = (sx - x0) // box
    my = (sy - x0) // box
    in_mod = in_qr & (mx >= border) & (mx < border + mc) & (my >= border) & (my < border + mc)
    k = (my - border) * mc + (mx - border)
    mm = np.where(in_mod, k, -1)
    return centers, mm


def _poc_geom(mc):
    if mc not in POC_CACHE:
        box = 2 if mc <= 56 else 1
        centers, mm = _module_geometry(POC_OUT, box, mc, out=POC_OUT)
        cnx = 2 * centers / (POC_OUT - 1) - 1
        xx, yy = np.meshgrid(cnx, cnx)
        cn = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
        POC_CACHE[mc] = (centers, mm, cn)
    return POC_CACHE[mc]


def _up(img, k=4):
    return cv2.resize(img, (img.shape[1] * k, img.shape[0] * k), interpolation=cv2.INTER_CUBIC)


def _to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _to_bgr(g):
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def _otsu(img):
    _, th = cv2.threshold(_to_gray(img), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return _to_bgr(th)


def _get_quad(img):
    deco = cv2.QRCodeDetector()
    for v in [img, _up(img, 4), _up(_otsu(img), 4), _up(_otsu(img), 2)]:
        ok, pts = deco.detect(v)
        if ok and pts is not None and len(pts):
            return pts[0].astype(np.float32).reshape(4, 2)
    return None


if TORCH_OK:  # pragma: no cover (sin torch en Render free nunca llega aquí)

    def _grid_soft(net, img256, mc, cn):
        x = torch.from_numpy(img256.transpose(2, 0, 1).astype(np.float32) / 255.0).unsqueeze(0)
        with torch.no_grad():
            logits = net(x.to(DEVICE))
            g = F.grid_sample(logits, cn.view(1, 1, mc * mc, 2).to(DEVICE),
                              mode="bilinear", align_corners=True)
        p = torch.sigmoid(g).view(mc, mc).cpu().numpy().astype(np.float32)
        return (p > 0.5).astype(np.uint8)


def _render_qr(grid, box=8, border=4):
    mc = grid.shape[0]
    h = (mc + 2 * border) * box
    img = np.full((h, h, 3), 255, np.uint8)
    for r in range(mc):
        for c in range(mc):
            if grid[r, c] == 0:
                img[(border + r) * box:(border + r + 1) * box,
                    (border + c) * box:(border + c + 1) * box] = 0
    return img


def _decode(img):
    try:
        d, _, _ = cv2.QRCodeDetector().detectAndDecode(img)
        if d:
            return d
    except Exception:
        pass
    try:
        from pyzbar.pyzbar import decode as zbar
        for r in zbar(img):
            if r.data:
                return r.data.decode("utf-8", "replace")
    except Exception:
        pass
    return None


# ---------- carga + ruta ----------

def _load_model():
    global _net
    if _net is not None:
        return _net, None
    if not TORCH_OK:
        return None, "torch no instalado"
    if not MODEL_PATH.exists():
        return None, "checkpoint v9b_net.pt no presente"
    try:
        ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
        net = V9Net().to(DEVICE)
        net.load_state_dict(ckpt)
        net.eval()
        _net = net
        return net, None
    except Exception as e:
        return None, f"no se pudo cargar el checkpoint: {e}"


def attempt(img: np.ndarray) -> dict:
    """Intenta la ruta v9b. Devuelve attempted/payload/note/reason. Nunca bloquea."""
    net, reason = _load_model()
    if net is None:
        return {"attempted": False, "payload": None,
                "note": "no disponible", "reason": reason}
    try:
        quad = _get_quad(img)
        if quad is None:
            return {"attempted": True, "payload": None,
                    "note": "no se detectó el quad del QR (cv2 ni en x4/otsu)", "reason": None}
        for v in range(1, 11):
            mc = 21 + 4 * v
            box = 2
            qrpx = (mc + 8) * box
            dst = np.array([[0, 0], [qrpx - 1, 0], [qrpx - 1, qrpx - 1], [0, qrpx - 1]],
                           dtype=np.float32)
            m = cv2.getPerspectiveTransform(quad, dst)
            w = cv2.warpPerspective(img, m, (qrpx, qrpx), flags=cv2.INTER_CUBIC)
            c = np.full((256, 256, 3), 255, np.uint8)
            x0 = (256 - qrpx) // 2
            c[x0:x0 + qrpx, x0:x0 + qrpx] = w
            cn = _poc_geom(mc)[2]
            grid = _grid_soft(net, c, mc, cn)
            clean = _render_qr(grid)
            pl = _decode(clean)
            if pl:
                return {"attempted": True, "payload": pl,
                        "note": f"v9b (grid soft -> QR limpio, versión v{v})", "reason": None}
        return {"attempted": True, "payload": None,
                "note": "v9b: probadas v1-v10, ninguna decodificó", "reason": None}
    except Exception as e:
        return {"attempted": True, "payload": None,
                "note": "error en la ruta del modelo", "reason": str(e)}