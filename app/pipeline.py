"""Pipeline clásico de Unloss: imagen -> métodos de restauración -> decodificación de QR.

Función pura (sin HTTP, sin I/O): recibe una imagen BGR (numpy) y devuelve
los métodos probados con su decodificación. Portado de notebooks/17 (docs §6.9):
la ruta clásica es el motor de producción; el modelo v9b es experimental.
"""

from __future__ import annotations

import time
from collections import Counter

import cv2
import numpy as np

_DECO = cv2.QRCodeDetector()


# ---------- transformaciones básicas ----------

def to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def to_bgr(g: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def up(img: np.ndarray, k: int = 4) -> np.ndarray:
    return cv2.resize(img, (img.shape[1] * k, img.shape[0] * k), interpolation=cv2.INTER_CUBIC)


def contrast(g: np.ndarray, f: float = 1.8) -> np.ndarray:
    return cv2.convertScaleAbs(g, alpha=f, beta=0)


def adaptive(g: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 31, 5)


def sharpen(img: np.ndarray) -> np.ndarray:
    k = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    return cv2.filter2D(img, -1, k)


def otsu(img: np.ndarray) -> np.ndarray:
    _, th = cv2.threshold(to_gray(img), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return to_bgr(th)


def invert(img: np.ndarray) -> np.ndarray:
    return cv2.bitwise_not(img)


# ---------- decodificación ----------

def decode(img: np.ndarray) -> str | None:
    """Decodifica un QR. Devuelve el payload o None. cv2 primero; pyzbar si existe."""
    try:
        d, _, _ = _DECO.detectAndDecode(img)
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


# ---------- pipeline ----------

def build_routes(img: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Orden de métodos probados (docs §6.9)."""
    x4 = up(img, 4)
    g = to_gray(img)
    return [
        ("Original", img),
        ("Upscale x4 (cubic)", x4),
        ("Escala de grises", to_bgr(g)),
        ("Gris + contraste x1.8", to_bgr(contrast(g, 1.8))),
        ("Afilar + contrastar", sharpen(contrast(img))),
        ("Umbral adaptativo", to_bgr(adaptive(g))),
        ("Otsu (original)", otsu(img)),
        ("Otsu x4", otsu(x4)),
        ("Invertido + contraste", invert(contrast(img))),
    ]


def run(img: np.ndarray) -> list[dict]:
    """Ejecuta todos los métodos sobre la imagen y devuelve resultados por método."""
    results = []
    for name, proc in build_routes(img):
        t0 = time.perf_counter()
        payload = decode(proc)
        results.append({
            "name": name,
            "decoded": bool(payload),
            "payload": payload,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
        })
    return results


def most_common_payload(results: list[dict]) -> str | None:
    """Payload más frecuente entre los métodos que decodifican (lecturas distintas -> la mayoría)."""
    vals = [r["payload"] for r in results if r["decoded"]]
    if not vals:
        return None
    return Counter(vals).most_common(1)[0][0]


def best_reconstruction(img: np.ndarray, results: list[dict]) -> tuple[str, np.ndarray]:
    """Mejor salida para mostrar: el método que decodificó; si ninguno, Otsu x4 (mejor esfuerzo)."""
    decoded = [r for r in results if r["decoded"]]
    if decoded:
        name = decoded[0]["name"]
        return name, dict(build_routes(img))[name]
    return "Otsu x4", otsu(up(img, 4))


def fit_max(img: np.ndarray, m: int = 2000) -> np.ndarray:
    """Acota el lado mayor para limitar el coste (los QR se degradan poco con downscale moderado)."""
    h, w = img.shape[:2]
    s = m / max(h, w)
    if s >= 1:
        return img
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


# ---------- métricas de píxel (secundarias: no predicen éxito funcional) ----------

def psnr(a: np.ndarray, b: np.ndarray) -> float:
    return float(cv2.PSNR(a, b))


def ssim(a: np.ndarray, b: np.ndarray, win: int = 7, sigma: float = 1.5) -> float:
    import scipy.ndimage as ndi

    if a.ndim == 3:
        a = to_gray(a)
    if b.ndim == 3:
        b = to_gray(b)
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    k1, k2 = 0.01, 0.03
    c1 = (k1 * 255) ** 2
    c2 = (k2 * 255) ** 2
    g = np.zeros((win, win), np.float64)
    for i in range(win):
        for j in range(win):
            g[i, j] = np.exp(-((i - win // 2) ** 2 + (j - win // 2) ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    mu1 = ndi.convolve(a, g)
    mu2 = ndi.convolve(b, g)
    s11 = ndi.convolve(a * a, g) - mu1 ** 2
    s22 = ndi.convolve(b * b, g) - mu2 ** 2
    s12 = ndi.convolve(a * b, g) - mu1 * mu2
    m = ((2 * mu1 * mu2 + c1) * (2 * s12 + c2)) / ((mu1 ** 2 + mu2 ** 2 + c1) * (s11 + s22 + c2))
    return float(m.mean())


def metrics(subida: np.ndarray, rec: np.ndarray) -> dict:
    """PSNR/SSIM subida vs reconstrucción (rec re-escalada al tamaño de la subida)."""
    rs = cv2.resize(rec, (subida.shape[1], subida.shape[0]), interpolation=cv2.INTER_CUBIC)
    return {
        "subida_vs_rec_psnr": round(psnr(subida, rs), 2),
        "subida_vs_rec_ssim": round(ssim(subida, rs), 4),
    }