"""Pipeline clásico de Unloss: imagen -> métodos de restauración -> decodificación de QR/texto.

Función pura (sin HTTP, sin I/O): recibe una imagen BGR (numpy) y devuelve
los métodos probados con su decodificación. Portado de notebooks/17 (docs §6.9):
la ruta clásica es el motor de producción; el modelo v9b es experimental.

Dos dominios (docs §6.8.2):
- QR  -> cv2 QRCodeDetector (+ pyzbar si existe)
- Texto -> Tesseract (mismo motor que la validación de la memoria: "Tesseract ×4"),
           con filtro de confianza de Tesseract y normalización OCR en verify.
"""

from __future__ import annotations

import time
from collections import Counter

import cv2
import numpy as np

_DECO = cv2.QRCodeDetector()

try:
    import pytesseract
    pytesseract.get_tesseract_version()
    _OCR_OK = True
except Exception:
    _OCR_OK = False

# Los upscale x4 generan imágenes enormes; cv2 los decodifica igualmente a un tamaño
# acotado (medido: > 3072 px no mejora la lectura y cuesta ~5x más).
DECODE_MAX = 3072
SSIM_MAX = 600  # SSIM se calcula sobre vista reducida: es métrica secundaria, no predice utilidad
TEXT_MAX = 2600  # cota del lado mayor para OCR (acota latencia; el detalle lo aporta el upscale)
OCR_MIN_LEN = 3   # mínimo de caracteres para no contar ruido como texto leído
OCR_MIN_CONF = 50.0  # confianza media mínima de Tesseract para contar el texto como leído
RAPID_MIN_CONF = 0.7  # gate de RapidOCR: su escala de confianza no es la de Tesseract. Calibrado
                      # en medición sintética local: lecturas legítimas ~0.97, ruido en Moiré
                      # extremo ~0.58. Experimental: no medido en el set real de la memoria.


def ocr_available() -> bool:
    return _OCR_OK


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

def _single(img: np.ndarray) -> str | None:
    """Decodifica con cv2 y, si existe, pyzbar. Devuelve el payload o None."""
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


def decode(img: np.ndarray) -> str | None:
    """Decodifica con vía rápida (vista acotada) y reintento a resolución completa.

    El upscale x4 genera imágenes enormes; la mayoría de los casos se leen en la vista
    acotada (DECODE_MAX) y terminan en milisegundos. Si no, se reintenta a resolución
    completa (necesario en QRs desenfocados donde el detalle importa, docs §6.9).
    """
    view = fit_max(img, DECODE_MAX) if max(img.shape[:2]) > DECODE_MAX else img
    payload = _single(view)
    if payload:
        return payload
    if view is not img:
        return _single(img)
    return None


# ---------- dominio texto (Tesseract, mismo motor que la memoria §6.8.2) ----------

def _ocr(img: np.ndarray) -> str | None:
    """Tesseract (psm 6: bloque de texto uniforme, típico de screenshots).

    Devuelve el texto leído con filtro de confianza media (evita contar ruido de una
    imagen sin texto como si fuera lectura) o None si no hay OCR disponible.
    """
    if not _OCR_OK:
        return None
    try:
        data = pytesseract.image_to_data(
            to_gray(img), output_type=pytesseract.Output.DICT, config="--oem 1 --psm 6")
        lines = []
        confs = []
        current_line = []
        last_block = -1
        last_par = -1
        last_line = -1

        for i, w in enumerate(data["text"]):
            w = w.strip()
            conf = data["conf"][i]
            
            if str(conf) in ("-1", ""):
                continue
            if not w:
                continue

            b = data["block_num"][i]
            p = data["par_num"][i]
            l = data["line_num"][i]

            if (b, p, l) != (last_block, last_par, last_line):
                if current_line:
                    lines.append(" ".join(current_line))
                    current_line = []
                last_block, last_par, last_line = b, p, l
                
            current_line.append(w)
            confs.append(float(conf))

        if current_line:
            lines.append(" ".join(current_line))
            
        if not lines:
            return None
            
        text = "\n".join(lines)
        
        if len(text) >= OCR_MIN_LEN and sum(confs) / len(confs) >= OCR_MIN_CONF:
            return text
        return None
    except Exception:
        return None


def decode_text(img: np.ndarray) -> str | None:
    """OCR sobre vista acotada (TEXT_MAX); el upscale x4 ya aporta el detalle."""
    view = fit_max(img, TEXT_MAX) if max(img.shape[:2]) > TEXT_MAX else img
    return _ocr(view)


# ---------- dominio texto, lector alternativo experimental (RapidOCR, ONNX/CPU) ----------
# Estado: EXPERIMENTAL. No está validado sobre el set de la memoria (§6.8.2) ni sobre datos
# reales; se documenta así hasta que se mida. Mismo gate de confianza que _ocr.

_RAPID = None
_RAPID_TRIED = False


def rapid_available() -> bool:
    """RapidOCR (modelos PaddleOCR en ONNX, Apache-2.0) disponible? Inicializa una sola vez.

    Perceptrón multicapa de detección + reconocimiento (no depende de la separación de glifos
    como Tesseract, por lo que aguanta mejor degradación/fotos de pantalla en principio).
    """
    global _RAPID, _RAPID_TRIED
    if _RAPID_TRIED:
        return _RAPID is not None
    _RAPID_TRIED = True
    try:
        from rapidocr_onnxruntime import RapidOCR
        _RAPID = RapidOCR()
    except Exception:
        _RAPID = None
    return _RAPID is not None


def _ocr_rapid(img: np.ndarray) -> str | None:
    if not rapid_available():
        return None
    try:
        res, _ = _RAPID(img)
    except Exception:
        return None
    if not res:
        return None
    items = []
    for box, text, conf in res:
        text = (text or "").strip()
        if not text:
            continue
        try:
            conf = float(conf)
        except (TypeError, ValueError):
            conf = 0.0
        items.append((box, text, conf))
    if not items:
        return None
    items.sort(key=lambda it: (round(min(p[1] for p in it[0]) / 12), min(p[0] for p in it[0])))
    lines: list[str] = []
    confs: list[float] = []
    last_y = None
    for box, text, conf in items:
        y = min(p[1] for p in box)
        if last_y is not None and abs(y - last_y) > 12:
            lines.append("")
        if not lines or lines[-1] == "":
            lines.append(text)
        else:
            lines[-1] += " " + text
        last_y = y
        confs.append(conf)
    text = "\n".join(l for l in lines if l)
    if len(text) >= OCR_MIN_LEN and sum(confs) / len(confs) >= RAPID_MIN_CONF:
        return text
    return None


def decode_text_rapid(img: np.ndarray) -> str | None:
    view = fit_max(img, TEXT_MAX) if max(img.shape[:2]) > TEXT_MAX else img
    return _ocr_rapid(view)


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


def build_text_routes(img: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Métodos del dominio texto (docs §6.8.2: Tesseract con upscale)."""
    x4 = up(img, 4)
    g = to_gray(img)
    return [
        ("Texto · original", img),
        ("Texto · upscale ×4 (cubic)", x4),
        ("Texto · grises + contraste", to_bgr(contrast(g, 1.8))),
        ("Texto · umbral adaptativo", to_bgr(adaptive(g))),
        ("Texto · Otsu ×4", otsu(x4)),
    ]


def run(img: np.ndarray, text_engine: str = "tesseract") -> list[dict]:
    """Ejecuta los métodos en orden y se detiene en el primero que decodifica.

    Acota la latencia: una vez hay payload verificado, los métodos restantes no aportan
    (el resultado funcional manda, docs §6.4). Los no ejecutados no aparecen en la tabla.

    Orden de dominios: primero QR; solo si ninguna variante QR decodifica se prueban las
    de texto (OCR). Así un QR legible no paga el coste del OCR, y una imagen de texto
    no puede contaminar la ruta QR.

    text_engine (lector de texto, experimental):
      - "tesseract" -> clásico, el validado en la memoria §6.8.2 (defecto)
      - "rapid"     -> RapidOCR (ONNX/CPU, Apache-2.0), experimental, no validado en el set
      - "auto"      -> clásico primero; si no lee, RapidOCR como refuerzo
    """
    readers = {"tesseract": ("tesseract",), "rapid": ("rapid",), "auto": ("tesseract", "rapid")}
    engine_order = readers.get(text_engine, readers["tesseract"])
    results = []
    for name, proc in build_routes(img):
        t0 = time.perf_counter()
        payload = decode(proc)
        results.append({
            "domain": "qr",
            "name": name,
            "decoded": bool(payload),
            "payload": payload,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
        })
        if payload:
            return results
    for engine in engine_order:
        for name, proc in build_text_routes(img):
            t0 = time.perf_counter()
            payload = (decode_text if engine == "tesseract" else decode_text_rapid)(proc)
            results.append({
                "domain": "text",
                "name": name,
                "engine": engine,
                "decoded": bool(payload),
                "payload": payload,
                "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
            })
            if payload:
                return results
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
        routes = dict(build_routes(img))
        routes.update(dict(build_text_routes(img)))
        return name, routes[name]
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
    """PSNR/SSIM subida vs reconstrucción. SSIM se calcula sobre vista reducida (SSIM_MAX):
    es una métrica secundaria y no predice el éxito funcional (docs §6.4)."""
    rs = cv2.resize(rec, (subida.shape[1], subida.shape[0]), interpolation=cv2.INTER_CUBIC)
    if max(subida.shape[:2]) > SSIM_MAX:
        s = SSIM_MAX / max(subida.shape[:2])
        rs = cv2.resize(rs, (int(rs.shape[1] * s), int(rs.shape[0] * s)), interpolation=cv2.INTER_AREA)
        subida = cv2.resize(subida, (int(subida.shape[1] * s), int(subida.shape[0] * s)),
                            interpolation=cv2.INTER_AREA)
    return {
        "subida_vs_rec_psnr": round(psnr(subida, rs), 2),
        "subida_vs_rec_ssim": round(ssim(subida, rs), 4),
    }