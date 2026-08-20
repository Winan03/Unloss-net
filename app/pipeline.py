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
from difflib import SequenceMatcher

import cv2
import numpy as np

from app.verify import normalize_text  # misma normalización OCR que la verificación funcional

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
# Selección del dominio texto (docs §6.8.2): se prueban todos los métodos y se elige la lectura
# con mayor confianza media. Umbral de parada temprana: una lectura por encima ya es "confiada";
# probar más métodos solo sumaría latencia sin ganancia medible (calibrado en caso real del poema:
# la mejor lectura es Otsu x4 con conf 82.4, por debajo de estos umbrales -> se prueban todos).
TEXT_HIGH_CONF = 90.0
RAPID_HIGH_CONF = 0.95
# Banda de confianza para seleccionar sin expected: entre las lecturas a <=CONF_BAND puntos de
# la máxima, gana la más larga. La confianza sola NO predice utilidad funcional (medido en el
# caso real del poema: upscale x4 conf 83.5 ratio 0.672 vs Otsu x4 conf 81.1 ratio 0.768);
# la banda + longitud elige a Otsu x4 (el óptimo real).
CONF_BAND = 3.0


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

def _ocr(img) -> tuple[str | None, float]:
    """Tesseract (psm 6: bloque de texto uniforme, típico de screenshots).

    Devuelve (texto, confianza media) con filtro de confianza (evita contar ruido de una
    imagen sin texto como si fuera lectura) o (None, 0.0) si no hay OCR disponible.
    """
    if not _OCR_OK:
        return None, 0.0
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
            return None, 0.0
            
        text = "\n".join(lines)
        mean_conf = sum(confs) / len(confs)
        
        if len(text) >= OCR_MIN_LEN and mean_conf >= OCR_MIN_CONF:
            return text, mean_conf
        return None, 0.0
    except Exception:
        return None, 0.0


def decode_text_full(img: np.ndarray) -> tuple[str | None, float]:
    """OCR sobre vista acotada (TEXT_MAX); el upscale x4 ya aporta el detalle.

    Devuelve (texto, confianza media) para que el dominio texto seleccione la mejor lectura.
    """
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


def _ocr_rapid(img: np.ndarray) -> tuple[str | None, float]:
    if not rapid_available():
        return None, 0.0
    try:
        res, _ = _RAPID(img)
    except Exception:
        return None, 0.0
    if not res:
        return None, 0.0
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
        return None, 0.0
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
    mean_conf = sum(confs) / len(confs) if confs else 0.0
    if len(text) >= OCR_MIN_LEN and mean_conf >= RAPID_MIN_CONF:
        return text, mean_conf
    return None, 0.0


def decode_text_rapid_full(img: np.ndarray) -> tuple[str | None, float]:
    """OCR (RapidOCR) sobre vista acotada (TEXT_MAX).

    Devuelve (texto, confianza media) para que el dominio texto seleccione la mejor lectura.
    """
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


def select_best_read(reads: list[tuple[float, str]]) -> int:
    """Índice de la mejor lectura sin expected: entre las de confianza a <=CONF_BAND puntos de
    la máxima, la más larga. -1 si vacío.

    Función pura (testeable sin OCR). La confianza sola no predice utilidad funcional
    (medido en el caso real del poema), por eso se combina con longitud dentro de la banda.
    """
    if not reads:
        return -1
    max_conf = max(c for c, _ in reads)
    band = [i for i, (c, _) in enumerate(reads) if c >= max_conf - CONF_BAND]
    return max(band, key=lambda i: len(reads[i][1]))


def select_best_match(reads: list[tuple[float, str]], expected: str) -> int:
    """Índice de la lectura con mayor similitud normalizada al contenido esperado.

    Ancla funcional (no de confianza): cuando el usuario declara qué contenido verificar,
    se elige la restauración cuyo texto más se acerca a lo esperado — misma normalización
    que verify.classify_text. Función pura. -1 si vacío.
    """
    if not reads:
        return -1
    ne = normalize_text(expected)
    best, best_r = 0, -1.0
    for i, (_, text) in enumerate(reads):
        r = SequenceMatcher(None, normalize_text(text), ne).ratio()
        if r > best_r:
            best, best_r = i, r
    return best


def run(img: np.ndarray, text_engine: str = "tesseract", expected: str | None = None) -> list[dict]:
    """Ejecuta los métodos en orden y se detiene en el primero que decodifica.

    Acota la latencia: una vez hay payload verificado, los métodos restantes no aportan
    (el resultado funcional manda, docs §6.4). Los no ejecutados no aparecen en la tabla.

    Orden de dominios: primero QR; solo si ninguna variante QR decodifica se prueban las
    de texto (OCR). Así un QR legible no paga el coste del OCR, y una imagen de texto
    no puede contaminar la ruta QR.

    Dominio texto (docs §6.8.2): se prueban TODOS los métodos de texto y se elige la lectura.
      - Con expected: la de mayor similitud normalizada al contenido esperado (ancla funcional,
        select_best_match). No hay parada temprana: para comparar hace falta probar todo.
      - Sin expected: banda de confianza + la más larga (select_best_read), con parada temprana
        solo si una lectura ya supera TEXT_HIGH_CONF / RAPID_HIGH_CONF.
    Medido en caso real (foto de poema): la selección pasa de la primera lectura (0.653) a
    Otsu x4 (0.768, el óptimo) — la confianza sola habría elegido upscale x4 (0.672).

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
        full = decode_text_full if engine == "tesseract" else decode_text_rapid_full
        high = TEXT_HIGH_CONF if engine == "tesseract" else RAPID_HIGH_CONF
        ne = normalize_text(expected) if expected else None
        reads: list[tuple[float, str]] = []
        candidates: list[dict] = []
        for name, proc in build_text_routes(img):
            t0 = time.perf_counter()
            payload, conf = full(proc)
            res = {
                "domain": "text",
                "name": name,
                "engine": engine,
                "decoded": bool(payload),
                "payload": payload,
                "confidence": round(conf, 1) if payload else None,
                "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
            }
            results.append(res)
            if payload:
                reads.append((conf, payload))
                candidates.append(res)
                if ne is not None:
                    if SequenceMatcher(None, normalize_text(payload), ne).ratio() == 1.0:
                        res["best"] = True
                        res["early"] = "exact"
                        return results
                elif conf >= high:
                    res["best"] = True
                    res["early"] = "conf"
                    return results
        if reads:
            idx = select_best_match(reads, expected) if expected else select_best_read(reads)
            candidates[idx]["best"] = True
            candidates[idx]["early"] = None
            return results
    return results


def most_common_payload(results: list[dict]) -> str | None:
    """Payload del método elegido (flag 'best' en texto) o el más frecuente entre los que decodifican."""
    for r in results:
        if r.get("best"):
            return r["payload"]
    vals = [r["payload"] for r in results if r["decoded"]]
    if not vals:
        return None
    return Counter(vals).most_common(1)[0][0]


def best_reconstruction(img: np.ndarray, results: list[dict]) -> tuple[str, np.ndarray]:
    """Mejor salida para mostrar: el método elegido ('best'); si ninguno, Otsu x4 (mejor esfuerzo)."""
    routes = dict(build_routes(img))
    routes.update(dict(build_text_routes(img)))
    for r in results:
        if r.get("best"):
            return r["name"], routes[r["name"]]
    decoded = [r for r in results if r["decoded"]]
    if decoded:
        return decoded[0]["name"], routes[decoded[0]["name"]]
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