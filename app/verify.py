"""Verificación funcional: clasifica el resultado del pipeline contra el contenido esperado.

Función pura (sin I/O, sin HTTP). Estados:
- verified        -> decodifica y coincide con lo esperado
- decoded         -> decodifica; no había contenido esperado
- false_positive  -> decodifica pero difiere de lo esperado (más peligroso que no leer)
- not_decoded     -> ninguna variante decodificó

Dominio texto (docs §6.8.2): la comparación usa normalización OCR estándar (LineAcc)
aplicada por igual a la salida del OCR y al contenido esperado. Solo se afirma "verified"
cuando la coincidencia normalizada es exacta; una lectura parecida pero no idéntica se
reporta como decoded con una nota (no se sobrevende la coincidencia).
"""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

PARTIAL_THRESHOLD = 0.9  # similitud mínima para reportar "lectura parecida, revisa"


def normalize_text(s: str) -> str:
    """Normalización OCR estándar (minúsculas, sin acentos, solo alfanuméricos, espacios colapsados)."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def classify(payload: str | None, expected: str | None) -> tuple[str, bool | None]:
    """Dominio QR: comparación exacta del payload."""
    if not payload:
        return "not_decoded", None
    if not expected:
        return "decoded", None
    if payload == expected:
        return "verified", True
    return "false_positive", False


def classify_text(payload: str | None, expected: str | None) -> tuple[str, bool | None, str | None]:
    """Dominio texto: comparación con normalización OCR. Devuelve (status, matches, note)."""
    if not payload or not payload.strip():
        return "not_decoded", None, None
    if not expected:
        return "decoded", None, None
    np_, ne = normalize_text(payload), normalize_text(expected)
    if np_ == ne:
        return "verified", True, None
    ratio = SequenceMatcher(None, np_, ne).ratio()
    if ratio >= PARTIAL_THRESHOLD:
        return "decoded", False, f"el OCR leyó algo parecido pero no idéntico (similitud {ratio:.0%}) — revisa el texto leído"
    return "false_positive", False, None