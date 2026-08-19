"""Verificación funcional: clasifica el resultado del pipeline contra el contenido esperado.

Función pura (sin I/O, sin HTTP). Estados:
- verified        -> decodifica y coincide con lo esperado
- decoded         -> decodifica; no había contenido esperado
- false_positive  -> decodifica pero difiere de lo esperado (más peligroso que no leer)
- not_decoded     -> ninguna variante decodificó
"""

from __future__ import annotations


def classify(payload: str | None, expected: str | None) -> tuple[str, bool | None]:
    """Devuelve (status, matches). matches es None cuando no hay con qué comparar."""
    if not payload:
        return "not_decoded", None
    if not expected:
        return "decoded", None
    if payload == expected:
        return "verified", True
    return "false_positive", False