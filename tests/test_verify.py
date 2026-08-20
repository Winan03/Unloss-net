import pytest

from app.verify import classify, classify_text, normalize_text


def test_sin_payload_es_not_decoded():
    assert classify(None, None) == ("not_decoded", None)


def test_decodifica_sin_esperado_es_decoded():
    assert classify("https://x.com", None) == ("decoded", None)


def test_decodifica_y_coincide_es_verified():
    assert classify("https://x.com", "https://x.com") == ("verified", True)


def test_decodifica_pero_difiere_es_false_positive():
    assert classify("https://a.com", "https://b.com") == ("false_positive", False)


def test_espacios_en_el_esperado_se_limpian_antes():
    assert classify("https://x.com", "https://x.com ") == ("false_positive", False)


# ---------- dominio texto (normalización OCR, LineAcc-style) ----------

def test_normalize_text_quita_mayusculas_acentos_y_puntuacion():
    assert normalize_text("Hola, ¿Cómo ESTÁS? — ok") == "hola como estas ok"


def test_texto_vacio_es_not_decoded():
    assert classify_text(None, "hola") == ("not_decoded", None, None)


def test_texto_sin_esperado_es_decoded():
    s, m, n = classify_text("hola mundo", None)
    assert s == "decoded" and m is None and n is None


def test_texto_coincide_normalizado_es_verified():
    s, m, n = classify_text("Hola, mundo!", "hola mundo")
    assert s == "verified" and m is True and n is None


def test_texto_parecido_no_identico_es_decoded_con_nota():
    s, m, n = classify_text("hola munde", "hola mundo")
    assert s == "decoded" and m is False and n is not None


def test_texto_difiere_es_false_positive():
    s, m, n = classify_text("adios mundo", "hola mundo")
    assert s == "false_positive" and m is False and n is None