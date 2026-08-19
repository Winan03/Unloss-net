import pytest

from app.verify import classify


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