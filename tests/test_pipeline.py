from pathlib import Path

import cv2
import numpy as np
import pytest
import qrcode

from app import pipeline

REAL_DIR = Path(__file__).parent.parent / "dataset_real_qr"
HF_PAYLOAD = "https://huggingface.co/unsloth/Phi-4-mini-instruct-GGUF"
DISNEY_PAYLOAD = "https://www.disneyplus.com/es-pe"


def make_qr(content: str) -> np.ndarray:
    qr = qrcode.QRCode(border=2)
    qr.add_data(content)
    qr.make(fit=True)
    pil = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


# ---------- sintético (funciona sin el dataset real, corre siempre) ----------

def test_pipeline_decodifica_qr_sintetico():
    payload = "https://example.com/unloss-demo"
    img = make_qr(payload)
    results = pipeline.run(img)
    assert any(r["decoded"] for r in results)
    assert pipeline.most_common_payload(results) == payload


def test_pipeline_sin_qr_no_decodifica():
    img = np.full((300, 300, 3), 255, np.uint8)
    results = pipeline.run(img)
    assert not any(r["decoded"] for r in results)


def test_best_reconstruction_escalada_al_tamano_de_la_subida():
    payload = "https://example.com/size"
    img = make_qr(payload)
    results = pipeline.run(img)
    name, rec = pipeline.best_reconstruction(img, results)
    assert rec.shape[:2] == img.shape[:2] or rec.shape[:2][0] >= img.shape[0]
    m = pipeline.metrics(img, rec)
    assert "subida_vs_rec_psnr" in m


def test_select_best_read_por_banda_y_longitud():
    # dentro de la banda (<= CONF_BAND de la máxima confianza) gana la más larga
    reads = [(50.0, "abc"), (90.0, "def"), (90.0, "def ghi")]
    assert pipeline.select_best_read(reads) == 2
    # fuera de la banda no compite: la de mayor confianza gana aunque sea más corta
    assert pipeline.select_best_read([(80.0, "corto"), (70.0, "otra lectura mas larga")]) == 0
    # dentro de la banda, la más larga supera a la de un punto más de confianza
    assert pipeline.select_best_read([(90.0, "aa"), (89.0, "bbbbbbbbbbbbbb")]) == 1
    assert pipeline.select_best_read([]) == -1


def test_select_best_match_elige_por_similitud_funcional():
    reads = [(90.0, "el perro corre por la calle"), (80.0, "otra cosa totalmente distinta")]
    assert pipeline.select_best_match(reads, "El perro corre por la calle") == 0
    assert pipeline.select_best_match(reads, "Otra cosa totalmente distinta") == 1
    assert pipeline.select_best_match([], "cualquiera") == -1


# ---------- caso real (requiere dataset_real_qr/, no versionado) ----------

def _load(name: str) -> np.ndarray:
    return cv2.imread(str(REAL_DIR / name), cv2.IMREAD_COLOR)


@pytest.mark.skipif(not (REAL_DIR / "imagen_despues_de_pasar_A_wsp.jpeg").exists(),
                    reason="dataset_real_qr no presente (vive en Drive)")
def test_real_whatsapp_recupera_payload_huggingface():
    img = _load("imagen_despues_de_pasar_A_wsp.jpeg")
    results = pipeline.run(img)
    assert pipeline.most_common_payload(results) == HF_PAYLOAD


@pytest.mark.skipif(not (REAL_DIR / "damaged_qr_04.png").exists(),
                    reason="dataset_real_qr no presente (vive en Drive)")
def test_real_disney_desenfocado_recupera_payload():
    img = _load("damaged_qr_04.png")
    results = pipeline.run(img)
    assert pipeline.most_common_payload(results) == DISNEY_PAYLOAD


# ---------- dominio texto (requiere tesseract en el entorno) ----------

def _make_text_img(text: str) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont
    import os

    img = Image.new("RGB", (1000, 220), "white")
    d = ImageDraw.Draw(img)
    font = None
    for p in (r"C:\Windows\Fonts\arial.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(p):
            font = ImageFont.truetype(p, 56)
            break
    if font is None:
        pytest.skip("sin fuente truetype para generar el fixture de texto")
    d.text((40, 60), text, fill="black", font=font)
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)


@pytest.mark.skipif(not pipeline.ocr_available(),
                    reason="tesseract no disponible en este entorno")
def test_pipeline_lee_texto_sintetico():
    img = _make_text_img("Hola, esto es una prueba")
    results = pipeline.run(img)
    assert any(r["domain"] == "text" and r["decoded"] for r in results)


@pytest.mark.skipif(not pipeline.rapid_available(),
                    reason="RapidOCR (ONNX) no disponible en este entorno")
def test_pipeline_lee_texto_sintetico_con_rapid():
    img = _make_text_img("Hola, esto es una prueba")
    results = pipeline.run(img, text_engine="rapid")
    assert any(r["domain"] == "text" and r["decoded"] for r in results)
    assert any(r.get("engine") == "rapid" and r["decoded"] for r in results)