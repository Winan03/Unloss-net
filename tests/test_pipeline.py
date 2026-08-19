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