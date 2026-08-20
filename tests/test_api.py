from pathlib import Path

import cv2
import numpy as np
import pytest
import qrcode
from fastapi.testclient import TestClient

from app.main import app
from app.pipeline import ocr_available, rapid_available

client = TestClient(app)

REAL_DIR = Path(__file__).parent.parent / "dataset_real_qr"
HF_PAYLOAD = "https://huggingface.co/unsloth/Phi-4-mini-instruct-GGUF"


def make_qr_bytes(content: str) -> bytes:
    qr = qrcode.QRCode(border=2)
    qr.add_data(content)
    qr.make(fit=True)
    pil = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes()


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_index():
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_restore_verified():
    payload = "https://example.com/unloss-api"
    r = client.post("/api/restore",
                    files={"image": ("qr.png", make_qr_bytes(payload), "image/png")},
                    data={"expected": payload})
    assert r.status_code == 200
    j = r.json()
    assert j["status"] == "verified"
    assert j["matches"] is True
    assert j["decoded_payload"] == payload
    assert j["reconstruction"]["data_url"].startswith("data:image/jpeg;base64,")
    assert j["reconstruction"]["decoded"] is True
    assert j["model"]["attempted"] is False


def test_restore_not_decoded():
    blank = np.full((300, 300, 3), 255, np.uint8)
    ok, buf = cv2.imencode(".png", blank)
    r = client.post("/api/restore",
                    files={"image": ("blank.png", buf.tobytes(), "image/png")})
    assert r.status_code == 200
    j = r.json()
    assert j["status"] == "not_decoded"
    assert j["reconstruction"]["decoded"] is False


def test_restore_con_modelo_nunca_define_el_estado():
    payload = "https://example.com/model"
    r = client.post("/api/restore",
                    files={"image": ("qr.png", make_qr_bytes(payload), "image/png")},
                    data={"expected": payload, "use_model": "true"})
    assert r.status_code == 200
    j = r.json()
    assert j["status"] == "verified"          # el clásico manda
    assert "attempted" in j["model"]
    assert "reason" in j["model"]


def test_restore_archivo_invalido():
    r = client.post("/api/restore",
                    files={"image": ("no_imagen.txt", b"esto no es una imagen", "text/plain")})
    assert r.status_code == 400


@pytest.mark.skipif(not ocr_available(), reason="tesseract no disponible en este entorno")
def test_restore_texto_verificado():
    from PIL import Image, ImageDraw, ImageFont
    import os

    text = "Hola desde el test de texto"
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
    arr = np.array(img.convert("RGB"))
    small = cv2.resize(arr, (arr.shape[1] // 2, arr.shape[0] // 2), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg",
                           cv2.cvtColor(small, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, 45])
    r = client.post("/api/restore",
                    files={"image": ("texto.jpg", buf.tobytes(), "image/jpeg")},
                    data={"expected": text})
    assert r.status_code == 200
    j = r.json()
    assert j["domain"] == "text"
    assert j["status"] == "verified"
    assert j["reconstruction"]["decoded"] is True


@pytest.mark.skipif(not rapid_available(), reason="RapidOCR (ONNX) no disponible en este entorno")
def test_restore_texto_con_rapid():
    from PIL import Image, ImageDraw, ImageFont
    import os

    text = "Hola desde el lector rapido"
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
    arr = np.array(img.convert("RGB"))
    small = cv2.resize(arr, (arr.shape[1] // 2, arr.shape[0] // 2), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg",
                           cv2.cvtColor(small, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, 45])
    r = client.post("/api/restore",
                    files={"image": ("texto.jpg", buf.tobytes(), "image/jpeg")},
                    data={"expected": text, "ocr_engine": "rapid"})
    assert r.status_code == 200
    j = r.json()
    assert j["domain"] == "text"
    assert j["status"] == "verified"
    assert any(m.get("engine") == "rapid" and m["decoded"] for m in j["methods"])


def test_restore_ocr_engine_invalido():
    payload = "https://example.com/invalido"
    r = client.post("/api/restore",
                    files={"image": ("qr.png", make_qr_bytes(payload), "image/png")},
                    data={"ocr_engine": "magico"})
    assert r.status_code == 400


@pytest.mark.skipif(not (REAL_DIR / "imagen_despues_de_pasar_A_wsp.jpeg").exists(),
                    reason="dataset_real_qr no presente (vive en Drive)")
def test_restore_caso_real_whatsapp_verificado():
    with open(REAL_DIR / "imagen_despues_de_pasar_A_wsp.jpeg", "rb") as f:
        content = f.read()
    r = client.post("/api/restore",
                    files={"image": ("wsp.jpeg", content, "image/jpeg")},
                    data={"expected": HF_PAYLOAD})
    assert r.status_code == 200
    j = r.json()
    assert j["status"] == "verified"
    assert j["decoded_payload"] == HF_PAYLOAD