"""API HTTP de Unloss: valida, orquesta y responde. NO contiene lógica de restauración.

- GET  /                -> UI (index.html)
- GET  /health          -> estado (target del ping cada 5 min en Render)
- POST /api/restore     -> imagen + expected (opcional) + use_model (opcional) -> JSON
"""

from __future__ import annotations

import base64
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app import pipeline, verify
from app.model_route import attempt as model_attempt

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_DIM = 2000                        # acota el coste de cómputo
REC_DATA_URL_MAX_DIM = 1600           # tope de la reconstrucción en la respuesta
TEXT_ENGINES = {"tesseract", "rapid", "auto"}

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="Unloss", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/restore")
def restore(image: UploadFile = File(...),
            expected: str = Form(""),
            use_model: str = Form("false"),
            ocr_engine: str = Form("tesseract")) -> JSONResponse:
    t0 = time.perf_counter()
    engine = ocr_engine.strip().lower()
    if engine not in TEXT_ENGINES:
        raise HTTPException(400, detail="ocr_engine debe ser tesseract, rapid o auto")
    raw = image.file.read()
    if not raw:
        raise HTTPException(400, detail="imagen vacía")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, detail=f"imagen demasiado grande (máx {MAX_UPLOAD_BYTES // (1024 * 1024)} MB)")

    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, detail="el archivo no es una imagen válida")

    img = pipeline.fit_max(img, MAX_DIM)

    results = pipeline.run(img, text_engine=engine)
    payload = pipeline.most_common_payload(results)
    exp = expected.strip() or None

    text_decoded = any(r.get("domain") == "text" and r["decoded"] for r in results)
    note = None
    if text_decoded:
        domain = "text"
        status, matches, note = verify.classify_text(payload, exp)
    else:
        domain = "qr"
        status, matches = verify.classify(payload, exp)

    rec_name, rec = pipeline.best_reconstruction(img, results)
    rec_small = pipeline.fit_max(rec, REC_DATA_URL_MAX_DIM)
    ok, buf = cv2.imencode(".jpg", rec_small, [cv2.IMWRITE_JPEG_QUALITY, 85])
    data_url = None
    if ok:
        data_url = "data:image/jpeg;base64," + base64.b64encode(buf).decode("ascii")

    use_model_bool = use_model.strip().lower() in ("true", "1", "yes", "on")
    model = {"attempted": False, "payload": None, "note": "desactivado", "reason": "use_model=false"}
    if use_model_bool:
        model = model_attempt(img)

    m = pipeline.metrics(img, rec) if data_url else {}

    return JSONResponse({
        "status": status,
        "domain": domain,
        "note": note,
        "methods": results,
        "decoded_payload": payload,
        "matches": matches,
        "reconstruction": {
            "data_url": data_url,
            "method": rec_name,
            "decoded": bool(payload),
            "width": int(rec_small.shape[1]),
            "height": int(rec_small.shape[0]),
        },
        "model": model,
        "metrics": m,
        "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
    })