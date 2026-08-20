# Unloss

Recupera el **contenido funcional** de códigos QR y texto degradados por apps de mensajería (WhatsApp y otras) y **verifica** que el resultado decodifica lo esperado. No busca que la imagen "se parezca" al original: busca que el QR **vuelva a decodificar** y el texto **se vuelva a leer**.

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](#)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](#)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.x-green.svg)](#)
[![Tesseract](https://img.shields.io/badge/Tesseract-5.x-darkgreen.svg)](#)
[![Licencia](https://img.shields.io/badge/Licencia-Apache_2.0-blue.svg)](LICENSE)

---

## ¿Para qué sirve? (casos de uso)

**QR — cuando no puedes escanearlo.**

- No tienes el celular a la mano o tu cámara no logra leer el QR (p. ej. el código de asistencia a una reunión en una pantalla lejana o un QR de la cámara del chat).
- Subes la foto del QR y el sistema te devuelve **la URL o el contenido** que decodifica.

**Texto — cuando necesitas copiar y pegar.**

- Tienes una foto con un **número de cuenta**, referencia, código o cualquier dato que necesitas usar en otro lado.
- O una foto con **mucho texto** que tendrías que tipear a mano y quieres hacerlo rápido.
- Subes la foto y obtienes **el texto leído y verificado**, listo para copiar.

En ambos casos el resultado se compara contra el contenido que esperabas: **decodifica y coincide**, **decodifica pero no coincide** (se advierte, no se esconde) o **no se pudo leer** (respuesta honesta, sin eufemismos).

---

## Cómo funciona

Las apps de mensajería redimensionan y recomprimen las imágenes al subirlas (medido aquí: un póster 5333×3000 con QR pasó a 1600×900 en el primer envío por WhatsApp, -77 %). Cuando el contenido fino (QR, texto pequeño) queda ilegible, **la única copia que circula es la degradada**. Unloss restaura esa copia y verifica el resultado.

```
Imagen degradada (QR / texto)
        │
        ▼
┌──────────────────────────────────────┐
│  Pipeline clásico: upscale ×4 + Otsu │
│  + variantes (contraste, umbral… )   │
└──────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────┐
│  Decodificación funcional            │
│   QR  → cv2 (+ zbar)                 │
│   Texto → Tesseract (psm 6)          │
│   (lector experimental: RapidOCR)    │
└──────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────┐
│  Verificación de contenido           │
│   ✓ verificado (coincide con esperado)│
│   ✓ decoded (lee, sin esperado)      │
│   ! false_positive (lee otro contenido)│
│   × not_decoded (ningún método)      │
└──────────────────────────────────────┘
```

- **QR primero, texto después**: si ningún método QR decodifica se ejecuta el OCR; un QR legible no paga el coste del lector de texto.
- **Verificación de texto honesta**: se compara el texto leído contra el esperado con normalización OCR (LineAcc-style). Solo se afirma `verified` si coincide; una lectura parecida pero no idéntica se reporta como `decoded` con nota de revisar.
- **Métricas de píxel son secundarias**: PSNR/SSIM bajos pueden convivir con decodificación exitosa. El resultado funcional (decodifica / coincide / no) manda.

---

## Resultados (medidos, con límites)

### Validación real (v16) — QR de WhatsApp

Póster 5333×3000 con QR v4/EC=M → WhatsApp → 1600×900 (-77 %), ilegible para `cv2`:

| Método | Resultado |
|---|---|
| cv2 sobre el degradado | no decodifica |
| **Upscale ×4 + cv2** | **payload exacto** |
| Otsu ×4 | payload exacto |
| Modelo V9Net + RS de borrados | no (gap sintético→real medido) |
| Link2QR (web) | payload exacto |

### Sintético (dominio QR, v14)

| Método | VDR |
|---|---|
| Clásico cv2 ×4 | 0.922 |
| Modelo V9Net (umbral realista) | **0.973** |
| Modelo V9Net (binario duro) | 0.730 |

### Texto (v15)

| Método | LineAcc |
|---|---|
| Tesseract ×4 | **0.990** (realista) / 0.983 (duro) |

### Lección principal

El techo lo pone el pipeline **clásico** (upscale + binarizar): barato, suficiente en casi todos los casos, y en la validación real es el único que recupera el QR. El modelo V9Net añade ~5 puntos en sintético (0.973 vs 0.922) pero **no transfiere** al caso real medido. La aportación del proyecto es el *qué* —pipeline de mensajería + verificación funcional de contenido con resultados verificados y honestos—, no el *cómo*.

---

## Límites conocidos (documentados en `docs/` §6–7)

- **Foto de pantalla con Moiré** (foto con celular a un monitor): el OCR clásico fusiona los glifos y falla; un lector CNN (RapidOCR) tampoco lo rescata en casos extremos (medido, N=1 + reproducción sintética). La verificación atrapa lecturas de ruido como `false_positive`.
- **Lector experimental RapidOCR** (`ocr_engine=rapid`): no está validado sobre el set de la memoria; se etiqueta como experimental en la UI.
- **Modelo V9Net**: solo QR, experimental y fuera de su dominio no transfiere; en Render está desactivado (sin GPU ni torch).
- **Las métricas de píxel no predicen utilidad funcional**; se reportan como secundarias.

---

## Web app: ejecutar y desplegar

**Local (con venv):**

```
python -m venv --system-site-packages .venv
.venv\Scripts\pip install -r requirements-dev.txt
.venv\Scripts\python -m uvicorn app.main:app --reload
# http://127.0.0.1:8000
```

Para el dominio texto necesitas el binario de Tesseract (Windows: `winget install UB-Mannheim.TesseractOCR`; en Render se instala con `apt-get` en `render.yaml`).

**Endpoint** `POST /api/restore` (multipart): `image` (obligatorio), `expected` (opcional), `use_model` (opcional, experimental), `ocr_engine` (opcional: `tesseract` | `rapid` | `auto`). Responde el estado funcional, el `domain` (`qr`/`text`), una `note` opcional (coincidencia parcial), la tabla de métodos por dominio, la reconstrucción como data URL y las métricas.

**Tests:**

```
.venv\Scripts\python -m pytest
```

**Despliegue (Render, free tier):** conecta el repo y Render usa `render.yaml`. El free tier duerme tras ~15 min de inactividad; se mantiene con un ping a `/health` cada 5 min (cronjob.org o UptimeRobot). El primer request tras el cold start tarda ~50 s.

---

## Estructura del repositorio

```
README.md
LICENSE                          # Apache-2.0
docs/proyecto-restauracion-imagenes.md   # memoria técnica completa (§1–§7)
app/                             # web app: API + pipeline + verificación + UI
  main.py                        #   rutas HTTP, validación, orquestación
  pipeline.py                    #   restauración + decodificación (QR y texto)
  verify.py                      #   clasificación funcional (verified/decoded/…)
  model_route.py                 #   ruta experimental v9b (opcional, desactivada)
  static/                        #   UI (español, accesible, sin librerías)
tests/                           # pytest: pipeline, verificación y API
models/v9b_net.pt                # checkpoint experimental (11 MB, ruta opcional)
results/                         # curvas, CSVs y resúmenes de métricas
legacy/                          # archivos acumulados del desarrollo (gitignored)
render.yaml                      # despliegue en Render (free tier, sin GPU)
```

El dataset real (`dataset_real_qr/`) y los notebooks de validación/demo (`notebooks/16`, `17`) se conservan en **Google Drive/Colab** y no se versionan aquí.

## Reproducción (investigación)

Los experimentos corren en **Google Colab** (GPU T4); el dataset es 100 % sintético y reproducible, y los checkpoints/logs viven en Drive.

- `16_v16_validacion_real.ipynb`: validación real autocontenida (requiere subir a Drive `Unloss-Net/dataset_real_qr/` y `models/`).
- `17_v17_mini_pipeline.ipynb`: demo interactiva (subir → restaurar → verificar → métricas).

## Roadmap

- [x] Pipeline de degradación sintética reproducible y dataset QR con severidades
- [x] Baselines (U-Net + L1, SR 2×): evidencia de que la L1 no basta
- [x] Red con FiLM + prior binario + pérdida de tarea
- [x] Dataset de texto con parámetros reales de las apps
- [x] Comparación SOTA (Real-ESRGAN) y validación real (v16)
- [x] Mini-pipeline interactivo (v17) con verificación funcional
- [x] Web app: antes/después con verificación funcional + API REST
- [x] Dominio texto (Tesseract) + lector experimental RapidOCR en la app

## Licencia

[Apache License 2.0](LICENSE).