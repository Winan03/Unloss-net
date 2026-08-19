# Unloss

Restauración **funcional** de códigos QR y texto degradados por el flujo real de mensajería (escalado + recompresión JPEG al subir, y pérdida acumulativa por screenshots y flujos cross-app).

El objetivo no es que la salida "se parezca" a la imagen limpia, sino que el QR **vuelva a decodificar** y el texto **se vuelva a leer**, con **verificación de contenido**: distingue éxito real, falso positivo y no-decodificación.

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](#)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.x-green.svg)](#)
[![Tesseract](https://img.shields.io/badge/Tesseract-OCR-darkgreen.svg)](#)
[![Colab](https://img.shields.io/badge/Google-Colab-orange.svg)](#)
[![Licencia](https://img.shields.io/badge/Licencia-Apache_2.0-blue.svg)](LICENSE)
[![Estado](https://img.shields.io/badge/Estado-experimental-yellow.svg)](#)

---

## Tabla de contenido

- [Problema](#problema)
- [Solución](#solucion)
- [Características](#caracteristicas)
- [Resultados](#resultados)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Reproducción](#reproduccion)
- [Roadmap](#roadmap)
- [Licencia](#licencia)

---

## Problema

Las apps de mensajería (WhatsApp, Telegram, Instagram) redimensionan y recomprimen las imágenes **al subirlas** (una sola vez). Medido en este proyecto: un póster 5333×3000 con QR pasó a 1600×900 en el primer envío por WhatsApp, y un segundo reenvío **no** lo degradó más. La degradación **acumulativa** ocurre cuando la imagen cruza fronteras:

- **Screenshots del chat**: cada captura es una nueva escala + re-codificación.
- **Descargar y re-subir**: cada subida re-codifica (aunque sin re-escalar si ya está dentro del límite).
- **Flujos cross-app**: cada app re-escala con sus propios límites.

En estos flujos, el contenido fino de alto contraste —códigos QR y texto pequeño en capturas— termina ilegible, y **la única copia que circula es la degradada** (no hay original que recuperar). La severidad la manda el **tamaño del QR/texto en origen**, no el número de reenvíos.

## Solución

Pipeline de restauración con éxito **verificado**:

| Componente | Rol |
|---|---|
| **Restauración clásica** (upscale ×4 + Otsu + variantes) | La ruta que funciona en datos reales (validación v16: payload exacto) |
| **Modelo V9Net** (solo QR) | Probabilidad por módulo (grid soft); ganancia medida en sintético, con límite documentado en real |
| **Decodificador propio + Reed–Solomon de borrados** | Marca codewords de baja confianza como borrados y los corrige |
| **Verificación funcional** | Clasifica cada salida en: decodifica y coincide / decodifica erróneo (falso positivo) / no decodifica |
| **Métrica VDR / CER** | Éxito verificado: el decoder lee **y** el contenido coincide con el ground truth |

```
Imagen degradada (QR / texto)
        │
        ▼
┌──────────────────────────────────────┐
│  Upscale ×4 (cubic) + Otsu + variantes │
└──────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────┐
│  Decoders: cv2 · zbar · propio + RS   │
│  (Reed–Solomon con borrados)          │
└──────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────┐
│  Verificación funcional de contenido  │
│   [✓] decodifica y coincide           │
│   [!] decodifica erróneo (falso +)    │
│   [×] no decodifica                   │
└──────────────────────────────────────┘
        │
        ▼
                        Resultado verificado
```

## Características

- **Dos dominios** en un mismo pipeline: QR y texto (OCR).
- **Verificación funcional de contenido**: no basta "decodifica"; se comprueba que coincida con el ground truth.
- **Decodificador QR propio**: Reed–Solomon sobre GF(256), soporte v1–10, 4 niveles de EC — 40/40 sintéticos exactos y el QR real exacto.
- **Métricas de tarea**: VDR (éxito verificado), tasa de falso positivo, CER/WER.
- **Demo interactiva** en Colab (`notebooks/17_v17_mini_pipeline.ipynb`).

## Resultados

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

El baseline clásico **satura** la tarea de texto → no se entrenó modelo de texto (decisión honesta documentada en `docs/` §6.8.2).

### Validación real (v16)

Poster 5333×3000 con QR v4/EC=M → WhatsApp → 1600×900 (-77 %), ilegible para `cv2`:

| Método | Resultado |
|---|---|
| cv2 sobre el degradado | no decodifica |
| **Upscale ×4 + cv2** | **payload exacto** |
| Otsu ×4 | payload exacto |
| Modelo V9Net + RS de borrados | no (gap sintético→real medido) |
| Link2QR (web) | payload exacto |

### Demo (v17) — caso con desenfoque

Un QR de Disney+ con desenfoque: frente al original, las métricas de píxel son malas (PSNR 14.24, SSIM 0.135), pero el pipeline **decodificó el payload exacto**. La misma imagen **no pudo ser leída por Link2QR** (20 pases, 20 decoders). Caso anecdótico (N=1, desenfoque moderado), reportado como demo: ilustra que **las métricas de píxel no predicen la utilidad funcional**.

En esa misma imagen, la **ruta del modelo v9b no decodificó** (el blur está fuera de su dominio de entrenamiento). Simetría medida: dentro de su dominio (sintético) el modelo gana 5 puntos; fuera de él, el clásico es el motor. Detalle y cautelas en `docs/` §6.9.

### Lección principal

El techo lo pone el pipeline **clásico** (upscale + binarizar): barato, suficiente en casi todos los casos, y en la validación real es el único que recupera el QR. El modelo añade **~5 puntos** en sintético (0.973 vs 0.922) pero **no transfiere** al caso real medido. La aportación real del proyecto es el *qué* —pipeline de mensajería + verificación funcional de contenido, con resultados verificados y honestos—, no el *cómo*.

## Estructura del repositorio

```
├── README.md
├── LICENSE                       # Apache-2.0
├── docs/
│   └── proyecto-restauracion-imagenes.md   # memoria completa (problema, estado del arte, diseño, métricas)
├── results/                      # CSVs, curvas, resúmenes de métricas
├── models/v9b_net.pt             # checkpoint experimental (11 MB, ruta opcional de la app)
├── app/                          # web app: API REST + pipeline clásico + verificación funcional
├── tests/                        # pytest (pipeline y API)
└── requirements.txt              # dependencias de la app (sin torch; el modelo es opcional)
```

El dataset real (`dataset_real_qr/`) y los notebooks de validación/demo se conservan en **Google Drive/Colab** (por tamaño y reproducibilidad); no están versionados aquí.

## Reproducción

Los experimentos se ejecutan en **Google Colab** con GPU (T4). El dataset es 100 % sintético y reproducible; los checkpoints y logs se conservan en Google Drive.

- `16_v16_validacion_real.ipynb`: autocontenido. Requiere subir a Drive `Unloss-Net/dataset_real_qr/` las 2 imágenes de validación y el checkpoint en `Unloss-Net/models/`.
- `17_v17_mini_pipeline.ipynb`: demo interactiva. Solo sube una imagen y procesa.

La **web app** (`app/`) expone el pipeline clásico como API REST (`/api/restore`) con verificación funcional en la respuesta, y la ruta experimental v9b desactivada por defecto. Corre en Render (free tier); el modelo no se instala en ese despliegue (sin GPU ni torch), se habilita solo en local con `models/v9b_net.pt`.

## Web app: ejecutar y desplegar

**Local (con venv):**

```
python -m venv --system-site-packages .venv
.venv\Scripts\pip install -r requirements-dev.txt
.venv\Scripts\python -m uvicorn app.main:app --reload
# http://127.0.0.1:8000
```

Endpoint `POST /api/restore` (multipart): `image` (obligatorio), `expected` (opcional), `use_model` (opcional, `true` para intentar la ruta experimental v9b). Responde el estado funcional (`verified`/`decoded`/`false_positive`/`not_decoded`), la tabla de métodos, la reconstrucción como data URL y las métricas.

**Tests:**

```
.venv\Scripts\python -m pytest
```

**Despliegue (Render, free tier):** conecta el repo y Render usará `render.yaml` (sin torch: la ruta v9b reporta "no disponible"). El free tier duerme tras ~15 min de inactividad; se mantiene despierto con un ping a `/health` cada 5 min (p. ej. cronjob.org o UptimeRobot). El primer request tras el cold start tarda ~50 s.

## Roadmap

- [x] Pipeline de degradación sintética reproducible y dataset QR con severidades
- [x] Baselines: U-Net vanilla + L1, SR 2× (evidencia de que la L1 no basta)
- [x] Red con FiLM + prior binario + pérdida de tarea; ablaciones por pieza
- [x] Cerrar el color de módulos y superar el control `LANCZOS 2×`
- [x] Dataset de texto y control texto-solo; parámetros reales de las apps
- [x] Comparación SOTA (Real-ESRGAN) y validación real (v16): 1er upload de WhatsApp medido
- [x] Validación real completa en Colab: clásico recupera el QR exacto; modelo+RS y Real-ESRGAN no (gap real medido, ver `docs/`)
- [x] Mini-pipeline interactivo (v17) con verificación funcional y métricas por método
- [x] Web app: antes/después con verificación funcional + API REST (`/api/restore`)

## Licencia

[Apache License 2.0](LICENSE).
