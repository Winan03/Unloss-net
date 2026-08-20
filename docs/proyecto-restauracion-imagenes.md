# Proyecto Unloss — Restauración funcional de QR/Texto degradados por pérdida generacional en apps de mensajería

## 0. Nombre del proyecto

**Unloss** = "un" (revertir) + "loss" (pérdida por compresión). Hace alusión directa al término técnico **generation loss**: la degradación acumulativa que sufre una imagen cada vez que una app de mensajería la redimensiona y recomprime. Verificado: sin conflictos de marca relevantes en tecnología/restauración de imágenes (solo un webapp escolar japonés y un perfil de foro usan "Unloss"; "Unlost" existe como empresa de servicios web en Francia y de logística en Suiza, pero es una palabra distinta).

---

## 1. Problema identificado

Las apps de mensajería (WhatsApp, Telegram, Instagram, etc.) redimensionan y recomprimen las imágenes **una vez, al subirlas**. El reenvío dentro de la misma app **no vuelve a re-codificar**: manda el archivo ya comprimido. Medido en este proyecto: un póster 5333×3000 con QR pasó a 1600×900 en el primer envío por WhatsApp, y un segundo reenvío no lo degradó más. La degradación **acumulativa** ocurre cuando la imagen cruza fronteras: **screenshots del chat** (cada captura es una nueva escala + re-codificación), **descargar y re-subir** (cada subida re-codifica, aunque sin re-escalar si ya está dentro del límite), y **flujos cross-app** (cada app re-escala con sus propios límites). En todos estos flujos la copia que circula ya no tiene el original y se degrada en cada salto.

Resultado: contenido fino de alto contraste — códigos QR y texto pequeño en screenshots — termina ilegible o no escaneable, y **no hay forma de recuperar el original** porque la única copia que circula es la degradada. La severidad del daño la manda sobre todo el **tamaño del QR/texto en origen** (cuánto lo reduce el downscale del primer upload), no el número de reenvíos.

### Caracterización del daño (distinto a lo que estudia la literatura)

| Tipo de daño | Estudiado en la literatura | Relevante a este proyecto |
|---|---|---|
| Daño físico (rayones, dobleces, tinta desvanecida, agujeros) | Sí, muy estudiado (QR industrial/logística) | No es el foco |
| Motion blur / desenfoque de cámara | Sí, muy estudiado (QR deblurring) | No es el foco |
| Baja resolución / super-resolución | Sí, muy estudiado | Parcial (un componente del pipeline) |
| Doble compresión JPEG (forense: detectar manipulación) | Sí, pero con fin de *detección*, no de *restauración* | Conceptual |
| **Pérdida por escalado+recompresión en el flujo real de mensajería (primer upload, screenshots, cross-app) + restauración funcional** | **No encontrado** | **Este proyecto** |

La combinación específica — degradación replicando los flujos reales de mensajería (con la corrección medida de que el reenvío dentro de la misma app no acumula) + aplicación a QR/texto + evaluación con métrica de *tarea* (decodificación/OCR verificada) — **no se encuentra publicada ni implementada como herramienta**.

---

## 2. Estado del arte

### 2.1 Papers relevantes (lo que ya existe)

| Trabajo | Degradación modelada | Arquitectura | Métrica principal | Brecha que deja |
|---|---|---|---|---|
| SR de QR con EDSR/VDSR/ESPCN/SRCNN (2024) | Baja resolución + artefactos de scanner | CNNs de super-resolución | PSNR/SSIM + tasa de detección | Sin pérdida generacional; sin recuperación generativa; métricas de píxel dominan |
| EHFP-GAN (Mathematics 2023) | Agujeros/daño físico 1–60% | GAN con pirámide de features + edge | Recognition Rate (95% daño leve) | Daño físico, inpainting; no compresión en cadena |
| Lightweight Pix2Pix para barcodes + QR logística (CMES 2025) | Bajo contraste, desgaste, interferencia logística | Pix2Pix / U-Net separable | Decode ratio 15%→68% (val) | El más cercano; degradación logística, no mensajería; sin verificación de contenido |
| EG-Restormer / ADNet (arXiv 2025) | Motion blur | Transformer con edge priors | Decoding Rate (90%) | Solo blur; sin pérdida generacional |
| LCM-QRIR (Nature Sci. Reports 2026) | Blur + ruido | Mamba ligera | PSNR/SSIM/RR | Solo blur; sin verificación de contenido |
| QRSuperResolutionNet (GitHub, 2025) | Degradación genérica | ESRGAN + RRDB + SE + TTA | Decode rate | Enfocado en upscaling; no cadena de reenvíos |
| CODiff (ICCV 2025) | Artefactos JPEG en fotos generales | Difusión de un paso | PSNR/LPIPS | Imágenes naturales, no tarea QR/texto |

**Conclusión del estado del arte:** el *cómo* (arquitecturas, pérdidas, métricas de decodificación) está resuelto. El *qué* (pipeline de mensajería + verificación funcional) no lo está.

### 2.2 Herramientas existentes (lo que un usuario usaría hoy) — lista verificada

Verificado por búsqueda y prueba directa (no todas las páginas de estos sitios son de reparación; muchas son generadores/escáneres/calculadoras con fines de marketing SEO):

| Herramienta | Enfoque | Método | Límite fundamental |
|---|---|---|---|
| **Link2QR** (`link2qr.com/decode`) | Reparar/decodificar QR dañado | Varias pasadas de mejora en navegador (grises, contraste, sharpen, umbral adaptativo, inversión) + múltiples decoders (jsQR/zxing/...) | 100 % basado en reglas; solo decodifica mejor, no reconstruye contenido ilegible |
| **QR Crafter** (`qr-crafter.com/en/qr-code-repair`) | Reparar QR | Decodifica el contenido y **regenera** un QR limpio | Documentado por ellos: *si el código no se puede decodificar, no se puede reparar* — no recupera contenido ilegible |
| **QRazyBox** (`merricx/qrazybox`, open source) | Análisis/recuperación QR | Editor manual píxel a píxel + Reed–Solomon con errores y borrados; soporta hasta versión 40 | Manual, lento, requiere experto |
| AKVIS Artifact Remover AI / Topaz (de-artefactado) | Fotos generales | ML de de-artefactado / SR | No evalúan si el QR decodifica o el OCR lee; no probados |

**Comparación medida con Link2QR (validación real, v16):** sobre el QR real degradado por WhatsApp (`imagen_despues_de_pasar_A_wsp.jpeg`, 1600×900), **Link2QR decodificó directamente** el payload exacto (`https://huggingface.co/unsloth/Phi-4-mini-instruct-GGUF`) sin upscaling previo. Nuestro pipeline clásico (upscale ×4 + cv2) también lo decodifica; cv2 sobre la imagen cruda no. El modelo v9b no transfiere (gap sintético→real). Conclusión honesta: el QR era recuperable por vía **clásica** con un buen stack de decoders; el valor del proyecto está en la verificación funcional y el pipeline medido, no en haber inventado la recuperación clásica.

### 2.3 Tabla comparativa Unloss vs. el panorama (benchmarking)

| Dimensión | Papers de QR (2.1) | Herramientas (2.2) | **Unloss** |
|---|---|---|---|
| Pipeline de degradación = flujo real de mensajería (upload, screenshots, cross-app) | No | Parcial (reglas manuales) | **Sí, modelado explícito y parametrizado** |
| Recuperación generativa cuando el decoder/OCR falla | Solo investigado (papers) | **No** (requieren contenido legible) | **Explorada (sintético 0.973 vs 0.922); no transferida al caso real (medido, §6.8.3)** |
| Métrica principal = tarea verificada (decodifica **y** coincide con ground truth) | Parcial (RR ≠ contenido correcto) | No | **Sí** |
| OCR de texto en screenshots reenviados | No | No | **Sí (pipeline clásico + medición LineAcc; el clásico satura)** |
| Demostrar que PSNR/SSIM/LPIPS no predicen la utilidad funcional | No es el objetivo | No | **Entregable explícito** |

---

## 3. Propuesta de valor novedosa ("lo que nadie hizo")

1. **Modelo de degradación fiel al flujo real de mensajería.** Medir los parámetros reales de WhatsApp/Telegram/Instagram (resolución de salida, quality factor, re-encode) y replicarlos — incluido el hallazgo medido de que el reenvío dentro de la misma app **no** re-codifica y que la pérdida acumulativa vive en screenshots y flujos cross-app. Ninguna herramienta ni paper modela estos flujos; solo reglas genéricas o blur/ruido.
2. **Salida por módulo con confianza + corrección Reed–Solomon de borrados.** El modelo emite una probabilidad por módulo (grid soft); con ella, el decodificador propio marca los codewords de menor confianza como *borrados* y RS los corrige. Medido: esto amplía la recuperación **en el rango sintético** (VDR 0.973 vs 0.922 del clásico), pero **no transfiere** al caso real (§6.8.3), donde el camino clásico (upscale ×4 + binarizar) es el que recupera el payload exacto. Es un resultado explorado con un límite medido, no un diferenciador asumido.
3. **Verificación funcional de contenido (no solo "decodifica").** Reportar por separado: (a) decodifica y el contenido coincide con el ground truth, (b) decodifica a contenido **erróneo** (falso positivo, peligroso), (c) no decodifica. La mayoría de la literatura reporta solo (a)/(c) mezclados.
4. **Doble dominio QR + texto.** Un mismo pipeline (restauración + verificación) cubre QRs y screenshots de texto. En texto el baseline clásico satura (LineAcc 0.990, §6.8.2) y se reporta el protocolo de medición; el modelo entrenado fue solo para el dominio QR.
5. **Contribución de metrología:** curva de correlación PSNR/SSIM/LPIPS vs. tasa de éxito funcional, para evidenciar que las métricas de píxel no capturan la utilidad. Esto por sí solo es un resultado publicable/defendible.

---

## 4. Arquitectura

### 4.1 Por qué no basta una CNN clásica

Los artefactos JPEG son **no locales**: el *ringing* aparece alrededor de cualquier borde de alto contraste y la cuadrícula 8×8 es una estructura global. Además, la señal útil (módulos del QR, trazos de texto) es esencialmente **binaria**. Una U-Net clásica con L1 tiende a suavizar bordes y deja halos, y no sabe que el objetivo final es "que decodifique", no "que se parezca en píxeles". Por eso la propuesta añade tres piezas propias:

1. **Condicionamiento por severidad (FiLM).** Un cabezal estima el nivel de degradación de la entrada (proxy: número de reenvíos N con el que se generó, más un auxiliar autónomo para inferencia) y condiciona cada etapa del encoder/decoder con *Feature-wise Linear Modulation* (escala y desplazamiento por canal). Ventajas:
   - Un solo modelo cubre todo el rango de severidades (los modelos clásicos suelen entrenarse para un nivel fijo).
   - En inferencia **no hace falta conocer N**: el cabezal lo predice del propio píxel.
   - UX de la demo: mostrar "degradación estimada: nivel 7/10".
2. **Prior binario.** El contenido es blanco/negro. La red emite, además de la imagen restaurada, un **mapa de probabilidad por píxel** (sigmoid) de "ser tinta", entrenado con BCE sobre el ground truth binarizable. En inferencia se umbraliza (umbral aprendido/adaptativo). Esto fuerza salidas binarizables, condición necesaria para que el QR decodifique y el OCR lea.
3. **Pérdida de tarea (decodificabilidad).** Un clasificador auxiliar pequeño predice "¿este resultado decodificará correctamente?" y su gradiente guía al restaurador. Es la idea de los trabajos de decodability-assessment, pero como **componente central de entrenamiento** (no solo métrica).

La novedad no está en cada pieza (cada una tiene precedente individual) sino en la **combinación para esta tarea concreta + el pipeline de degradación propio**: eso es lo defendible.

### 4.2 Baseline "vanilla" (lo que TODO debe superar)

- U-Net simple, entrada 256×256.
- Encoder: stem (conv 3×3, 64 canales, ReLU) + 4 etapas de [ResBlock×2 + conv stride-2]. Canales 64 → 128 → 256 → 512.
- Bottleneck: ResBlock×1.
- Decoder simétrico con skip connections (concat): upsampling 2× + ResBlock×2 por etapa.
- Salida: conv 1×1 en modo residuo (predice la diferencia).
- Pérdida: L1 + perceptual VGG.
- Mismo dataset y aumento que las demás filas.

### 4.3 "Unloss-Net" (propuesta)

> Diseño explorado. Durante el desarrollo se simplificó (§6.8.4): el modelo final **V9Net** conserva la salida por módulo (grid soft) y la binarización, y prescinde de FiLM y atención. Las filas comparadas en 4.5 son las del diseño, con sus ablaciones.

| Bloque | Detalle |
|---|---|
| Entrada | 256×256, RGB o escala de grises (mismo tamaño para todas las filas comparadas) |
| Cabezal de severidad | CNN pequeña (3× conv 3×3 stride-2 + GAP) → embedding FiLM de 32 dims; regresión a N (proxy) + cabeza autónoma de severidad para inferencia |
| Encoder | Stem 3×3→64; etapas 1–4 de 2× ResBlock + down 2×; canales 64/128/256/512. **FiLM** (embedding de severidad) tras cada etapa |
| Bottleneck | Bloque de atención compacto (window-attention o Transformer ligero, 2 capas, 4 cabezas, dim 512) para dependencias de largo alcance y estructura de rejilla del QR |
| Decoder | 4 etapas: up 2× + concat skip + 2× ResBlock + FiLM |
| Salidas | conv 1×1 → imagen restaurada (residuo) **y** cabeza paralela → mapa de probabilidad binaria |
| Cabezal de tarea | Clasificador CNN binario "decodifica correctamente / no" — solo en entrenamiento |
| Pérdida total | L = λ1·L1 + λ2·Lperceptual(VGG) + λ3·Lbin(BCE) + λ4·Ltarea(decodificabilidad) + λ5·Lregión(FMAE, peso alto en zona QR/texto) |

Tamaño estimado: ~25–35M parámetros. Si el cómputo es limitado: versión **Unloss-Net-lite** (sin bottleneck de atención → ResBlock; canales 32/64/128/256) manteniendo FiLM + binario + pérdida de tarea.

### 4.4 ¿Pix2Pix / discriminador?

Se **compara**, no se asume. Primera fase sin adversarial; si la nitidez percibida (LPIPS) es el cuello de botella, se activa pérdida adversarial (estilo Pix2Pix) y se mide si mejora o empeora la VDR (un GAN puede afinar pero también alucinar). Se trata como variable de estudio, no como componente por defecto.

### 4.5 Tabla de comparación de arquitecturas (mismo test set, mismas N)

| Arquitectura | Condicionamiento | Prior binario | Pérdida de tarea | Atención global | Para qué sirve |
|---|---|---|---|---|---|
| Otsu + morfología (clásico) | – | – | – | – | Baseline mínimo de viabilidad |
| U-Net vanilla | – | – | – | – | "¿Qué pasa si no añado nada?" |
| U-Net + FiLM severidad | ✓ | – | – | – | Aísla el aporte del condicionamiento |
| U-Net + FiLM + binario | ✓ | ✓ | – | – | Aísla el aporte del prior discreto |
| Pix2Pix | – | – | – | – | Referencia GAN clásica |
| Real-ESRGAN (fine-tune) | – | – | – | – | Referencia SOTA general de upscaling |
| Restormer (si cómputo lo permite) | – | – | – | ✓ | Referencia Transformer SOTA |
| **Unloss-Net (completa)** | ✓ | ✓ | ✓ | ✓ | La propuesta |

**Decisión honesta:** el objetivo no es SOTA genérico; es que **Unloss-Net supere a todas las filas anteriores en éxito funcional verificado (VDR/OCR) con recursos acotados**, y que las filas intermedias (ablaciones) demuestren qué pieza aporta cada ganancia.

## 5. Datos

Generación 100% sintética y automática (sin etiquetado manual ni scraping):

1. QRs sintéticos con contenido variado (URLs, texto, WiFi, vCards) y distintos version/error-correction level + screenshots de texto reales sin degradar.
2. Simular el pipeline real de mensajería, **medido y corregido por validación real**:
   - Resize a las resoluciones reales que aplican las apps (e.g. WhatsApp: 1600×900 medido en este proyecto; 1280/960/640 según el origen).
   - Recompresión JPEG con el quality factor real medido.
   - **Un solo par resize+recompresa por envío** (medido: el reenvío dentro de la misma app no re-codifica), con la severidad barrida por el **tamaño del QR/texto en origen**; aparte, cadenas **cross-app / screenshots** (descargar → re-subir) para cubrir la pérdida acumulativa real.
3. Cada par (limpia, degradada) es una muestra de entrenamiento.
4. **Validación crítica:** un conjunto de imágenes que realmente pasen por la app real (enviar, recibir, reenviar por WhatsApp/Telegram, descargar) para medir el gap sintético→real. La validez del proyecto depende de que el modelo sintético transfiera a datos reales.

## 6. Métricas de evaluación (marco completo)

### Nivel A — Funcionales (PRIMARIAS: definen el éxito del proyecto)

| Métrica | Definición | Por qué importa |
|---|---|---|
| **Tasa de éxito verificado (VDR)** | % de imágenes donde el decoder lee **y** el contenido coincide con el ground truth | Distingue éxito real de alucinación |
| **Tasa de falso positivo** | % donde decodifica a contenido **erróneo** | Riesgo de seguridad; nadie la reporta |
| **Tasa de no-decodificación** | % donde el decoder clásico falla | Complementa a las dos anteriores |
| **CER / WER / Exact Match** (texto) | Error de caracteres/palabras y coincidencia exacta del OCR | Métrica estándar de OCR, no "precisión" genérica |
| **Desglose por severidad** | VDR por nivel N de reenvío (N=0,1,2,...) | Revela dónde se rompe el modelo (leve vs severo) |
| **Precisión binaria de módulos** (QR) | % de celdas correctas tras binarizar salida vs. GT | Enlaza directamente con el límite de corrección Reed–Solomon |

### Nivel B — Perceptuales (SECUNDARIAS)

| Métrica | Definición |
|---|---|
| PSNR / SSIM | Similitud de píxeles (para comparar con literatura) |
| LPIPS | Similitud perceptual aprendida (estándar moderno) |

### Nivel C — Validez del claim (lo que hace creíble el proyecto)

| Métrica/experimento | Definición |
|---|---|
| **Baselines clásicos** | Otsu, sharpening, morfología, upscale bicúbico. Si la U-Net no supera a Otsu en VDR, hay un problema por descubrir antes de presentar |
| **Herramientas libres** | Link2QR, QR Crafter, EAN Check, Fuzana, QR Sharpener sobre el mismo test set |
| **Arquitecturas (tabla 4.5)** | U-Net vanilla, U-Net+FiLM, U-Net+FiLM+binario, Pix2Pix, Real-ESRGAN, Restormer sobre el mismo test set |
| **Correlación de métricas** | Scatter + coeficiente PSNR/SSIM/LPIPS vs. VDR (objetivo: correlación débil → evidencia de que la métrica de tarea aporta) |
| **Ablaciones (arquitectura + pérdida)** | Aporte individual de cada pieza de Unloss-Net (FiLM, binario, atención, pérdida de tarea) y de cada pérdida (L1, perceptual, región, tarea) |

### Nivel D — Robustez / generalización

| Métrica/experimento | Definición |
|---|---|
| **Test set real** | Imágenes pasadas por la app real (QR + screenshots) — el gap sintético→real |
| **Variabilidad** | Distintas versiones de QR, niveles de EC, densidades, tamaños de texto, apps (WhatsApp vs Telegram vs Instagram) |
| **Casos límite** | QR parcialmente recortado, bajo contraste, foto de pantalla, QR con mucho contenido |

### Nivel E — Eficiencia (la app web lo exige)

| Métrica | Definición |
|---|---|
| Tiempo de inferencia (ms/imagen) | Debe caber en una web app |
| Parámetros y FLOPs | Coste de cómputo |
| Latencia de la demo | UX de la comparación antes/después |

### Nivel F — Rigor estadístico

- Intervalos de confianza / test de significancia entre métodos.
- Tamaño de muestra por nivel de severidad (mínimo ~500 por celda de severidad).
- Separación estricta train/val/test (sin solapamiento de contenidos).

**Resumen ejecutivo de métricas:** las 5 que definen la historia son (1) tasa de éxito verificado, (2) desglose por severidad, (3) comparación vs. baselines clásicos y herramientas libres, (4) correlación métricas-de-píxel vs. éxito funcional, y (5) rendimiento en test set real.

### 6.6 Resultados intermedios — baseline U-Net vanilla (evidencia)

Dataset: 1500 QR sintéticos (dominio: **solo QR**), `max_dim=256`, `N∈{0,1,2,4,6}`, `q∈[22,48]`, split 70/15/15 (test=225). Decode verificado (OpenCV + zbar, contenido == ground truth). Entrenamiento: 50 epochs, ~28 min en T4, val_L1 0.0062 (ep 40), sin overfitting.

| N | VDR_raw | VDR_clasico | VDR_unet | VDR_unet_clasico | VDR_2x_clasico |
|---|---|---|---|---|---|
| 2 | 0.253 | 0.387 | 0.267 | **0.413** | **0.507** |
| 4 | 0.147 | 0.240 | 0.133 | **0.267** | **0.413** |
| 6 | 0.120 | 0.213 | 0.120 | **0.213** | **0.320** |

Hallazgos (validan y corrigen el diseño):

1. **La comparación justa importa.** La U-Net decodificada en grises no gana a nada (`VDR_unet` ≈ `VDR_raw`); pero si a su salida se le aplica la **misma binarización clásica** que a los baselines (`VDR_unet_clasico`), **sí supera al clásico** (0.413 vs 0.387 en N=2; 0.267 vs 0.240 en N=4). La ganancia es denoising → binarización más limpia. Cualquier reporte futuro debe usar esta comparación, no la salida cruda en grises.
2. **El cuello de botella es la resolución, no el ruido.** El control barato (LANCZOS 2× + binarizar) gana a todo (0.507/0.413/0.320). A 256 px el QR tiene pocos píxeles/módulo y ningún denoiser recupera eso: la información perdida por el subsampling no existe en la entrada.
3. **Decisión tomada:** la formulación cambia a **super-resolución 2×** (entrada 256 → salida 512, GT al screenshot limpio a 512). Un SR aprendido que recupera los bordes reales de los módulos debe superar el control `VDR_2x_clasico`.

### 6.7 Resultado del SR vanilla (03) — la L1 no es el objetivo correcto

SR 2× con U-Net vanilla + L1/perceptual (base 32, ~3.7M params, 30 epochs, ~15 min en T4):

| N | VDR_raw | VDR_2x_clasico (control) | VDR_sr | VDR_sr_clasico |
|---|---|---|---|---|
| 2 | 0.253 | 0.520 | 0.053 | 0.360 |
| 4 | 0.147 | 0.427 | 0.013 | 0.080 |
| 6 | 0.120 | 0.347 | 0.000 | 0.040 |

**El SR con L1 destruye la decodificación** (peor que el crudo). La L1 optimiza el error de píxel promedio → salidas suaves/borrosas; para un decodificador QR la nitidez de bordes lo es todo. Conclusión de diseño: **la pérdida de píxel sola no sirve**; la ganancia vive en la cabeza binaria + una pérdida que fuerce salidas binarizables/nítidas. Eso es exactamente lo que implementa Unloss-Net (sección 4.3), no el SR vanilla.

**Nota de dominio y comparabilidad.** Las secciones 6.6 y 6.7 reportan resultados **del dominio QR** (modelos entrenados solo con QR); son baselines de dominio único y no pretenden medir texto. El hallazgo "la palanca es la resolución + el binario" es una hipótesis medida **solo en QR**; el 04/05 la testean en texto. Además, los números QR del 04 (modelo entrenado con datos **combinados** QR+texto) **no son directamente comparables** con los de 6.6/6.7: la comparación controlada es la ablación interna del propio 04 (`film`/`bin`/`full` sobre los mismos datos). En la tabla de benchmarking final, cada fila lleva su dominio explícito.

## 6.8 Resultados finales (v14-v16): dos dominios cerrados + validación real

### 6.8.1 Dominio QR (v14, modelo combinado QR+texto)

Modelo final **V9Net**: encoder convolucional (stem 48 → 96 → 192 → 384) + **GridHead** (ConvTranspose ×2 → 128×128 logits) que emite una **probabilidad por módulo** (grid_sample en las coordenadas canónicas de cada módulo, `box=2`, `POC_OUT=128`). Test set sintético:

| Método | VDR (umbral realista) | VDR (binario duro 0.5) |
|---|---|---|
| Clásico cv2 ×4 | 0.922 | 0.922 |
| **Modelo v9b** | **0.973** | 0.730 |

La ganancia del modelo sobre el clásico es **~5 %** en QR (sobre el **test set sintético**; en la validación real, §6.8.3, el modelo no transfiere — gap sintético→real medido). El umbral realista (curva) aprovecha la salida soft por módulo; el binario duro pierde en módulos borderline. El margen útil se amplía con **RS de borrados**: se marcan como borrados los codewords de menor confianza del modelo y RS los corrige (decodificador propio, sección 6.8.3) — funciona cuando la degradación es del rango sintético; no transfiere al caso real medido.

### 6.8.2 Dominio texto (v15)

Pipeline clásico **Tesseract ×4**: LineAcc **0.990** (realista) / **0.983** (duro). El baseline clásico **satura** la tarea → **no se entrena modelo de texto**. La contribución en este dominio es el protocolo de medición (LineAcc con normalización OCR estándar aplicada por igual a GT y salida) y la decisión honesta de no añadir complejidad sin ganancia medida.

### 6.8.3 Validación real (v16) — la prueba de fuego

- Poster real **5333×3000 px** con QR **v4/EC=M** incrustado (payload `https://huggingface.co/unsloth/Phi-4-mini-instruct-GGUF`).
- Reenviado por **WhatsApp** → **1600×900 px** (-77 %). El QR queda **ilegible**: `cv2` no lo decodifica y la lectura módulo a módulo naive da `mod_acc` = 0.504 (≈ azar). Es degradación real, no sintética.
- **Corrección de premisa medida:** un segundo reenvío del mismo archivo por WhatsApp no lo degradó más (WhatsApp re-codifica al subir, no al reenviar). La pérdida acumulativa se observa al descargar/re-subir o por screenshots — ver §1.
- **Rectificación canónica calibrada** contra el grid verdadero (`mod_acc` = 1.0 en el ORIGINAL). Nota de honestidad: la calibración usa el original porque el degradado no expone el quad (cv2 no lo detecta); en despliegue la geometría se estimaría del propio degradado, que el upscale ×4 sí detecta.
- **Recuperación verificada**: upscale ×4 cubic + lector → **payload EXACTO**; Otsu ×4 idéntico. Además, la herramienta web **Link2QR** decodificó la imagen degradada real directamente (sin upscaling) — ver comparación medida en §2.2.
- **Decodificador QR propio** (código completo: formato/versión/máscaras, deinterleave de bloques de longitud mixta —los cortos primero—, Reed–Solomon sobre GF(256), `RS_TABLE` v1-10, `is_func` con alineaciones múltiples y version info v7+): **40/40** QRs sintéticos v1-10 × 4 niveles EC decodifican exacto, y el QR real decodifica **exacto**.
- **Resultado del modelo + RS de borrados sobre el QR real destruido (medido en Colab)**: `mod_acc` = **0.497** (≈ azar, la lectura naive da 0.504), decisión dura **FALLO**, RS de borrados **FALLO**. El modelo entrenado en degradación sintética **no transfiere** a este caso real: el gap sintético→real (riesgo documentado en §8) quedó **demostrado** con este experimento. El camino clásico (×4 + cv2) sí recupera el payload exacto. Real-ESRGAN no pudo evaluarse en Colab (incompatibilidad de `basicsr`/`realesrgan` con el torchvision instalado; la celda está protegida con try/except y no afecta al resto).

### 6.8.4 Nota de arquitectura (vs. 4.3)

El diseño explorado en 4.3 (FiLM de severidad + atención + pérdida de tarea) se **simplificó durante el desarrollo**: el modelo final V9Net no usa FiLM ni atención; la severidad se maneja por la curva realista del umbral sobre la salida soft y la recuperación por RS de borrados. Cada iteración quedó registrada en sus notebooks y checkpoints de Drive; las ablaciones y hallazgos (L1 no basta, la resolución es el cuello de botella, el clásico es el techo) son el valor medido.

### 6.9 Mini-pipeline interactivo (v17) — la propuesta en vivo

`17_v17_mini_pipeline.ipynb` es la pieza de demo: un panel interactivo (Colab) donde el usuario sube una imagen (QR o screenshot de texto), opcionalmente la degrada sintéticamente (blur σ, escala, JPEG q) para barrer cuánto aguanta el pipeline, y ve: la imagen subida vs. la reconstrucción, si decodificó, si coincide con el payload esperado (verificación funcional), la tabla de métodos y las métricas por método. Usa la ruta clásica como motor (la que la evidencia de §6.8.3 señala como la que funciona en datos reales) y, de forma **opcional y experimental**, la ruta del modelo v9b (carga `v9b_net.pt` desde Drive si existe) para comparar ambos caminos sobre la misma imagen.

**Caso medido (QR de Disney+, desenfoque moderado, payload `https://www.disneyplus.com/es-pe`):**

| Método | ¿Decodifica? |
|---|---|
| Original / grises / contraste / afilar / umbral adaptativo / invertido | No |
| **Upscale ×4 (cubic)** | **Sí → payload exacto** |
| **Otsu (original)** | **Sí → payload exacto** |
| **Otsu ×4** | **Sí → payload exacto** |

| Métrica | Valor |
|---|---|
| PSNR subida vs reconstrucción | 48.46 |
| SSIM subida vs reconstrucción | 0.9993 |
| PSNR original vs subida (daño) | 14.30 |
| PSNR original vs reconstrucción (recuperación) | 14.24 |
| SSIM original vs reconstrucción | 0.1355 |

Lectura: frente al original limpio, la reconstrucción tiene PSNR 14.24 y SSIM 0.1355 (píxel a píxel es mala), y **aun así decodifica exacto**. Es la correlación débil métricas-de-píxel vs. éxito funcional (objetivo de §6.4) demostrada en un caso concreto, no solo en agregado. Además, la misma imagen **no pudo ser recuperada por Link2QR** (20 pases de mejora, 20 decoders sin éxito); la ruta clásica de este pipeline (upscale ×4 + Otsu) sí la leyó. Es una muestra anecdótica (N=1, desenfoque moderado), por lo que se reporta como demo y no como afirmación estadística; refuerza el hallazgo de que el *qué* (upscale + binarización + verificación) es la aportación operativa del proyecto.

**Comparación clásico vs. modelo v9b (misma imagen):**

| Ruta | ¿Decodifica? | Payload |
|---|---|---|
| **Clásica** (upscale ×4 + Otsu) | **Sí** | exacto |
| **Modelo v9b** (grid soft → QR limpio, probadas v1–v10) | No | — |

Resultado: el clásico decodificó y el modelo no. Lectura honesta, con tres cautelas: (1) el daño es **blur** (fuera de la distribución de entrenamiento del modelo, que es resize+JPEG en cadena); (2) la ruta del modelo en v17 es *best-effort* (warp canónico sin la calibración de §6.8.3, no el pipeline formal); (3) N=1. La simetría que lo hace útil: **dentro de su dominio (sintético, §6.8.1) el modelo gana 5 puntos (0.973 vs 0.922); fuera de él, el clásico gana**. Esto delimita el dominio de validez del modelo de forma medida, no asumida, y respalda la decisión de producción (opción A): el clásico es el motor, el modelo queda como ruta experimental.

## 7. Demo / aplicación

Web app construida (FastAPI + HTML/JS vanilla) donde el usuario sube una imagen degradada (QR o screenshot con texto) y ve:

- **Veredicto protagonista**: "el QR decodifica → coincide con la URL esperada" o "OCR lee: ...", con el tiempo de cómputo integrado.
- **Antes/después** con slider y lupa 1:1 (inspección, secundario).
- **Métricas en HUD** (PSNR/SSIM/vía que decodificó/latencia) + tabla por método.
- En falso positivo o coincidencia parcial, **advertencia explícita**.
- **Accesibilidad**: tema claro/oscuro (respeta `prefers-color-scheme`, override de 3 estados), tamaño de letra, reducción de animaciones, foco visible y contrastes WCAG AA verificados.
- Privacidad: las imágenes se procesan en memoria y no se guardan.

**Dos dominios en la misma ruta (coherente con §6.8):** primero se prueban los métodos QR (cv2 + zbar si existe); solo si ninguno decodifica se prueban los de **texto con Tesseract** (mismo motor que el baseline validado de §6.8.2: psm 6 + upscale ×4). La verificación de texto usa la **normalización OCR estándar (LineAcc-style) aplicada por igual al OCR y al esperado**; solo se afirma `verified` si la coincidencia normalizada es exacta. Una lectura parecida pero no idéntica (similitud ≥ 0.9) se reporta como `decoded` con nota de revisar — no se sobrevende la coincidencia. El campo `domain` de la respuesta distingue `qr`/`text`.

**Caso real medido: foto de pantalla (N=1) — límite del OCR clásico confirmado y reproducido.** El usuario probó una foto con celular de un voucher mostrado en un monitor (no un screenshot). Las letras se ven nítidas al ojo, pero la rejilla de píxeles de la pantalla genera Moiré; al limpiar/binarizar, los glifos "engordan" y se fusionan. Tesseract intentó leer un bloque real pero su confianza media fue **47.91 %** — bajo el umbral de la app (**50.0 %**) → la vía quedó en `decoded=False`. La misma imagen, que un humano lee sin esfuerzo (usa contexto; el OCR clásico no), no supera el filtro. **Reproducción sintética del mecanismo en el repo** (texto limpio → rejilla nearest-neighbor + ruido + JPEG q32): confianza base **95.6 %** → **35.0 %** leído directo y **23.8 %** tras Otsu ×4 (la binarización empeora la fusión de glifos). Mitigaciones clásicas baratas probadas y **no** rescatan el caso: denoise + Otsu (**36.0 %**), open morfológico sobre Otsu ×4 (**22–24 %**) — todas bajo 50. Conclusión: el OCR clásico depende de la separación de glifos; el dominio texto funciona en screenshots reenviados y fotos de papel (glifos separados) y falla en foto de pantalla (Moiré). Se reporta como anécdota (N=1) + reproducción sintética, no como estadística; la UI ya es honesta al mostrar "no se pudo leer" con la vía por método en la tabla.

**Lector alternativo experimental (RapidOCR, ONNX/CPU):** para el dominio texto se integra un segundo lector, **RapidOCR** (modelos PaddleOCR convertidos a ONNX, Apache-2.0, corre en CPU sin GPU), seleccionable por API/UI (`ocr_engine`: `tesseract` | `rapid` | `auto`). El clásico Tesseract sigue siendo el defecto y el motor de la memoria; RapidOCR es **experimental**: no está validado en el set de §6.8.2 ni en datos reales, y así se reporta en la UI. Mediciones sintéticas locales (N pequeño, no estadístico): sus lecturas legítimas dan confianza media **~0.97**, el ruido en Moiré extremo **~0.58** — por eso usa su propio umbral (**0.7**), ya que la escala de confianza no es la de Tesseract. Hallazgo de verificación: cuando el lector experimental produjo texto de ruido (conf ~0.58, sobre el límite), la verificación contra `expected` lo atrapó como `false_positive` antes de calibrar el umbral — evidencia de que la verificación funcional es lo que sostiene la propuesta, no el motor. En el Moiré extremo ninguno de los dos lectores rescata la lectura (límite §7, caso foto de pantalla). La foto real del voucher del usuario (N=1) sigue sin validar con el lector nuevo: la UI permite elegirlo para probarlo.

**Primer caso real medido con VLM de GPU (N=1, demo — no estadística):** el usuario procesó su voucher (foto de pantalla) con el demo gratuito de **Unlimited-OCR** (Baidu, 3.3B, MIT, requiere GPU NVIDIA ≥8 GB) y el resultado comparado con Tesseract en la misma imagen (app): Tesseract leyó `feBOTICA DEXEUS`, `**k4eteke eg OTT`, `-DEROSITO`, `S/ seal` (30-40 % de caracteres corruptos); el VLM leyó `AGENTE BCP`, `BOTICA DEXEUS`, `NO.OPE:068892`, `-DEPÓSITO`, `A CTA. AHORROS S/`, `NRO: 44011` — esencialmente perfecto, incluyendo las líneas de máscara (`NACARINO AN*****`, asteriscos reales del voucher, no error). **Dato honesto clave:** ambos fallan en `MONTO RECIBIDO: S/` — el importe no es legible en la foto en sí; ningún motor lo inventa (lo que refuerza el valor de la verificación). Conclusión medible: el VLM con prior de lenguaje (contexto) resuelve el régimen foto-de-pantalla donde el OCR clásico falla (§7, caso foto de pantalla). Costo/arquitectura (ver §7, lector alternativo): no justifica una VPS permanente (~$73-255/mes); la escalada serverless opcional (Modal/RunPod, por segundo) costaría ~$0.01 por voucher. Pendiente de validación formal: sigue siendo N=1; no hay dataset de foto-de-pantalla para estadística (el set de Drive es de screenshots, donde el clásico ya funciona).

**Límite honesto de la app en texto (medición):** la web app reutiliza el mecanismo del baseline de §6.8.2, no vuelve a medir sobre el dataset de validación (que vive en Drive). La verificación local de la ruta de texto usa un fixture sintético (screenshot de texto generado con PIL y degradado con JPEG), no el set real de la memoria; el despliegue (Render) instala Tesseract vía `apt-get` en `render.yaml`.

**Selección de la mejor lectura de texto (medición en caso real, N=1 — no estadística):** la app solía detenerse en el **primer** método que devolvía texto (umbral de confianza 50), y en una imagen real de un poema (screenshot 739×415, texto pequeño) eso mostraba "original" (similitud normalizada 0.653 vs. el texto real). Al probar **todos** los métodos y medir contra el texto del poema, **Otsu ×4** lee mejor (**0.768**) que el original (0.653) y que upscale ×4 (0.672). **Hallazgo medible clave:** la confianza media de Tesseract **no** predice la utilidad funcional — upscale ×4 da conf 83.5 con ratio 0.672 mientras que Otsu ×4 da conf 81.1 (peor en **todas** las señales de confianza: media, mediana, p25, p10) y aun así lee mejor. Es el mismo patrón ya documentado para PSNR/SSIM aplicado a la confianza del OCR: la señal de píxel no manda, el contenido sí. Diseño resultante: (a) **con `expected`** el pipeline selecciona la lectura con mayor similitud normalizada al contenido esperado (ancla funcional, `select_best_match`); (b) **sin `expected`** usa banda de confianza + la más larga (`select_best_read`, `CONF_BAND=3.0`), que en este caso elige a Otsu ×4 (el óptimo). La parada temprana solo se dispara con lectura de confianza alta (90 / 0.95) o con coincidencia exacta al esperado. Con `expected`=poema el resultado sigue siendo **`false_positive`** (0.768 < 0.9): el OCR lee la mayor parte pero con errores (tildes, `yen` por `y en`, `mds` por `más`) — la verificación lo reporta con honestidad, no lo esconde.

**Rutas:** `GET /` (UI), `GET /health` (ping de mantenimiento), `POST /api/restore` (imagen + expected opcional + use_model opcional → JSON con estado, domain, note, métodos, reconstrucción data URL, métricas).

## 8. Riesgos y mitigaciones

| Riesgo | Mitigación |
|---|---|
| Un baseline clásico (Otsu) ya recupera casi todo el VDR | Se mide desde el día 1 y se reporta; el modelo solo se defiende si lo supera |
| La corrección Reed–Solomon (hasta ~30%) ya tolera el daño | Se separan las imágenes "recuperables por EC" de las "no recuperables"; el valor de Unloss está en las segundas |
| Gap sintético→real | Test set real de los flujos reales (primer upload, screenshots, cross-app); se itera el pipeline de degradación hasta cerrar la brecha |
| Problema percibido como artificial ("envía como documento") | Se documenta el caso de uso real: capturas ya degradadas que circulan, memes, screenshots compartidos sin el original |
| Alucinaciones que decodifican a contenido erróneo | Métrica de falso positivo explícita y advertencia en la demo |

## 9. Próximos pasos

Estado real (resultados en 6.6-6.9):

- [x] Pipeline de degradación sintética parametrizado y reproducible (00, 01).
- [x] Dataset QR con severidades (manifest 1500, split 70/15/15) (01).
- [x] U-Net vanilla + L1/perceptual (02) — val_L1 0.0062, 28 min en T4.
- [x] POC de métrica de tarea: VDR por severidad vs. crudo/clásico/2x (02).
- [x] SR vanilla 2× (03) — la L1 sola destruye la decodificación (resultado en 6.7).
- [x] Medir parámetros reales de WhatsApp/Telegram/Instagram y replicarlos (06).
- [x] **Unloss-Net** (04 → v14): SR + prior binario + salida por módulo, entrenado sobre dataset combinado QR+texto; ablaciones por pieza con checkpoints y logs en Drive. Meta superada: VDR 0.973 (realista) vs clásico 0.922.
- [x] Dataset de texto + baseline texto-solo (05, v15): LineAcc 0.990/0.983 — el clásico satura, sin modelo de texto.
- [x] Comparación SOTA (Real-ESRGAN) y **validación real** (06, v16): QR real roto por WhatsApp recuperado con payload exacto (clásico); modelo+RS y Real-ESRGAN no (gap real medido).
- [x] Validación real completa en Colab (celdas finales de `16_v16_validacion_real.ipynb`) con tabla resumen de todos los métodos.
- [x] **Mini-pipeline interactivo** (17_v17): demo en Colab con degradación sintética opcional, verificación funcional y métricas por método; caso con desenfoque que Link2QR no leyó y el clásico sí (§6.9).
- [ ] Demo web + API REST del pipeline (con verificación funcional en la respuesta).
