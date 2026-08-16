# Proyecto Unloss — Restauración funcional de QR/Texto degradados por pérdida generacional en apps de mensajería

## 0. Nombre del proyecto

**Unloss** = "un" (revertir) + "loss" (pérdida por compresión). Hace alusión directa al término técnico **generation loss**: la degradación acumulativa que sufre una imagen cada vez que una app de mensajería la redimensiona y recomprime. Verificado: sin conflictos de marca relevantes en tecnología/restauración de imágenes (solo un webapp escolar japonés y un perfil de foro usan "Unloss"; "Unlost" existe como empresa de servicios web en Francia y de logística en Suiza, pero es una palabra distinta).

---

## 1. Problema identificado

Las apps de mensajería (WhatsApp, Telegram, Instagram, etc.) redimensionan y recomprimen automáticamente las imágenes enviadas. Cuando una imagen se reenvía varias veces, este proceso se repite en cadena: resize + recompresión JPEG sobre una imagen ya degradada. Resultado: **pérdida acumulativa de calidad** que se nota especialmente en contenido fino de alto contraste como códigos QR o texto pequeño en screenshots, llegando a hacerlos ilegibles.

### Caracterización del daño (distinto a lo que estudia la literatura)

| Tipo de daño | Estudiado en la literatura | Relevante a este proyecto |
|---|---|---|
| Daño físico (rayones, dobleces, tinta desvanecida, agujeros) | Sí, muy estudiado (QR industrial/logística) | No es el foco |
| Motion blur / desenfoque de cámara | Sí, muy estudiado (QR deblurring) | No es el foco |
| Baja resolución / super-resolución | Sí, muy estudiado | Parcial (un componente del pipeline) |
| Doble compresión JPEG (forense: detectar manipulación) | Sí, pero con fin de *detección*, no de *restauración* | Conceptual |
| **Compresión en cadena (N recomprimaiones) de apps de mensajería + restauración funcional** | **No encontrado** | **Este proyecto** |

La combinación específica — degradación sintetizada replicando el pipeline real de mensajería + aplicación a QR/texto + evaluación con métrica de *tarea* (decodificación/OCR verificada) — **no se encuentra publicada ni implementada como herramienta**.

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

### 2.2 Herramientas existentes (lo que un usuario usaría hoy)

| Herramienta | Enfoque | Método | Límite fundamental |
|---|---|---|---|
| Link2QR (link2qr.com/decode) | Reparar/decodificar QR | Varias pasadas clásicas (grises, contraste, sharpen, umbral adaptativo) + múltiples decoders | 100% basado en reglas; si el decoder falla, falla |
| QR Crafter (qr-crafter.com) | Reparar QR | Decodifica y **regenera** un QR limpio con los mismos datos | Solo funciona si el contenido aún se decodifica; **no puede recuperar contenido ilegible** |
| EAN Check QR Recovery | Recuperar QR dañado | 13 estrategias de mejora + redundancia Reed–Solomon | Sin ML; no supera el límite de corrección de errores del QR |
| Fuzana AI QR Restorer | Restaurar QR borroso | Upscaler IA genérico (estilo Real-ESRGAN) | Upscaling genérico; no entrenado en pérdida generacional; sin verificación funcional |
| QR Sharpener | Reconstruir QR borroso | Segmentación en rejilla n×n y clasificación blanco/negro por celda | Heurístico simple; requiere parámetros de rejilla manuales |
| QRazyBox | Recuperar QR | Editor manual píxel a píxel + decodificador Reed–Solomon universal | Manual, lento, requiere experto |
| AKVIS Artifact Remover AI | Artefactos JPEG ("modo extreme" para re-guardados) | ML clásico de de-artefactado | Fotos generales; no enfocado en tarea QR/texto |
| Topaz JPEG to RAW / Oakgen / PhotoSharpener | De-artefactado + upscale de fotos | ML (difusión / Real-ESRGAN) | Imágenes naturales; no evalúan si el QR decodifica o el OCR lee |

### 2.3 Tabla comparativa Unloss vs. el panorama (benchmarking)

| Dimensión | Papers de QR (2.1) | Herramientas (2.2) | **Unloss** |
|---|---|---|---|
| Pipeline de degradación = cadena real de mensajería (resize + JPEG × N) | No | Parcial (reglas manuales) | **Sí, modelado explícito y parametrizado** |
| Recuperación generativa cuando el decoder/OCR falla por completo | Solo investigado (papers) | **No** (QR Crafter requiere datos legibles) | **Sí: reconstrucción aprendida + verificación** |
| Métrica principal = tarea verificada (decodifica **y** coincide con ground truth) | Parcial (RR ≠ contenido correcto) | No | **Sí** |
| OCR de texto en screenshots reenviados | No | No | **Sí (segundo dominio del mismo modelo)** |
| Demostrar que PSNR/SSIM/LPIPS no predicen la utilidad funcional | No es el objetivo | No | **Entregable explícito** |

---

## 3. Propuesta de valor novedosa ("lo que nadie hizo")

1. **Modelo de degradación fiel a la cadena de reenvíos.** Medir primero los parámetros reales de WhatsApp/Telegram (resolución de salida, quality factor, pipeline de re-encode) y replicarlos. Ninguna herramienta ni paper modela esta cadena; solo reglas genéricas o blur/ruido.
2. **Recuperación generativa más allá del límite de los decoders y de la corrección Reed–Solomon.** Las herramientas existentes operan "intenta mejorar y decodifica"; si el contenido es ilegible, se rinden. Unloss aprende a *reconstruir módulos/letras* a partir de la apariencia, incluso cuando el decoder clásico falla. Es el diferenciador más claro contra QR Crafter, Link2QR y EAN Check.
3. **Verificación funcional de contenido (no solo "decodifica").** Reportar por separado: (a) decodifica y el contenido coincide con el ground truth, (b) decodifica a contenido **erróneo** (falso positivo, peligroso), (c) no decodifica. La mayoría de la literatura reporta solo (a)/(c) mezclados.
4. **Doble dominio QR + texto.** Un solo modelo para QRs y screenshots de texto da cobertura que ninguna herramienta cubre.
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
2. Simular el pipeline real de mensajería, **medido**:
   - Resize a las resoluciones reales que aplican las apps (e.g. WhatsApp: 1280px max / 960px / 640px).
   - Recompresión JPEG con el quality factor real medido.
   - Repetir el par resize+recompresa **N veces** (N = 0..10) para barrer severidades.
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

## 7. Demo / aplicación

Web app donde el usuario sube una imagen degradada (QR o screenshot con texto) y ve:
- Comparación antes/después con slider.
- Verificación funcional: "el QR ahora decodifica → URL correcto" o "OCR lee: ...".
- En caso de falso positivo, advertencia explícita.

## 8. Riesgos y mitigaciones

| Riesgo | Mitigación |
|---|---|
| Un baseline clásico (Otsu) ya recupera casi todo el VDR | Se mide desde el día 1 y se reporta; el modelo solo se defiende si lo supera |
| La corrección Reed–Solomon (hasta ~30%) ya tolera el daño | Se separan las imágenes "recuperables por EC" de las "no recuperables"; el valor de Unloss está en las segundas |
| Gap sintético→real | Test set real de reenvíos por la app; se itera el pipeline de degradación hasta cerrar la brecha |
| Problema percibido como artificial ("envía como documento") | Se documenta el caso de uso real: capturas ya degradadas que circulan, memes, screenshots compartidos sin el original |
| Alucinaciones que decodifican a contenido erróneo | Métrica de falso positivo explícita y advertencia en la demo |

## 9. Próximos pasos

Estado real (resultados en 6.6):

- [x] Pipeline de degradación sintética parametrizado y reproducible (00, 01).
- [x] Dataset QR con severidades (manifest 1500, split 70/15/15) (01).
- [x] U-Net vanilla + L1/perceptual (02) — val_L1 0.0062, 28 min en T4.
- [x] POC de métrica de tarea: VDR por severidad vs. crudo/clásico/2x (02).
- [x] SR vanilla 2× (03) — la L1 sola destruye la decodificación (resultado en 6.7).
- [ ] Medir parámetros reales de WhatsApp/Telegram/Instagram (resolución de salida, quality factor) — protocolo en 06.
- [ ] **Unloss-Net real** (04): SR + FiLM de severidad + prior binario (BCE) + pérdida de tarea, entrenado sobre el dataset **combinado QR + texto**; ablaciones por pieza (`film`/`bin`/`full`) en **celdas de código explícitas** (evidencia, no configuración manual), cada una con checkpoint y log incremental en Drive. Meta = superar `VDR_2x_clasico` (0.520/0.427/0.347) y `EM_2x` de texto.
- [ ] Dataset de texto con el mismo pipeline (manifest_text.csv, 600 muestras, seed reproducible) + baseline texto-solo como control (05).
- [ ] Comparación SOTA: Real-ESRGAN, Pix2Pix, herramientas libres sobre el mismo test set (07).
- [ ] Test set real (reenvíos por la app) y ajuste del pipeline (06).
- [ ] Demo web.
