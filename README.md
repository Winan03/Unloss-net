# Unloss

> Restauración **funcional** de códigos QR y texto degradados por *generation loss*: el daño acumulativo que produce el reenvío de imágenes entre apps de mensajería (resize + recompresión JPEG en cadena).

El objetivo no es que la salida "se parezca" a la imagen limpia, sino que **vuelva a decodificar** (QR) o **a leerse** (OCR) de forma verificada. La calidad se mide con métricas de tarea, no solo de píxel.

---

## Problema

WhatsApp, Telegram, Instagram, etc. redimensionan y recomprimen automáticamente las imágenes que envías. Al reenviarse varias veces, ese proceso se repite sobre una imagen ya degradada: **loss en cadena**. El contenido fino y de alto contraste —códigos QR, texto pequeño en capturas— se vuelve ilegible sin que exista el archivo original.

## Propuesta

Una red de **super-resolución + restauración** entrenada sobre un pipeline de degradación sintética que replica el de las apps, evaluada con éxito *funcional verificado*:

| Componente | Rol |
|---|---|
| **FiLM por severidad** | Un cabezal estima el nivel de degradación de la entrada y condiciona encoder/decoder; un solo modelo cubre todo el rango de severidades sin conocer `N` en inferencia |
| **Prior binario** | Salida adicional de probabilidad "es tinta" por píxel, entrenada con BCE — fuerza salidas binarizables, condición necesaria para decodificar |
| **Pérdida de tarea** | Guía el entrenamiento hacia la decodificabilidad (no hacia el error de píxel, que la L1 sola destruye) |
| **Métrica VDR / CER** | Éxito verificado: el decoder lee **y** el contenido coincide con el ground truth |

## Métricas

Las primarias son **funcionales**:

- **VDR** (Verified Decoding Rate): % de imágenes que decodifican y coinciden con el ground truth.
- **CER / WER / Exact Match**: error de caracteres/palabras del OCR sobre texto.
- **Precisión binaria de módulos**: % de celdas correctas del QR tras binarizar — enlaza con el límite de corrección Reed–Solomon.
- **Desglose por severidad** (`N` = número de reenvíos), falsos positivos y comparación contra baselines clásicos (Otsu, LANCZOS 2×, herramientas libres).

## Estado y resultados

En desarrollo (el detalle completo vive en [`docs/`](docs/proyecto-restauracion-imagenes.md)).

Hallazgos medidos hasta ahora:

- El SR vanilla con **L1 sola destruye la decodificación** (peor que el crudo): la pérdida de píxel no es el objetivo correcto.
- La comparación justa exige aplicar la **misma binarización clásica** a todas las salidas; así la U-Net sí supera al clásico, y el control barato **LANCZOS 2× + binarizar** es el techo a batir (VDR 0.520 / 0.427 / 0.347 para `N = 2, 4, 6`).
- Versión actual (04, dominio combinado QR+texto): la geometría quedó resuelta (precisión de módulos ~0.76) y el cuello de botella es el **color de los módulos**, atacado con pérdida de módulo + salida binaria a 1024 px y ablación contra un U-Net vanilla sobre los mismos datos.

## Reproducción

Los experimentos se ejecutan en Google Colab con GPU (T4). El dataset es 100 % sintético y reproducible, y los checkpoints y logs de entrenamiento se conservan en Google Drive.

Próximo paso: exponer el modelo como una API REST que la futura web app consumirá para restaurar imágenes a demanda.

## Estructura

```
├── README.md
├── LICENSE                       # Apache-2.0
└── docs/
    └── proyecto-restauracion-imagenes.md   # memoria completa (problema, estado del arte, diseño, métricas)
```

## Roadmap

- [x] Pipeline de degradación sintética reproducible y dataset QR con severidades
- [x] Baselines: U-Net vanilla + L1, SR 2× (evidencia de que la L1 no basta)
- [x] Red con FiLM + prior binario + pérdida de tarea; ablaciones por pieza
- [ ] Cerrar el color de módulos (pérdida de módulo + bin 1024) y superar el control `LANCZOS 2×`
- [ ] Dataset de texto y control texto-solo; parámetros reales de las apps
- [ ] Comparación SOTA (Real-ESRGAN, Pix2Pix) y test set real de reenvíos
- [ ] Demo web (antes/después con verificación funcional)

## Licencia

[Apache License 2.0](LICENSE).