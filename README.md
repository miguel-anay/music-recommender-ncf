# Sistema de Recomendación Musical — NCF Híbrido

Sistema de recomendación basado en **Neural Collaborative Filtering (NCF)** con features de contenido semántico, entrenado sobre el dataset **HetRec 2011 Last.fm**.

Resuelve las limitaciones del SVD baseline identificadas en el módulo:
- Cold-start parcial → features de tags para artistas sin historial
- Datos ricos sin usar → tags TF-IDF integrados como tercer embedding
- Modelo estático → arquitectura preparada para `partial_fit` incremental

---

## Estructura

```
proyecto/
├── data/lastfm/          ← Archivos .dat del dataset
├── notebooks/
│   ├── 01_EDA.ipynb      ← Análisis exploratorio
│   ├── 02_data_prep.ipynb← Pipeline de preprocesamiento
│   ├── 03_modelo_ncf.ipynb← Entrenamiento NCF
│   └── 04_evaluacion.ipynb← Comparativa SVD vs NCF
├── src/
│   ├── config.py         ← Rutas e hiperparámetros
│   ├── dataset.py        ← Carga, filtrado, binarización, split, Dataset PyTorch
│   ├── model.py          ← Clase NCFHybrid(nn.Module)
│   ├── metrics.py        ← hit_rate_at_k, ndcg_at_k, evaluate_svd_baseline
│   └── train.py          ← Loop de entrenamiento, guardado de checkpoint
├── results/              ← best_model.pt, gráficas
└── requirements.txt
```

---

## Dataset

**HetRec 2011 Last.fm** — 1892 usuarios, 17632 artistas, 92834 interacciones.

Los archivos `.dat` NO están en el repositorio. Descárgalos desde:
[https://grouplens.org/datasets/hetrec-2011/](https://grouplens.org/datasets/hetrec-2011/)

Colócalos en `data/lastfm/`.

---

## Instalación

```bash
pip install -r requirements.txt
```

> **Nota:** `numpy<2` es obligatorio por compatibilidad con `scikit-surprise` si se usa el baseline.

---

## Uso

Ejecuta los notebooks en orden:

```
01_EDA.ipynb        → Exploración del dataset
02_data_prep.ipynb  → Preprocesamiento paso a paso
03_modelo_ncf.ipynb → Entrenamiento (recomendado: Google Colab con GPU T4)
04_evaluacion.ipynb → Métricas Hit Rate@10 y NDCG@10
```

### En Google Colab

1. Sube los archivos `.dat` a Drive
2. Al inicio de cada notebook, ajusta `DATA_DIR_OVERRIDE`:
   ```python
   DATA_DIR_OVERRIDE = '/content/drive/MyDrive/lastfm'
   ```
3. Activa GPU: **Entorno de ejecución → Cambiar tipo → GPU (T4)**

---

## Arquitectura del modelo

```
user_idx  → Embedding(64) ─┐
artist_idx → Embedding(64) ─┼→ concat(192) → MLP → Sigmoid → [0,1]
tag_features → Linear(50→64)┘

MLP: 192 → 128 → ReLU → 64 → ReLU → 1 → Sigmoid
Pérdida: BCELoss | Optimizer: Adam | Negative sampling: 4:1
```

---

## Evaluación

Protocolo leave-one-out: 1 positivo + 99 negativos aleatorios por usuario.

| Métrica | Descripción |
|---------|-------------|
| Hit Rate@10 | % usuarios con su artista positivo en top-10 |
| NDCG@10 | Ganancia de descuento normalizada con ponderación de posición |

---

## Trabajo futuro

- Incorporar red social (`user_friends`) como grafo de conocimiento
- Negative sampling por popularidad para negativos más difíciles
- Entrenamiento incremental con `partial_fit` (online learning)
