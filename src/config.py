from pathlib import Path
import os

_COLAB_DATA    = Path('/content/lastfm')
_COLAB_RESULTS = Path('/content/music-recommender-ncf/results')
_LOCAL_DATA    = Path(__file__).parent.parent / "data" / "lastfm"
_LOCAL_RESULTS = Path(__file__).parent.parent / "results"

DATA_DIR    = _COLAB_DATA    if _COLAB_DATA.exists()    else _LOCAL_DATA
RESULTS_DIR = _COLAB_RESULTS if _COLAB_DATA.exists()    else _LOCAL_RESULTS

# Filtrado
MIN_INTERACTIONS = 5

# Features de tags
TAG_FEATURES = 50

# Hiperparámetros del modelo
EMBED_DIM  = 64
LEARNING_RATE = 1e-3
EPOCHS     = 20
BATCH_SIZE = 256
NEG_RATIO  = 4

# Evaluación
TOP_K        = 10
N_CANDIDATES = 100
TEST_SIZE    = 0.2
RANDOM_SEED  = 42
