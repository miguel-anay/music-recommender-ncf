from pathlib import Path

DATA_DIR    = Path(__file__).parent.parent / "data" / "lastfm"
RESULTS_DIR = Path(__file__).parent.parent / "results"

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
