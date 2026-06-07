from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model1_sederhana.h5"

IMG_SIZE = 64
CLASSES = ["buildings", "forest", "mountain"]
