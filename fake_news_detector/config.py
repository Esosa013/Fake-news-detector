import logging
from pathlib import Path

MODEL_PATH = Path("models/fake_news_detector.pkl")
MAX_FEATURES = 5000
NGRAM_RANGE = (1, 3)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
