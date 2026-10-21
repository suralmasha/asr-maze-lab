"""Обучение локального классификатора команд из собранного датасета."""

import logging
from pathlib import Path

from trainer import load_dataset, save_classifier, train

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s | path=%(path)s | count=%(count)s | %(params)s",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
X, y, paths = load_dataset(ROOT / "dataset")
classifier = train(X, y)
path = save_classifier(classifier, ROOT / "models", X, y, paths=paths)
logger.info(
    "Trained keyword classifier",
    extra={"path": str(path), "count": len(y), "params": {"feature_count": X.shape[1]}},
)
