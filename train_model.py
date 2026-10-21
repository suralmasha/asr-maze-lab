"""Offline backup if Jupyter isn't available: python train_model.py"""
from pathlib import Path
from trainer import load_dataset, train, save_classifier

ROOT = Path(__file__).resolve().parent
X, y, _ = load_dataset(ROOT / "dataset")
classifier = train(X, y)
path = save_classifier(classifier, ROOT / "models", X, y)
print(f"Модель готова: {path} ({len(y)} записей, {X.shape[1]} признаков)")
