"""Background local training and explainable k-NN visualization for the workshop UI."""
from __future__ import annotations

import io
import os
import tempfile
import threading
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from features import LABELS, extract_features, load_wav
from trainer import save_classifier, train

PALETTE = {"up": "#5277c3", "down": "#dc8350", "left": "#6a9a6c", "right": "#aa6fb5"}


class TrainingManager:
    """One running job at a time; UI can poll real audio-processing progress."""
    def __init__(self):
        self.lock = threading.Lock()
        self.state = {"state": "idle", "stage": "", "completed": 0, "total": 0,
                      "percent": 0, "message": "Добавь записи, затем нажми «Обучить»."}
        self.last_vector = None

    def status(self):
        with self.lock:
            return dict(self.state)

    def change(self, **kwargs):
        with self.lock:
            self.state.update(kwargs)

    def start(self, dataset: Path, model: Path):
        with self.lock:
            if self.state["state"] == "running":
                raise RuntimeError("Обучение уже идёт.")
            paths = [(label, p) for label in LABELS
                     for p in sorted((dataset / label).glob("*.wav"))]
            counts = {label: sum(l == label for l, _ in paths) for label in LABELS}
            if min(counts.values(), default=0) < 2:
                raise ValueError("Для обучения нужно минимум по 2 записи каждой команды. "
                                 + ", ".join(f"{LABELS[k]}: {v}" for k, v in counts.items()))
            self.state = {"state": "running", "stage": "features", "completed": 0,
                          "total": len(paths), "percent": 0,
                          "message": "Извлекаем 26 MFCC-признаков из каждого WAV…"}
            self.last_vector = None
        threading.Thread(target=self._run, args=(paths, model), daemon=True).start()
        return self.status()

    def _run(self, paths, model):
        try:
            X, y, names = [], [], []
            total = len(paths)
            for index, (label, path) in enumerate(paths, 1):
                try:
                    vector = extract_features(load_wav(path))
                except (ValueError, OSError) as exc:
                    raise ValueError(f"Ошибка {label}/{path.name}: {exc}") from exc
                X.append(vector)
                y.append(label)
                names.append(path)
                self.change(completed=index, percent=round(index / total * 85),
                            message=f"Извлечены признаки: {index} из {total} записей")
            X, y = np.vstack(X), np.asarray(y)
            self.change(stage="fit", percent=90,
                        message="Нормализуем признаки и запоминаем ближайших соседей…")
            classifier = train(X, y)
            self.change(stage="save", percent=96, message="Сохраняем обученную модель…")
            model.parent.mkdir(parents=True, exist_ok=True)
            # Atomic replacement prevents the game from loading a partially-written NPZ.
            with tempfile.TemporaryDirectory(dir=model.parent) as tmp:
                temporary = save_classifier(classifier, Path(tmp), X, y, paths=names)
                os.replace(temporary, model)
            self.change(state="done", stage="done", percent=100,
                        message=f"Модель обучена на {total} записях и готова к игре!")
        except Exception as exc:
            self.change(state="error", stage="error", message=str(exc))

    def set_probe(self, vector):
        with self.lock:
            self.last_vector = np.asarray(vector).copy()

    def get_probe(self):
        with self.lock:
            return None if self.last_vector is None else self.last_vector.copy()


def explain(model: dict, vector: np.ndarray) -> dict:
    """True neighbours/distances in 26D, not PCA-projected distances."""
    normalized = (np.asarray(vector).reshape(-1) - model["mean"]) / model["scale"]
    distances = np.linalg.norm(model["train_X"] - normalized, axis=1)
    nearest = np.argsort(distances, kind="stable")[:int(model["k"])]
    ds = distances[nearest]
    weights = (ds < 1e-10).astype(float) if np.any(ds < 1e-10) else 1.0 / ds
    totals = {}
    neighbors = []
    files = model.get("train_names")
    for idx, distance, weight in zip(nearest, ds, weights):
        label = str(model["train_y"][idx])
        totals[label] = totals.get(label, 0.0) + float(weight)
        name = str(files[idx]) if files is not None else f"Пример №{idx + 1}"
        neighbors.append({"label": label, "command": LABELS[label], "file": name,
                          "distance": round(float(distance), 3)})
    predicted = sorted(totals, key=lambda label: (-totals[label], label))[0]
    return {"label": predicted, "command": LABELS[predicted], "neighbors": neighbors,
            "note": "Расстояния вычислены по всем 26 признакам; PCA на графике — только иллюстрация."}


def projection_png(model: dict, vector: np.ndarray | None = None) -> bytes:
    """Visualize 26-dimensional stored feature vectors projected onto two axes."""
    X = model["train_X"].astype(float)
    y = model["train_y"]
    transform = PCA(n_components=2)
    # Center the training points; new voice projected using the same PCA fitted on training points.
    points = transform.fit_transform(X)
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    fig.patch.set_facecolor("#ffffff")
    for label, title in LABELS.items():
        mask = y == label
        ax.scatter(points[mask, 0], points[mask, 1], s=76,
                   color=PALETTE[label], edgecolors="white", linewidth=0.9,
                   alpha=0.9, label=f"{title} ({int(mask.sum())})")
    if vector is not None:
        normalized = (np.asarray(vector).reshape(-1) - model["mean"]) / model["scale"]
        x, z = transform.transform(normalized.reshape(1, -1))[0]
        ax.scatter([x], [z], marker="*", s=350, color="#e22754", zorder=8,
                   edgecolors="white", linewidth=1.3, label="Новая запись")
    ax.set(title="Записи на карте признаков (проекция PCA)",
           xlabel="Главная компонента 1", ylabel="Главная компонента 2")
    ax.axhline(0, color="#dddddd", lw=0.7)
    ax.axvline(0, color="#dddddd", lw=0.7)
    ax.grid(alpha=0.15)
    ax.legend(loc="best", fontsize=9)
    fig.text(0.5, 0.01, "На рисунке 2 координаты; k-NN сравнивает записи по всем 26 признакам.",
             ha="center", fontsize=10, color="#555555")
    output = io.BytesIO()
    try:
        fig.savefig(output, format="png", dpi=120)
        return output.getvalue()
    finally:
        plt.close(fig)
