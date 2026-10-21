"""A tiny keyword spotter: MFCC statistics -> normalization -> nearest neighbours.

This is NOT a general speech-to-text recognizer. It classifies four trained words.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from features import LABELS, extract_features, load_wav


def load_dataset(directory: Path, show_progress: bool = True):
    directory = Path(directory)
    pairs = [(label, file) for label in LABELS for file in sorted((directory / label).glob("*.wav"))]
    by_class = {label: sum(c == label for c, _ in pairs) for label in LABELS}
    if min(by_class.values()) < 2:
        raise ValueError("Нужно минимум 2 WAV-файла для КАЖДОЙ команды. Сейчас: " + str(by_class))
    iterator = tqdm(pairs, desc="Извлекаем признаки из WAV", unit="запись", disable=not show_progress)
    features, classes, paths = [], [], []
    for label, path in iterator:
        try:
            vector = extract_features(load_wav(path))
        except ValueError as e:
            raise ValueError(f"Ошибка файла {path}: {e}") from e
        features.append(vector)
        classes.append(label)
        paths.append(path)
    return np.vstack(features), np.array(classes), paths


def train(X: np.ndarray, y: np.ndarray):
    # A simple explainable model: no neural-network pretraining, no network required.
    k = min(3, len(y))
    classifier = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k, weights="distance"))
    classifier.fit(X, y)
    return classifier


def save_classifier(classifier, directory: Path, X: np.ndarray, y: np.ndarray, paths: list[Path] | None = None) -> Path:
    """Write portable numeric model, compatible between Colab and local Python."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "keyword_classifier.npz"
    scaler = classifier.named_steps["standardscaler"]
    normalized = scaler.transform(X)
    np.savez_compressed(
        output, train_X=normalized.astype(np.float32), train_y=np.asarray(y, dtype="<U10"),
        mean=scaler.mean_, scale=scaler.scale_, k=np.array(min(3, len(y))),
        feature_version=np.array(1),
        train_names=np.array([f"{p.parent.name}/{p.name}" for p in paths] if paths else [f"Пример №{i+1}" for i in range(len(y))]),
    )
    return output


def portable_predict(model: dict, vector: np.ndarray):
    """Same distance-weighted kNN rule as sklearn; no unpickling or sklearn needed."""
    normalized = (np.asarray(vector).reshape(-1) - model["mean"]) / model["scale"]
    distances = np.linalg.norm(model["train_X"] - normalized, axis=1)
    nearest = np.argsort(distances, kind="stable")[: int(model["k"])]
    nearest_distances = distances[nearest]
    # sklearn distance weighting: an exact match gets all weight.
    if np.any(nearest_distances < 1e-10):
        weights = (nearest_distances < 1e-10).astype(float)
    else:
        weights = 1.0 / nearest_distances
    totals = {}
    for idx, weight in zip(nearest, weights):
        label = str(model["train_y"][idx])
        totals[label] = totals.get(label, 0.0) + float(weight)
    predicted = sorted(totals, key=lambda label: (-totals[label], label))[0]
    return predicted, nearest_distances


def draw_projection(X: np.ndarray, y: np.ndarray, ax=None):
    """2D PCA is for illustration, NOT the actual decision boundary."""
    from matplotlib import pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    projected = pca.fit_transform(scaled)
    palette = {"up": "#5277c3", "down": "#dc8350", "left": "#6a9a6c", "right": "#aa6fb5"}
    for label, title in LABELS.items():
        mask = y == label
        ax.scatter(projected[mask, 0], projected[mask, 1],
                   s=90, edgecolor="white", linewidth=1.1, color=palette[label], label=title)
    ax.axhline(0, color="#dddddd", lw=0.8)
    ax.axvline(0, color="#dddddd", lw=0.8)
    ax.set(xlabel="Главная компонента 1", ylabel="Главная компонента 2",
           title="Проекция признаков WAV в 2D (PCA)")
    ax.legend(title="Команда", loc="best")
    ax.grid(alpha=0.2)
    return ax


def explain_prediction(classifier, X: np.ndarray, y: np.ndarray, paths: list[Path], wav_path: Path):
    """Show *real* nearest-neighbour distances in 26-dimensional scaled space."""
    query = extract_features(load_wav(wav_path)).reshape(1, -1)
    predicted = classifier.predict(query)[0]
    scaler = classifier.named_steps["standardscaler"]
    nn = classifier.named_steps["kneighborsclassifier"]
    dist, indices = nn.kneighbors(scaler.transform(query))
    fig, ax = plt.subplots(figsize=(9, 4))
    descriptions = [f"{paths[int(i)].parent.name}/{paths[int(i)].name} · {LABELS[y[int(i)]]}"
                    for i in indices[0]]
    bars = ax.barh(descriptions[::-1], dist[0][::-1], color="#5277c3")
    ax.bar_label(bars, fmt="%.2f", padding=4)
    ax.set(xlabel="Расстояние в нормализованном пространстве 26 признаков",
           title=f"Ближайшие записи → выбран класс «{LABELS[predicted]}»")
    ax.set_xlim(0, max(1.0, float(dist.max()) * 1.3))
    ax.grid(axis="x", alpha=0.15)
    fig.tight_layout()
    return predicted, fig
