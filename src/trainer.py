"""Объяснимая классификация команд по статистикам MFCC и ближайшим соседям."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from experiment_settings import MAX_SAMPLES, MIN_SAMPLES, validate_settings
from features import (
    LABELS,
    N_FFT,
    N_MELS,
    N_MFCC,
    extract_features,
    extract_features_with_settings,
    load_wav,
)


def load_dataset(
    directory: Path, show_progress: bool = True
) -> tuple[np.ndarray, np.ndarray, list[Path]]:
    """
    Загружает записи каждой команды в пределах конфигурации лаборатории.

    :param directory: Корневой каталог датасета.
    :param show_progress: Отображение прогресса в терминале.
    :return: Матрица признаков, классы и пути записей.
    """
    directory = Path(directory)
    pairs = [
        (label, file)
        for label in LABELS
        for file in sorted((directory / label).glob("*.wav"))[:MAX_SAMPLES]
    ]
    by_class = {
        label: sum(command == label for command, _ in pairs) for label in LABELS
    }
    if min(by_class.values()) < MIN_SAMPLES:
        raise ValueError(
            f"Нужно минимум {MIN_SAMPLES} WAV-файлов для каждой команды. Сейчас: {by_class}"
        )
    iterator = tqdm(
        pairs,
        desc="Извлечение признаков из WAV",
        unit="запись",
        disable=not show_progress,
    )
    features, classes, paths = [], [], []
    for label, path in iterator:
        try:
            vector = extract_features(load_wav(path))
        except ValueError as exc:
            raise ValueError(f"Ошибка файла {path}: {exc}") from exc
        features.append(vector)
        classes.append(label)
        paths.append(path)
    return np.vstack(features), np.array(classes), paths


def train(X: np.ndarray, y: np.ndarray) -> Pipeline:
    """
    Обучает исходный классификатор с 3 взвешенными соседями.

    :param X: Матрица признаков.
    :param y: Классы обучающих записей.
    :return: Обученный конвейер нормализации и классификации.
    """
    classifier = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=min(3, len(y)), weights="distance"),
    )
    classifier.fit(X, y)
    return classifier


def train_with_settings(
    X: np.ndarray, y: np.ndarray, settings: dict[str, Any]
) -> Pipeline:
    """
    Обучает классификатор с выбранными параметрами эксперимента.

    :param X: Матрица признаков.
    :param y: Классы обучающих записей.
    :param settings: Параметры эксперимента.
    :return: Обученный конвейер с сохраненными параметрами.
    """
    settings = validate_settings(settings)
    if settings["k"] > len(y):
        raise ValueError(
            "Количество соседей не должно превышать число обучающих примеров."
        )
    classifier = make_pipeline(
        StandardScaler(with_mean=settings["normalize"], with_std=settings["normalize"]),
        KNeighborsClassifier(
            n_neighbors=settings["k"],
            weights=settings["weights"],
            metric=settings["metric"],
        ),
    )
    classifier.fit(X, y)
    classifier.asr_settings = settings
    return classifier


def save_classifier(
    classifier: Pipeline,
    directory: Path,
    X: np.ndarray,
    y: np.ndarray,
    paths: list[Path] | None = None,
) -> Path:
    """
    Сохраняет числовую модель без pickle вместе с параметрами признаков.

    :param classifier: Обученный конвейер.
    :param directory: Каталог модели.
    :param X: Исходные обучающие признаки.
    :param y: Классы обучающих записей.
    :param paths: Пути обучающих записей.
    :return: Путь сохраненного NPZ.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "keyword_classifier.npz"
    scaler = classifier.named_steps["standardscaler"]
    neighbors = classifier.named_steps["kneighborsclassifier"]
    settings = getattr(classifier, "asr_settings", None)
    feature_count = X.shape[1]
    np.savez_compressed(
        output,
        train_X=scaler.transform(X).astype(np.float32),
        train_y=np.asarray(y, dtype="<U10"),
        mean=scaler.mean_ if scaler.mean_ is not None else np.zeros(feature_count),
        scale=scaler.scale_ if scaler.scale_ is not None else np.ones(feature_count),
        k=np.array(neighbors.n_neighbors),
        metric=np.array(neighbors.metric),
        weights=np.array(neighbors.weights),
        feature_version=np.array(2 if settings is not None else 1),
        settings_json=np.array(json.dumps(settings or {}, ensure_ascii=False)),
        train_augmented=np.asarray(
            getattr(classifier, "asr_augmented", np.zeros(len(y), dtype=bool))
        ),
        train_names=np.array(
            [f"{path.parent.name}/{path.name}" for path in paths]
            if paths
            else [f"Пример №{index + 1}" for index in range(len(y))]
        ),
    )
    return output


def model_settings(model: dict[str, Any]) -> dict[str, Any]:
    """
    Восстанавливает параметры новой или исходной модели.

    :param model: Числовые данные NPZ.
    :return: Проверенные параметры модели.
    """
    if int(model["feature_version"]) == 1:
        # Метаданные версии 1 сохраняют исторические значения независимо от JSON.
        return {
            "n_mels": N_MELS,
            "n_mfcc": N_MFCC,
            "n_fft": N_FFT,
            "hop_length": N_FFT // 4,
            "denoise_strength": 0.0,
            "denoise_n_fft": N_FFT,
            "normalize": True,
            "metric": "euclidean",
            "weights": "distance",
            "k": int(model["k"]),
            "samples_per_class": 50,
            "trim_silence": True,
            "noise": 0.0,
        }
    supplied = (
        json.loads(str(model["settings_json"])) if "settings_json" in model else {}
    )
    settings = validate_settings(supplied)
    settings["k"] = int(model["k"])
    return settings


def model_features(model: dict[str, Any], wav: np.ndarray) -> np.ndarray:
    """
    Извлекает признаки тем же способом, что использовался при обучении.

    :param model: Числовые данные NPZ.
    :param wav: Отсчеты проверяемой записи.
    :return: Вектор признаков нужной версии и размерности.
    """
    if int(model["feature_version"]) == 1:
        return extract_features(wav)
    return extract_features_with_settings(wav, model_settings(model))


def portable_neighbors(
    model: dict[str, Any], vector: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Находит соседей с метрикой и весами обученной модели.

    :param model: Числовые данные NPZ.
    :param vector: Вектор новой записи.
    :return: Индексы, расстояния и голоса ближайших соседей.
    """
    normalized = (np.asarray(vector).reshape(-1) - model["mean"]) / model["scale"]
    difference = model["train_X"] - normalized
    metric = str(model.get("metric", "euclidean"))
    distances = (
        np.abs(difference).sum(axis=1)
        if metric == "manhattan"
        else np.linalg.norm(difference, axis=1)
    )
    nearest = np.argsort(distances, kind="stable")[: int(model["k"])]
    selected = distances[nearest]
    if str(model.get("weights", "distance")) == "uniform":
        weights = np.ones(selected.size)
    elif np.any(selected < 1e-10):
        weights = (selected < 1e-10).astype(float)
    else:
        weights = 1.0 / selected
    return nearest, selected, weights


def portable_predict(model: dict, vector: np.ndarray) -> tuple[str, np.ndarray]:
    """
    Повторяет голосование k-NN без загрузки pickle и sklearn.

    :param model: Числовые данные NPZ.
    :param vector: Вектор новой записи.
    :return: Выбранный класс и расстояния до ближайших соседей.
    """
    nearest, distances, weights = portable_neighbors(model, vector)
    totals: dict[str, float] = {}
    for index, weight in zip(nearest, weights):
        label = str(model["train_y"][index])
        totals[label] = totals.get(label, 0.0) + float(weight)
    predicted = min(totals, key=lambda label: (-totals[label], label))
    return predicted, distances


def draw_projection(X: np.ndarray, y: np.ndarray, ax: Axes | None = None) -> Axes:
    """
    Показывает признаки в 2 координатах PCA.

    :param X: Матрица признаков.
    :param y: Классы записей.
    :param ax: Ось графика или None для новой фигуры.
    :return: Ось построенного графика.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    projected = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(X))
    palette = {
        "up": "#5277c3",
        "down": "#dc8350",
        "left": "#6a9a6c",
        "right": "#aa6fb5",
    }
    for label, title in LABELS.items():
        mask = y == label
        ax.scatter(
            projected[mask, 0],
            projected[mask, 1],
            s=90,
            edgecolor="white",
            linewidth=1.1,
            color=palette[label],
            label=title,
        )
    ax.axhline(0, color="#dddddd", lw=0.8)
    ax.axvline(0, color="#dddddd", lw=0.8)
    ax.set(
        xlabel="Главная компонента 1",
        ylabel="Главная компонента 2",
        title="Проекция признаков WAV в 2D (PCA)",
    )
    ax.legend(title="Команда", loc="best")
    ax.grid(alpha=0.2)
    return ax


def explain_prediction(
    classifier: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    paths: list[Path],
    wav_path: Path,
) -> tuple[str, Figure]:
    """
    Показывает реальные расстояния до соседей для отдельного WAV.

    :param classifier: Обученный конвейер.
    :param X: Обучающие признаки.
    :param y: Классы записей.
    :param paths: Пути обучающих записей.
    :param wav_path: Путь проверяемой записи.
    :return: Предсказанный класс и график соседей.
    """
    settings = getattr(classifier, "asr_settings", None)
    wav = load_wav(wav_path)
    query = (
        extract_features_with_settings(wav, settings)
        if settings is not None
        else extract_features(wav)
    ).reshape(1, -1)
    predicted = str(classifier.predict(query)[0])
    scaler = classifier.named_steps["standardscaler"]
    neighbors = classifier.named_steps["kneighborsclassifier"]
    distances, indices = neighbors.kneighbors(scaler.transform(query))
    figure, axis = plt.subplots(figsize=(9, 4))
    descriptions = [
        f"{paths[int(index)].parent.name}/{paths[int(index)].name} - {LABELS[y[int(index)]]}"
        for index in indices[0]
    ]
    bars = axis.barh(descriptions[::-1], distances[0][::-1], color="#5277c3")
    axis.bar_label(bars, fmt="%.2f", padding=4)
    axis.set(
        xlabel=f"Расстояние в пространстве {X.shape[1]} признаков",
        title=f"Ближайшие записи: выбран класс «{LABELS[predicted]}»",
    )
    axis.set_xlim(0, max(1.0, float(distances.max()) * 1.3))
    axis.grid(axis="x", alpha=0.15)
    figure.tight_layout()
    return predicted, figure
