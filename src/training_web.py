"""Фоновое обучение и объяснение ближайших соседей для интерфейса лаборатории."""

from __future__ import annotations

import io
import logging
import os
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from experiment_settings import MAX_SAMPLES, MIN_SAMPLES, validate_settings
from features import (
    LABELS,
    extract_features,
    extract_features_with_settings,
    load_wav,
    suppress_noise,
)
from model_history import ModelHistory
from trainer import (
    portable_neighbors,
    portable_predict,
    save_classifier,
    train,
    train_with_settings,
)

PALETTE = {"up": "#5277c3", "down": "#dc8350", "left": "#6a9a6c", "right": "#aa6fb5"}
logger = logging.getLogger(__name__)


class TrainingManager:
    """Выполняет 1 задачу обучения одновременно и хранит ее прогресс."""

    def __init__(self) -> None:
        """Создает состояние обучения и блокировку доступа."""
        self.lock = threading.Lock()
        self.state: dict[str, Any] = {
            "state": "idle",
            "stage": "",
            "completed": 0,
            "total": 0,
            "percent": 0,
            "message": f"Запиши хотя бы по {MIN_SAMPLES} примеров каждой команды и нажми «Обучить модель».",
        }
        self.last_vector: np.ndarray | None = None
        self.history: ModelHistory | None = None
        self.feature_callback: Callable[[Path, dict[str, Any]], np.ndarray] | None = (
            None
        )
        self.generation = 0

    def configure_history(self, history: ModelHistory | None) -> None:
        """
        Настраивает историю для режима предрасчета.

        :param history: Хранилище или None для расчета по запросу.
        """
        with self.lock:
            self.history = history

    def configure_features(
        self, callback: Callable[[Path, dict[str, Any]], np.ndarray] | None
    ) -> None:
        """
        Подключает кеш извлечения исходных признаков.

        :param callback: Вычисление признаков WAV по настройкам или None.
        """
        with self.lock:
            self.feature_callback = callback

    def invalidate(self) -> None:
        """Отменяет устаревшее обучение при изменении датасета."""
        with self.lock:
            self.generation += 1
            self.last_vector = None
            self.state = {
                "state": "idle",
                "stage": "",
                "completed": 0,
                "total": 0,
                "percent": 0,
                "message": "Датасет изменился. Обучи модель заново.",
            }

    def _progress(self, generation: int, **kwargs: Any) -> bool:
        """
        Обновляет состояние только актуальной задачи.

        :param generation: Поколение задачи.
        :param kwargs: Поля состояния.
        :return: Признак актуальной задачи.
        """
        with self.lock:
            if self.generation != generation:
                return False
            self.state.update(kwargs)
            return True

    def status(self) -> dict[str, Any]:
        """
        Возвращает независимый снимок состояния.

        :return: Текущее состояние обучения.
        """
        with self.lock:
            return dict(self.state)

    def change(self, **kwargs: Any) -> None:
        """
        Обновляет поля состояния потока обучения.

        :param kwargs: Измененные поля состояния.
        """
        with self.lock:
            self.state.update(kwargs)

    def start(self, dataset: Path, model: Path) -> dict[str, Any]:
        """
        Запускает обучение с исходными признаками версии 1.

        :param dataset: Каталог датасета.
        :param model: Путь результирующего NPZ.
        :return: Начальное состояние задачи.
        """
        return self._start(dataset, model, None)

    def start_experiment(
        self, dataset: Path, model: Path, settings: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Запускает обучение с выбранными параметрами эксперимента.

        :param dataset: Каталог датасета.
        :param model: Путь результирующего NPZ.
        :param settings: Параметры эксперимента.
        :return: Начальное состояние задачи.
        """
        return self._start(dataset, model, validate_settings(settings))

    def _start(
        self, dataset: Path, model: Path, settings: dict[str, Any] | None
    ) -> dict[str, Any]:
        """
        Проверяет датасет и запускает фоновый поток.

        :param dataset: Каталог датасета.
        :param model: Путь результирующего NPZ.
        :param settings: Новые параметры или None для исходного алгоритма.
        :return: Начальное состояние задачи.
        """
        options = settings or validate_settings()
        with self.lock:
            if self.state["state"] == "running":
                raise RuntimeError("Модель сейчас учится. Подожди немного.")
            paths = [
                (label, path)
                for label in LABELS
                for path in sorted((dataset / label).glob("*.wav"))[
                    : min(options["samples_per_class"], MAX_SAMPLES)
                ]
            ]
            counts = {
                label: sum(command == label for command, _ in paths) for label in LABELS
            }
            if min(counts.values(), default=0) < MIN_SAMPLES:
                raise ValueError(
                    f"Добавь хотя бы по {MIN_SAMPLES} записей каждой команды. Сейчас у тебя: "
                    + ", ".join(
                        f"{LABELS[label]}: {count}" for label, count in counts.items()
                    )
                )
            training_count = len(paths) * (2 if options["noise"] > 0 else 1)
            if options["k"] > training_count:
                raise ValueError(
                    "Выбери меньше соседей: их не может быть больше, чем примеров для обучения."
                )
            history = self.history
            experiment = (
                history.experiment(options, settings is None) if history else None
            )
            if history and experiment:
                try:
                    metadata = history.select(experiment[0])
                except FileNotFoundError:
                    pass
                else:
                    self.last_vector = None
                    self.state = {
                        "state": "done",
                        "stage": "done",
                        "completed": len(paths),
                        "total": len(paths),
                        "percent": 100,
                        "settings": metadata["settings"],
                        "training_count": training_count,
                        "source_count": len(paths),
                        "model_id": experiment[0],
                        "reused": True,
                        "message": "Активирована сохраненная модель с этими настройками.",
                    }
                    return dict(self.state)
            self.state = {
                "state": "running",
                "stage": "features",
                "completed": 0,
                "total": len(paths),
                "percent": 0,
                "settings": options,
                "training_count": training_count,
                "source_count": len(paths),
                "message": f"Считаю {2 * options['n_mfcc']} признаков для каждой твоей записи",
            }
            self.last_vector = None
            generation = self.generation
        threading.Thread(
            target=self._run_experiment,
            args=(paths, model, settings, generation, history, experiment),
            daemon=True,
        ).start()
        return self.status()

    def _run(self, paths: list[tuple[str, Path]], model: Path) -> None:
        """
        Выполняет исходный алгоритм обучения в фоновом потоке.

        :param paths: Классы и пути записей.
        :param model: Путь результирующего NPZ.
        """
        self._run_experiment(paths, model, None)

    def _run_experiment(
        self,
        paths: list[tuple[str, Path]],
        model: Path,
        settings: dict[str, Any] | None,
        generation: int | None = None,
        history: ModelHistory | None = None,
        experiment: tuple[str, int] | None = None,
    ) -> None:
        """
        Извлекает признаки, обучает модель и атомарно сохраняет результат.

        :param paths: Классы и пути записей.
        :param model: Путь результирующего NPZ.
        :param settings: Параметры эксперимента или исходный алгоритм.
        :param generation: Поколение задачи или текущее поколение.
        :param history: Хранилище результатов или None.
        :param experiment: Ключ и поколение истории.
        """
        if generation is None:
            generation = self.generation
        try:
            features, labels, names, augmented = [], [], [], []
            options = settings or validate_settings()
            random = np.random.default_rng(42)
            for index, (label, path) in enumerate(paths, 1):
                if not self._progress(generation):
                    return
                try:
                    samples = load_wav(path)
                    if settings is None:
                        vector = extract_features(samples)
                    elif self.feature_callback is not None:
                        vector = self.feature_callback(path, settings)
                    else:
                        vector = extract_features_with_settings(samples, settings)
                except (ValueError, OSError) as exc:
                    raise ValueError(f"Ошибка {label}/{path.name}: {exc}") from exc
                features.append(vector)
                labels.append(label)
                names.append(path)
                augmented.append(False)
                if options["noise"] > 0:
                    # Шумовой дубль создается после выбранного шумоподавления.
                    prepared = suppress_noise(
                        samples, options["denoise_strength"], options["denoise_n_fft"]
                    )
                    noise = random.normal(
                        0,
                        float(np.sqrt(np.mean(prepared**2))) * options["noise"],
                        prepared.size,
                    )
                    features.append(
                        extract_features_with_settings(
                            np.clip(prepared + noise, -1, 1),
                            {**options, "denoise_strength": 0},
                        )
                    )
                    labels.append(label)
                    names.append(path)
                    augmented.append(True)
                if not self._progress(
                    generation,
                    completed=index,
                    percent=round(index / len(paths) * 85),
                    message=f"Считаю признаки твоих записей: готово {index} из {len(paths)}",
                ):
                    return
            matrix, classes = np.vstack(features), np.asarray(labels)
            if not self._progress(
                generation,
                stage="fit",
                percent=90,
                message="Запоминаю твои примеры, чтобы узнавать новые команды",
            ):
                return
            classifier = (
                train(matrix, classes)
                if settings is None
                else train_with_settings(matrix, classes, settings)
            )
            classifier.asr_augmented = np.asarray(augmented, dtype=bool)  # ty: ignore[unresolved-attribute]
            if not self._progress(
                generation, stage="save", percent=96, message="Сохраняю твою модель"
            ):
                return
            model.parent.mkdir(parents=True, exist_ok=True)
            # Атомарная замена не позволяет прочитать недописанный NPZ во время игры.
            with tempfile.TemporaryDirectory(dir=model.parent) as temporary:
                output = save_classifier(
                    classifier, Path(temporary), matrix, classes, paths=names
                )
                with self.lock:
                    if self.generation != generation:
                        return
                    if history and experiment:
                        if not history.save(
                            experiment[0], experiment[1], output, options
                        ):
                            self.state.update(
                                state="idle",
                                stage="",
                                percent=0,
                                message="Датасет изменился. Обучи модель заново.",
                            )
                            return
                    else:
                        os.replace(output, model)
            self._progress(
                generation,
                state="done",
                stage="done",
                percent=100,
                message=f"Готово! Твоя модель запомнила {len(matrix)} примеров из {len(paths)} записей. Теперь проверь ее на новой команде",
            )
        except Exception as exc:
            # Граница фоновой задачи: любое исключение должно стать видимым интерфейсу.
            logger.exception(
                "Failed to train model", extra={"path": str(model), "error": str(exc)}
            )
            self._progress(generation, state="error", stage="error", message=str(exc))

    def set_probe(self, vector: np.ndarray) -> None:
        """
        Запоминает признаки последней проверочной записи.

        :param vector: Признаки проверочной записи.
        """
        with self.lock:
            self.last_vector = np.asarray(vector).copy()

    def get_probe(self) -> np.ndarray | None:
        """
        Возвращает копию последней проверочной записи.

        :return: Признаки записи или None.
        """
        with self.lock:
            return None if self.last_vector is None else self.last_vector.copy()


def explain(model: dict, vector: np.ndarray) -> dict:
    """
    Показывает реальные расстояния и результат голосования соседей.

    :param model: Числовые данные NPZ.
    :param vector: Признаки новой записи.
    :return: Класс, имена соседних WAV и их расстояния.
    """
    nearest, distances, _ = portable_neighbors(model, vector)
    predicted, _ = portable_predict(model, vector)
    files = model.get("train_names")
    neighbors = []
    for index, distance in zip(nearest, distances):
        label = str(model["train_y"][index])
        name = str(files[index]) if files is not None else f"Пример №{index + 1}"
        neighbors.append(
            {
                "label": label,
                "command": LABELS[label],
                "file": name,
                "distance": round(float(distance), 3),
            }
        )
    return {
        "label": predicted,
        "command": LABELS[predicted],
        "neighbors": neighbors,
        "note": f"Здесь сравниваем все {model['train_X'].shape[1]} признаков. Карта PCA помогает тебе рассмотреть записи, но не показывает все расстояния точно.",
    }


def projection_data(
    model: dict[str, Any], vector: np.ndarray | None = None
) -> dict[str, Any]:
    """
    Возвращает координаты PCA с именем каждого исходного WAV.

    :param model: Числовые данные NPZ.
    :param vector: Признаки новой записи или None.
    :return: Координаты, подписи и пояснение для интерактивного графика.
    """
    matrix = model["train_X"].astype(float)
    transform = PCA(n_components=2)
    projected = transform.fit_transform(matrix)
    names = model.get("train_names")
    augmented = model.get("train_augmented", np.zeros(len(matrix), dtype=bool))
    points = [
        {
            "x": float(x),
            "y": float(y),
            "label": str(model["train_y"][index]),
            "command": LABELS[str(model["train_y"][index])],
            "file": str(names[index]) if names is not None else f"Пример №{index + 1}",
            "augmented": bool(augmented[index]),
        }
        for index, (x, y) in enumerate(projected)
    ]
    probe = None
    if vector is not None:
        normalized = (np.asarray(vector).reshape(-1) - model["mean"]) / model["scale"]
        x, y = transform.transform(normalized.reshape(1, -1))[0]
        probe = {"x": float(x), "y": float(y)}
    feature_count = matrix.shape[1]
    return {
        "points": points,
        "probe": probe,
        "axes": ["Главная компонента 1", "Главная компонента 2"],
        "feature_count": feature_count,
        "note": f"На карте ты видишь 2 координаты. Твоя модель сравнивает все {feature_count} признаков.",
    }


def projection_png(model: dict, vector: np.ndarray | None = None) -> bytes:
    """
    Рисует проекцию признаков с отдельным местом для пояснения.

    :param model: Числовые данные NPZ.
    :param vector: Признаки новой записи или None.
    :return: График PNG.
    """
    data = projection_data(model, vector)
    figure, axis = plt.subplots(figsize=(10, 5.8))
    figure.patch.set_facecolor("#ffffff")
    figure.subplots_adjust(left=0.10, right=0.97, top=0.90, bottom=0.22)
    for label, title in LABELS.items():
        points = [point for point in data["points"] if point["label"] == label]
        axis.scatter(
            [point["x"] for point in points],
            [point["y"] for point in points],
            s=76,
            color=PALETTE[label],
            edgecolors="white",
            linewidth=0.9,
            alpha=0.9,
            label=f"{title} ({len(points)})",
        )
    if data["probe"] is not None:
        axis.scatter(
            [data["probe"]["x"]],
            [data["probe"]["y"]],
            marker="*",
            s=350,
            color="#e22754",
            zorder=8,
            edgecolors="white",
            linewidth=1.3,
            label="Новая запись",
        )
    axis.set(
        title="Записи на карте признаков (проекция PCA)",
        xlabel=data["axes"][0],
        ylabel=data["axes"][1],
    )
    axis.axhline(0, color="#dddddd", lw=0.7)
    axis.axvline(0, color="#dddddd", lw=0.7)
    axis.grid(alpha=0.15)
    axis.legend(loc="best", fontsize=9)
    figure.text(0.5, 0.055, data["note"], ha="center", fontsize=10, color="#555555")
    output = io.BytesIO()
    try:
        figure.savefig(output, format="png", dpi=120)
        return output.getvalue()
    finally:
        plt.close(figure)
