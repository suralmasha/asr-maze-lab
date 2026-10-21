"""Настройки лаборатории, подготовка результатов и фоновый предрасчет."""

from __future__ import annotations

import hashlib
import io
import json
import os
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from computation_cache import DiskCache
from experiment_settings import validate_settings
from features import (
    LABELS,
    encode_wav,
    extract_features_with_settings,
    load_wav,
    make_feature_signal_figure,
    make_mel_figure,
    make_preprocessing_figure,
    suppress_noise,
    trim_silence,
)
from lab_config import CONFIG

FEATURE_KEYS = (
    "n_mels",
    "n_mfcc",
    "n_fft",
    "hop_length",
    "denoise_strength",
    "denoise_n_fft",
    "trim_silence",
)


def dataset_stamp(dataset: Path) -> str:
    """
    Вычисляет отпечаток имен и содержимого записей.

    :param dataset: Каталог записей.
    :return: Отпечаток текущего датасета.
    """
    digest = hashlib.sha256()
    for label in LABELS:
        for path in sorted((dataset / label).glob("*.wav")):
            digest.update(f"{label}/{path.name}\0".encode())
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


class ExperimentRuntime:
    """Хранит общие настройки и вычисления одного локального сервера."""

    def __init__(
        self, dataset: Path, directory: Path, mode: str, plot_lock: Any
    ) -> None:
        """
        Загружает настройки и создает кеш только для режима предрасчета.

        :param dataset: Каталог записей.
        :param directory: Каталог состояния лаборатории.
        :param mode: Режим вычислений.
        :param plot_lock: Общая блокировка matplotlib.
        """
        self.dataset = dataset
        self.directory = directory
        self.mode = mode
        self.plot_lock = plot_lock
        self.lock = threading.RLock()
        self.path = directory / "lab_settings.json"
        self.settings = validate_settings()
        self.stamp = dataset_stamp(dataset)
        self.persisted_stamp = self.stamp
        if self.path.exists():
            saved = json.loads(self.path.read_text())
            self.settings = validate_settings(saved.get("settings", {}))
            self.persisted_stamp = saved.get("dataset", self.stamp)
        self.cache = DiskCache(directory / "cache") if mode == "precompute" else None
        if self.cache is not None and self.persisted_stamp != self.stamp:
            self.cache.clear()
        elif self.cache is None and self.persisted_stamp != self.stamp:
            self._clear_inactive_cache()
        self.save()

    def save(self) -> None:
        """Атомарно сохраняет настройки и отпечаток датасета."""
        with self.lock:
            self.directory.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps({"settings": self.settings, "dataset": self.stamp})
            )
            os.replace(temporary, self.path)

    def get_settings(self) -> dict[str, Any]:
        """
        Возвращает независимый снимок настроек.

        :return: Настройки и режим сервера.
        """
        with self.lock:
            return {"settings": dict(self.settings), "mode": self.mode}

    def update_settings(self, values: dict[str, Any]) -> dict[str, Any]:
        """
        Проверяет и сохраняет новые примененные настройки.

        :param values: Изменяемые параметры.
        :return: Новые настройки и режим.
        """
        with self.lock:
            updated = validate_settings({**self.settings, **values})
            changed = any(updated[key] != self.settings[key] for key in FEATURE_KEYS)
            self.settings = updated
            self.save()
            if changed and self.cache is not None:
                self.cache.pause()
            return self.get_settings()

    def sync_dataset(self) -> bool:
        """
        Сбрасывает кеш при изменении исходных записей.

        :return: Признак изменения датасета.
        """
        with self.lock:
            stamp = dataset_stamp(self.dataset)
            if stamp == self.stamp:
                return False
            self.stamp = stamp
            if self.cache is not None:
                self.cache.clear()
            else:
                self._clear_inactive_cache()
            self.save()
            return True

    def _clear_inactive_cache(self) -> None:
        """Удаляет кеш прежнего режима после изменения датасета без запуска worker."""
        for path in (self.directory / "cache").glob("*.cache"):
            path.unlink(missing_ok=True)

    def key(self, kind: str, path: Path, options: dict[str, Any]) -> str:
        """
        Создает ключ только по параметрам, влияющим на результат.

        :param kind: Тип сохраняемого результата.
        :param path: Исходная запись.
        :param options: Проверенные параметры.
        :return: Однозначный ключ результата.
        """
        keys = list(FEATURE_KEYS)
        if kind in ("preprocess", "audio"):
            keys = ["denoise_strength", "denoise_n_fft", "trim_silence"]
        elif kind == "features":
            keys.remove("n_mfcc")
            keys.remove("n_mels")
        elif kind == "mel":
            keys.remove("n_mfcc")
        return json.dumps(
            [
                "v1",
                self.stamp,
                str(path.relative_to(self.dataset)),
                kind,
                {key: options[key] for key in keys},
            ],
            sort_keys=True,
        )

    def compute(self, kind: str, path: Path, options: dict[str, Any]) -> bytes:
        """
        Создает числовые признаки, WAV или готовый PNG.

        :param kind: Тип результата.
        :param path: Исходная запись.
        :param options: Проверенные параметры.
        :return: Байты результата.
        """
        samples = load_wav(path)
        output = io.BytesIO()
        if kind == "vector":
            np.save(
                output,
                extract_features_with_settings(samples, options),
                allow_pickle=False,
            )
            return output.getvalue()
        if kind in ("preprocess", "audio"):
            processed = suppress_noise(
                samples, options["denoise_strength"], options["denoise_n_fft"]
            )
            if options["trim_silence"]:
                processed = trim_silence(processed)
            if kind == "audio":
                return encode_wav(processed)
        with self.plot_lock:
            if kind == "preprocess":
                figure = make_preprocessing_figure(
                    samples, processed, options["denoise_n_fft"]
                )
            elif kind == "features":
                figure = make_feature_signal_figure(
                    samples, options, f"{path.parent.name}/{path.name}"
                )
            elif kind == "mel":
                figure = make_mel_figure(
                    (samples,), options, (f"{path.parent.name}/{path.name}",)
                )
            else:
                raise ValueError("Неизвестный тип результата.")
            try:
                figure.savefig(output, format="png", dpi=115)
            finally:
                plt.close(figure)
        return output.getvalue()

    def result(self, kind: str, path: Path, options: dict[str, Any]) -> bytes:
        """
        Получает результат по запросу с учетом режима сервера.

        :param kind: Тип результата.
        :param path: Исходная запись.
        :param options: Проверенные параметры.
        :return: Байты готового результата.
        """
        if self.cache is None:
            return self.compute(kind, path, options)
        return self.cache.get_or_compute(
            self.key(kind, path, options), lambda: self.compute(kind, path, options)
        )

    def vector(self, path: Path, options: dict[str, Any]) -> np.ndarray:
        """
        Получает признаки для обучения из того же кеша, что и предрасчет.

        :param path: Исходная запись.
        :param options: Параметры признаков.
        :return: Вектор признаков.
        """
        return np.load(
            io.BytesIO(self.result("vector", path, options)), allow_pickle=False
        )

    def tasks(
        self, page: str, selected: Path, base: dict[str, Any]
    ) -> Iterator[tuple[str, Callable[[], bytes]]]:
        """
        Перебирает варианты выбранной записи, затем остальных записей.

        :param page: Этап предрасчета.
        :param selected: Запись с приоритетом.
        :param base: Снимок примененных настроек.
        :return: Ленивые задания расчета.
        """
        paths = [selected] + [
            path
            for label in LABELS
            for path in sorted((self.dataset / label).glob("*.wav"))
            if path != selected
        ]
        seen: set[str] = set()
        scales = CONFIG["feature_scales"]
        strength_minimum, strength_maximum = CONFIG["limits"]["denoise_strength"]
        strength_step = CONFIG["preprocessing"]["strength_step"]
        strength_count = (
            int((strength_maximum - strength_minimum) / strength_step + 1e-9) + 1
        )
        for path in paths:
            variants = [base]
            if page == "preprocess":
                variants.extend(
                    {
                        **base,
                        "denoise_strength": round(
                            strength_minimum + strength * strength_step, 12
                        ),
                        "denoise_n_fft": fft,
                        "trim_silence": trim,
                    }
                    for fft in scales["n_fft"]
                    for strength in range(strength_count)
                    for trim in (base["trim_silence"], not base["trim_silence"])
                )
            else:
                variants.extend(
                    {
                        **base,
                        "n_fft": fft,
                        "hop_length": hop,
                        "n_mels": mel,
                        "n_mfcc": mfcc,
                    }
                    for fft in scales["n_fft"]
                    for hop in scales["hop_length"]
                    if hop <= fft
                    for mel in scales["n_mels"]
                    for mfcc in scales["n_mfcc"]
                    if mfcc <= mel
                )
            for options in variants:
                kinds = (
                    ("preprocess", "audio")
                    if page == "preprocess"
                    else ("features", "mel", "vector")
                )
                for kind in kinds:
                    key = self.key(kind, path, options)
                    if key not in seen:
                        seen.add(key)
                        yield (
                            key,
                            lambda kind=kind, path=path, options=options: self.compute(
                                kind, path, options
                            ),
                        )

    def warm(self, page: str, selected: Path) -> None:
        """
        Запускает предрасчет при входе на страницу.

        :param page: Имя этапа.
        :param selected: Первая запись.
        """
        if page not in ("preprocess", "features"):
            raise ValueError("Предрасчет недоступен для этого этапа.")
        if not selected.is_file():
            raise FileNotFoundError(selected)
        with self.lock:
            if self.cache is not None:
                self.cache.start(self.tasks(page, selected, dict(self.settings)))

    def clear_cache(self) -> None:
        """Останавливает предрасчет и удаляет кеш до следующего входа на этап."""
        with self.lock:
            if self.cache is not None:
                self.cache.clear()

    def cache_status(self) -> dict[str, Any]:
        """
        Возвращает состояние кеша для интерфейса.

        :return: Размер, лимит и состояние фоновой работы.
        """
        if self.cache is None:
            return {
                "enabled": False,
                "bytes": 0,
                "limit": CONFIG["cache_max_bytes"],
                "state": "idle",
                "completed": 0,
                "total": 0,
            }
        state = self.cache.status()
        return {
            **state,
            "enabled": True,
            "bytes": state["size"],
            "state": "error"
            if state.get("error")
            else "full"
            if state["paused"] and state["pending"]
            else "paused"
            if state["paused"]
            else "running"
            if state["pending"]
            else "idle",
            "completed": state.get("count", 0),
            "total": 0,
        }
