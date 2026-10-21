"""Дисковая история экспериментов с контролем версии датасета."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from features import LABELS


class ModelHistory:
    """Хранит независимые модели вне кеша графиков."""

    def __init__(self, root: Path, active_model: Path, dataset: Path) -> None:
        """
        Открывает историю моделей.

        :param root: Каталог сохраненных экспериментов.
        :param active_model: Путь модели для распознавания.
        :param dataset: Каталог исходных WAV.
        """
        self.root = root
        self.active_model = active_model
        self.dataset = dataset
        self.lock = threading.RLock()
        self.generation = 0
        root.mkdir(parents=True, exist_ok=True)
        self.sync_dataset()

    def fingerprint(self) -> str:
        """
        Считает отпечаток имен и содержимого всего датасета.

        :return: SHA-256 полного датасета.
        """
        digest = hashlib.sha256()
        for label in LABELS:
            for path in sorted((self.dataset / label).glob("*.wav")):
                name = f"{label}/{path.name}".encode()
                digest.update(len(name).to_bytes(8, "big"))
                digest.update(name)
                digest.update(hashlib.sha256(path.read_bytes()).digest())
        return digest.hexdigest()

    def sync_dataset(self) -> str:
        """
        Удаляет устаревшие модели при изменении записей.

        :return: Текущий отпечаток датасета.
        """
        with self.lock:
            fingerprint = self.fingerprint()
            stamp = self.root / "dataset.txt"
            if stamp.exists() and stamp.read_text() != fingerprint:
                self.clear()
                self.active_model.unlink(missing_ok=True)
            stamp.write_text(fingerprint)
            return fingerprint

    def experiment(
        self, settings: dict[str, Any], legacy: bool = False
    ) -> tuple[str, int]:
        """
        Создает ключ эксперимента и запоминает поколение истории.

        :param settings: Полные настройки эксперимента.
        :param legacy: Признак исходного алгоритма извлечения признаков.
        :return: Идентификатор и поколение для проверки сохранения.
        """
        with self.lock:
            fingerprint = self.sync_dataset()
            payload = json.dumps([fingerprint, settings, legacy], sort_keys=True)
            return hashlib.sha256(payload.encode()).hexdigest(), self.generation

    def list_models(self) -> list[dict[str, Any]]:
        """
        Возвращает сохраненные эксперименты от новых к старым.

        :return: Модели с настройками и признаком активной модели.
        """
        with self.lock:
            self.sync_dataset()
            marker = self.root / "active.txt"
            active = marker.read_text() if marker.exists() else ""
            models = []
            for path in self.root.glob("*.json"):
                if path.with_suffix(".npz").exists():
                    item = json.loads(path.read_text())
                    models.append({**item, "active": item["id"] == active})
            return sorted(models, key=lambda item: item["created"], reverse=True)

    def _metadata_path(self, identifier: str) -> Path:
        """
        Проверяет безопасный идентификатор модели.

        :param identifier: SHA-256 эксперимента.
        :return: Путь метаданных внутри истории.
        """
        if len(identifier) != 64 or any(
            char not in "0123456789abcdef" for char in identifier
        ):
            raise ValueError("ASR-DAT-01: Invalid model identifier (Hint: fix request)")
        return self.root / f"{identifier}.json"

    def _activate(self, identifier: str) -> None:
        """
        Атомарно копирует выбранную модель в рабочий путь.

        :param identifier: Идентификатор сохраненной модели.
        """
        self.active_model.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=self.active_model.parent) as temporary:
            output = Path(temporary) / "active.npz"
            shutil.copyfile(self.root / f"{identifier}.npz", output)
            os.replace(output, self.active_model)
        (self.root / "active.txt").write_text(identifier)

    def select(self, identifier: str) -> dict[str, Any]:
        """
        Активирует модель и возвращает настройки для интерфейса.

        :param identifier: Идентификатор эксперимента.
        :return: Метаданные выбранного эксперимента.
        """
        with self.lock:
            self.sync_dataset()
            metadata = json.loads(self._metadata_path(identifier).read_text())
            self._activate(identifier)
            return {**metadata, "active": True}

    def save(
        self, identifier: str, generation: int, source: Path, settings: dict[str, Any]
    ) -> bool:
        """
        Сохраняет результат только для неизменившегося датасета.

        :param identifier: Ключ эксперимента.
        :param generation: Поколение на момент начала обучения.
        :param source: Временная обученная модель.
        :param settings: Полные настройки эксперимента.
        :return: Успешность сохранения результата.
        """
        with self.lock:
            self.sync_dataset()
            if generation != self.generation:
                return False
            metadata = self._metadata_path(identifier)
            output = metadata.with_suffix(".npz")
            if not output.exists():
                with tempfile.TemporaryDirectory(dir=self.root) as temporary:
                    prepared = Path(temporary) / "model.npz"
                    shutil.copyfile(source, prepared)
                    os.replace(prepared, output)
                with np.load(source, allow_pickle=False) as model:
                    training_count = len(model["train_y"])
                    source_count = len(set(model["train_names"].tolist()))
                created = time.time()
                metadata.write_text(
                    json.dumps(
                        {
                            "id": identifier,
                            "created": created,
                            "created_at": created,
                            "settings": settings,
                            "training_count": training_count,
                            "source_count": source_count,
                        },
                        ensure_ascii=False,
                    )
                )
            self._activate(identifier)
            return True

    def delete(self, identifier: str) -> None:
        """
        Удаляет выбранный эксперимент и его активную копию.

        :param identifier: Идентификатор эксперимента.
        """
        with self.lock:
            metadata = self._metadata_path(identifier)
            marker = self.root / "active.txt"
            if marker.exists() and marker.read_text() == identifier:
                marker.unlink()
                self.active_model.unlink(missing_ok=True)
            metadata.unlink(missing_ok=True)
            metadata.with_suffix(".npz").unlink(missing_ok=True)

    def clear(self) -> None:
        """Удаляет все модели и отменяет сохранение текущего поколения."""
        with self.lock:
            self.generation += 1
            for path in self.root.glob("*.json"):
                self.delete(path.stem)
            (self.root / "active.txt").unlink(missing_ok=True)
            self.active_model.unlink(missing_ok=True)
