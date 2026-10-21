"""Проверки истории моделей, повторного обучения и отмены устаревших задач."""

from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiment_settings import validate_settings
from features import LABELS
from model_history import ModelHistory
from training_web import TrainingManager


class TestModelHistory(unittest.TestCase):
    """Проверяет сохранение независимых экспериментов на диске."""

    def setUp(self) -> None:
        """Создает изолированный датасет и историю."""
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.dataset = self.root / "dataset"
        self.active = self.root / "active.npz"
        for label in LABELS:
            (self.dataset / label).mkdir(parents=True)
            for index in range(5):
                (self.dataset / label / f"{index}.wav").write_bytes(b"wav")
        self.history = ModelHistory(self.root / "history", self.active, self.dataset)
        self.settings = validate_settings()
        self.source = self.root / "source.npz"
        np.savez(
            self.source, train_y=np.array(["up"]), train_names=np.array(["up/0.wav"])
        )

    def tearDown(self) -> None:
        """Удаляет временные файлы."""
        self.temporary.cleanup()

    def wait_training(self, manager: TrainingManager) -> dict:
        """
        Ожидает конечного состояния обучения.

        :param manager: Менеджер фоновой задачи.
        :return: Конечное состояние задачи.
        """
        deadline = time.monotonic() + 10
        while manager.status()["state"] == "running":
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.01)
        return manager.status()

    def test_restart_select_and_delete(self) -> None:
        """Сохраняет модели между открытиями и удаляет активную копию."""
        identifier, generation = self.history.experiment(self.settings)
        self.assertTrue(
            self.history.save(identifier, generation, self.source, self.settings)
        )
        history = ModelHistory(self.history.root, self.active, self.dataset)
        self.assertEqual(history.list_models()[0]["settings"], self.settings)
        self.active.unlink()
        self.assertTrue(history.select(identifier)["active"])
        self.assertEqual(self.active.read_bytes(), self.source.read_bytes())
        history.delete(identifier)
        self.assertEqual(history.list_models(), [])
        self.assertFalse(self.active.exists())

    def test_dataset_content_invalidates_history_and_worker(self) -> None:
        """Меняет содержимое WAV и отклоняет сохранение прежнего поколения."""
        identifier, generation = self.history.experiment(self.settings)
        self.history.save(identifier, generation, self.source, self.settings)
        (self.dataset / "up" / "0.wav").write_bytes(b"new")
        self.assertFalse(
            self.history.save(identifier, generation, self.source, self.settings)
        )
        self.assertEqual(self.history.list_models(), [])
        self.assertFalse(self.active.exists())

    def test_clear_cancels_pending_save(self) -> None:
        """Ручная очистка запрещает повторное появление текущего результата."""
        identifier, generation = self.history.experiment(self.settings)
        self.history.clear()
        self.assertFalse(
            self.history.save(identifier, generation, self.source, self.settings)
        )

    def test_identical_experiment_reuses_model(self) -> None:
        """Повторно активирует модель без запуска вычисления признаков."""
        manager = TrainingManager()
        manager.configure_history(self.history)
        with (
            patch("training_web.load_wav", return_value=np.ones(160)),
            patch(
                "training_web.extract_features_with_settings",
                return_value=np.arange(26, dtype=float),
            ),
        ):
            manager.start_experiment(self.dataset, self.active, self.settings)
            self.assertEqual(self.wait_training(manager)["state"], "done")
        with patch(
            "training_web.extract_features_with_settings",
            side_effect=AssertionError("recomputed"),
        ):
            status = manager.start_experiment(self.dataset, self.active, self.settings)
        self.assertTrue(status["reused"])
        self.assertEqual(len(self.history.list_models()), 1)

    def test_invalidation_does_not_overwrite_new_state(self) -> None:
        """Отмененный worker не сохраняет модель и не меняет новое состояние."""
        manager = TrainingManager()
        entered, release = threading.Event(), threading.Event()

        def blocked_features(path: Path, settings: dict) -> np.ndarray:
            """
            Задерживает вычисление до отмены задачи.

            :param path: Путь исходного WAV.
            :param settings: Настройки вычисления.
            :return: Проверочный вектор.
            """
            entered.set()
            release.wait(5)
            return np.arange(26, dtype=float)

        manager.configure_features(blocked_features)
        with patch("training_web.load_wav", return_value=np.ones(160)):
            manager.start_experiment(self.dataset, self.active, self.settings)
            self.assertTrue(entered.wait(5))
            manager.invalidate()
            manager.change(message="Новое состояние")
            release.set()
            time.sleep(0.05)
        self.assertEqual(manager.status()["message"], "Новое состояние")
        self.assertFalse(self.active.exists())


if __name__ == "__main__":
    unittest.main()
