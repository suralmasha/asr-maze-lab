"""Проверки общего конфигурационного файла лаборатории."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from lab_config import CONFIG, load_config


class TestLabConfig(unittest.TestCase):
    """Проверяет загрузку и согласованность параметров."""

    def load(self, config: dict[str, Any]) -> dict[str, Any]:
        """
        Загружает конфиг из временного файла.

        :param config: Проверяемая конфигурация.
        :return: Загруженная конфигурация.
        """
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lab.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            return load_config(path)

    def test_custom_limits_and_scales(self) -> None:
        """Принимает согласованные значения вне исходных настроек."""
        config = copy.deepcopy(CONFIG)
        config["cache_max_bytes"] = 1234567
        config["audio"].update(min_samples=3, max_samples=7)
        config["feature_scales"].update(n_mels=[16, 64], n_mfcc=[8, 20])
        config["defaults"].update(n_mels=16, n_mfcc=8)
        self.assertEqual(self.load(config), config)

    def test_rejects_inconsistent_configuration(self) -> None:
        """Не допускает расхождения связанных ограничений и шкал."""
        invalid = (
            ("audio", "min_samples", CONFIG["audio"]["max_samples"] + 1),
            ("audio", "max_bytes", 100),
            ("feature_scales", "n_fft", [256, 500, 512]),
            ("feature_scales", "n_mfcc", [4, 8]),
            ("limits", "noise", [0.01, 0.1]),
            ("preprocessing", "strength_step", 0.3),
            ("defaults", "noise", float("nan")),
        )
        for section, key, value in invalid:
            with self.subTest(section=section, key=key):
                config = copy.deepcopy(CONFIG)
                config[section][key] = value
                with self.assertRaisesRegex(ValueError, "ASR-CFG-02"):
                    self.load(config)

    def test_missing_file_has_configuration_error(self) -> None:
        """Объясняет отсутствие конфигурации при запуске."""
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "ASR-CFG-02"):
                load_config(Path(directory) / "missing.json")
