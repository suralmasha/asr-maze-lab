"""Проверки HTTP-сценариев кеша, настроек и истории экспериментов."""

from __future__ import annotations

import io
import json
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
import wave
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np

import features
import lab_server
from experiment_settings import validate_settings
from features import LABELS, encode_wav
from training_web import TrainingManager


class TestRuntimeWeb(unittest.TestCase):
    """Изолирует новый режим сервера от пользовательского датасета."""

    def setUp(self) -> None:
        """Запускает сервер предрасчета на временных файлах."""
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.dataset = root / "dataset"
        self.model = root / "models" / "keyword_classifier.npz"
        self.samples = np.concatenate(
            (np.zeros(5000), 0.3 * np.sin(np.arange(6000) * 0.1), np.zeros(5000))
        )
        self.audio = encode_wav(self.samples)
        for label in LABELS:
            (self.dataset / label).mkdir(parents=True)
            for index in range(5):
                (self.dataset / label / f"{index:02d}.wav").write_bytes(self.audio)
        self.replacements = patch.multiple(
            lab_server,
            DATA=self.dataset,
            MODEL=self.model,
            MODE="precompute",
            training=TrainingManager(),
            _runtime=None,
            _history=None,
            _classifier=None,
            _classifier_stamp=None,
        )
        self.replacements.start()
        self.http = ThreadingHTTPServer(("127.0.0.1", 0), lab_server.Handler)
        self.base = f"http://127.0.0.1:{self.http.server_port}"
        self.thread = threading.Thread(target=self.http.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        """Останавливает потоки и удаляет временный датасет."""
        self.http.shutdown()
        self.http.server_close()
        self.thread.join()
        if lab_server._runtime is not None and lab_server._runtime.cache is not None:
            lab_server._runtime.cache.stop()
        lab_server.training.invalidate()
        self.replacements.stop()
        self.temporary.cleanup()

    def request(
        self, path: str, body: bytes | dict[str, Any] | None = None
    ) -> tuple[int, bytes]:
        """
        Выполняет запрос с JSON либо WAV.

        :param path: Путь HTTP API.
        :param body: JSON-объект, WAV либо None для GET.
        :return: Статус и тело ответа.
        """
        mime = "application/json" if isinstance(body, dict) else "audio/wav"
        data = json.dumps(body).encode() if isinstance(body, dict) else body
        request = urllib.request.Request(
            self.base + path,
            data=data,
            headers={"Content-Type": mime} if data is not None else {},
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read()

    def json_request(
        self, path: str, body: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Читает успешный JSON-ответ.

        :param path: Путь API.
        :param body: Тело POST либо None для GET.
        :return: Данные JSON.
        """
        status, raw = self.request(path, body)
        self.assertIn(status, (200, 202), raw)
        return json.loads(raw)

    def train(self, settings: dict[str, Any]) -> dict[str, Any]:
        """
        Обучает модель через HTTP и ожидает результата.

        :param settings: Полные настройки эксперимента.
        :return: Конечное состояние обучения.
        """
        started = self.json_request("/api/train/start", settings)
        deadline = time.monotonic() + 20
        while True:
            state = self.json_request("/api/train/status")
            if state["state"] != "running":
                self.assertEqual(state["state"], "done", state)
                return {**started, **state}
            self.assertLess(time.monotonic(), deadline, state)
            time.sleep(0.03)

    def test_settings_restart_trim_and_cache_hit(self) -> None:
        """Сохраняет настройки, обрезает WAV и повторно использует кеш."""
        saved = self.json_request(
            "/api/settings", {"trim_silence": False, "n_mels": 64}
        )
        self.assertEqual(saved["settings"]["n_mels"], 64)
        self.assertEqual(saved["mode"], "precompute")
        lab_server._runtime.cache.stop()
        lab_server._runtime = None
        self.assertEqual(
            self.json_request("/api/settings")["settings"], saved["settings"]
        )
        runtime = lab_server.runtime()
        path = "/api/preprocess/audio?label=up&file=00.wav&strength=0&trim_silence=true"
        with patch.object(runtime, "compute", wraps=runtime.compute) as compute:
            status, trimmed = self.request(path)
            self.assertEqual(status, 200, trimmed)
            self.assertEqual(self.request(path), (200, trimmed))
            self.assertEqual(compute.call_count, 1)
        with wave.open(io.BytesIO(trimmed)) as recording:
            self.assertLess(recording.getnframes(), len(self.samples))
        status, original = self.request(path.replace("true", "false"))
        self.assertEqual(status, 200)
        with wave.open(io.BytesIO(original)) as recording:
            self.assertEqual(recording.getnframes(), len(self.samples))
        self.assertGreater(self.json_request("/api/cache/status")["bytes"], 0)
        self.assertEqual(self.json_request("/api/cache/clear", {})["bytes"], 0)

    def test_history_reuse_restore_and_dataset_change(self) -> None:
        """Возвращает готовую модель и сбрасывает результаты после изменения WAV."""
        settings = validate_settings({"n_mels": 16, "n_mfcc": 8, "trim_silence": False})
        self.train(settings)
        first = self.json_request("/api/models")["models"][0]
        with patch(
            "training_web.train_with_settings", side_effect=AssertionError("retrained")
        ):
            self.assertTrue(self.train(settings)["reused"])
        self.train({**settings, "k": 5})
        self.assertEqual(len(self.json_request("/api/models")["models"]), 2)
        selected = self.json_request("/api/models/select", {"id": first["id"]})
        self.assertEqual(selected["active_id"], first["id"])
        self.assertEqual(self.json_request("/api/settings")["settings"], settings)
        self.assertGreater(self.json_request("/api/cache/status")["bytes"], 0)
        status, raw = self.request("/api/samples?label=up", self.audio)
        self.assertEqual(status, 201, raw)
        self.assertEqual(self.json_request("/api/models")["models"], [])
        self.assertEqual(self.json_request("/api/cache/status")["bytes"], 0)
        self.assertFalse(self.model.exists())

    def test_recognition_can_disable_denoising(self) -> None:
        """Отключает шумоподавление новой команды без изменения модели."""
        settings = validate_settings({"denoise_strength": 1.2})
        self.train(settings)
        with patch("features.suppress_noise", wraps=features.suppress_noise) as denoise:
            status, raw = self.request("/api/predict?denoise=false", self.audio)
            self.assertEqual(status, 200, raw)
            self.assertEqual(denoise.call_args.args[1], 0)
        self.assertEqual(
            self.json_request("/api/settings")["settings"]["denoise_strength"], 1.2
        )
        self.assertEqual(
            self.request("/api/predict?denoise=invalid", self.audio)[0], 400
        )

    def test_invalid_model_identifiers(self) -> None:
        """Отклоняет некорректные пути и неизвестные модели."""
        for identifier in ("../outside", "", "z" * 64):
            self.assertEqual(
                self.request("/api/models/select", {"id": identifier})[0], 400
            )
            self.assertEqual(
                self.request("/api/models/delete", {"id": identifier})[0], 400
            )
        self.assertEqual(self.request("/api/models/select", {"id": "a" * 64})[0], 404)

    def test_background_prepares_requested_variant(self) -> None:
        """Готовит PNG в фоне и не кеширует ознакомительное сравнение."""
        runtime = lab_server.runtime()
        self.json_request(
            "/api/cache/warm",
            {"page": "preprocess", "label": "up", "file": "00.wav"},
        )
        deadline = time.monotonic() + 10
        while not runtime.cache_status()["completed"]:
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.01)
        runtime.cache.pause()
        with patch.object(runtime, "compute", side_effect=AssertionError("cache miss")):
            status, image = self.request(
                "/api/preprocess?label=up&file=00.wav&strength=0&trim_silence=true"
            )
            self.assertEqual(status, 200, image)
            self.assertTrue(image.startswith(b"\x89PNG"))
        before = runtime.cache_status()["bytes"]
        self.assertEqual(
            self.request(
                "/api/compare?label=up&file=00.wav&label2=down&file2=00.wav&view=signal"
            )[0],
            200,
        )
        self.assertEqual(runtime.cache_status()["bytes"], before)
        self.json_request("/api/cache/clear", {})
        self.assertEqual(runtime.cache_status()["bytes"], 0)


if __name__ == "__main__":
    unittest.main()
