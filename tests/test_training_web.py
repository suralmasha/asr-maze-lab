"""Интеграционные проверки обучения и новых аудиоэкспериментов."""

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
import zipfile
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np

import lab_server
from experiment_settings import validate_settings
from features import (
    load_wav,
    representation,
    representation_with_settings,
    suppress_noise,
)
from lab_config import CONFIG
from trainer import (
    model_features,
    portable_predict,
    save_classifier,
    train_with_settings,
)
from training_web import TrainingManager


def tone(freq: float) -> bytes:
    """
    Создает проверочный WAV с синусоидой.

    :param freq: Частота синусоиды в герцах.
    :return: Байты WAV длительностью 1 секунда.
    """
    times = np.arange(16000, dtype=float) / 16000
    samples = (
        0.28 * np.sin(2 * np.pi * freq * times) * np.hanning(16000) * 32767
    ).astype("<i2")
    output = io.BytesIO()
    with wave.open(output, "wb") as recording:
        recording.setframerate(16000)
        recording.setnchannels(1)
        recording.setsampwidth(2)
        recording.writeframes(samples.tobytes())
    return output.getvalue()


class TestWebTraining(unittest.TestCase):
    """Проверяет HTTP-контракт на отдельном временном датасете."""

    def test_browser_configuration(self) -> None:
        """Передает браузеру общий конфиг и загрузчик элементов управления."""
        status, script = self.request("/lab-config.js")
        self.assertEqual(status, 200)
        payload = script.decode().removeprefix("const labConfig = ").strip()
        self.assertEqual(json.loads(payload.removesuffix(";")), CONFIG)
        status, script = self.request("/configure.js")
        self.assertEqual(status, 200)
        self.assertIn(b"labConfig.defaults", script)

    def setUp(self) -> None:
        """Запускает изолированный сервер на свободном порту."""
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.p_data = patch.object(lab_server, "DATA", root / "dataset")
        self.p_model = patch.object(
            lab_server, "MODEL", root / "models" / "keyword_classifier.npz"
        )
        self.p_training = patch.object(lab_server, "training", TrainingManager())
        for replacement in (self.p_data, self.p_model, self.p_training):
            replacement.start()
        lab_server._classifier = None
        lab_server._classifier_stamp = None
        for label in ("up", "down", "left", "right"):
            (lab_server.DATA / label).mkdir(parents=True)
        self.http = ThreadingHTTPServer(("127.0.0.1", 0), lab_server.Handler)
        self.base = f"http://127.0.0.1:{self.http.server_port}"
        self.thread = threading.Thread(target=self.http.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        """Останавливает сервер и удаляет только проверочные данные."""
        self.http.shutdown()
        self.http.server_close()
        self.thread.join()
        for replacement in (self.p_training, self.p_model, self.p_data):
            replacement.stop()
        self.temp.cleanup()

    def request(
        self, path: str, body: bytes | None = None, mime: str = "audio/wav"
    ) -> tuple[int, bytes]:
        """
        Выполняет HTTP-запрос к проверочному серверу.

        :param path: Адрес API или статического файла.
        :param body: Тело POST или None для GET.
        :param mime: Тип тела запроса.
        :return: HTTP-статус и байты ответа.
        """
        request = urllib.request.Request(
            self.base + path,
            data=body,
            headers={"Content-Type": mime} if body is not None else {},
            method="POST" if body is not None else "GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read()

    def populate(self, count: int) -> None:
        """
        Заполняет временный датасет проверочными WAV.

        :param count: Количество записей каждой команды.
        """
        for label, frequency in (
            ("up", 220),
            ("down", 380),
            ("left", 510),
            ("right", 730),
        ):
            for index in range(count):
                (lab_server.DATA / label / f"{index + 1:02d}.wav").write_bytes(
                    tone(frequency + index * 11)
                )

    def wait_training(self) -> dict[str, Any]:
        """
        Ожидает завершения реального фонового обучения.

        :return: Конечное состояние задачи.
        """
        deadline = time.monotonic() + 30
        while True:
            status, raw = self.request("/api/train/status")
            data = json.loads(raw)
            self.assertEqual(status, 200)
            if data["state"] != "running":
                self.assertEqual(data["state"], "done", data)
                return data
            self.assertLess(time.monotonic(), deadline, data)
            time.sleep(0.05)

    def test_train_and_explain(self) -> None:
        """Сохраняет прежний HTTP-сценарий и исходную версию признаков."""
        code, message = self.request("/api/train/start", b"", "application/json")
        self.assertEqual(code, 400)
        self.assertIn("5 записей", json.loads(message)["error"])
        for label, frequency in (
            ("up", 220),
            ("down", 380),
            ("left", 510),
            ("right", 730),
        ):
            for index in range(5):
                status, data = self.request(
                    f"/api/samples?label={label}", tone(frequency + index * 11)
                )
                self.assertEqual(status, 201, data)
        status, started = self.request("/api/train/start", b"", "application/json")
        self.assertEqual(status, 202, started)
        self.assertEqual(json.loads(started)["state"], "running")
        data = self.wait_training()
        self.assertEqual(data["percent"], 100)
        self.assertTrue(data["ready"])
        with np.load(lab_server.MODEL, allow_pickle=False) as values:
            self.assertEqual(int(values["feature_version"]), 1)
            self.assertEqual(values["train_names"][0], "up/01.wav")
        for endpoint in ("/api/train/plot", "/api/plot?label=up&file=01.wav"):
            status, image = self.request(endpoint)
            self.assertEqual(status, 200)
            self.assertTrue(image.startswith(b"\x89PNG"))
        status, result = self.request("/api/train/probe", tone(220))
        self.assertEqual(status, 200)
        data = json.loads(result)
        self.assertEqual(data["label"], "up")
        self.assertEqual(len(data["neighbors"]), 3)
        self.assertEqual(data["neighbors"][0]["file"], "up/01.wav")
        code, html = self.request("/train.html")
        self.assertEqual(code, 200)
        self.assertIn(b"train.js", html)
        code, script = self.request("/train.js")
        self.assertEqual(code, 200)
        self.assertIn(b"/api/train/start", script)
        status, _ = self.request("/api/train/start", b"", "application/json")
        self.assertEqual(status, 202)
        self.wait_training()

    def test_experiment_settings_and_audio_views(self) -> None:
        """Проверяет параметры, сохранение модели, PCA и новые графики через HTTP."""
        self.populate(7)
        settings = {
            "n_mels": 8,
            "n_mfcc": 7,
            "n_fft": 256,
            "hop_length": 128,
            "denoise_strength": 1.2,
            "denoise_n_fft": 512,
            "normalize": False,
            "metric": "manhattan",
            "weights": "uniform",
            "k": 15,
            "samples_per_class": 5,
            "trim_silence": False,
            "noise": 0.05,
        }
        status, started = self.request(
            "/api/train/start", json.dumps(settings).encode(), "application/json"
        )
        self.assertEqual(status, 202, started)
        completed = self.wait_training()
        self.assertEqual(completed["training_count"], 40)
        self.assertEqual(completed["source_count"], 20)
        self.assertEqual(completed["model_settings"], settings)
        with np.load(lab_server.MODEL, allow_pickle=False) as saved:
            self.assertEqual(int(saved["feature_version"]), 2)
            self.assertEqual(saved["train_X"].shape, (40, 14))
            np.testing.assert_array_equal(saved["mean"], np.zeros(14))
            np.testing.assert_array_equal(saved["scale"], np.ones(14))
            self.assertEqual(int(saved["train_augmented"].sum()), 20)
            for index, name in enumerate(saved["train_names"][::2]):
                np.testing.assert_allclose(
                    saved["train_X"][2 * index],
                    model_features(dict(saved), load_wav(lab_server.DATA / str(name))),
                    rtol=1e-6,
                    atol=1e-6,
                )
        status, raw = self.request("/api/train/probe", tone(220))
        self.assertEqual(status, 200, raw)
        self.assertEqual(len(json.loads(raw)["neighbors"]), 15)
        predicted = json.loads(raw)["label"]
        status, raw = self.request("/api/predict", tone(220))
        self.assertEqual(status, 200, raw)
        self.assertEqual(json.loads(raw)["label"], predicted)
        status, raw = self.request("/api/train/projection")
        self.assertEqual(status, 200, raw)
        projection = json.loads(raw)
        self.assertEqual(len(projection["points"]), 40)
        self.assertEqual(projection["points"][0]["file"], "up/01.wav")
        self.assertIsNotNone(projection["probe"])
        for endpoint in (
            "/api/plot?label=up&file=01.wav&n_mels=10&n_fft=2048",
            "/api/plot?label=up&file=01.wav&view=signal&trim_silence=false",
            "/api/plot?label=up&file=01.wav&view=features&hop_length=64&denoise_strength=1.2",
            "/api/plot?label=up&file=01.wav&view=mel&n_mfcc=4&n_mels=8&hop_length=256",
            "/api/compare?label=up&file=01.wav&label2=down&file2=02.wav&view=signal",
            "/api/compare?label=up&file=01.wav&label2=down&file2=02.wav&n_mels=128&n_fft=256",
            "/api/preprocess?label=up&file=01.wav&strength=1.2&n_fft=1024",
        ):
            status, image = self.request(endpoint)
            self.assertEqual(status, 200, image[:200])
            self.assertTrue(image.startswith(b"\x89PNG"))
        status, audio = self.request(
            "/api/preprocess/audio?label=up&file=01.wav&strength=1&n_fft=1024"
        )
        self.assertEqual(status, 200, audio[:200])
        with wave.open(io.BytesIO(audio)) as processed:
            self.assertEqual(processed.getnframes(), 16000)
        status, archive = self.request("/api/dataset.zip")
        self.assertEqual(status, 200)
        with zipfile.ZipFile(io.BytesIO(archive)) as dataset:
            self.assertIn("experiment_settings.py", dataset.namelist())

    def test_recording_limit_and_default_dataset_size(self) -> None:
        """Блокирует запись сверх 50 и использует все имеющиеся WAV до лимита."""
        self.populate(51)
        status, data = self.request("/api/samples?label=up", tone(220))
        self.assertEqual(status, 400, data)
        self.assertEqual(len(list((lab_server.DATA / "up").glob("*.wav"))), 51)
        status, started = self.request("/api/train/start", b"{}", "application/json")
        self.assertEqual(status, 202, started)
        completed = self.wait_training()
        self.assertEqual(completed["source_count"], 200)
        with np.load(lab_server.MODEL, allow_pickle=False) as saved:
            self.assertEqual(saved["train_X"].shape[0], 200)
            self.assertNotIn("up/51.wav", saved["train_names"])

    def test_invalid_settings(self) -> None:
        """Отклоняет параметры вне диапазонов до начала обучения."""
        for options in (
            {"n_mels": 7},
            {"n_mels": 129},
            {"n_fft": 700},
            {"hop_length": 0},
            {"n_fft": 256, "hop_length": 512},
            {"denoise_strength": "nan"},
            {"denoise_strength": 2.1},
            {"denoise_n_fft": 700},
            {"n_mfcc": 41},
            {"k": 16},
            {"samples_per_class": 4},
            {"noise": "nan"},
            {"metric": "cosine"},
        ):
            status, response = self.request(
                "/api/train/start", json.dumps(options).encode(), "application/json"
            )
            self.assertEqual(status, 400, response)
        (lab_server.DATA / "up" / "01.wav").write_bytes(tone(220))
        status, _ = self.request(
            "/api/preprocess/audio?label=up&file=01.wav&strength=nan"
        )
        self.assertEqual(status, 400)


class TestFeatureExperiments(unittest.TestCase):
    """Проверяет реальные числовые изменения новых параметров."""

    def test_window_mel_and_denoising(self) -> None:
        """Меняет разрешение STFT и число мел-фильтров и подавляет фоновый шум."""
        samples = load_wav(tone(350))
        short = representation_with_settings(samples, {"n_fft": 256, "n_mels": 8})
        long = representation_with_settings(samples, {"n_fft": 2048, "n_mels": 128})
        self.assertEqual(short["mel_power"].shape[0], 8)
        self.assertEqual(long["mel_power"].shape[0], 128)
        self.assertGreater(short["power"].shape[1], long["power"].shape[1])
        self.assertAlmostEqual(float(short["times"][1] - short["times"][0]), 64 / 16000)
        self.assertAlmostEqual(float(long["times"][1] - long["times"][0]), 512 / 16000)
        spaced = representation_with_settings(
            samples, {"n_fft": 256, "hop_length": 256, "n_mfcc": 4}
        )
        self.assertAlmostEqual(
            float(spaced["times"][1] - spaced["times"][0]), 256 / 16000
        )
        self.assertEqual(spaced["mfcc"].shape[0], 4)
        self.assertLess(spaced["power"].shape[1], short["power"].shape[1])
        legacy = representation(samples)
        self.assertAlmostEqual(
            float(legacy["times"][1] - legacy["times"][0]), 160 / 16000
        )
        random = np.random.default_rng(5)
        noisy = samples + random.normal(0, 0.025, samples.size)
        processed = suppress_noise(noisy, strength=1.5)
        self.assertEqual(processed.shape, samples.shape)
        self.assertLess(
            float(np.mean(processed[:1000] ** 2)), float(np.mean(noisy[:1000] ** 2))
        )
        np.testing.assert_array_equal(suppress_noise(samples, strength=0), samples)

    def test_portable_classifier_matches_settings(self) -> None:
        """Сравнивает сохраненное голосование с sklearn при разных настройках."""
        random = np.random.default_rng(4)
        matrix = random.normal(size=(20, 8)).astype(np.float32)
        labels = np.asarray(["up"] * 5 + ["down"] * 5 + ["left"] * 5 + ["right"] * 5)
        query = random.normal(size=8).astype(np.float32)
        with tempfile.TemporaryDirectory() as temporary:
            for normalize, metric, weights in (
                (True, "euclidean", "distance"),
                (False, "manhattan", "uniform"),
                (True, "manhattan", "distance"),
            ):
                settings = validate_settings(
                    {
                        "normalize": normalize,
                        "metric": metric,
                        "weights": weights,
                        "k": 5,
                        "n_mfcc": 4,
                    }
                )
                classifier = train_with_settings(matrix, labels, settings)
                output = save_classifier(classifier, Path(temporary), matrix, labels)
                with np.load(output, allow_pickle=False) as saved:
                    model = dict(saved)
                predicted, _ = portable_predict(model, query)
                self.assertEqual(predicted, classifier.predict(query.reshape(1, -1))[0])


if __name__ == "__main__":
    unittest.main()
