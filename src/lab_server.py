"""Локальный сервер лаборатории сбора аудио, обучения и голосового лабиринта."""

from __future__ import annotations

import argparse
import io
import json
import logging
import re
import threading
import zipfile
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from experiment_runtime import ExperimentRuntime
from experiment_settings import DEFAULT_SETTINGS, MAX_SAMPLES, validate_settings
from features import (
    LABELS,
    load_wav,
    make_comparison_figure,
    make_configured_figure,
    make_feature_signal_figure,
    make_figure,
    make_mel_figure,
    make_signal_figure,
)
from lab_config import CONFIG
from model_history import ModelHistory
from trainer import model_features, model_settings, portable_predict
from training_web import TrainingManager, explain, projection_data, projection_png

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset"
MODEL = ROOT / "models" / "keyword_classifier.npz"
HOST, PORT = "127.0.0.1", 8000
MAX_BYTES = CONFIG["audio"]["max_bytes"]
ALLOWED_FILES = {
    "/",
    "/index.html",
    "/style.css",
    "/app.js",
    "/collect.html",
    "/collect.js",
    "/visualize.html",
    "/visualize.js",
    "/audio-settings.js",
    "/configure.js",
    "/sound.html",
    "/sound.js",
    "/workshop.css",
    "/recorder.js",
    "/train.html",
    "/train.js",
    "/compare.html",
    "/compare.js",
    "/preprocess.html",
    "/preprocess.js",
}
_lock = threading.Lock()
training = TrainingManager()
_classifier = None
_classifier_stamp = None
logger = logging.getLogger(__name__)
MODE = "on-demand"
_runtime: ExperimentRuntime | None = None
_history: ModelHistory | None = None
_runtime_lock = threading.RLock()


def runtime() -> ExperimentRuntime:
    """
    Получает состояние текущего сервера с поддержкой изолированных тестов.

    :return: Настройки и кеш лаборатории.
    """
    global _runtime, _history
    with _runtime_lock:
        if _runtime is None or (
            _runtime.dataset,
            _runtime.directory,
            _runtime.mode,
        ) != (DATA, MODEL.parent, MODE):
            if _runtime is not None and _runtime.cache is not None:
                _runtime.cache.stop()
            _runtime = ExperimentRuntime(DATA, MODEL.parent, MODE, _lock)
            _history = (
                ModelHistory(MODEL.parent / "history", MODEL, DATA)
                if MODE == "precompute"
                else None
            )
            if MODE == "on-demand":
                (MODEL.parent / "history" / "active.txt").unlink(missing_ok=True)
        training.configure_history(_history)
        training.configure_features(
            _runtime.vector if _runtime.cache is not None else None
        )
        if _runtime.sync_dataset():
            training.invalidate()
            if _history is not None:
                _history.sync_dataset()
        return _runtime


def models_response() -> dict[str, Any]:
    """
    Собирает список моделей для интерфейса.

    :return: Доступность истории, модели и активный идентификатор.
    """
    runtime()
    models = _history.list_models() if _history is not None else []
    return {
        "enabled": _history is not None,
        "models": models,
        "active_id": next((item["id"] for item in models if item["active"]), None),
    }


def model_loaded() -> dict[str, Any] | None:
    """
    Обновляет модель после обучения с поддержкой исходного формата NPZ.

    :return: Числовые данные проверенной модели или None.
    """
    global _classifier, _classifier_stamp
    if not MODEL.is_file():
        _classifier, _classifier_stamp = None, None
        return None
    stamp = MODEL.stat().st_mtime_ns
    if stamp == _classifier_stamp:
        return _classifier
    try:
        with np.load(MODEL, allow_pickle=False) as saved:
            bundle = {key: saved[key] for key in saved.files}
        version = int(bundle["feature_version"])
        if version not in (1, 2):
            raise ValueError(
                "Версия признаков не поддерживается. Требуется повторное обучение модели."
            )
        expected = 26 if version == 1 else 2 * model_settings(bundle)["n_mfcc"]
        matrix = bundle["train_X"]
        if matrix.ndim != 2 or matrix.shape[1] != expected or not len(matrix):
            raise ValueError("Неверная форма обучающих признаков.")
        if bundle["mean"].shape != (expected,) or bundle["scale"].shape != (expected,):
            raise ValueError("Неверная форма параметров нормализации.")
        if np.any(bundle["scale"] <= 0) or not all(
            np.isfinite(bundle[key]).all() for key in ("train_X", "mean", "scale")
        ):
            raise ValueError("Некорректные значения обучающих признаков.")
        if len(bundle["train_y"]) != len(matrix) or not set(bundle["train_y"]).issubset(
            LABELS
        ):
            raise ValueError("Некорректные классы обучающих записей.")
        if not 1 <= int(bundle["k"]) <= min(CONFIG["limits"]["k"][1], len(matrix)):
            raise ValueError("Некорректное количество ближайших соседей.")
        for key in ("train_names", "train_augmented"):
            if key in bundle and len(bundle[key]) != len(matrix):
                raise ValueError("Неверное количество имен обучающих записей.")
        _classifier, _classifier_stamp = bundle, stamp
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.exception(
            "Failed to load trained model",
            extra={"path": str(MODEL), "error": str(exc)},
        )
        _classifier, _classifier_stamp = None, None
    return _classifier


def validate_label(value: str) -> str:
    """
    Проверяет название категории.

    :param value: Название категории.
    :return: Проверенное название.
    """
    if value not in LABELS:
        raise ValueError("Неизвестная команда.")
    return value


def validate_file(value: str) -> str:
    """
    Проверяет безопасное имя WAV.

    :param value: Имя файла.
    :return: Проверенное имя.
    """
    if not re.fullmatch(r"[0-9]{2,6}\.wav", value):
        raise ValueError("Неверное имя файла.")
    return value


def sample_path(params: dict[str, list[str]]) -> Path:
    """
    Собирает путь записи из проверенных параметров запроса.

    :param params: Параметры label и file.
    :return: Путь WAV внутри датасета.
    """
    label = validate_label(params.get("label", [""])[0])
    name = validate_file(params.get("file", [""])[0])
    return DATA / label / name


class Handler(SimpleHTTPRequestHandler):
    """Обрабатывает локальные страницы и API лаборатории."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Создает обработчик с фиксированным корнем статических файлов.

        :param args: Позиционные аргументы HTTP-сервера.
        :param kwargs: Именованные аргументы HTTP-сервера.
        """
        super().__init__(*args, directory=str(ROOT / "web"), **kwargs)

    def json_response(self, status: int, value: Any) -> None:
        """
        Отправляет ответ JSON.

        :param status: HTTP-статус.
        :param value: Сериализуемые данные.
        """
        self.bytes_response(
            status,
            json.dumps(value, ensure_ascii=False).encode(),
            "application/json; charset=utf-8",
        )

    def bytes_response(
        self, status: int, data: bytes, content_type: str, filename: str | None = None
    ) -> None:
        """
        Отправляет двоичный ответ без кеширования.

        :param status: HTTP-статус.
        :param data: Байты ответа.
        :param content_type: MIME-тип ответа.
        :param filename: Имя загружаемого файла при необходимости.
        """
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        if filename:
            self.send_header(
                "Content-Disposition", f'attachment; filename="{filename}"'
            )
        self.end_headers()
        self.wfile.write(data)

    def params(self) -> dict[str, list[str]]:
        """
        Читает строку параметров текущего запроса.

        :return: Значения query-параметров.
        """
        return parse_qs(urlsplit(self.path).query)

    def figure_response(self, figure: Any) -> None:
        """
        Отправляет и закрывает созданную под блокировкой фигуру.

        :param figure: Фигура matplotlib.
        """
        output = io.BytesIO()
        try:
            figure.savefig(output, format="png", dpi=115)
        finally:
            plt.close(figure)
        self.bytes_response(200, output.getvalue(), "image/png")

    def do_GET(self) -> None:
        """Обрабатывает чтение страниц, графиков и состояния обучения."""
        path = urlsplit(self.path).path
        try:
            if path == "/lab-config.js":
                script = (
                    "const labConfig = "
                    + json.dumps(CONFIG, ensure_ascii=False)
                    + ";\n"
                )
                return self.bytes_response(
                    200, script.encode(), "application/javascript; charset=utf-8"
                )
            state = runtime()
            if path == "/api/settings":
                return self.json_response(200, state.get_settings())
            if path == "/api/cache/status":
                return self.json_response(200, state.cache_status())
            if path == "/api/models":
                return self.json_response(200, models_response())
            if path == "/api/health":
                model = model_loaded()
                return self.json_response(
                    200,
                    {
                        "ready": model is not None,
                        "message": "Твоя модель готова к игре."
                        if model is not None
                        else "Сначала обучи свою модель на этапе «Обучение», а потом возвращайся играть.",
                    },
                )
            if path == "/api/train/status":
                model = model_loaded()
                details = (
                    {
                        "model_settings": model_settings(model),
                        "training_count": len(model["train_X"]),
                        "source_count": int(
                            (
                                ~model.get(
                                    "train_augmented",
                                    np.zeros(len(model["train_X"]), dtype=bool),
                                )
                            ).sum()
                        ),
                    }
                    if model is not None
                    else {}
                )
                return self.json_response(
                    200, {**details, **training.status(), "ready": model is not None}
                )
            if path in ("/api/train/plot", "/api/train/projection"):
                model = model_loaded()
                if model is None:
                    return self.json_response(
                        409,
                        {
                            "error": "Сначала нажми «Обучить модель» на этапе «Обучение»."
                        },
                    )
                if path.endswith("projection"):
                    return self.json_response(
                        200, projection_data(model, training.get_probe())
                    )
                with _lock:
                    image = projection_png(model, training.get_probe())
                return self.bytes_response(200, image, "image/png")
            if path == "/api/samples":
                return self.json_response(
                    200,
                    {
                        label: [
                            sample.name
                            for sample in sorted((DATA / label).glob("*.wav"))
                        ]
                        for label in LABELS
                    },
                )
            if path == "/api/audio":
                wav = sample_path(self.params())
                return self.bytes_response(200, wav.read_bytes(), "audio/wav")
            if path in (
                "/api/plot",
                "/api/compare",
                "/api/preprocess",
                "/api/preprocess/audio",
            ):
                params = self.params()
                source = sample_path(params)
                samples = load_wav(source)
                options = validate_settings(
                    {
                        key: values[0]
                        for key, values in params.items()
                        if key in DEFAULT_SETTINGS
                    }
                )
                if path.startswith("/api/preprocess"):
                    options = validate_settings(
                        {
                            **options,
                            "denoise_strength": float(
                                params.get(
                                    "strength",
                                    [str(CONFIG["preprocessing"]["preview_strength"])],
                                )[0]
                            ),
                            "denoise_n_fft": options["n_fft"],
                            "trim_silence": params.get("trim_silence", ["false"])[0],
                        }
                    )
                    image = state.result(
                        "audio" if path.endswith("/audio") else "preprocess",
                        source,
                        options,
                    )
                    return self.bytes_response(
                        200,
                        image,
                        "audio/wav" if path.endswith("/audio") else "image/png",
                    )
                if path == "/api/plot" and params.get("view", [""])[0] in (
                    "features",
                    "mel",
                ):
                    return self.bytes_response(
                        200,
                        state.result(params["view"][0], source, options),
                        "image/png",
                    )
                view = params.get("view", [""])[0]
                if view == "features":
                    with _lock:
                        return self.figure_response(
                            make_feature_signal_figure(
                                samples, options, f"{source.parent.name}/{source.name}"
                            )
                        )
                if view in ("signal", "mel"):
                    recordings = (samples,)
                    names = (f"{source.parent.name}/{source.name}",)
                    if path == "/api/compare":
                        other = sample_path(
                            {
                                "label": params.get("label2", [""]),
                                "file": params.get("file2", [""]),
                            }
                        )
                        recordings += (load_wav(other),)
                        names += (f"{other.parent.name}/{other.name}",)
                    with _lock:
                        builder = (
                            make_mel_figure if view == "mel" else make_signal_figure
                        )
                        return self.figure_response(builder(recordings, options, names))
                if path == "/api/compare":
                    other = sample_path(
                        {
                            "label": params.get("label2", [""]),
                            "file": params.get("file2", [""]),
                        }
                    )
                    other_samples = load_wav(other)
                    with _lock:
                        return self.figure_response(
                            make_comparison_figure(
                                samples,
                                other_samples,
                                options,
                                (
                                    f"{source.parent.name}/{source.name}",
                                    f"{other.parent.name}/{other.name}",
                                ),
                            )
                        )
                with _lock:
                    figure = (
                        make_configured_figure(samples, options)
                        if any(key in DEFAULT_SETTINGS for key in params)
                        else make_figure(samples)
                    )
                    return self.figure_response(figure)
            if path == "/api/dataset.zip":
                output = io.BytesIO()
                with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
                    for label in LABELS:
                        for wav in sorted((DATA / label).glob("*.wav")):
                            archive.write(wav, arcname=f"dataset/{label}/{wav.name}")
                    for name in (
                        "features.py",
                        "trainer.py",
                        "experiment_settings.py",
                        "lab_config.py",
                    ):
                        archive.write(ROOT / "src" / name, arcname=name)
                    archive.write(
                        ROOT / "config" / "lab.json", arcname="config/lab.json"
                    )
                return self.bytes_response(
                    200, output.getvalue(), "application/zip", "asr-dataset.zip"
                )
            if path in ALLOWED_FILES:
                self.path = "/index.html" if path == "/" else path
                return super().do_GET()
            return self.send_error(404)
        except (ValueError, FileNotFoundError) as exc:
            return self.json_response(
                400 if isinstance(exc, ValueError) else 404, {"error": str(exc)}
            )
        except Exception as exc:
            logger.exception(
                "Failed to process request", extra={"path": path, "error": str(exc)}
            )
            return self.json_response(
                500,
                {
                    "error": "Не получилось выполнить действие. Попробуй еще раз. Подробности есть в терминале сервера."
                },
            )

    def get_audio(self) -> tuple[bytes, np.ndarray]:
        """
        Читает и проверяет тело запроса с WAV.

        :return: Исходные байты WAV и декодированные отсчеты.
        """
        if self.headers.get_content_type() not in ("audio/wav", "audio/x-wav"):
            raise ValueError("Нужен Content-Type: audio/wav")
        try:
            size = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            size = 0
        if not 44 <= size <= MAX_BYTES:
            raise ValueError(f"Некорректный размер WAV (максимум {MAX_BYTES} байт).")
        payload = self.rfile.read(size)
        if len(payload) != size:
            raise ValueError("WAV получен не полностью.")
        return payload, load_wav(payload)

    def training_options(self) -> dict[str, Any]:
        """
        Читает дополнительные параметры обучения из query и JSON.

        :return: Параметры эксперимента или пустой словарь для старого API.
        """
        options: dict[str, Any] = {
            key: values[0]
            for key, values in self.params().items()
            if key in DEFAULT_SETTINGS
        }
        size = int(self.headers.get("Content-Length", "0"))
        if not 0 <= size <= 16_384:
            raise ValueError("Некорректный размер параметров обучения.")
        if size:
            if self.headers.get_content_type() != "application/json":
                raise ValueError(
                    "Для параметров обучения нужен Content-Type: application/json."
                )
            try:
                body = json.loads(self.rfile.read(size))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise ValueError("Не получилось прочитать параметры обучения.") from exc
            if not isinstance(body, dict):
                raise ValueError("Параметры обучения должны быть объектом JSON.")
            options.update(body)
        return options

    def do_POST(self) -> None:
        """Обрабатывает запись WAV, обучение, проверку и удаление записей."""
        path = urlsplit(self.path).path
        try:
            state = runtime()
            if path in (
                "/api/settings",
                "/api/cache/warm",
                "/api/cache/clear",
                "/api/models/select",
                "/api/models/delete",
                "/api/models/clear",
            ):
                body = self.training_options()
                if path == "/api/settings":
                    return self.json_response(200, state.update_settings(body))
                if path == "/api/cache/warm":
                    selected = sample_path(
                        {
                            "label": [str(body.get("label", ""))],
                            "file": [str(body.get("file", ""))],
                        }
                    )
                    state.warm(str(body.get("page", "")), selected)
                    return self.json_response(200, state.cache_status())
                if path == "/api/cache/clear":
                    state.clear_cache()
                    return self.json_response(200, state.cache_status())
                if _history is None:
                    raise ValueError("История моделей доступна в режиме предрасчета.")
                training.invalidate()
                if path == "/api/models/select":
                    metadata = _history.select(str(body.get("id", "")))
                    state.update_settings(metadata["settings"])
                elif path == "/api/models/delete":
                    _history.delete(str(body.get("id", "")))
                else:
                    _history.clear()
                active = model_loaded()
                training.change(
                    state="done" if active is not None else "idle",
                    stage="done" if active is not None else "",
                    percent=100 if active is not None else 0,
                    message="Выбранная модель готова к проверке и игре."
                    if active is not None
                    else "Модели удалены. Обучи новый вариант.",
                    settings=model_settings(active)
                    if active is not None
                    else state.settings,
                )
                return self.json_response(200, models_response())
            if path == "/api/train/start":
                options = self.training_options()
                if options:
                    checked = validate_settings(options)
                    state.update_settings(checked)
                state = (
                    training.start_experiment(DATA, MODEL, options)
                    if options
                    else training.start(DATA, MODEL)
                )
                return self.json_response(202, state)
            if path in ("/api/train/probe", "/api/predict"):
                model = model_loaded()
                if model is None:
                    return self.json_response(
                        409 if path.endswith("probe") else 503,
                        {
                            "error": "Сначала нажми «Обучить модель» на этапе «Обучение»."
                        },
                    )
                _, samples = self.get_audio()
                denoise = self.params().get("denoise", ["true"])[0]
                if denoise not in ("true", "false"):
                    raise ValueError("denoise: требуется true или false.")
                if denoise == "false" and int(model["feature_version"]) == 2:
                    from features import extract_features_with_settings

                    vector = extract_features_with_settings(
                        samples, {**model_settings(model), "denoise_strength": 0}
                    )
                else:
                    vector = model_features(model, samples)
                if path.endswith("probe"):
                    response = explain(model, vector)
                    training.set_probe(vector)
                    return self.json_response(200, response)
                label, distances = portable_predict(model, vector.reshape(1, -1))
                return self.json_response(
                    200,
                    {
                        "label": label,
                        "text": LABELS[label].lower(),
                        "neighbor_distances": [
                            round(float(value), 2) for value in distances
                        ],
                    },
                )
            if path == "/api/samples":
                label = validate_label(self.params().get("label", [""])[0])
                payload, samples = self.get_audio()
                folder = DATA / label
                folder.mkdir(parents=True, exist_ok=True)
                with _lock:
                    recordings = list(folder.glob("*.wav"))
                    if len(recordings) >= MAX_SAMPLES:
                        raise ValueError(
                            f"У тебя уже {MAX_SAMPLES} записей этой команды. Удали ненужный пример, чтобы записать новый."
                        )
                    existing = [
                        int(sample.stem)
                        for sample in recordings
                        if sample.stem.isdigit()
                    ]
                    number = max(existing, default=0) + 1
                    while True:
                        name = f"{number:02d}.wav"
                        try:
                            with (folder / name).open("xb") as recording:
                                recording.write(payload)
                            break
                        except FileExistsError:
                            number += 1
                runtime()
                return self.json_response(
                    201,
                    {
                        "label": label,
                        "filename": name,
                        "duration": round(len(samples) / 16000, 2),
                    },
                )
            if path == "/api/delete":
                sample = sample_path(self.params())
                with _lock:
                    sample.unlink(missing_ok=False)
                runtime()
                return self.json_response(200, {"deleted": sample.name})
            return self.json_response(404, {"error": "Неизвестный адрес."})
        except (ValueError, RuntimeError) as exc:
            return self.json_response(
                409 if isinstance(exc, RuntimeError) else 400, {"error": str(exc)}
            )
        except FileNotFoundError:
            return self.json_response(404, {"error": "Запись не найдена."})
        except Exception as exc:
            logger.exception(
                "Failed to process request", extra={"path": path, "error": str(exc)}
            )
            return self.json_response(
                500,
                {
                    "error": "Не получилось выполнить действие. Попробуй еще раз. Подробности есть в терминале сервера."
                },
            )


def main() -> None:
    """Запускает локальный сервер лаборатории на порту 8000."""
    global MODE
    parser = argparse.ArgumentParser(description="Локальная ASR-лаборатория")
    parser.add_argument(
        "--mode",
        choices=("on-demand", "precompute"),
        default="on-demand",
        help="Режим вычислений",
    )
    MODE = parser.parse_args().mode
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    for label in LABELS:
        (DATA / label).mkdir(parents=True, exist_ok=True)
    runtime()
    with ThreadingHTTPServer((HOST, PORT), Handler) as server:
        logger.info(
            "Service started", extra={"url": f"http://localhost:{PORT}/collect.html"}
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            logger.info("Service stopped")
        finally:
            if _runtime is not None and _runtime.cache is not None:
                _runtime.cache.stop()


if __name__ == "__main__":
    main()
