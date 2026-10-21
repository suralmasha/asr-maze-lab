"""One-laptop offline workshop server. Start: python lab_server.py.

Only binds to localhost. Audio stays on the laptop. No third-party speech API.
"""
from __future__ import annotations

import io
import json
import logging
import re
import threading
import zipfile
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import matplotlib
matplotlib.use("Agg")  # Safe server-side plotting on macOS/background threads.

from features import LABELS, load_wav, extract_features, make_figure
from trainer import portable_predict
from training_web import TrainingManager, explain, projection_png

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dataset"
MODEL = ROOT / "models" / "keyword_classifier.npz"
HOST, PORT = "127.0.0.1", 8000
MAX_BYTES = 200_000
ALLOWED_FILES = {"/", "/index.html", "/style.css", "/app.js", "/collect.html",
                 "/collect.js", "/visualize.html", "/visualize.js", "/workshop.css", "/recorder.js",
                 "/train.html", "/train.js"}
_lock = threading.Lock()
training = TrainingManager()
_classifier = None
_classifier_stamp = None


def model_loaded():
    """Refresh trained model without restarting the server after training."""
    global _classifier, _classifier_stamp
    if not MODEL.is_file():
        _classifier, _classifier_stamp = None, None
        return None
    stamp = MODEL.stat().st_mtime_ns
    if stamp == _classifier_stamp:
        return _classifier
    import numpy as np
    try:
        with np.load(MODEL, allow_pickle=False) as saved:
            bundle = {k: saved[k] for k in saved.files}
        if int(bundle["feature_version"]) != 1:
            raise ValueError("Версия признаков не совпала; переобучи модель.")
        if bundle["train_X"].ndim != 2 or bundle["train_X"].shape[1] != 26:
            raise ValueError("Неверная форма обучающих признаков.")
        _classifier = bundle
        _classifier_stamp = stamp
    except Exception:
        logging.exception("Cannot load trained model")
        _classifier, _classifier_stamp = None, None
    return _classifier


def validate_label(value: str) -> str:
    if value not in LABELS:
        raise ValueError("Неизвестная команда.")
    return value


def validate_file(value: str) -> str:
    if not re.fullmatch(r"[0-9]{2,6}\.wav", value):
        raise ValueError("Неверное имя файла.")
    return value


def sample_path(params) -> Path:
    label = validate_label(params.get("label", [""])[0])
    name = validate_file(params.get("file", [""])[0])
    return DATA / label / name


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def json_response(self, status, value):
        data = json.dumps(value, ensure_ascii=False).encode()
        self.bytes_response(status, data, "application/json; charset=utf-8")

    def bytes_response(self, status, data, content_type, filename=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.end_headers()
        self.wfile.write(data)

    def params(self):
        return parse_qs(urlsplit(self.path).query)

    def do_GET(self):
        path = urlsplit(self.path).path
        try:
            if path == "/api/health":
                model = model_loaded()
                return self.json_response(200, {"ready": model is not None,
                                                "message": "Классификатор загружен." if model is not None
                                                else "Модель ещё не обучена. Открой страницу «Обучение»."})
            if path == "/api/train/status":
                return self.json_response(200, {**training.status(), "ready": model_loaded() is not None})
            if path == "/api/train/plot":
                model = model_loaded()
                if model is None:
                    return self.json_response(409, {"error": "Сначала обучи модель."})
                with _lock:
                    data = projection_png(model, training.get_probe())
                return self.bytes_response(200, data, "image/png")
            if path == "/api/samples":
                return self.json_response(200, {label: [p.name for p in sorted((DATA / label).glob("*.wav"))]
                                                for label in LABELS})
            if path == "/api/audio":
                wav = sample_path(self.params())
                if not wav.is_file():
                    raise FileNotFoundError("Запись не найдена.")
                return self.bytes_response(200, wav.read_bytes(), "audio/wav")
            if path == "/api/plot":
                wav = sample_path(self.params())
                if not wav.is_file():
                    raise FileNotFoundError("Запись не найдена.")
                import matplotlib.pyplot as plt
                with _lock:  # pyplot is not thread-safe
                    fig = make_figure(load_wav(wav))
                    try:
                        output = io.BytesIO()
                        fig.savefig(output, format="png", dpi=115)
                    finally:
                        plt.close(fig)
                return self.bytes_response(200, output.getvalue(), "image/png")
            if path == "/api/dataset.zip":
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                    for label in LABELS:
                        for wav in sorted((DATA / label).glob("*.wav")):
                            z.write(wav, arcname=f"dataset/{label}/{wav.name}")
                    for name in ("features.py", "trainer.py"):
                        z.write(ROOT / name, arcname=name)
                return self.bytes_response(200, buf.getvalue(), "application/zip", "asr-dataset.zip")
            if path in ALLOWED_FILES:
                self.path = "/index.html" if path == "/" else path
                return super().do_GET()
            return self.send_error(404)
        except (ValueError, FileNotFoundError) as e:
            return self.json_response(400 if isinstance(e, ValueError) else 404, {"error": str(e)})
        except Exception:
            logging.exception("GET failed")
            return self.json_response(500, {"error": "Ошибка обработки. См. терминал."})

    def get_audio(self):
        if self.headers.get_content_type() not in ("audio/wav", "audio/x-wav"):
            raise ValueError("Нужен Content-Type: audio/wav")
        try:
            size = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            size = 0
        if not 44 <= size <= MAX_BYTES:
            raise ValueError("Некорректный размер WAV (максимум 200 КБ).")
        payload = self.rfile.read(size)
        if len(payload) != size:
            raise ValueError("WAV получен не полностью.")
        samples = load_wav(payload)
        return payload, samples

    def do_POST(self):
        path = urlsplit(self.path).path
        try:
            if path == "/api/train/start":
                return self.json_response(202, training.start(DATA, MODEL))
            if path == "/api/train/probe":
                model = model_loaded()
                if model is None:
                    return self.json_response(409, {"error": "Сначала обучи модель."})
                _, samples = self.get_audio()
                vector = extract_features(samples)
                response = explain(model, vector)
                training.set_probe(vector)
                return self.json_response(200, response)
            if path == "/api/samples":
                label = validate_label(self.params().get("label", [""])[0])
                payload, samples = self.get_audio()
                folder = DATA / label
                folder.mkdir(parents=True, exist_ok=True)
                with _lock:
                    existing = [int(p.stem) for p in folder.glob("*.wav") if p.stem.isdigit()]
                    number = max(existing, default=0) + 1
                    while True:
                        name = f"{number:02d}.wav"
                        try:
                            with (folder / name).open("xb") as f:
                                f.write(payload)
                            break
                        except FileExistsError:
                            number += 1
                return self.json_response(201, {"label": label, "filename": name,
                                                "duration": round(len(samples) / 16000, 2)})
            if path == "/api/predict":
                model = model_loaded()
                if model is None:
                    return self.json_response(503, {"error": "Сначала обучи модель на странице train.html."})
                _, samples = self.get_audio()
                feature = extract_features(samples).reshape(1, -1)
                label, distances = portable_predict(model, feature)
                return self.json_response(200, {"label": label, "text": LABELS[label].lower(),
                                                "neighbor_distances": [round(float(v), 2) for v in distances]})
            if path == "/api/delete":
                sample = sample_path(self.params())
                with _lock:
                    sample.unlink(missing_ok=False)
                return self.json_response(200, {"deleted": sample.name})
            return self.json_response(404, {"error": "Неизвестный адрес."})
        except (ValueError, RuntimeError) as e:
            return self.json_response(409 if isinstance(e, RuntimeError) else 400, {"error": str(e)})
        except FileNotFoundError:
            return self.json_response(404, {"error": "Запись не найдена."})
        except Exception:
            logging.exception("POST failed")
            return self.json_response(500, {"error": "Ошибка обработки. См. терминал."})


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    for label in LABELS:
        (DATA / label).mkdir(parents=True, exist_ok=True)
    with ThreadingHTTPServer((HOST, PORT), Handler) as server:
        print(f"Лаборатория ASR: http://localhost:{PORT}/collect.html")
        print("Только этот компьютер; интернет для работы сервера не нужен.")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nСервер остановлен.")


if __name__ == "__main__":
    main()
