"""Оценка сохраненной модели на отдельном наборе WAV без обучения."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import zipfile
from importlib.metadata import version
from pathlib import Path
from platform import python_version
from typing import Any

import numpy as np

from features import LABELS, load_wav
from trainer import model_features, model_settings, portable_predict

ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger(__name__)


def load_model(contents: bytes) -> dict[str, Any]:
    """
    Загружает числовую модель и проверяет данные для предсказания.

    :param contents: Содержимое сохраненной модели NPZ.
    :return: Проверенные массивы модели.
    """
    with np.load(io.BytesIO(contents), allow_pickle=False) as saved:
        model = dict(saved)
    required = {"train_X", "train_y", "mean", "scale", "k", "feature_version"}
    if not required.issubset(model):
        raise ValueError("ASR-MDL-01: Missing model arrays (Hint: fix request)")
    matrix, labels = model["train_X"], model["train_y"]
    if (
        matrix.ndim != 2
        or not matrix.size
        or labels.shape != (matrix.shape[0],)
        or set(labels.tolist()) != set(LABELS)
        or model["mean"].shape != (matrix.shape[1],)
        or model["scale"].shape != (matrix.shape[1],)
        or not all(
            np.isfinite(model[key]).all() for key in ("train_X", "mean", "scale")
        )
        or np.any(model["scale"] <= 0)
    ):
        raise ValueError("ASR-MDL-02: Invalid model arrays (Hint: fix request)")
    if (
        model["k"].shape != ()
        or not 1 <= int(model["k"]) <= len(labels)
        or model["k"] != int(model["k"])
        or model["feature_version"].shape != ()
        or int(model["feature_version"]) not in (1, 2)
        or model["feature_version"] != int(model["feature_version"])
        or str(model.get("metric", "euclidean"))
        not in ("euclidean", "manhattan", "minkowski")
        or str(model.get("weights", "distance")) not in ("uniform", "distance")
    ):
        raise ValueError("ASR-MDL-03: Invalid model settings (Hint: fix request)")
    feature_count = (
        2 * model_settings(model)["n_mfcc"]
        if int(model["feature_version"]) == 2
        else 26
    )
    if matrix.shape[1] != feature_count:
        raise ValueError("ASR-MDL-04: Incompatible model features (Hint: fix request)")
    return model


def evaluate(model_path: Path, dataset: Path) -> dict[str, Any]:
    """
    Считает метрики на всех WAV четырех классов без изменения модели.

    :param model_path: Путь обученной модели NPZ.
    :param dataset: Каталог независимых проверочных записей.
    :return: Воспроизводимый отчет с метриками и отпечатками входных данных.
    """
    model_contents = model_path.read_bytes()
    model = load_model(model_contents)
    if dataset.resolve() == (ROOT / "dataset").resolve():
        raise ValueError("ASR-DAT-02: Training dataset selected (Hint: fix request)")
    if not dataset.is_dir():
        raise ValueError(
            "ASR-DAT-03: Evaluation directory not found (Hint: fix request)"
        )
    if {entry.name for entry in dataset.iterdir() if entry.is_dir()} != set(
        LABELS
    ) or any(
        path.suffix.lower() == ".wav" for path in dataset.iterdir() if path.is_file()
    ):
        raise ValueError(
            "ASR-DAT-04: Expected four class directories (Hint: fix request)"
        )
    training_hashes = {
        hashlib.sha256(path.read_bytes()).hexdigest()
        for label in LABELS
        for path in (ROOT / "dataset" / label).glob("*.wav")
    }
    labels = list(LABELS)
    confusion = np.zeros((len(labels), len(labels)), dtype=int)
    seen: set[str] = set()
    predictions = []
    for row, label in enumerate(labels):
        files = sorted((dataset / label).iterdir())
        if not files or any(
            not path.is_file() or path.suffix.lower() != ".wav" for path in files
        ):
            raise ValueError(
                "ASR-DAT-05: Each class must contain only WAV files "
                f"(Hint: fix request): {label}"
            )
        for path in files:
            contents = path.read_bytes()
            digest = hashlib.sha256(contents).hexdigest()
            if digest in seen or digest in training_hashes:
                raise ValueError(
                    "ASR-DAT-06: Duplicate or training WAV found "
                    f"(Hint: fix request): {path}"
                )
            seen.add(digest)
            try:
                vector = model_features(model, load_wav(contents))
            except ValueError as exc:
                raise ValueError(
                    f"ASR-DAT-07: Invalid evaluation WAV (Hint: fix request): {path}: {exc}"
                ) from exc
            if vector.shape != model["mean"].shape or not np.isfinite(vector).all():
                raise ValueError(
                    "ASR-MDL-04: Incompatible model features (Hint: fix request)"
                )
            predicted, _ = portable_predict(model, vector)
            confusion[row, labels.index(predicted)] += 1
            predictions.append(
                {
                    "file": path.relative_to(dataset).as_posix(),
                    "sha256": digest,
                    "expected": label,
                    "predicted": predicted,
                }
            )
    per_class = {}
    for index, label in enumerate(labels):
        correct = int(confusion[index, index])
        support = int(confusion[index].sum())
        predicted_count = int(confusion[:, index].sum())
        precision = correct / predicted_count if predicted_count else 0.0
        recall = correct / support
        per_class[label] = {
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0,
        }
    return {
        "schema_version": 1,
        "environment": {
            "python": python_version(),
            **{name: version(name) for name in ("numpy", "scipy", "scikit-learn")},
        },
        "model_sha256": hashlib.sha256(model_contents).hexdigest(),
        "feature_version": int(model["feature_version"]),
        "settings": model_settings(model)
        if int(model["feature_version"]) == 2
        else None,
        "labels": labels,
        "count": len(predictions),
        "accuracy": float(np.trace(confusion) / confusion.sum()),
        "confusion_matrix": confusion.tolist(),
        "per_class": per_class,
        "predictions": predictions,
        "overlap_check": {"current_training_files_checked": len(training_hashes)},
        "limitations": [
            "Совпадения WAV проверены только с текущим локальным dataset и внутри оценки",
            "NPZ не хранит отпечатки исходных WAV: независимость от исторического обучения не подтверждена",
            "Другие имена, кодировки, аугментации и совпадение дикторов автоматически не исключаются",
            "Отчет не определяет, содержит ли WAV реальную речь или синтетический сигнал",
            "Модель всегда выбирает одну из четырех команд и не оценивает отказ на посторонней речи",
        ],
    }


def main() -> None:
    """Сохраняет JSON-отчет и сообщает итог оценки через logger."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, default=ROOT / "models/keyword_classifier.npz"
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/evaluation.json"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        report = evaluate(args.model, args.dataset)
        if (
            args.output.resolve() in {args.model.resolve()}
            or args.output.suffix != ".json"
            or args.output.is_symlink()
            or (args.output.exists() and args.output.stat().st_nlink > 1)
        ):
            raise ValueError("ASR-DAT-08: Invalid report path (Hint: fix request)")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open(
            "w" if args.overwrite else "x", encoding="utf-8"
        ) as output:
            output.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
        logger.error("Failed to evaluate model", extra={"error": str(exc)})
        parser.exit(1, f"{exc}\n")
    logger.info(
        "Evaluated keyword classifier",
        extra={
            "path": str(args.output),
            "count": report["count"],
            "result": report["accuracy"],
        },
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
