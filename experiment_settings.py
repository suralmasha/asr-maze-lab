"""Параметры воспроизводимых экспериментов с аудио и ближайшими соседями."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

MIN_SAMPLES = 5
MAX_SAMPLES = 50
DEFAULT_SETTINGS: dict[str, Any] = {
    "n_mels": 32,
    "n_mfcc": 13,
    "n_fft": 512,
    "hop_length": None,
    "denoise_strength": 0.0,
    "denoise_n_fft": 512,
    "normalize": True,
    "metric": "euclidean",
    "weights": "distance",
    "k": 3,
    "samples_per_class": MAX_SAMPLES,
    "trim_silence": True,
    "noise": 0.0,
}


def _integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    """
    Проверяет целое значение в заданном диапазоне.

    :param value: Полученное значение.
    :param name: Название параметра.
    :param minimum: Нижняя граница.
    :param maximum: Верхняя граница.
    :return: Проверенное число.
    """
    try:
        number = int(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}: требуется целое число.") from exc
    if not minimum <= number <= maximum:
        raise ValueError(f"{name}: допустимы значения от {minimum} до {maximum}.")
    return number


def _boolean(value: Any, name: str) -> bool:
    """
    Проверяет переключатель из JSON или строки запроса.

    :param value: Полученное значение.
    :param name: Название параметра.
    :return: Состояние переключателя.
    """
    if isinstance(value, bool):
        return value
    if str(value).lower() in ("true", "1"):
        return True
    if str(value).lower() in ("false", "0"):
        return False
    raise ValueError(f"{name}: требуется true или false.")


def validate_settings(values: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """
    Проверяет параметры и дополняет их значениями по умолчанию.

    :param values: Параметры эксперимента.
    :return: Полный набор проверенных параметров.
    """
    supplied = dict(values or {})
    settings = {
        **DEFAULT_SETTINGS,
        **{key: value for key, value in supplied.items() if key in DEFAULT_SETTINGS},
    }
    settings["n_mels"] = _integer(settings["n_mels"], "n_mels", 8, 128)
    if "n_mfcc" not in supplied:
        settings["n_mfcc"] = min(settings["n_mfcc"], settings["n_mels"])
    settings["n_mfcc"] = _integer(
        settings["n_mfcc"], "n_mfcc", 2, min(40, settings["n_mels"])
    )
    settings["n_fft"] = _integer(settings["n_fft"], "n_fft", 256, 2048)
    if settings["n_fft"] not in (256, 512, 1024, 2048):
        raise ValueError("n_fft: допустимы 256, 512, 1024 или 2048.")
    if settings["hop_length"] is None:
        settings["hop_length"] = settings["n_fft"] // 4
    settings["hop_length"] = _integer(
        settings["hop_length"], "hop_length", 1, settings["n_fft"]
    )
    settings["denoise_n_fft"] = _integer(
        settings["denoise_n_fft"], "denoise_n_fft", 256, 2048
    )
    if settings["denoise_n_fft"] not in (256, 512, 1024, 2048):
        raise ValueError("denoise_n_fft: допустимы 256, 512, 1024 или 2048.")
    try:
        settings["denoise_strength"] = float(settings["denoise_strength"])
    except (TypeError, ValueError) as exc:
        raise ValueError("denoise_strength: требуется число от 0 до 2.") from exc
    if (
        not math.isfinite(settings["denoise_strength"])
        or not 0 <= settings["denoise_strength"] <= 2
    ):
        raise ValueError("denoise_strength: требуется число от 0 до 2.")
    settings["k"] = _integer(settings["k"], "k", 1, 15)
    settings["samples_per_class"] = _integer(
        settings["samples_per_class"], "samples_per_class", MIN_SAMPLES, MAX_SAMPLES
    )
    for name in ("normalize", "trim_silence"):
        settings[name] = _boolean(settings[name], name)
    if settings["metric"] not in ("euclidean", "manhattan"):
        raise ValueError("metric: требуется euclidean или manhattan.")
    if settings["weights"] not in ("uniform", "distance"):
        raise ValueError("weights: требуется uniform или distance.")
    try:
        settings["noise"] = float(settings["noise"])
    except (TypeError, ValueError) as exc:
        raise ValueError("noise: требуется число от 0 до 0.1.") from exc
    if not math.isfinite(settings["noise"]) or not 0 <= settings["noise"] <= 0.1:
        raise ValueError("noise: требуется число от 0 до 0.1.")
    return settings
