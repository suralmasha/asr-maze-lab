"""Параметры воспроизводимых экспериментов с аудио и ближайшими соседями."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from lab_config import CONFIG

MIN_SAMPLES = CONFIG["audio"]["min_samples"]
MAX_SAMPLES = CONFIG["audio"]["max_samples"]
DEFAULT_SETTINGS: dict[str, Any] = {
    **CONFIG["defaults"],
    "samples_per_class": MAX_SAMPLES,
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
    settings["n_mels"] = _integer(
        settings["n_mels"], "n_mels", *CONFIG["limits"]["n_mels"]
    )
    if "n_mfcc" not in supplied:
        settings["n_mfcc"] = min(settings["n_mfcc"], settings["n_mels"])
    settings["n_mfcc"] = _integer(
        settings["n_mfcc"],
        "n_mfcc",
        CONFIG["limits"]["n_mfcc"][0],
        min(CONFIG["limits"]["n_mfcc"][1], settings["n_mels"]),
    )
    fft_values = CONFIG["feature_scales"]["n_fft"]
    settings["n_fft"] = _integer(
        settings["n_fft"], "n_fft", min(fft_values), max(fft_values)
    )
    if settings["n_fft"] not in fft_values:
        raise ValueError(f"n_fft: допустимы {', '.join(map(str, fft_values))}.")
    if settings["hop_length"] is None:
        settings["hop_length"] = settings["n_fft"] // 4
    settings["hop_length"] = _integer(
        settings["hop_length"], "hop_length", 1, settings["n_fft"]
    )
    settings["denoise_n_fft"] = _integer(
        settings["denoise_n_fft"], "denoise_n_fft", min(fft_values), max(fft_values)
    )
    if settings["denoise_n_fft"] not in fft_values:
        raise ValueError(f"denoise_n_fft: допустимы {', '.join(map(str, fft_values))}.")
    settings["k"] = _integer(settings["k"], "k", *CONFIG["limits"]["k"])
    settings["samples_per_class"] = _integer(
        settings["samples_per_class"], "samples_per_class", MIN_SAMPLES, MAX_SAMPLES
    )
    for name in ("normalize", "trim_silence"):
        settings[name] = _boolean(settings[name], name)
    if settings["metric"] not in ("euclidean", "manhattan"):
        raise ValueError("metric: требуется euclidean или manhattan.")
    if settings["weights"] not in ("uniform", "distance"):
        raise ValueError("weights: требуется uniform или distance.")
    for name in ("denoise_strength", "noise"):
        minimum, maximum = CONFIG["limits"][name]
        message = f"{name}: требуется число от {minimum:g} до {maximum:g}."
        try:
            settings[name] = float(settings[name])
        except (TypeError, ValueError) as exc:
            raise ValueError(message) from exc
        if (
            not math.isfinite(settings[name])
            or not minimum <= settings[name] <= maximum
        ):
            raise ValueError(message)
    return settings
