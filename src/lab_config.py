"""Загрузка общих ограничений и шкал локальной лаборатории."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

DIRECTORY = Path(__file__).resolve().parent
CONFIG_PATH = DIRECTORY / "config/lab.json"
if not CONFIG_PATH.is_file():
    CONFIG_PATH = DIRECTORY.parent / "config/lab.json"


def _number(value: Any) -> float:
    """
    Проверяет конечное число конфигурации.

    :param value: Значение из JSON.
    :return: Проверенное число.
    """
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("Ожидалось конечное число.")
    return float(value)


def _require(condition: bool, description: str) -> None:
    """
    Проверяет согласованность связанных настроек.

    :param condition: Условие корректности.
    :param description: Описание нарушенного ограничения.
    """
    if not condition:
        raise ValueError(description)


def _validate(config: dict[str, Any]) -> None:
    """
    Проверяет ограничения, шкалы и начальные параметры.

    :param config: Содержимое конфигурационного файла.
    """
    audio, scales, limits, defaults = (
        config["audio"],
        config["feature_scales"],
        config["limits"],
        config["defaults"],
    )
    rates = config["sampling_rates"]
    _require(
        isinstance(rates, list)
        and bool(rates)
        and all(type(rate) is int and rate > 0 for rate in rates)
        and rates == sorted(set(rates))
        and 16000 in rates,
        "Частоты демонстрации должны быть положительными, возрастать и включать 16000.",
    )
    for value in (
        config["cache_max_bytes"],
        audio["min_samples"],
        audio["max_samples"],
        audio["max_bytes"],
    ):
        _require(
            type(value) is int and value > 0,
            "Лимиты должны быть положительными целыми числами.",
        )
    _require(
        audio["min_samples"] <= audio["max_samples"],
        "Минимум записей превышает максимум.",
    )
    _require(
        0
        < _number(audio["min_seconds"])
        <= _number(audio["max_seconds"])
        <= _number(audio["validation_max_seconds"]),
        "Неверный диапазон длительности записи.",
    )
    _require(
        audio["max_bytes"]
        >= math.ceil(audio["validation_max_seconds"] * 16000 * 2) + 44,
        "Лимит байтов не вмещает максимальный WAV.",
    )
    for name in ("n_mels", "n_mfcc", "k", "noise", "denoise_strength"):
        bounds = limits[name]
        _require(
            isinstance(bounds, list)
            and len(bounds) == 2
            and _number(bounds[0]) <= _number(bounds[1]),
            f"Неверные границы {name}.",
        )
        if name in ("n_mels", "n_mfcc", "k"):
            _require(
                all(type(value) is int and value > 0 for value in bounds),
                f"Границы {name} должны быть положительными целыми числами.",
            )
        else:
            _require(bounds[0] == 0, f"Нижняя граница {name} должна быть нулевой.")
        value = defaults[name]
        _require(
            bounds[0] <= _number(value) <= bounds[1],
            f"Начальное значение {name} вне границ.",
        )
    for name in ("n_mels", "n_mfcc", "n_fft", "hop_length"):
        values = scales[name]
        _require(
            isinstance(values, list)
            and bool(values)
            and all(type(value) is int and value > 0 for value in values),
            f"Шкала {name} должна содержать положительные целые числа.",
        )
        _require(
            values == sorted(set(values)), f"Шкала {name} должна строго возрастать."
        )
        if name in limits:
            _require(
                all(limits[name][0] <= value <= limits[name][1] for value in values),
                f"Шкала {name} выходит за границы.",
            )
        selected = defaults[name]
        if selected is None and name == "hop_length":
            selected = defaults["n_fft"] // 4
        _require(
            type(selected) is int and selected in values,
            f"Начальное значение {name} отсутствует в шкале.",
        )
    _require(
        all(value >= 4 and value & (value - 1) == 0 for value in scales["n_fft"]),
        "Окна FFT должны быть степенями двух от 4.",
    )
    _require(
        defaults["denoise_n_fft"] in scales["n_fft"],
        "Окно шумоподавления отсутствует в шкале FFT.",
    )
    _require(
        defaults["n_mfcc"] <= defaults["n_mels"]
        and min(scales["n_mfcc"]) <= min(scales["n_mels"]),
        "MFCC превышает число мел-фильтров.",
    )
    _require(
        min(scales["hop_length"]) <= min(scales["n_fft"])
        and max(scales["hop_length"]) <= max(scales["n_fft"]),
        "Шаг анализа не согласован с окнами FFT.",
    )
    _require(
        (defaults["hop_length"] or defaults["n_fft"] // 4) <= defaults["n_fft"],
        "Начальный шаг превышает окно FFT.",
    )
    step = _number(config["preprocessing"]["strength_step"])
    low, high = limits["denoise_strength"]
    _require(
        0 < step <= high - low
        and math.isclose((high - low) / step, round((high - low) / step)),
        "Шаг шумоподавления должен делить его диапазон.",
    )
    _require(
        low <= _number(config["preprocessing"]["preview_strength"]) <= high,
        "Начальный просмотр шумоподавления вне границ.",
    )
    noise = config["noise_levels"]
    _require(
        isinstance(noise, list)
        and bool(noise)
        and all(
            limits["noise"][0] <= _number(value) <= limits["noise"][1]
            for value in noise
        ),
        "Уровни шума выходят за границы.",
    )
    _require(
        noise == sorted(set(noise)) and defaults["noise"] in noise,
        "Шкала шума не согласована с начальным значением.",
    )
    _require(
        type(defaults["k"]) is int
        and isinstance(defaults["normalize"], bool)
        and isinstance(defaults["trim_silence"], bool),
        "Неверные типы начальных настроек.",
    )
    _require(
        defaults["metric"] in ("euclidean", "manhattan")
        and defaults["weights"] in ("uniform", "distance"),
        "Неизвестная метрика или взвешивание.",
    )


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """
    Загружает и проверяет общий конфиг без резервных скрытых значений.

    :param path: Путь конфигурационного JSON.
    :return: Проверенная конфигурация.
    """
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
        _require(isinstance(config, dict), "Конфигурация должна быть объектом JSON.")
        _validate(config)
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
        raise ValueError(
            f"ASR-CFG-02: Invalid laboratory configuration: {exc} (Hint: fix configuration)"
        ) from exc
    return config


CONFIG = load_config()
