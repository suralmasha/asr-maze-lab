"""Общие преобразования WAV для графиков, обучения и распознавания команд."""

from __future__ import annotations

import io
import wave
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.fft import dct
from scipy.ndimage import uniform_filter1d
from scipy.signal import istft, stft

from experiment_settings import validate_settings

if TYPE_CHECKING:
    from matplotlib.figure import Figure

SAMPLE_RATE = 16_000
LABELS = {"up": "Вверх", "down": "Вниз", "left": "Влево", "right": "Вправо"}
N_FFT = 512
N_MELS = 32
N_MFCC = 13
EPS = 1e-10


def load_wav(data: bytes | Path | str) -> np.ndarray:
    """
    Проверяет WAV и возвращает отсчеты в диапазоне от -1 до 1.

    :param data: Байты WAV или путь к файлу.
    :return: Массив монофонических отсчетов с частотой 16000 Гц.
    """
    source = io.BytesIO(data) if isinstance(data, bytes) else str(data)
    try:
        with wave.open(source, "rb") as recording:
            channels = recording.getnchannels()
            width = recording.getsampwidth()
            rate = recording.getframerate()
            encoding = recording.getcomptype()
            frames = recording.getnframes()
            if (channels, width, rate, encoding) != (1, 2, SAMPLE_RATE, "NONE"):
                raise ValueError("Нужен WAV PCM 16-bit, mono, 16000 Гц.")
            if not int(0.4 * rate) <= frames <= int(5.5 * rate):
                raise ValueError(
                    "Запиши слово длиной от 0,4 до 5 секунд и попробуй еще раз."
                )
            raw = recording.readframes(frames)
    except (wave.Error, EOFError) as exc:
        raise ValueError("Не получилось прочитать WAV-файл.") from exc
    if len(raw) != frames * 2:
        raise ValueError("WAV-файл неполный.")
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if np.sqrt(np.mean(samples * samples)) < 0.0015:
        raise ValueError(
            "Почти не слышно голоса. Подойди ближе к микрофону и запиши слово еще раз."
        )
    return samples


def trim_silence(x: np.ndarray) -> np.ndarray:
    """
    Удаляет краевую тишину с запасом 120 мс вокруг речи.

    :param x: Исходные отсчеты.
    :return: Отсчеты после удаления краевой тишины.
    """
    envelope = uniform_filter1d(np.abs(x), size=320, mode="nearest")
    threshold = max(0.005, float(np.max(envelope)) * 0.10)
    active = np.flatnonzero(envelope > threshold)
    if not active.size:
        return x
    margin = int(0.12 * SAMPLE_RATE)
    first = max(0, int(active[0]) - margin)
    last = min(x.size, int(active[-1]) + margin + 1)
    return x[first:last]


def hz_to_mel(f: np.ndarray | float) -> np.ndarray:
    """
    Переводит частоты из герц в мелы.

    :param f: Частоты в герцах.
    :return: Частоты в мелах.
    """
    return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)


def mel_to_hz(m: np.ndarray | float) -> np.ndarray:
    """
    Переводит частоты из мелов в герцы.

    :param m: Частоты в мелах.
    :return: Частоты в герцах.
    """
    return 700.0 * (10.0 ** (np.asarray(m) / 2595.0) - 1.0)


def configured_mel_filterbank(n_mels: int, n_fft: int) -> np.ndarray:
    """
    Строит треугольные мел-фильтры для выбранного размера спектра.

    :param n_mels: Количество фильтров.
    :param n_fft: Размер преобразования Фурье.
    :return: Матрица весов мел-фильтров.
    """
    frequencies = np.fft.rfftfreq(n_fft, 1.0 / SAMPLE_RATE)
    points = mel_to_hz(np.linspace(hz_to_mel(0), hz_to_mel(8000), n_mels + 2))
    bank = np.zeros((n_mels, len(frequencies)), dtype=np.float32)
    for index in range(n_mels):
        left, middle, right = points[index : index + 3]
        bank[index] = np.maximum(
            0,
            np.minimum(
                (frequencies - left) / (middle - left),
                (right - frequencies) / (right - middle),
            ),
        )
        bank[index] /= max(float(bank[index].sum()), EPS)
    return bank


def mel_filterbank() -> np.ndarray:
    """
    Строит исходный набор из 32 мел-фильтров.

    :return: Матрица весов для спектра размером 512.
    """
    return configured_mel_filterbank(N_MELS, N_FFT)


MEL_BANK = mel_filterbank()


def _representation(
    wav: np.ndarray, settings: dict[str, Any], legacy: bool = False
) -> dict[str, Any]:
    """
    Рассчитывает согласованные представления аудио.

    :param wav: Исходные отсчеты.
    :param settings: Проверенные параметры признаков.
    :param legacy: Сохранение исходного окна 400 и шага 160 отсчетов.
    :return: Волна, спектры и коэффициенты MFCC.
    """
    samples = suppress_noise(
        wav, settings["denoise_strength"], settings["denoise_n_fft"]
    )
    if settings["trim_silence"]:
        samples = trim_silence(samples)
    samples = samples / max(float(np.max(np.abs(samples))), 0.001)
    window = 400 if legacy else settings["n_fft"]
    hop = 160 if legacy else settings["hop_length"]
    if samples.size < window:
        samples = np.pad(samples, (0, window - samples.size))
    frequencies, times, spectrum = stft(
        samples,
        fs=SAMPLE_RATE,
        window="hann",
        nperseg=window,
        noverlap=window - hop,
        nfft=settings["n_fft"],
        boundary=None,
        padded=False,
    )
    power = np.abs(spectrum) ** 2
    mel_power = configured_mel_filterbank(settings["n_mels"], settings["n_fft"]) @ power
    log_mel = np.log(mel_power + EPS)
    mfcc = dct(log_mel, type=2, axis=0, norm="ortho")[: settings["n_mfcc"]]
    return {
        "samples": samples,
        "freq": frequencies,
        "times": times,
        "power": power,
        "mel_power": mel_power,
        "log_mel": log_mel,
        "mfcc": mfcc,
    }


def representation(wav: np.ndarray) -> dict:
    """
    Рассчитывает исходные представления для совместимости модели версии 1.

    :param wav: Исходные отсчеты.
    :return: Волна, спектры и 13 коэффициентов MFCC.
    """
    return _representation(wav, validate_settings(), legacy=True)


def representation_with_settings(
    wav: np.ndarray, settings: dict[str, Any]
) -> dict[str, Any]:
    """
    Рассчитывает представления с интерактивными параметрами.

    :param wav: Исходные отсчеты.
    :param settings: Параметры эксперимента.
    :return: Волна, спектры и коэффициенты MFCC.
    """
    return _representation(wav, validate_settings(settings))


def extract_features(wav: np.ndarray) -> np.ndarray:
    """
    Извлекает исходные 26 статистических признаков MFCC.

    :param wav: Исходные отсчеты.
    :return: Средние и стандартные отклонения 13 коэффициентов.
    """
    coefficients = representation(wav)["mfcc"]
    return np.concatenate([coefficients.mean(axis=1), coefficients.std(axis=1)]).astype(
        np.float32
    )


def extract_features_with_settings(
    wav: np.ndarray, settings: dict[str, Any]
) -> np.ndarray:
    """
    Извлекает статистические признаки с параметрами эксперимента.

    :param wav: Исходные отсчеты.
    :param settings: Параметры эксперимента.
    :return: Вектор из 2 умножить на n_mfcc признаков.
    """
    coefficients = representation_with_settings(wav, settings)["mfcc"]
    return np.concatenate([coefficients.mean(axis=1), coefficients.std(axis=1)]).astype(
        np.float32
    )


def _draw_views(info: dict[str, Any], axes: Any) -> None:
    """
    Рисует представления звука на переданных осях.

    :param info: Рассчитанные представления аудио.
    :param axes: Оси волны, спектра, STFT и необязательной мел-спектрограммы.
    """
    samples = info["samples"]
    axes[0].plot(
        np.arange(samples.size) / SAMPLE_RATE, samples, color="#3458a0", linewidth=0.7
    )
    axes[0].set(
        title="1 - Волновая форма",
        xlabel="Время, с",
        ylabel="Амплитуда",
        ylim=(-1.05, 1.05),
    )
    axes[1].plot(
        info["freq"],
        10 * np.log10(info["power"].mean(axis=1) + EPS),
        color="#ce6d35",
        linewidth=1.1,
    )
    axes[1].set(
        title="2 - Усредненный спектр",
        xlabel="Частота, Гц",
        ylabel="Мощность, дБ",
        xlim=(0, 8000),
    )
    views = [(axes[2], info["power"], "3 - Спектрограмма (STFT)", [0, 8000])]
    if len(axes) > 3:
        views.append(
            (
                axes[3],
                info["mel_power"],
                "4 - Мел-спектрограмма",
                [0, info["mel_power"].shape[0]],
            )
        )
    for axis, power, title, extent in views:
        decibels = 10 * np.log10(power + EPS)
        times = info["times"]
        image = axis.imshow(
            decibels,
            origin="lower",
            aspect="auto",
            cmap="viridis",
            extent=[
                float(times[0]),
                max(float(times[-1]), float(times[0]) + 0.001),
                *extent,
            ],
            vmin=decibels.max() - 75,
            vmax=decibels.max(),
        )
        axis.set(
            title=title,
            xlabel="Время, с",
            ylabel="Частота, Гц" if axis is axes[2] else "Номер мел-фильтра",
        )
        axis.figure.colorbar(image, ax=axis, label="Мощность, дБ", shrink=0.83)
    for axis in axes:
        axis.grid(alpha=0.15)


def make_figure(wav: np.ndarray) -> Figure:
    """
    Строит исходные 4 графика аудио.

    :param wav: Исходные отсчеты.
    :return: Фигура, которую вызывающая сторона должна закрыть.
    """
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(13.5, 7.3), constrained_layout=True)
    figure.patch.set_facecolor("#ffffff")
    _draw_views(representation(wav), axes.flat)
    return figure


def make_configured_figure(wav: np.ndarray, settings: dict[str, Any]) -> Figure:
    """
    Строит 4 графика с выбранными параметрами.

    :param wav: Исходные отсчеты.
    :param settings: Параметры эксперимента.
    :return: Фигура, которую вызывающая сторона должна закрыть.
    """
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(13.5, 7.3), constrained_layout=True)
    figure.patch.set_facecolor("#ffffff")
    _draw_views(representation_with_settings(wav, settings), axes.flat)
    return figure


def make_comparison_figure(
    wav: np.ndarray, other: np.ndarray, settings: dict[str, Any], names: tuple[str, str]
) -> Figure:
    """
    Сопоставляет 2 записи по одинаковым признакам и масштабам.

    :param wav: Отсчеты 1 записи.
    :param other: Отсчеты 2 записи.
    :param settings: Общие параметры эксперимента.
    :param names: Имена записей для подписей.
    :return: Фигура из 2 рядов по 4 графика.
    """
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 4, figsize=(19, 7.8), constrained_layout=True)
    figure.patch.set_facecolor("#ffffff")
    representations = [
        representation_with_settings(samples, settings) for samples in (wav, other)
    ]
    for row, (info, name) in enumerate(zip(representations, names)):
        _draw_views(info, axes[row])
        axes[row, 0].set_title(f"{name}\n1 - Волновая форма")
    duration = max(len(info["samples"]) for info in representations) / SAMPLE_RATE
    for row in range(2):
        for column in (0, 2, 3):
            axes[row, column].set_xlim(0, duration)
    low = min(axes[row, 1].get_ylim()[0] for row in range(2))
    high = max(axes[row, 1].get_ylim()[1] for row in range(2))
    for row in range(2):
        axes[row, 1].set_ylim(low, high)
    for column in (2, 3):
        maximum = max(
            float(axes[row, column].images[0].get_clim()[1]) for row in range(2)
        )
        for row in range(2):
            axes[row, column].images[0].set_clim(maximum - 75, maximum)
    return figure


def make_signal_figure(
    recordings: tuple[np.ndarray, ...], settings: dict[str, Any], names: tuple[str, ...]
) -> Figure:
    """
    Рисует исходные представления звука отдельно от мел-фильтров.

    :param recordings: Отсчеты выбранных записей.
    :param settings: Общие параметры анализа.
    :param names: Подписи выбранных записей.
    :return: Волны, спектры и спектрограммы с общими шкалами.
    """
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        len(recordings),
        3,
        figsize=(15, 3.6 * len(recordings)),
        squeeze=False,
        sharex="col",
        sharey="col",
        constrained_layout=True,
    )
    representations = [
        representation_with_settings(samples, settings) for samples in recordings
    ]
    duration = max(len(info["samples"]) for info in representations) / SAMPLE_RATE
    for row, (info, name) in enumerate(zip(representations, names)):
        _draw_views(info, axes[row])
        axes[row, 0].set_title(f"{name}\n1 - Волновая форма")
        for column in (0, 2):
            axes[row, column].set_xlim(0, duration)
    maximum = max(float(axis.images[0].get_clim()[1]) for axis in axes[:, 2])
    for axis in axes[:, 2]:
        axis.images[0].set_clim(maximum - 75, maximum)
    return figure


def make_feature_signal_figure(
    wav: np.ndarray, settings: dict[str, Any], name: str
) -> Figure:
    """
    Показывает волну и спектрограмму для подбора признаков.

    :param wav: Отсчеты выбранной записи.
    :param settings: Параметры обработки и признаков.
    :param name: Подпись выбранной записи.
    :return: Фигура с волной и спектрограммой.
    """
    import matplotlib.pyplot as plt

    info = representation_with_settings(wav, settings)
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 3.8), constrained_layout=True)
    duration = len(info["samples"]) / SAMPLE_RATE
    axes[0].plot(
        np.arange(len(info["samples"])) / SAMPLE_RATE,
        info["samples"],
        color="#3458a0",
        linewidth=0.7,
    )
    axes[0].set(
        title=f"{name}\nВолновая форма",
        xlabel="Время, с",
        ylabel="Амплитуда",
        ylim=(-1.05, 1.05),
        xlim=(0, duration),
    )
    decibels = 10 * np.log10(info["power"] + EPS)
    image = axes[1].imshow(
        decibels,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        extent=[
            float(info["times"][0]),
            max(float(info["times"][-1]), float(info["times"][0]) + 0.001),
            0,
            8000,
        ],
        vmin=float(decibels.max()) - 75,
        vmax=float(decibels.max()),
    )
    axes[1].set(
        title="Спектрограмма",
        xlabel="Время, с",
        ylabel="Частота, Гц",
        xlim=(0, duration),
    )
    figure.colorbar(image, ax=axes[1], label="Мощность, дБ", shrink=0.83)
    for axis in axes:
        axis.grid(alpha=0.15)
    return figure


def make_mel_figure(
    recordings: tuple[np.ndarray, ...], settings: dict[str, Any], names: tuple[str, ...]
) -> Figure:
    """
    Связывает форму мел-фильтров с результатом их применения к записям.

    :param recordings: Отсчеты выбранных записей.
    :param settings: Общие параметры анализа.
    :param names: Подписи выбранных записей.
    :return: Банк фильтров и мел-спектрограммы с общей шкалой мощности.
    """
    import matplotlib.pyplot as plt

    options = validate_settings(settings)
    n_mels = options["n_mels"]
    figure = plt.figure(figsize=(12.5, 3.8 * len(recordings)), constrained_layout=True)
    grid = figure.add_gridspec(len(recordings), 2)
    bank_axis = figure.add_subplot(grid[:, 0])
    points = mel_to_hz(np.linspace(hz_to_mel(0), hz_to_mel(8000), n_mels + 2))
    colors = plt.get_cmap("viridis")(np.linspace(0.1, 0.9, n_mels))
    for index, color in enumerate(colors):
        bank_axis.plot(points[index : index + 3], [0, 1, 0], color=color, linewidth=0.9)
    bank_axis.set(
        title=f"Банк мел-фильтров: {n_mels}",
        xlabel="Частота, Гц",
        ylabel="Относительный вес в фильтре",
        xlim=(0, 8000),
        ylim=(0, 1.05),
    )
    bank_axis.grid(alpha=0.15)
    representations = [
        representation_with_settings(samples, options) for samples in recordings
    ]
    powers = [10 * np.log10(info["mel_power"] + EPS) for info in representations]
    maximum = max(float(power.max()) for power in powers)
    duration = max(len(info["samples"]) for info in representations) / SAMPLE_RATE
    mel_axes = []
    for row, (info, power, name) in enumerate(zip(representations, powers, names)):
        axis = figure.add_subplot(grid[row, 1])
        image = axis.imshow(
            power,
            origin="lower",
            aspect="auto",
            cmap="viridis",
            extent=[
                float(info["times"][0]),
                max(float(info["times"][-1]), float(info["times"][0]) + 0.001),
                0,
                n_mels,
            ],
            vmin=maximum - 75,
            vmax=maximum,
        )
        axis.set(
            title=f"{name}\nМел-спектрограмма",
            xlabel="Время, с",
            ylabel="Номер мел-фильтра",
            xlim=(0, duration),
        )
        mel_axes.append(axis)
    figure.colorbar(image, ax=mel_axes, label="Мощность, дБ", shrink=0.83)
    return figure


def suppress_noise(
    wav: np.ndarray, strength: float = 1.0, n_fft: int = N_FFT
) -> np.ndarray:
    """
    Подавляет стационарный шум мягким спектральным фильтром.

    :param wav: Исходные отсчеты.
    :param strength: Сила подавления от 0 до 2.
    :param n_fft: Размер окна преобразования Фурье.
    :return: Обработанные отсчеты исходной длины.
    """
    if not np.isfinite(strength) or not 0 <= strength <= 2:
        raise ValueError("Сила шумоподавления должна быть от 0 до 2.")
    if n_fft not in (256, 512, 1024, 2048):
        raise ValueError("n_fft: допустимы 256, 512, 1024 или 2048.")
    if strength == 0:
        return wav.copy()
    _, _, spectrum = stft(wav, fs=SAMPLE_RATE, nperseg=n_fft, noverlap=3 * n_fft // 4)
    power = np.abs(spectrum) ** 2
    quiet_frames = power.mean(axis=0) <= np.quantile(power.mean(axis=0), 0.2)
    noise_power = np.mean(power[:, quiet_frames], axis=1, keepdims=True)
    gain = np.maximum(0.08, 1.0 - strength * noise_power / (power + EPS))
    gain = uniform_filter1d(gain, size=3, axis=1, mode="nearest")
    _, processed = istft(
        spectrum * gain, fs=SAMPLE_RATE, nperseg=n_fft, noverlap=3 * n_fft // 4
    )
    return np.clip(processed[: wav.size], -1, 1).astype(np.float32)


def make_preprocessing_figure(
    wav: np.ndarray, processed: np.ndarray, n_fft: int
) -> Figure:
    """
    Сравнивает волну и спектрограмму до и после обработки.

    :param wav: Исходные отсчеты.
    :param processed: Обработанные отсчеты.
    :param n_fft: Размер окна преобразования Фурье.
    :return: Фигура с общими шкалами для 2 записей.
    """
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(13.5, 7.3), constrained_layout=True)
    spectra = [
        stft(samples, fs=SAMPLE_RATE, nperseg=n_fft, noverlap=3 * n_fft // 4)
        for samples in (wav, processed)
    ]
    maximum = max(
        float((10 * np.log10(np.abs(spec) ** 2 + EPS)).max()) for _, _, spec in spectra
    )
    amplitude_limit = (
        max(0.005, float(np.max(np.abs(wav))), float(np.max(np.abs(processed)))) * 1.1
    )
    for column, (samples, title, (frequencies, times, spectrum)) in enumerate(
        zip((wav, processed), ("Исходное аудио", "После шумоподавления"), spectra)
    ):
        axes[0, column].plot(
            np.arange(samples.size) / SAMPLE_RATE,
            samples,
            color="#3458a0",
            linewidth=0.7,
        )
        axes[0, column].set(
            title=f"{title}: волновая форма",
            xlabel="Время, с",
            ylabel="Амплитуда",
            ylim=(-amplitude_limit, amplitude_limit),
            xlim=(0, wav.size / SAMPLE_RATE),
        )
        image = axes[1, column].pcolormesh(
            times,
            frequencies,
            10 * np.log10(np.abs(spectrum) ** 2 + EPS),
            shading="auto",
            cmap="viridis",
            vmin=maximum - 75,
            vmax=maximum,
        )
        axes[1, column].set(
            title=f"{title}: спектрограмма",
            xlabel="Время, с",
            ylabel="Частота, Гц",
            xlim=(0, wav.size / SAMPLE_RATE),
        )
        figure.colorbar(image, ax=axes[1, column], label="Мощность, дБ", shrink=0.83)
    return figure


def encode_wav(samples: np.ndarray) -> bytes:
    """
    Кодирует отсчеты в PCM WAV для прослушивания в браузере.

    :param samples: Отсчеты монофонического аудио.
    :return: Байты WAV PCM 16-bit с частотой 16000 Гц.
    """
    output = io.BytesIO()
    with wave.open(output, "wb") as recording:
        recording.setnchannels(1)
        recording.setsampwidth(2)
        recording.setframerate(SAMPLE_RATE)
        recording.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
    return output.getvalue()


def plot_wav(path: Path, target: Path) -> None:
    """
    Сохраняет исходные 4 графика WAV в PNG.

    :param path: Путь исходного WAV.
    :param target: Путь PNG.
    """
    import matplotlib.pyplot as plt

    figure = make_figure(load_wav(path))
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=150)
    plt.close(figure)
