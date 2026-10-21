"""Shared audio processing for recording plots, training and inference.

Dependencies: numpy, scipy, matplotlib (matplotlib is only imported by the plot function).
All recordings use mono, 16-bit PCM WAV, 16 kHz, 0.4–5 seconds.
"""
from __future__ import annotations

import io
import wave
from pathlib import Path

import numpy as np
from scipy.fft import dct
from scipy.ndimage import uniform_filter1d
from scipy.signal import stft

SAMPLE_RATE = 16_000
LABELS = {"up": "Вверх", "down": "Вниз", "left": "Влево", "right": "Вправо"}
N_FFT = 512
N_MELS = 32
N_MFCC = 13
EPS = 1e-10


def load_wav(data: bytes | Path | str) -> np.ndarray:
    """Validate WAV on disk/in memory and return samples as float32 [-1, 1]."""
    if isinstance(data, bytes):
        inp = io.BytesIO(data)
    else:
        inp = str(data)
    try:
        with wave.open(inp, "rb") as f:
            channels = f.getnchannels()
            width = f.getsampwidth()
            rate = f.getframerate()
            encoding = f.getcomptype()
            frames = f.getnframes()
            if (channels, width, rate, encoding) != (1, 2, SAMPLE_RATE, "NONE"):
                raise ValueError("Нужен WAV PCM 16-bit, mono, 16000 Гц.")
            if not int(0.4 * rate) <= frames <= int(5.5 * rate):
                raise ValueError("Запиши слово длиной примерно 0.4–5 секунд.")
            raw = f.readframes(frames)
    except (wave.Error, EOFError) as e:
        raise ValueError("Не получилось прочитать WAV-файл.") from e
    if len(raw) != frames * 2:
        raise ValueError("WAV-файл неполный.")
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    rms = np.sqrt(np.mean(x * x))
    if rms < 0.0015:
        raise ValueError("Почти тишина. Подойди ближе к микрофону и повтори.")
    return x


def trim_silence(x: np.ndarray) -> np.ndarray:
    """Remove leading/trailing near-silence while retaining a 120 ms context margin."""
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
    return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)


def mel_to_hz(m: np.ndarray | float) -> np.ndarray:
    return 700.0 * (10.0 ** (np.asarray(m) / 2595.0) - 1.0)


def mel_filterbank() -> np.ndarray:
    """HTK-style triangular filters; also used by the visualization."""
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / SAMPLE_RATE)
    points = mel_to_hz(np.linspace(hz_to_mel(0), hz_to_mel(8000), N_MELS + 2))
    bank = np.zeros((N_MELS, len(freqs)), dtype=np.float32)
    for i in range(N_MELS):
        left, middle, right = points[i : i + 3]
        bank[i] = np.maximum(0, np.minimum((freqs - left) / (middle - left), (right - freqs) / (right - middle)))
        bank[i] /= max(float(bank[i].sum()), EPS)
    return bank


MEL_BANK = mel_filterbank()


def representation(wav: np.ndarray) -> dict:
    """Compute magnitude spectrum, STFT, log-mel and MFCC consistently."""
    x = trim_silence(wav)
    x = x / max(float(np.max(np.abs(x))), 0.001)
    if x.size < 400:
        x = np.pad(x, (0, 400 - x.size))
    freq, times, complex_spec = stft(
        x, fs=SAMPLE_RATE, window="hann", nperseg=400, noverlap=240,
        nfft=N_FFT, boundary=None, padded=False,
    )
    power = np.abs(complex_spec) ** 2
    mel_power = MEL_BANK @ power
    log_mel = np.log(mel_power + EPS)
    mfcc = dct(log_mel, type=2, axis=0, norm="ortho")[:N_MFCC]
    return {"samples": x, "freq": freq, "times": times, "power": power,
            "mel_power": mel_power, "log_mel": log_mel, "mfcc": mfcc}


def extract_features(wav: np.ndarray) -> np.ndarray:
    """26-dimensional MFCC statistics for an intentionally simple k-NN model."""
    coefs = representation(wav)["mfcc"]
    return np.concatenate([coefs.mean(axis=1), coefs.std(axis=1)]).astype(np.float32)


def make_figure(wav: np.ndarray):
    """Create four accurate, labeled views. Caller is responsible for plt.close(fig)."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    info = representation(wav)
    x, sr = info["samples"], SAMPLE_RATE
    time_axis = np.arange(x.size) / sr
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 7.3), constrained_layout=True)
    fig.patch.set_facecolor("#ffffff")
    ax = axes[0, 0]
    ax.plot(time_axis, x, color="#3458a0", linewidth=0.7)
    ax.set(title="1 · Волновая форма", xlabel="Время, с", ylabel="Амплитуда", ylim=(-1.05, 1.05))

    ax = axes[0, 1]
    mean_power = info["power"].mean(axis=1)
    ax.plot(info["freq"], 10 * np.log10(mean_power + EPS), color="#ce6d35", linewidth=1.1)
    ax.set(title="2 · Усреднённый спектр", xlabel="Частота, Гц", ylabel="Мощность, дБ")
    ax.set_xlim(0, 8000)

    ax = axes[1, 0]
    db = 10 * np.log10(info["power"] + EPS)
    img = ax.pcolormesh(info["times"], info["freq"], db, shading="auto", cmap="magma", vmin=db.max()-75, vmax=db.max())
    ax.set(title="3 · Спектрограмма (STFT)", xlabel="Время, с", ylabel="Частота, Гц")
    fig.colorbar(img, ax=ax, label="Относительная мощность, дБ", shrink=0.83)

    ax = axes[1, 1]
    mel_db = 10 * np.log10(info["mel_power"] + EPS)
    im = ax.imshow(mel_db, origin="lower", aspect="auto", cmap="viridis",
                   extent=[info["times"][0], info["times"][-1], 0, N_MELS],
                   vmin=mel_db.max()-75, vmax=mel_db.max())
    ax.set(title="4 · Мел-спектрограмма", xlabel="Время, с", ylabel="Мел-фильтр (номер)")
    fig.colorbar(im, ax=ax, label="Относительная мощность, дБ", shrink=0.83)
    for axis in axes.flat:
        axis.grid(alpha=0.15)
    return fig


def plot_wav(path: Path, target: Path) -> None:
    import matplotlib.pyplot as plt
    fig = make_figure(load_wav(path))
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=150)
    plt.close(fig)
