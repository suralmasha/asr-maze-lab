"""Построение четырех графиков WAV с частотой 16000 Гц без браузера."""

import argparse
import logging
from pathlib import Path

from features import plot_wav

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description="WAV: waveform, spectrum, spectrogram, mel spectrogram"
    )
    parser.add_argument("wav", type=Path, help="Source PCM WAV, 16000 Hz, mono, 16-bit")
    parser.add_argument(
        "--out", type=Path, default=Path("plots.png"), help="Output PNG file"
    )
    args = parser.parse_args()
    plot_wav(args.wav, args.out)
    logger.info("Created audio plots", extra={"path": str(args.out.resolve())})
