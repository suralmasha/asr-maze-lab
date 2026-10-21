"""Build four audio plots for any 16kHz, 16bit, mono WAV (without the browser)."""
import argparse
from pathlib import Path
from features import plot_wav

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WAV → waveform, spectrum, spectrogram, mel spectrogram")
    parser.add_argument("wav", type=Path, help="Source PCM WAV, 16000 Hz, mono, 16-bit")
    parser.add_argument("--out", type=Path, default=Path("plots.png"), help="Output PNG file")
    args = parser.parse_args()
    plot_wav(args.wav, args.out)
    print(f"Готово: {args.out.resolve()}")
