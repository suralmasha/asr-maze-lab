"use strict";

const audioSettings = {
  melCounts: [8, 16, 32, 64, 128],
  mfccCounts: [4, 8, 13, 20, 32],
  fftSizes: [256, 512, 1024, 2048],
  hopSizes: [64, 128, 256, 512],
  defaults: {n_mels: 32, n_mfcc: 13, n_fft: 512, hop_length: 128, denoise_strength: 0, denoise_n_fft: 512},
  storageKey: "asr-maze-audio-settings-v1",
  preprocessingKey: "asr-maze-preprocessing-v1",

  load() {
    try {
      const settings = JSON.parse(localStorage.getItem(this.storageKey));
      if (settings && this.melCounts.includes(settings.n_mels) && this.fftSizes.includes(settings.n_fft)) {
        return {
          n_mels: settings.n_mels, n_fft: settings.n_fft,
          n_mfcc: this.mfccCounts.includes(settings.n_mfcc) && settings.n_mfcc <= settings.n_mels ? settings.n_mfcc : Math.min(13, settings.n_mels),
          hop_length: this.hopSizes.includes(settings.hop_length) && settings.hop_length <= settings.n_fft ? settings.hop_length : settings.n_fft / 4,
        };
      }
    } catch { /* При недоступном хранилище остаются настройки на странице. */ }
    return null;
  },

  save(settings) {
    try {
      const {n_mels, n_mfcc, n_fft, hop_length} = settings;
      localStorage.setItem(this.storageKey, JSON.stringify({n_mels, n_mfcc, n_fft, hop_length}));
      return true;
    } catch { return false; }
  },

  loadPreprocessing() {
    try {
      const settings = JSON.parse(localStorage.getItem(this.preprocessingKey));
      if (settings && Number.isFinite(settings.denoise_strength) && settings.denoise_strength >= 0 && settings.denoise_strength <= 2 && this.fftSizes.includes(settings.denoise_n_fft)) return {denoise_strength: settings.denoise_strength, denoise_n_fft: settings.denoise_n_fft};
    } catch { /* При недоступном хранилище остаются исходные записи. */ }
    return null;
  },

  savePreprocessing(settings) {
    try {
      localStorage.setItem(this.preprocessingKey, JSON.stringify(settings));
      return true;
    } catch { return false; }
  },

};
