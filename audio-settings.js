"use strict";

const audioSettings = {
  melCounts: [8, 16, 32, 64, 128],
  mfccCounts: [4, 8, 13, 20, 32],
  fftSizes: [256, 512, 1024, 2048],
  hopSizes: [64, 128, 256, 512],
  defaults: {n_mels: 32, n_mfcc: 13, n_fft: 512, hop_length: 128, denoise_strength: 0, denoise_n_fft: 512, trim_silence: true},
  settings: null,
  mode: "on-demand",

  async request(url, init) {
    const response = await fetch(url, {cache: "no-store", ...init});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Не получилось выполнить запрос. Попробуй еще раз.");
    return data;
  },

  async init() {
    const data = await this.request("/api/settings");
    this.settings = data.settings;
    this.mode = data.mode;
    return this.settings;
  },

  load() {
    if (!this.settings) return null;
    const {n_mels, n_mfcc, n_fft, hop_length} = this.settings;
    return {n_mels, n_mfcc, n_fft, hop_length};
  },

  async save(settings) {
    const data = await this.request("/api/settings", {
      method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(settings),
    });
    this.settings = data.settings;
    this.mode = data.mode;
  },

  loadPreprocessing() {
    if (!this.settings) return null;
    const {denoise_strength, denoise_n_fft, trim_silence} = this.settings;
    return {denoise_strength, denoise_n_fft, trim_silence};
  },

  async savePreprocessing(settings) {
    await this.save(settings);
  },

  async showCache(page, label, file) {
    const panel = document.createElement("section");
    panel.className = "cache-panel";
    panel.setAttribute("aria-label", "Режим вычислений");
    const status = document.createElement("p");
    status.setAttribute("role", "status");
    status.setAttribute("aria-live", "polite");
    status.textContent = "Расчет по запросу: результаты графиков не сохраняются.";
    panel.append(status);
    document.querySelector(".hero").after(panel);
    if (this.mode !== "precompute") return;
    const button = document.createElement("button");
    button.className = "action secondary";
    button.type = "button";
    button.textContent = "Очистить кеш";
    panel.append(button);
    let cleared = false;
    let checking = false;
    let stopped = false;
    const refresh = async () => {
      if (checking || stopped) return;
      checking = true;
      try {
        const data = await this.request("/api/cache/status");
        const states = {idle: "Ожидание", running: "Готовлю варианты в фоне", full: "Кеш заполнен, фон остановлен", paused: "Фон остановлен", error: "Ошибка предрасчета"};
        status.textContent = `Предрасчет: ${states[data.state] || data.state}. Кеш: ${(data.bytes / 1e6).toFixed(1)} из ${(data.limit / 1e6).toFixed(0)} МБ. Готово результатов: ${data.completed || 0}${data.total ? ` из ${data.total}` : ""}.${data.error ? ` ${data.error}` : ""}`;
      } catch (error) {status.textContent = error.message;}
      finally {checking = false;}
    };
    button.addEventListener("click", async () => {
      cleared = true;
      button.disabled = true;
      try {
        await this.request("/api/cache/clear", {method: "POST"});
        await refresh();
      } catch (error) {status.textContent = error.message;}
      finally {button.disabled = false;}
    });
    await refresh();
    if (page && file && !cleared) {
      try {
        await this.request("/api/cache/warm", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({page, label, file})});
        await refresh();
      } catch (error) {status.textContent = error.message;}
    }
    const timer = setInterval(refresh, 1800);
    window.addEventListener("pagehide", () => {stopped = true; clearInterval(timer);}, {once: true});
  },
};
