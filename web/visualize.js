"use strict";
const labelInput = document.getElementById("label");
const fileInput = document.getElementById("filename");
const plot = document.getElementById("plot");
const melPlot = document.getElementById("mel-plot");
const plotStatus = document.getElementById("plot-status");
const audio = document.getElementById("sample-audio");
let recordings = {};
let redrawTimer = null;
let requestNumber = 0;
let activeController = null;
let imageUrls = [];
let preprocessing = null;
const sliderIds = {
  n_mels: "n-mels",
  n_mfcc: "n-mfcc",
  n_fft: "n-fft",
  hop_length: "hop-length",
};

function readSettings() {
  return Object.fromEntries(
    Object.entries(sliderIds).map(([key, id]) => {
      const input = document.getElementById(id);
      return [key, JSON.parse(input.dataset.values)[Number(input.value)]];
    }),
  );
}

function setSlider(id, values, selected) {
  const input = document.getElementById(id);
  const nearest = values.reduce((closest, value) =>
    Math.abs(value - selected) < Math.abs(closest - selected) ? value : closest,
  );
  input.dataset.values = JSON.stringify(values);
  input.max = values.length - 1;
  input.value = values.indexOf(nearest);
  const label = ["n-fft", "hop-length"].includes(id)
    ? `${nearest} отсчетов · ${(nearest / 16).toLocaleString("ru-RU")} мс`
    : String(nearest);
  input.setAttribute("aria-valuetext", label);
  document.getElementById(`${id}-value`).value = label;
  const ticks = document.getElementById(`${id}-ticks`);
  ticks.replaceChildren();
  for (const value of values) {
    const tick = document.createElement("span");
    tick.textContent = value;
    ticks.append(tick);
  }
  return nearest;
}

function configureSliders(settings) {
  const nMels = setSlider("n-mels", audioSettings.melCounts, settings.n_mels);
  setSlider(
    "n-mfcc",
    audioSettings.mfccCounts.filter(
      (value) => value <= Math.min(nMels, labConfig.limits.n_mfcc[1]),
    ),
    settings.n_mfcc,
  );
  const nFft = setSlider("n-fft", audioSettings.fftSizes, settings.n_fft);
  setSlider(
    "hop-length",
    audioSettings.hopSizes.filter((value) => value <= nFft),
    settings.hop_length,
  );
}

function showSettingsStatus(applied) {
  const settings = readSettings();
  document.getElementById("audio-settings-status").textContent =
    `${settings.n_mels} мел-фильтров, ${settings.n_mfcc} MFCC (${2 * settings.n_mfcc} признаков), окно ${settings.n_fft}, шаг ${settings.hop_length}. ${applied ? "Применено ко всем записям для обучения и распознавания." : "Это предварительный просмотр. Нажми «Применить», чтобы использовать эти признаки дальше."}`;
}

async function redraw() {
  if (!preprocessing) return;
  const request = ++requestNumber;
  activeController?.abort();
  plotStatus.hidden = false;
  document.getElementById("plot-box").setAttribute("aria-busy", "false");
  const filename = fileInput.value;
  if (!filename) {
    plot.hidden = true;
    melPlot.hidden = true;
    plotStatus.textContent =
      "Пока здесь пусто. Вернись на этап «Данные» и запиши эту команду.";
    audio.removeAttribute("src");
    audio.load();
    return;
  }
  const sampleParams = new URLSearchParams({
    label: labelInput.value,
    file: filename,
  });
  const nextAudio =
    preprocessing.denoise_strength > 0 || preprocessing.trim_silence
      ? `/api/preprocess/audio?${sampleParams}&strength=${preprocessing.denoise_strength}&n_fft=${preprocessing.denoise_n_fft}&trim_silence=${preprocessing.trim_silence}`
      : `/api/audio?${sampleParams}`;
  if (audio.getAttribute("src") !== nextAudio) {
    audio.src = nextAudio;
    audio.load();
  }
  for (const [key, value] of Object.entries({
    ...readSettings(),
    ...preprocessing,
  }))
    sampleParams.set(key, value);
  plotStatus.textContent = "Рисую графики твоей записи…";
  document.getElementById("plot-box").setAttribute("aria-busy", "true");
  activeController = new AbortController();
  try {
    const responses = await Promise.all(
      ["features", "mel"].map((view) =>
        fetch(`/api/plot?${sampleParams}&view=${view}`, {
          signal: activeController.signal,
        }),
      ),
    );
    for (const response of responses) {
      if (!response.ok)
        throw new Error(
          (await response.json()).error ||
            "Не получилось построить графики. Выбери запись еще раз.",
        );
    }
    const blobs = await Promise.all(
      responses.map((response) => response.blob()),
    );
    if (request !== requestNumber) return;
    imageUrls.forEach((url) => URL.revokeObjectURL(url));
    imageUrls = blobs.map((blob) => URL.createObjectURL(blob));
    plot.src = imageUrls[0];
    melPlot.src = imageUrls[1];
    await Promise.all([plot.decode(), melPlot.decode()]);
    if (request !== requestNumber) return;
    plot.hidden = false;
    melPlot.hidden = false;
    plotStatus.hidden = true;
  } catch (error) {
    if (error.name === "AbortError" || request !== requestNumber) return;
    plotStatus.textContent = error.message;
  } finally {
    if (request === requestNumber)
      document.getElementById("plot-box").setAttribute("aria-busy", "false");
  }
}

function choose() {
  const old = fileInput.value;
  fileInput.replaceChildren();
  for (const filename of recordings[labelInput.value] || []) {
    const option = document.createElement("option");
    option.value = filename;
    option.textContent = filename;
    fileInput.add(option);
  }
  if ((recordings[labelInput.value] || []).includes(old)) fileInput.value = old;
  fileInput.disabled = !fileInput.options.length;
  void redraw();
}

labelInput.addEventListener("change", choose);
fileInput.addEventListener("change", redraw);
for (const id of Object.values(sliderIds))
  document.getElementById(id).addEventListener("input", () => {
    configureSliders(readSettings());
    showSettingsStatus(false);
    document.getElementById("plot-box").setAttribute("aria-busy", "true");
    plotStatus.hidden = false;
    plotStatus.textContent = "Обновляю графики с твоими параметрами…";
    clearTimeout(redrawTimer);
    redrawTimer = setTimeout(redraw, 180);
  });
document
  .getElementById("apply-features")
  .addEventListener("click", async () => {
    const button = document.getElementById("apply-features");
    button.disabled = true;
    try {
      await audioSettings.save(readSettings());
      showSettingsStatus(true);
    } catch (error) {
      document.getElementById("audio-settings-status").textContent =
        error.message;
    } finally {
      button.disabled = false;
    }
  });

async function initialize() {
  const controls = [
    ...document.querySelectorAll(
      ".plot-panel input, .plot-panel select, .plot-panel button",
    ),
  ];
  controls.forEach((control) => {
    control.disabled = true;
  });
  document.getElementById("apply-features").disabled = true;
  try {
    await audioSettings.init();
    preprocessing = audioSettings.loadPreprocessing();
    configureSliders(audioSettings.load());
    showSettingsStatus(true);
    document.getElementById("preprocessing-note").textContent =
      `Примененная обработка: шумоподавление ${preprocessing.denoise_strength}, окно ${preprocessing.denoise_n_fft}; обрезка тишины ${preprocessing.trim_silence ? "включена" : "выключена"}. Здесь ты видишь и слышишь этот вариант записи.`;
    recordings = await audioSettings.request("/api/samples");
    controls.forEach((control) => {
      control.disabled = false;
    });
    choose();
    await audioSettings.showCache(
      "features",
      labelInput.value,
      fileInput.value,
    );
  } catch (error) {
    plotStatus.textContent = error.message;
    document.getElementById("plot-box").setAttribute("aria-busy", "false");
  }
}
void initialize();
