"use strict";
const $ = (id) => document.getElementById(id);
let recordings = {};
let redrawTimer = null;
let requestNumber = 0;
let activeController = null;
let imageUrl = null;

function updateStrengthLabel() {
  const strength = Number($("strength").value);
  let level;
  if (strength === 0) level = "отключено";
  else if (strength < 1) level = "слабое";
  else if (strength === 1) level = "среднее";
  else if (strength < 2) level = "сильное";
  else level = "очень сильное";
  const label = `${strength.toLocaleString("ru-RU")} · ${level}`;
  $("strength-value").value = label;
  $("strength").setAttribute("aria-valuetext", label);
}

function choose() {
  const input = $("filename");
  const old = input.value;
  const files = recordings[$("label").value] || [];
  input.replaceChildren();
  for (const filename of files) {
    const option = document.createElement("option");
    option.value = filename;
    option.textContent = filename;
    input.add(option);
  }
  if (files.includes(old)) input.value = old;
  input.disabled = !files.length;
  void redraw();
}

async function redraw() {
  const request = ++requestNumber;
  activeController?.abort();
  $("plot").hidden = true;
  $("plot-status").hidden = false;
  $("plot-box").setAttribute("aria-busy", "false");
  $("processed-audio").pause();
  $("processed-audio").removeAttribute("src");
  $("processed-audio").load();
  if (!$("filename").value) {
    $("plot-status").textContent =
      "Пока здесь пусто. Вернись на этап «Данные» и запиши эту команду.";
    $("original-audio").removeAttribute("src");
    $("original-audio").load();
    return;
  }
  const params = new URLSearchParams({
    label: $("label").value,
    file: $("filename").value,
  });
  const originalUrl = `/api/audio?${params}`;
  if ($("original-audio").getAttribute("src") !== originalUrl) {
    $("original-audio").src = originalUrl;
    $("original-audio").load();
  }
  params.set("strength", $("strength").value);
  params.set("n_fft", audioSettings.loadPreprocessing().denoise_n_fft);
  params.set("trim_silence", $("trim-silence").checked);
  $("plot-status").textContent = "Убираю шум и рисую графики…";
  $("plot-box").setAttribute("aria-busy", "true");
  activeController = new AbortController();
  try {
    const response = await fetch(`/api/preprocess?${params}`, {
      signal: activeController.signal,
    });
    if (!response.ok)
      throw new Error(
        (await response.json()).error ||
          "Не получилось обработать запись. Попробуй выбрать ее еще раз.",
      );
    const blob = await response.blob();
    if (request !== requestNumber) return;
    if (imageUrl) URL.revokeObjectURL(imageUrl);
    imageUrl = URL.createObjectURL(blob);
    $("plot").onload = () => {
      if (request !== requestNumber) return;
      $("plot").hidden = false;
      $("plot-status").hidden = true;
      $("plot-box").setAttribute("aria-busy", "false");
      $("processed-audio").src = `/api/preprocess/audio?${params}`;
      $("processed-audio").load();
    };
    $("plot").onerror = () => {
      $("plot-status").textContent =
        "Не получилось показать графики. Попробуй выбрать запись еще раз.";
      $("plot-box").setAttribute("aria-busy", "false");
    };
    $("plot").src = imageUrl;
  } catch (error) {
    if (error.name === "AbortError" || request !== requestNumber) return;
    $("plot-status").textContent = error.message;
    $("plot-box").setAttribute("aria-busy", "false");
  }
}

$("label").addEventListener("change", choose);
$("filename").addEventListener("change", redraw);
function markUnapplied() {
  $("preprocessing-status").textContent =
    "Это предварительный просмотр. Нажми «Применить», чтобы использовать эти настройки для всех записей дальше.";
}

$("trim-silence").addEventListener("change", () => {
  markUnapplied();
  void redraw();
});
$("strength").addEventListener("input", () => {
  markUnapplied();
  updateStrengthLabel();
  clearTimeout(redrawTimer);
  redrawTimer = setTimeout(redraw, 180);
});
$("apply-preprocessing").addEventListener("click", async () => {
  const button = $("apply-preprocessing");
  button.disabled = true;
  const settings = {
    denoise_strength: Number($("strength").value),
    denoise_n_fft: audioSettings.loadPreprocessing().denoise_n_fft,
    trim_silence: $("trim-silence").checked,
  };
  try {
    await audioSettings.savePreprocessing(settings);
    $("preprocessing-status").textContent =
      `Для всех записей применено: шумоподавление ${settings.denoise_strength}, обрезка тишины ${settings.trim_silence ? "включена" : "выключена"}. Настройки сохранены на сервере для признаков, обучения и распознавания.`;
  } catch (error) {
    $("preprocessing-status").textContent = error.message;
  } finally {
    button.disabled = false;
  }
});

async function initialize() {
  updateStrengthLabel();
  $("trim-silence").checked = audioSettings.defaults.trim_silence;
  const controls = [
    ...document.querySelectorAll(
      ".plot-panel input, .plot-panel select, .plot-panel button",
    ),
  ];
  controls.forEach((control) => {
    control.disabled = true;
  });
  $("apply-preprocessing").disabled = true;
  try {
    await audioSettings.init();
    const applied = audioSettings.loadPreprocessing();
    $("strength").value = applied.denoise_strength;
    $("trim-silence").checked = applied.trim_silence;
    updateStrengthLabel();
    $("preprocessing-status").textContent = "";
    recordings = await audioSettings.request("/api/samples");
    controls.forEach((control) => {
      control.disabled = false;
    });
    choose();
    await audioSettings.showCache(
      "preprocess",
      $("label").value,
      $("filename").value,
    );
  } catch (error) {
    $("plot-status").textContent = error.message;
    $("plot-box").setAttribute("aria-busy", "false");
  }
}
void initialize();
