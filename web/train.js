"use strict";
const $ = (id) => document.getElementById(id);
const labels = {
  up: "↑ Вверх",
  down: "↓ Вниз",
  left: "← Влево",
  right: "→ Вправо",
};
const palette = {
  up: "#2659ce",
  down: "#a35a05",
  left: "#087763",
  right: "#914a9d",
};
const recorder = new VoiceRecorder();
let busy = false;
let probeBusy = false;
let modelReady = false;
let pollTimer = null;
let sampleCounts = {};
let countsLoaded = false;
let projectionData = null;
let projectionRequest = 0;
let projectionNeedsRefresh = true;
let selectedPoint = 0;
let resizeTimer = null;
let featureSettings = { ...audioSettings.defaults };

async function api(url, init) {
  const response = await fetch(url, { cache: "no-store", ...init });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}

function readSettings() {
  return {
    ...featureSettings,
    k: Number($("k").value),
    normalize: $("normalize").checked,
    metric: $("metric").value,
    weights: $("weights").value,
    samples_per_class: Number($("samples-per-class").value),
    noise: Number($("noise").value),
  };
}

function applySettings(settings) {
  if (!settings) return;
  const fields = {
    k: "k",
    metric: "metric",
    weights: "weights",
    samples_per_class: "samples-per-class",
  };
  for (const key of Object.keys(featureSettings))
    if (settings[key] !== undefined) featureSettings[key] = settings[key];
  for (const [key, id] of Object.entries(fields)) {
    if (settings[key] !== undefined) $(id).value = settings[key];
  }
  if (settings.normalize !== undefined)
    $("normalize").checked = Boolean(settings.normalize);
  if (settings.noise !== undefined) {
    const choices = labConfig.noise_levels;
    $("noise").value = String(
      choices.reduce((closest, value) =>
        Math.abs(value - settings.noise) < Math.abs(closest - settings.noise)
          ? value
          : closest,
      ),
    );
  }
  updateSettingLabels();
}

function updateSettingLabels() {
  const perClass = Number($("samples-per-class").value);
  const sourceCount = Object.keys(labels).reduce(
    (total, label) =>
      total +
      Math.min(perClass, labConfig.audio.max_samples, sampleCounts[label] || 0),
    0,
  );
  const count = sourceCount * (Number($("noise").value) > 0 ? 2 : 1);
  const [minimumK, maximumK] = labConfig.limits.k;
  $("k").max = countsLoaded
    ? Math.max(minimumK, Math.min(maximumK, count))
    : maximumK;
  if (Number($("k").value) > Number($("k").max)) $("k").value = $("k").max;
  $("k-value").value = $("k").value;
  $("samples-value").value =
    perClass === labConfig.audio.max_samples
      ? `Все, до ${labConfig.audio.max_samples}`
      : `До ${perClass}`;
  $("training-size").textContent = countsLoaded
    ? `Для обучения возьмем ${sourceCount} твоих WAV. Обучающих примеров: ${count}, признаков: ${featureSettings.n_mfcc * 2}. Соседей можно выбрать до ${Math.min(maximumK, count)}.`
    : "Считаю твои записи…";
  $("applied-settings").textContent =
    `Для всех записей: ${featureSettings.n_mels} мел-фильтров, ${featureSettings.n_mfcc} MFCC, окно ${featureSettings.n_fft}, шаг ${featureSettings.hop_length}. Шумоподавление: ${featureSettings.denoise_strength > 0 ? `сила ${featureSettings.denoise_strength}, окно ${featureSettings.denoise_n_fft}` : "выключено"}. Обрезка тишины: ${featureSettings.trim_silence ? "включена" : "выключена"}. Изменить их можно на этапах 04 и 05 кнопкой «Применить».`;
  updateActions();
}

function updateActions() {
  const enoughSamples =
    countsLoaded &&
    Object.keys(labels).every(
      (label) => (sampleCounts[label] || 0) >= labConfig.audio.min_samples,
    );
  $("train-button").disabled =
    busy || probeBusy || recorder.isRecording || !enoughSamples;
  $("training-settings").disabled = busy || probeBusy || recorder.isRecording;
  $("probe-button").disabled = busy || probeBusy || !modelReady;
  $("probe-denoise").disabled = busy || probeBusy || recorder.isRecording;
  for (const id of [
    "select-model",
    "delete-model",
    "clear-models",
    "model-select",
  ])
    $(id).disabled =
      busy ||
      probeBusy ||
      recorder.isRecording ||
      !$("model-select").options.length;
}

async function updateCounts() {
  const data = await api("/api/samples");
  $("counts").replaceChildren();
  for (const [key, name] of Object.entries(labels)) {
    const count = data[key]?.length || 0;
    sampleCounts[key] = count;
    const item = document.createElement("div");
    item.className =
      "training-count" + (count < labConfig.audio.min_samples ? " warn" : "");
    const title = document.createElement("span");
    title.textContent = name;
    const value = document.createElement("strong");
    value.textContent = `${count} WAV`;
    const minimum = document.createElement("span");
    minimum.textContent =
      count < labConfig.audio.min_samples
        ? `Запиши еще: ${labConfig.audio.min_samples - count}`
        : "Можно начинать обучение";
    item.append(title, value, minimum);
    $("counts").append(item);
  }
  countsLoaded = true;
  updateSettingLabels();
}

function describeModel(settings, count) {
  if (!settings) return "";
  const normalization = settings.normalize
    ? "с нормализацией"
    : "без нормализации";
  const metric =
    settings.metric === "manhattan"
      ? "манхэттенское расстояние"
      : "евклидово расстояние";
  const weights =
    settings.weights === "uniform" ? "равные голоса" : "взвешенные голоса";
  return `Твоя модель: k = ${settings.k}, MFCC = ${settings.n_mfcc}, мел-фильтры = ${settings.n_mels}, окно = ${settings.n_fft}, шаг = ${settings.hop_length}, шумоподавление = ${settings.denoise_strength}, ${normalization}, ${metric}, ${weights}${count ? `, ${count} примеров` : ""}.`;
}

function setTrainStatus(state) {
  const wasBusy = busy;
  busy = state.state === "running";
  $("progress").value = state.percent || 0;
  $("progress-percent").textContent = `${state.percent || 0}%`;
  $("progress-stage").textContent =
    {
      idle: "Можно начинать",
      features: "Считаю признаки",
      fit: "Запоминаю примеры",
      save: "Сохраняю модель",
      done: "Готово!",
      error: "Не получилось обучить",
    }[state.stage] || (state.ready ? "Модель готова" : "Ожидание");
  $("train-message").textContent =
    state.message ||
    `Запиши хотя бы по ${labConfig.audio.min_samples} примеров каждой команды, затем нажми «Обучить модель».`;
  $("train-message").classList.toggle("error", state.state === "error");
  modelReady = Boolean(state.ready) && !busy;
  $("model-settings").textContent = describeModel(
    state.model_settings,
    state.training_count,
  );
  if (wasBusy && !busy && state.state === "done") {
    $("settings-note").textContent =
      "Готово! Твои настройки сохранены. Теперь попробуй произнести новую команду.";
    projectionNeedsRefresh = true;
    void refreshModels();
  }
  if (!modelReady && !busy) {
    projectionRequest++;
    projectionData = null;
    $("projection").setAttribute("hidden", "");
    $("projection-selection").hidden = true;
    $("probe-result").hidden = true;
    $("projection-legend").replaceChildren();
    $("projection-status").hidden = false;
    $("projection-status").textContent =
      "Обучи модель или выбери сохраненную, чтобы увидеть карту записей.";
  }
  if (modelReady && !recorder.isRecording && !probeBusy)
    $("probe-status").textContent =
      "Твоя модель готова. Нажми «Произнести команду» и проверь ее на новой записи.";
  updateActions();
  if (modelReady && projectionNeedsRefresh) {
    projectionNeedsRefresh = false;
    void drawProjection();
  }
  if (busy && !pollTimer) pollTimer = setInterval(refreshStatus, 600);
  else if (!busy && pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}

async function refreshStatus() {
  try {
    setTrainStatus(await api("/api/train/status"));
  } catch (error) {
    $("train-message").textContent = error.message;
    $("train-message").classList.add("error");
  }
}

function svgNode(tag, attributes = {}, text) {
  const element = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [key, value] of Object.entries(attributes))
    element.setAttribute(key, String(value));
  if (text !== undefined) element.textContent = text;
  return element;
}

function pointShape(label, x, y, attributes = {}) {
  if (label === "down")
    return svgNode("rect", {
      x: x - 7,
      y: y - 7,
      width: 14,
      height: 14,
      rx: 2,
      ...attributes,
    });
  if (label === "left")
    return svgNode("polygon", {
      points: `${x},${y - 9} ${x - 9},${y + 7} ${x + 9},${y + 7}`,
      ...attributes,
    });
  if (label === "right")
    return svgNode("polygon", {
      points: `${x},${y - 10} ${x - 9},${y} ${x},${y + 10} ${x + 9},${y}`,
      ...attributes,
    });
  return svgNode("circle", { cx: x, cy: y, r: 8, ...attributes });
}

function renderProjection() {
  if (busy || !projectionData?.points?.length) return;
  const chart = $("projection");
  chart.removeAttribute("hidden");
  chart.replaceChildren();
  const width = Math.max(280, chart.parentElement.clientWidth - 2);
  const height = Math.min(440, Math.max(300, width * 0.48));
  chart.setAttribute("viewBox", `0 0 ${width} ${height}`);
  chart.style.height = `${height}px`;
  const left = 65,
    right = width - 24,
    top = 28,
    bottom = height - 64;
  const points = projectionData.points;
  const allPoints = projectionData.probe
    ? [...points, projectionData.probe]
    : points;
  const xs = allPoints.map((point) => Number(point.x)),
    ys = allPoints.map((point) => Number(point.y));
  const xMin = Math.min(...xs),
    xMax = Math.max(...xs),
    yMin = Math.min(...ys),
    yMax = Math.max(...ys);
  const xPad = Math.max((xMax - xMin) * 0.1, 0.5),
    yPad = Math.max((yMax - yMin) * 0.12, 0.5);
  const xPosition = (value) =>
    left + ((value - xMin + xPad) / (xMax - xMin + 2 * xPad)) * (right - left);
  const yPosition = (value) =>
    bottom -
    ((value - yMin + yPad) / (yMax - yMin + 2 * yPad)) * (bottom - top);
  for (let step = 0; step <= 4; step++) {
    const ratio = step / 4;
    const x = left + ratio * (right - left),
      y = bottom - ratio * (bottom - top);
    chart.append(
      svgNode("line", {
        x1: x,
        x2: x,
        y1: top,
        y2: bottom,
        stroke: "var(--border)",
      }),
    );
    chart.append(
      svgNode("line", {
        x1: left,
        x2: right,
        y1: y,
        y2: y,
        stroke: "var(--border)",
      }),
    );
    chart.append(
      svgNode(
        "text",
        { x, y: bottom + 23, "text-anchor": "middle", "font-size": 12 },
        (xMin - xPad + ratio * (xMax - xMin + 2 * xPad)).toFixed(1),
      ),
    );
    chart.append(
      svgNode(
        "text",
        { x: left - 10, y: y + 4, "text-anchor": "end", "font-size": 12 },
        (yMin - yPad + ratio * (yMax - yMin + 2 * yPad)).toFixed(1),
      ),
    );
  }
  chart.append(
    svgNode(
      "text",
      {
        x: (left + right) / 2,
        y: height - 14,
        "text-anchor": "middle",
        "font-size": 13,
      },
      "Главная компонента 1",
    ),
  );
  chart.append(
    svgNode(
      "text",
      {
        transform: `translate(18 ${(top + bottom) / 2}) rotate(-90)`,
        "text-anchor": "middle",
        "font-size": 13,
      },
      "Главная компонента 2",
    ),
  );
  points.forEach((point, index) => {
    const shape = pointShape(
      point.label,
      xPosition(Number(point.x)),
      yPosition(Number(point.y)),
      {
        fill: palette[point.label] || "var(--accent)",
        class: "projection-point",
        "data-index": index,
        role: "button",
        tabindex: index === selectedPoint ? 0 : -1,
        "aria-label": `${point.command || labels[point.label]}: ${point.file}${point.augmented ? ", шумовой дубль" : ""}`,
        "aria-pressed": "false",
      },
    );
    shape.append(
      svgNode(
        "title",
        {},
        `${point.command || labels[point.label]} · ${point.file}${point.augmented ? " · шумовой дубль" : ""}`,
      ),
    );
    shape.addEventListener("click", () => selectPoint(index));
    shape.addEventListener("keydown", (event) => {
      if (
        [
          "ArrowLeft",
          "ArrowDown",
          "ArrowRight",
          "ArrowUp",
          "Home",
          "End",
        ].includes(event.key)
      ) {
        event.preventDefault();
        const next =
          event.key === "Home"
            ? 0
            : event.key === "End"
              ? points.length - 1
              : (index +
                  (["ArrowLeft", "ArrowDown"].includes(event.key) ? -1 : 1) +
                  points.length) %
                points.length;
        selectPoint(next);
        chart.querySelector(`[data-index="${next}"]`)?.focus();
      } else if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        selectPoint(index);
      }
    });
    chart.append(shape);
  });
  if (projectionData.probe) {
    chart.append(
      svgNode(
        "text",
        {
          x: xPosition(Number(projectionData.probe.x)),
          y: yPosition(Number(projectionData.probe.y)) + 9,
          "text-anchor": "middle",
          "font-size": 30,
          style: "fill: var(--error)",
          role: "img",
          "aria-label": "Новая команда",
        },
        "★",
      ),
    );
  }
  selectPoint(Math.min(selectedPoint, points.length - 1));
}

function selectPoint(index) {
  const point = projectionData?.points[index];
  if (!point) return;
  selectedPoint = index;
  $("point-select").value = String(index);
  $("selected-file").textContent =
    `${point.command || labels[point.label]} · ${point.file}`;
  const filename = String(point.audio_file || point.file)
    .split(/[\\/]/)
    .pop();
  const audioUrl = `/api/audio?${new URLSearchParams({ label: point.label, file: filename })}`;
  if ($("selected-audio").getAttribute("src") !== audioUrl) {
    $("selected-audio").src = audioUrl;
    $("selected-audio").load();
  }
  $("selected-note").textContent = point.augmented
    ? "Это шумовая копия. Здесь ты можешь послушать исходный WAV."
    : `Можешь послушать запись ${point.file}.`;
  $("projection")
    .querySelectorAll(".projection-point")
    .forEach((shape) => {
      const selected = Number(shape.dataset.index) === index;
      shape.classList.toggle("selected", selected);
      shape.setAttribute("aria-pressed", String(selected));
      shape.setAttribute("tabindex", selected ? "0" : "-1");
    });
}

async function drawProjection() {
  const request = ++projectionRequest;
  $("projection-status").hidden = false;
  $("projection-status").textContent = "Рисую карту твоих записей…";
  $("projection-box").setAttribute("aria-busy", "true");
  try {
    const data = await api("/api/train/projection");
    if (request !== projectionRequest) return;
    if (!data.points?.length)
      throw new Error(
        "Пока нечего показать. Добавь записи и обучи свою модель.",
      );
    projectionData = data;
    $("point-select").replaceChildren();
    data.points.forEach((point, index) => {
      const option = document.createElement("option");
      option.value = index;
      option.textContent = `${point.command || labels[point.label]} · ${point.file}${point.augmented ? " · шумовой дубль" : ""}`;
      $("point-select").add(option);
    });
    $("projection-legend").replaceChildren();
    const symbols = { up: "●", down: "■", left: "▲", right: "◆" };
    for (const [label, name] of Object.entries(labels)) {
      const item = document.createElement("span");
      const symbol = document.createElement("b");
      symbol.textContent = symbols[label];
      symbol.style.color = palette[label];
      symbol.setAttribute("aria-hidden", "true");
      item.append(
        symbol,
        document.createTextNode(
          `${name}: ${data.points.filter((point) => point.label === label).length}`,
        ),
      );
      $("projection-legend").append(item);
    }
    if (data.probe) {
      const item = document.createElement("span");
      item.textContent = "★ Новая команда";
      $("projection-legend").append(item);
    }
    $("projection-note").textContent =
      data.note ||
      `На карте ты видишь только 2 координаты. Модель сравнивает все ${data.feature_count} признаков, поэтому близость точек не всегда означает одинаковый ответ.`;
    $("projection-status").hidden = true;
    $("projection-selection").hidden = false;
    renderProjection();
  } catch (error) {
    if (request !== projectionRequest) return;
    projectionData = null;
    $("projection-status").textContent = error.message;
    $("projection").setAttribute("hidden", "");
    $("projection-selection").hidden = true;
    $("projection-legend").replaceChildren();
  } finally {
    if (request === projectionRequest)
      $("projection-box").setAttribute("aria-busy", "false");
  }
}

$("point-select").addEventListener("change", () =>
  selectPoint(Number($("point-select").value)),
);
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(renderProjection, 100);
});
$("training-settings").addEventListener("input", () => {
  updateSettingLabels();
  $("settings-note").textContent = modelReady
    ? "Чтобы применить новые настройки, нажми «Обучить модель». До этого твоя модель работает с прежними настройками."
    : "Меняй настройки и нажми «Обучить модель». Сохраним их вместе с твоей моделью.";
});
$("train-button").addEventListener("click", async () => {
  if (busy || probeBusy || recorder.isRecording) return;
  busy = true;
  updateActions();
  projectionData = null;
  projectionRequest++;
  $("probe-result").hidden = true;
  $("projection").setAttribute("hidden", "");
  $("projection-selection").hidden = true;
  $("projection-legend").replaceChildren();
  $("projection-status").hidden = false;
  $("projection-status").textContent =
    "Обучу модель и обновлю карту твоих записей.";
  try {
    const state = await api("/api/train/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(readSettings()),
    });
    projectionNeedsRefresh = true;
    selectedPoint = 0;
    setTrainStatus(state);
  } catch (error) {
    busy = false;
    $("train-message").textContent = error.message;
    $("train-message").classList.add("error");
    updateActions();
  }
});

function showPrediction(result) {
  $("probe-result").hidden = false;
  $("predicted").textContent = result.command;
  $("neighbors-heading").textContent =
    `Ближайшие записи: ${result.neighbors.length}`;
  $("neighbors").replaceChildren();
  result.neighbors.forEach((neighbor, index) => {
    const item = document.createElement("div");
    item.className = "neighbor-item";
    const name = document.createElement("span");
    name.textContent = `${index + 1}. ${neighbor.command} · ${neighbor.file}`;
    const distance = document.createElement("strong");
    distance.textContent = `Расстояние: ${neighbor.distance}`;
    item.append(name, distance);
    $("neighbors").append(item);
  });
}

async function toggleProbe() {
  if (probeBusy || busy || (!modelReady && !recorder.isRecording)) return;
  probeBusy = true;
  updateActions();
  const button = $("probe-button");
  if (!recorder.isRecording) {
    $("probe-status").textContent = "Включаю микрофон…";
    try {
      await recorder.start({
        maxSeconds: labConfig.audio.max_seconds,
        onLimit: () => {
          if (recorder.isRecording) void toggleProbe();
        },
      });
      button.textContent = "■ Остановить и распознать";
      button.setAttribute("aria-pressed", "true");
      $("probe-status").textContent =
        "Записываю. Скажи «вверх», «вниз», «влево» или «вправо», затем останови запись.";
    } catch (error) {
      $("probe-status").textContent =
        `Не получилось включить микрофон. Проверь доступ к нему в браузере и попробуй еще раз. ${error.message}`;
    } finally {
      probeBusy = false;
      updateActions();
    }
    return;
  }
  button.textContent = "Сравниваю с твоими примерами…";
  button.setAttribute("aria-pressed", "false");
  $("probe-status").textContent = "Ищу записи, похожие на твою команду…";
  try {
    const blob = await recorder.stop();
    const result = await api(
      `/api/train/probe?denoise=${$("probe-denoise").checked}`,
      { method: "POST", headers: { "Content-Type": "audio/wav" }, body: blob },
    );
    showPrediction(result);
    $("probe-status").textContent =
      "Готово! Посмотри, какие записи оказались похожими. Твоя новая команда отмечена звездой на карте.";
    await drawProjection();
  } catch (error) {
    $("probe-status").textContent =
      `Не получилось узнать команду. Попробуй записать ее еще раз. ${error.message}`;
  } finally {
    probeBusy = false;
    button.textContent = "Произнести команду еще раз";
    updateActions();
  }
}

$("probe-button").addEventListener("click", toggleProbe);
async function refreshModels() {
  try {
    const data = await api("/api/models");
    $("model-history").hidden = !data.enabled;
    const selected = $("model-select").value;
    $("model-select").replaceChildren();
    for (const model of data.models) {
      const option = document.createElement("option");
      option.value = model.id;
      const settings = model.settings;
      const date = new Date(
        typeof model.created_at === "number"
          ? model.created_at * 1000
          : model.created_at,
      );
      const stamp = Number.isNaN(date.getTime())
        ? String(model.created_at)
        : date.toLocaleString("ru-RU");
      option.textContent = `${model.active || model.id === data.active_id ? "Активна · " : ""}${stamp} · k=${settings.k}, MFCC=${settings.n_mfcc}, окно=${settings.n_fft}, ${model.training_count} примеров`;
      $("model-select").add(option);
    }
    if (data.models.some((model) => model.id === selected))
      $("model-select").value = selected;
    else if (data.active_id) $("model-select").value = data.active_id;
    if (!data.models.length)
      $("history-status").textContent =
        "Пока нет сохраненных моделей. Обучи первый вариант.";
    updateActions();
  } catch (error) {
    $("model-history").hidden = false;
    $("history-status").textContent = error.message;
  }
}

async function changeHistory(action) {
  if (busy || probeBusy || recorder.isRecording) return;
  const id = $("model-select").value;
  if (!id && action !== "clear") return;
  if (
    action !== "select" &&
    !window.confirm(
      action === "clear"
        ? "Удалить все сохраненные модели? Исходные WAV сохранятся."
        : "Удалить выбранную модель? Исходные WAV сохранятся.",
    )
  )
    return;
  for (const control of [
    "select-model",
    "delete-model",
    "clear-models",
    "model-select",
    "train-button",
    "probe-button",
  ])
    $(control).disabled = true;
  try {
    await api(`/api/models/${action}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id }),
    });
    if (action === "select") {
      await audioSettings.init();
      applySettings(audioSettings.settings);
      projectionNeedsRefresh = true;
      selectedPoint = 0;
      $("probe-result").hidden = true;
      $("settings-note").textContent =
        "Восстановлены настройки выбранной модели, включая обработку и признаки на предыдущих этапах.";
    }
    $("history-status").textContent =
      action === "select"
        ? "Модель активна для проверки и игры."
        : "Модели удалены. Исходные записи сохранены.";
    await refreshModels();
    await refreshStatus();
  } catch (error) {
    $("history-status").textContent = error.message;
  } finally {
    updateActions();
  }
}

$("select-model").addEventListener("click", () => {
  void changeHistory("select");
});
$("delete-model").addEventListener("click", () => {
  void changeHistory("delete");
});
$("clear-models").addEventListener("click", () => {
  void changeHistory("clear");
});

async function initialize() {
  $("training-settings").disabled = true;
  for (const [index, level] of labConfig.noise_levels.entries()) {
    const option = document.createElement("option");
    option.value = level;
    const description =
      labConfig.noise_levels.length === 3
        ? ["", "слабый ", "сильный "][index]
        : "";
    option.textContent =
      level === 0
        ? "Оставляем без шума"
        : `Добавляем ${description}шум - ${(level * 100).toLocaleString("ru-RU")}%`;
    $("noise").add(option);
  }
  applySettings(audioSettings.defaults);
  try {
    await audioSettings.init();
    applySettings(audioSettings.settings);
    await Promise.all([
      updateCounts(),
      refreshStatus(),
      refreshModels(),
      audioSettings.showCache(),
    ]);
  } catch (error) {
    $("counts").textContent = error.message;
    $("train-message").textContent = error.message;
  }
}
void initialize();
