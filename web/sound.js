"use strict";
const $ = (id) => document.getElementById(id);
let recordings = {};
let requestNumber = 0;
const samplingRates = labConfig.sampling_rates;
$("sample-rate").max = samplingRates.length - 1;
$("sample-rate").value = samplingRates.indexOf(16000);
for (const [index, rate] of samplingRates.entries()) {
  const tick = document.createElement("span");
  tick.textContent = `${rate / 1000} кГц`;
  $("sample-rate-ticks").append(tick);
  const step = document.createElement("option");
  step.value = index;
  $("sample-rate-steps").append(step);
}

function choose() {
  const previous = $("filename").value;
  $("filename").replaceChildren();
  const files = recordings[$("label").value] || [];
  for (const filename of files) {
    const option = document.createElement("option");
    option.value = filename;
    option.textContent = filename;
    $("filename").add(option);
  }
  if (files.includes(previous)) $("filename").value = previous;
  $("filename").disabled = !files.length;
  redraw();
}

function redraw() {
  const sampleRate = samplingRates[Number($("sample-rate").value)];
  $("sample-rate-value").value = `${sampleRate / 1000} кГц`;
  $("sampling-audio-rate").textContent = `${sampleRate / 1000} кГц`;
  $("sampling-audio-status").textContent = "";
  $("sample-rate").setAttribute("aria-valuetext", `${sampleRate / 1000} кГц`);
  const request = ++requestNumber;
  $("plot").hidden = true;
  $("plot-status").hidden = false;
  $("plot-box").setAttribute("aria-busy", "false");
  if (!$("filename").value) {
    $("plot-status").textContent =
      "Пока здесь пусто. Вернись на этап «Данные» и запиши эту команду.";
    $("sample-audio").removeAttribute("src");
    $("sample-audio").load();
    $("sampling-audio").removeAttribute("src");
    $("sampling-audio").load();
    return;
  }
  const params = new URLSearchParams({
    label: $("label").value,
    file: $("filename").value,
  });
  const audioUrl = `/api/audio?${params}`;
  if ($("sample-audio").getAttribute("src") !== audioUrl)
    $("sample-audio").src = audioUrl;
  $("plot-status").textContent = "Рисую графики твоей записи…";
  $("plot-box").setAttribute("aria-busy", "true");
  $("plot").onload = () => {
    if (request !== requestNumber) return;
    $("plot").hidden = false;
    $("plot-status").hidden = true;
    $("plot-box").setAttribute("aria-busy", "false");
  };
  $("plot").onerror = () => {
    if (request !== requestNumber) return;
    $("plot-status").textContent =
      "Не получилось показать графики. Попробуй выбрать запись еще раз.";
    $("plot-box").setAttribute("aria-busy", "false");
  };
  params.set("sample_rate", sampleRate);
  const samplingUrl = `/api/sampling/audio?${params}`;
  if ($("sampling-audio").getAttribute("src") !== samplingUrl) {
    $("sampling-audio").pause();
    $("sampling-audio").src = samplingUrl;
    $("sampling-audio").load();
  }
  $("plot").src = `/api/sampling?${params}`;
}

$("sampling-audio").addEventListener("error", () => {
  if ($("sampling-audio").getAttribute("src"))
    $("sampling-audio-status").textContent =
      "Не получилось загрузить аудио. Попробуй выбрать запись еще раз.";
});
for (const id of ["sample-audio", "sampling-audio"])
  $(id).addEventListener("play", () => {
    $(id === "sample-audio" ? "sampling-audio" : "sample-audio").pause();
  });

$("label").addEventListener("change", choose);
$("filename").addEventListener("change", redraw);
$("sample-rate").addEventListener("input", redraw);
fetch("/api/samples", { cache: "no-store" })
  .then((response) => {
    if (!response.ok)
      throw new Error(
        "Не получилось загрузить записи. Попробуй обновить страницу.",
      );
    return response.json();
  })
  .then((data) => {
    recordings = data;
    choose();
  })
  .catch((error) => {
    $("plot-status").textContent = error.message;
    $("plot-box").setAttribute("aria-busy", "false");
  });
