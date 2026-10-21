"use strict";
const $ = id => document.getElementById(id);
let recordings = {};
let requestNumber = 0;
let activeController = null;
let imageUrl = null;

function choose(number) {
  const input = $(`filename-${number}`);
  const old = input.value;
  const files = recordings[$(`label-${number}`).value] || [];
  input.replaceChildren();
  for (const filename of files) {
    const option = document.createElement("option"); option.value = filename; option.textContent = filename; input.add(option);
  }
  if (files.includes(old)) input.value = old;
  input.disabled = !files.length;
}

async function redraw() {
  const request = ++requestNumber;
  activeController?.abort();
  $("plot").hidden = true; $("plot-status").hidden = false;
  $("plot-box").setAttribute("aria-busy", "false");
  let complete = true;
  for (const number of [1, 2]) {
    const audio = $(`audio-${number}`);
    const filename = $(`filename-${number}`).value;
    if (!filename) {audio.removeAttribute("src"); audio.load(); complete = false; continue;}
    const url = `/api/audio?${new URLSearchParams({label: $(`label-${number}`).value, file: filename})}`;
    if (audio.getAttribute("src") !== url) {audio.src = url; audio.load();}
  }
  if (!complete) {$("plot-status").textContent = "Выбери 2 записи слева. Если нужной записи пока нет, добавь ее на этапе «Данные»."; return;}
  const params = new URLSearchParams({
    label: $("label-1").value, file: $("filename-1").value,
    label2: $("label-2").value, file2: $("filename-2").value,
    n_mels: 32, n_fft: 512, view: "signal", trim_silence: false,
  });
  $("plot-status").textContent = "Рисую графики для твоих 2 записей…";
  $("plot-box").setAttribute("aria-busy", "true");
  activeController = new AbortController();
  try {
    const response = await fetch(`/api/compare?${params}`, {signal: activeController.signal});
    if (!response.ok) throw new Error((await response.json()).error || "Не получилось сравнить записи. Попробуй выбрать их еще раз.");
    const blob = await response.blob();
    if (request !== requestNumber) return;
    if (imageUrl) URL.revokeObjectURL(imageUrl);
    imageUrl = URL.createObjectURL(blob);
    $("plot").onload = () => {
      if (request !== requestNumber) return;
      $("plot").hidden = false; $("plot-status").hidden = true;
      $("plot-box").setAttribute("aria-busy", "false");
    };
    $("plot").onerror = () => {$("plot-status").textContent = "Не получилось показать графики. Попробуй выбрать записи еще раз."; $("plot-box").setAttribute("aria-busy", "false");};
    $("plot").src = imageUrl;
  } catch (error) {
    if (error.name === "AbortError" || request !== requestNumber) return;
    $("plot-status").textContent = error.message;
    $("plot-box").setAttribute("aria-busy", "false");
  }
}

for (const number of [1, 2]) {
  $(`label-${number}`).addEventListener("change", () => {choose(number); void redraw();});
  $(`filename-${number}`).addEventListener("change", redraw);
}
fetch("/api/samples", {cache: "no-store"}).then(response => {
  if (!response.ok) throw new Error("Не получилось загрузить твои записи. Попробуй обновить страницу.");
  return response.json();
}).then(data => {
  recordings = data;
  for (const number of [1, 2]) choose(number);
  void redraw();
}).catch(error => {$("plot-status").textContent = error.message; $("plot-box").setAttribute("aria-busy", "false");});
