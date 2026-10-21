"use strict";
const $ = id => document.getElementById(id);
let recordings = {};
let requestNumber = 0;

function choose() {
  const previous = $("filename").value;
  $("filename").replaceChildren();
  const files = recordings[$("label").value] || [];
  for (const filename of files) {
    const option = document.createElement("option"); option.value = filename; option.textContent = filename; $("filename").add(option);
  }
  if (files.includes(previous)) $("filename").value = previous;
  $("filename").disabled = !files.length;
  redraw();
}

function redraw() {
  const request = ++requestNumber;
  $("plot").hidden = true; $("plot-status").hidden = false;
  $("plot-box").setAttribute("aria-busy", "false");
  if (!$("filename").value) {
    $("plot-status").textContent = "Пока здесь пусто. Вернись на этап «Данные» и запиши эту команду.";
    $("sample-audio").removeAttribute("src"); $("sample-audio").load(); return;
  }
  const params = new URLSearchParams({label: $("label").value, file: $("filename").value});
  $("sample-audio").src = `/api/audio?${params}`;
  $("plot-status").textContent = "Рисую графики твоей записи…";
  $("plot-box").setAttribute("aria-busy", "true");
  $("plot").onload = () => {
    if (request !== requestNumber) return;
    $("plot").hidden = false; $("plot-status").hidden = true;
    $("plot-box").setAttribute("aria-busy", "false");
  };
  $("plot").onerror = () => {
    $("plot-status").textContent = "Не получилось показать графики. Попробуй выбрать запись еще раз.";
    $("plot-box").setAttribute("aria-busy", "false");
  };
  $("plot").src = `/api/plot?${params}&view=signal&trim_silence=false`;
}

$("label").addEventListener("change", choose);
$("filename").addEventListener("change", redraw);
fetch("/api/samples", {cache: "no-store"}).then(response => {
  if (!response.ok) throw new Error("Не получилось загрузить записи. Попробуй обновить страницу.");
  return response.json();
}).then(data => {recordings = data; choose();}).catch(error => {$("plot-status").textContent = error.message; $("plot-box").setAttribute("aria-busy", "false");});
