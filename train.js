"use strict";
const $ = id => document.getElementById(id);
const labels = {up:"↑ Вверх", down:"↓ Вниз", left:"← Влево", right:"→ Вправо"};
const recorder = new VoiceRecorder();
let busy = false, modelReady = false, pollTimer = null;

async function api(url, init) {
  const response = await fetch(url, {cache:"no-store", ...init});
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}
async function updateCounts() {
  const data = await api("/api/samples");
  const box = $("counts"); box.replaceChildren();
  for (const [key, name] of Object.entries(labels)) {
    const count = data[key]?.length || 0;
    const item = document.createElement("div");
    item.className = "training-count" + (count < 2 ? " warn" : "");
    const title = document.createElement("span"); title.textContent = name;
    const strong = document.createElement("strong"); strong.textContent = `${count} WAV`;
    item.append(title, strong); box.append(item);
  }
}
function setTrainStatus(s) {
  busy = s.state === "running";
  $("train-button").disabled = busy;
  $("progress").value = s.percent || 0;
  $("progress-percent").textContent = `${s.percent || 0}%`;
  $("progress-stage").textContent = {
    idle:"Ожидание",features:"Извлечение MFCC",fit:"Обучение k-NN",
    save:"Сохранение",done:"Готово",error:"Ошибка"
  }[s.stage] || (s.ready ? "Модель готова" : "Ожидание");
  $("train-message").textContent = s.message;
  $("train-message").classList.toggle("error",s.state === "error");
  modelReady = Boolean(s.ready) && !busy;
  $("probe-button").disabled = !modelReady;
  if (modelReady && $("projection").hidden) drawProjection();
  if (s.state === "running" && !pollTimer) {
    pollTimer = setInterval(refreshStatus, 500);
  } else if (s.state !== "running" && pollTimer) {
    clearInterval(pollTimer); pollTimer = null;
    if (s.state === "done") drawProjection();
  }
}
async function refreshStatus() {
  try { setTrainStatus(await api("/api/train/status")); }
  catch (e) {$("train-message").textContent=e.message;}
}
function drawProjection() {
  const image = $("projection"), status = $("projection-status");
  status.hidden = false; status.textContent = "Строю карту признаков…";
  image.hidden = true;
  image.onload = () => {image.hidden=false;status.hidden=true;};
  image.onerror = () => {status.hidden=false;status.textContent="Не удалось построить график. Проверь терминал.";};
  image.src = `/api/train/plot?ts=${Date.now()}`;
}
$("train-button").addEventListener("click", async () => {
  if(busy)return;
  try {
    $("train-button").disabled=true;
    $("probe-result").hidden=true;
    $("projection").hidden=true;
    setTrainStatus(await api("/api/train/start",{method:"POST"}));
  } catch(e) {
    $("train-message").textContent = e.message;
    $("train-message").classList.add("error");
    $("train-button").disabled=false;
  }
});
function showPrediction(result) {
  $("probe-result").hidden = false;
  $("predicted").textContent = result.command;
  const list = $("neighbors"); list.replaceChildren();
  result.neighbors.forEach((n,i) => {
    const item=document.createElement("div");item.className="neighbor-item";
    const name=document.createElement("span");name.textContent=`${i+1}. ${n.command} · ${n.file}`;
    const distance=document.createElement("strong");distance.textContent=`Расстояние: ${n.distance}`;
    item.append(name,distance);list.append(item);
  });
}
async function toggleProbe() {
  if (!modelReady && !recorder.isRecording) return;
  const button = $("probe-button");
  if (!recorder.isRecording) {
    try {
      await recorder.start({maxSeconds: 5, onLimit: () => {if(recorder.isRecording)toggleProbe();}});
      button.textContent = "■ Остановить и распознать";
      $("probe-status").textContent = "Произнеси: вверх / вниз / влево / вправо.";
    } catch(e) {$("probe-status").textContent=e.message;}
    return;
  }
  button.disabled=true;
  button.textContent = "Обрабатываю запись…";
  try {
    const blob=await recorder.stop();
    const result=await api("/api/train/probe",{method:"POST",headers:{"Content-Type":"audio/wav"},body:blob});
    showPrediction(result);
    $("probe-status").textContent="Команда классифицирована. Посмотри на ближайшие записи и на звезду на графике.";
    drawProjection();
  } catch(e) {$("probe-status").textContent=`Не удалось распознать: ${e.message}`;}
  finally {button.disabled=!modelReady;button.textContent="🎙 Произнести ещё раз";}
}
$("probe-button").addEventListener("click",toggleProbe);
updateCounts().catch(e=>{$("counts").textContent=e.message;});
refreshStatus();
