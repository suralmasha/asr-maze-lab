"use strict";
const labels = { up: ["↑", "Вверх"], down: ["↓", "Вниз"], left: ["←", "Влево"], right: ["→", "Вправо"] };
const recorder = new VoiceRecorder();
const el = id => document.getElementById(id);
let currentLabel = null;
let busy = false;
let sampleState = {};

function status(text, error = false) {
  el("notice").textContent = text;
  el("notice").classList.toggle("error", error);
}
function makeCard(key, icon, title) {
  const card = document.createElement("section");
  card.className = "record-card";
  card.innerHTML = `<div class="card-head"><div class="direction">${icon}</div><div><h2>${title}</h2><small>Скажи: «${title.toLowerCase()}»</small></div><span class="count" id="count-${key}">0 / 5</span></div>
    <button id="button-${key}" class="record-button" type="button">● Начать запись</button>
    <div class="file-grid" id="files-${key}"></div>`;
  el("cards").append(card);
  el(`button-${key}`).addEventListener("click", () => toggle(key));
}
function updateButtons() {
  for (const key of Object.keys(labels)) {
    const button = el(`button-${key}`);
    button.disabled = busy && currentLabel !== key;
    button.textContent = recorder.isRecording && currentLabel === key ? "■ Остановить и сохранить" : "● Начать запись";
    button.classList.toggle("recording", recorder.isRecording && currentLabel === key);
  }
}
async function refresh() {
  const response = await fetch("/api/samples", { cache: "no-store" });
  if (!response.ok) throw new Error("Не удалось загрузить список записей.");
  sampleState = await response.json();
  const sum = Object.values(sampleState).reduce((s, a) => s + a.length, 0);
  el("total").textContent = sum;
  el("goal").textContent = `${Object.values(sampleState).reduce((s, a) => s + Math.min(5, a.length), 0)} / 20`;
  for (const key of Object.keys(labels)) {
    const list = el(`files-${key}`);
    list.replaceChildren();
    el(`count-${key}`).textContent = `${sampleState[key].length} / 5`;
    if (!sampleState[key].length) { const p=document.createElement("p");p.className="empty";p.textContent="Пока нет записей";list.append(p); }
    for (const filename of sampleState[key]) {
      const item = document.createElement("div");item.className="file";
      const audio = document.createElement("audio");audio.controls=true;audio.preload="none";
      audio.src=`/api/audio?label=${key}&file=${filename}`;
      audio.title=`${labels[key][1]} ${filename}`;
      const name=document.createElement("span");name.textContent=filename;
      const remove=document.createElement("button");remove.className="delete";remove.type="button";remove.textContent="×";
      remove.setAttribute("aria-label",`Удалить ${filename}`);
      remove.addEventListener("click", async () => {
        if (busy || !window.confirm(`Удалить ${key}/${filename}?`)) return;
        try {
          const res = await fetch(`/api/delete?label=${key}&file=${filename}`, {method:"POST"});
          if(!res.ok) throw new Error((await res.json()).error);
          await refresh();status(`Удалена запись ${key}/${filename}`);
        } catch(e) {status(e.message, true);}
      });
      item.append(name,audio,remove);list.append(item);
    }
  }
}
async function toggle(key) {
  if (busy && currentLabel !== key) return;
  if (recorder.isRecording) {
    if (currentLabel !== key) return;
    busy = true;updateButtons();
    try {
      const wav = await recorder.stop();
      const response = await fetch(`/api/samples?label=${key}`, { method:"POST", headers:{"Content-Type":"audio/wav"}, body:wav });
      const data=await response.json();
      if (!response.ok) throw new Error(data.error || "Не получилось сохранить.");
      await refresh();status(`Сохранено: dataset/${key}/${data.filename} (${data.duration} с)`);
    } catch(e) {status(`Ошибка: ${e.message}`,true);}
    finally {currentLabel=null;busy=false;updateButtons();}
    return;
  }
  if(busy) return;
  busy=true;currentLabel=key;updateButtons();
  try {
    await recorder.start({maxSeconds: 5, onLimit: () => {if(recorder.isRecording) void toggle(key);} });
    status(`Идёт запись «${labels[key][1].toLowerCase()}». Нажми ■ для сохранения.`);
  } catch(e) {status(`Не удалось включить микрофон: ${e.message}`,true);currentLabel=null;}
  finally {busy=false;updateButtons();}
}
for (const [key, [icon, title]] of Object.entries(labels)) makeCard(key,icon,title);
refresh().catch(e=>status(e.message,true));
