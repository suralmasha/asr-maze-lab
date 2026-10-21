"use strict";
const labels = {
  up: ["↑", "Вверх"],
  down: ["↓", "Вниз"],
  left: ["←", "Влево"],
  right: ["→", "Вправо"],
};
const recorder = new VoiceRecorder();
const el = (id) => document.getElementById(id);
const maximumSamples = labConfig.audio.max_samples;
const minimumSamples = labConfig.audio.min_samples;
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
  card.innerHTML = `<div class="card-head"><div class="direction" aria-hidden="true">${icon}</div><div><h2>${title}</h2></div><span class="count" id="count-${key}">0 / ${maximumSamples}</span></div>
    <p class="record-state" id="state-${key}">Запиши хотя бы ${minimumSamples} примеров</p>
    <button id="button-${key}" class="record-button" type="button">● Начать запись</button>
    <div class="file-grid" id="files-${key}"></div>`;
  el("cards").append(card);
  el(`button-${key}`).addEventListener("click", () => toggle(key));
}

function updateButtons() {
  for (const key of Object.keys(labels)) {
    const button = el(`button-${key}`);
    const active = recorder.isRecording && currentLabel === key;
    const count = sampleState[key]?.length || 0;
    button.disabled =
      busy ||
      (recorder.isRecording && !active) ||
      (!active && count >= maximumSamples);
    button.textContent = active
      ? "■ Остановить и сохранить"
      : count >= maximumSamples
        ? `У тебя уже ${maximumSamples} записей`
        : "● Начать запись";
    button.classList.toggle("recording", active);
    button.setAttribute("aria-pressed", String(active));
    el(`state-${key}`).textContent = active
      ? "Записываю твой голос"
      : count < minimumSamples
        ? `Осталось записать: ${minimumSamples - count}`
        : "";
    el(`state-${key}`).hidden = !active && count >= minimumSamples;
  }
  document.querySelectorAll(".delete").forEach((button) => {
    button.disabled = busy || recorder.isRecording;
  });
}

async function refresh() {
  const response = await fetch("/api/samples", { cache: "no-store" });
  if (!response.ok)
    throw new Error(
      "Не получилось загрузить твои записи. Попробуй обновить страницу.",
    );
  sampleState = await response.json();
  el("total").textContent = Object.values(sampleState).reduce(
    (sum, samples) => sum + samples.length,
    0,
  );
  for (const key of Object.keys(labels)) {
    const list = el(`files-${key}`);
    list.replaceChildren();
    const files = sampleState[key] || [];
    el(`count-${key}`).textContent = `${files.length} / ${maximumSamples}`;
    if (!files.length) {
      const empty = document.createElement("p");
      empty.className = "empty";
      empty.textContent = "Здесь появятся твои записи";
      list.append(empty);
    }
    for (const filename of files) {
      const item = document.createElement("div");
      item.className = "file";
      const audio = document.createElement("audio");
      audio.controls = true;
      audio.preload = "none";
      audio.src = `/api/audio?${new URLSearchParams({ label: key, file: filename })}`;
      audio.setAttribute("aria-label", `${labels[key][1]}: ${filename}`);
      const name = document.createElement("span");
      name.textContent = filename;
      const remove = document.createElement("button");
      remove.className = "delete";
      remove.type = "button";
      remove.textContent = "×";
      remove.setAttribute(
        "aria-label",
        `Удалить ${labels[key][1]} ${filename}`,
      );
      remove.addEventListener("click", async () => {
        if (
          busy ||
          recorder.isRecording ||
          !window.confirm(`Удалить твою запись ${key}/${filename}?`)
        )
          return;
        busy = true;
        updateButtons();
        try {
          const result = await fetch(
            `/api/delete?${new URLSearchParams({ label: key, file: filename })}`,
            { method: "POST" },
          );
          if (!result.ok) throw new Error((await result.json()).error);
          await refresh();
          status(
            `Запись ${key}/${filename} удалена. Можешь записать новый пример.`,
          );
        } catch (error) {
          status(error.message, true);
        } finally {
          busy = false;
          updateButtons();
        }
      });
      item.append(name, audio, remove);
      list.append(item);
    }
  }
  updateButtons();
}

async function toggle(key) {
  if (busy || (recorder.isRecording && currentLabel !== key)) return;
  if (recorder.isRecording) {
    busy = true;
    updateButtons();
    status("Сохраняю твою запись…");
    try {
      const wav = await recorder.stop();
      const response = await fetch(`/api/samples?label=${key}`, {
        method: "POST",
        headers: { "Content-Type": "audio/wav" },
        body: wav,
      });
      const data = await response.json();
      if (!response.ok)
        throw new Error(
          data.error || "Не получилось сохранить запись. Попробуй еще раз.",
        );
      await refresh();
      status(
        `Готово! Твоя запись: ${key}/${data.filename} (${data.duration} с). Можешь записать следующий пример.`,
      );
    } catch (error) {
      status(`Не получилось сохранить запись: ${error.message}`, true);
    } finally {
      currentLabel = null;
      busy = false;
      updateButtons();
    }
    return;
  }
  if ((sampleState[key]?.length || 0) >= maximumSamples) return;
  busy = true;
  currentLabel = key;
  updateButtons();
  status("Включаю микрофон…");
  try {
    await recorder.start({
      maxSeconds: labConfig.audio.max_seconds,
      onLimit: () => {
        if (recorder.isRecording) void toggle(key);
      },
    });
    status(
      `Скажи «${labels[key][1].toLowerCase()}», затем нажми «Остановить и сохранить».`,
    );
  } catch (error) {
    status(
      `Не получилось включить микрофон. Проверь доступ к нему в браузере и попробуй еще раз. ${error.message}`,
      true,
    );
    currentLabel = null;
  } finally {
    busy = false;
    updateButtons();
  }
}

for (const [key, [icon, title]] of Object.entries(labels))
  makeCard(key, icon, title);
refresh()
  .then(() => status(""))
  .catch((error) => status(error.message, true));
