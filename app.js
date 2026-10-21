"use strict";

// # 1. Игровое поле: # — стена, S — старт, F — финиш, . — дорожка.
// Менять лабиринт можно прямо здесь. Длина всех строк должна совпадать.
const MAP = [
  "#########",
  "#S..#...#",
  "#.#.#.#.#",
  "#.#...#.#",
  "#.###.#.#",
  "#.....#F#",
  "#########",
];

const DIRECTIONS = {
  up: { dx: 0, dy: -1, label: "↑", name: "вверх" },
  down: { dx: 0, dy: 1, label: "↓", name: "вниз" },
  left: { dx: -1, dy: 0, label: "←", name: "влево" },
  right: { dx: 1, dy: 0, label: "→", name: "вправо" },
};

const STORAGE_KEY = "asr-maze-keywords-v2";
const DEFAULT_RULES = [
  {phrase: "вверх", direction: "up", steps: 1},
  {phrase: "вниз", direction: "down", steps: 1},
  {phrase: "влево", direction: "left", steps: 1},
  {phrase: "вправо", direction: "right", steps: 1},
];
const $ = (id) => document.getElementById(id);
const startY = MAP.findIndex(row => row.includes("S"));
const START = { x: MAP[startY].indexOf("S"), y: startY };
let position = { ...START };
let moves = 0;
let commands = 0;
let finished = false;
let rules = loadRules();

// # 2. Нормализация фразы нужна, чтобы регистр и знаки препинания не мешали.
function normalize(value) {
  return String(value)
    .toLocaleLowerCase("ru-RU")
    .replace(/ё/g, "е")
    .replace(/[^\p{L}\p{N}]+/gu, " ")
    .trim()
    .replace(/\s+/g, " ");
}

function loadRules() {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored === null) return DEFAULT_RULES.map(rule => ({...rule}));
    const data = JSON.parse(stored);
    if (!Array.isArray(data)) return [];
    return data.filter(rule =>
      rule && typeof rule.phrase === "string" && rule.phrase.length > 0 &&
      Object.hasOwn(DIRECTIONS, rule.direction) &&
      Number.isInteger(rule.steps) && rule.steps >= 1 && rule.steps <= 3
    ).slice(0, 30);
  } catch {
    return [];
  }
}

function saveRules() {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(rules)); }
  catch { /* Сайт работает и при отключённом localStorage. */ }
}

function setStatus(title, detail, kind = "neutral") {
  $("status-title").textContent = title;
  $("status-text").textContent = detail;
  $("status-dot").parentElement.dataset.kind = kind;
}

function renderMaze() {
  const container = $("maze");
  container.replaceChildren();
  MAP.forEach((row, y) => {
    [...row].forEach((char, x) => {
      const cell = document.createElement("div");
      cell.className = "cell " + (char === "#" ? "wall" : char === "F" ? "finish" : char === "S" ? "start" : "path");
      if (position.x === x && position.y === y) {
        const robot = document.createElement("div");
        robot.className = "robot";
        robot.setAttribute("aria-hidden", "true");
        cell.append(robot);
      } else if (char === "F") {
        const flag = document.createElement("span");
        flag.className = "goal-flag";
        flag.textContent = "⚑";
        cell.append(flag);
      }
      container.append(cell);
    });
  });
  $("move-count").textContent = String(moves);
  $("command-count").textContent = String(commands);
  container.setAttribute("aria-label", `Лабиринт. Робот в клетке ${position.x + 1}, ${position.y + 1}. ${finished ? "Финиш достигнут." : "Финиш пока не достигнут."}`);
}

function renderRules() {
  $("rule-count").textContent = String(rules.length);
  const list = $("rule-list");
  list.replaceChildren();
  if (rules.length === 0) {
    const empty = document.createElement("div");
    empty.className = "empty-rules";
    empty.textContent = "Правил пока нет. Начни с четырёх направлений: «вверх», «вниз», «влево», «вправо».";
    list.append(empty);
    return;
  }
  rules.forEach((rule, index) => {
    const card = document.createElement("div");
    card.className = "rule-card";
    const ifTag = document.createElement("span");
    ifTag.className = "rule-if";
    ifTag.textContent = "ЕСЛИ";
    const word = document.createElement("span");
    word.className = "rule-word";
    word.textContent = `«${rule.phrase}»`;
    const effect = document.createElement("span");
    effect.className = "rule-effect";
    effect.textContent = `ТО ${DIRECTIONS[rule.direction].label} × ${rule.steps}`;
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "icon-button";
    remove.textContent = "×";
    remove.title = "Удалить правило";
    remove.setAttribute("aria-label", `Удалить правило ${rule.phrase}`);
    remove.addEventListener("click", () => {
      rules.splice(index, 1);
      saveRules();
      renderRules();
      setStatus("Правило удалено", `Теперь фраза «${rule.phrase}» больше не управляет роботом.`, "neutral");
    });
    card.append(ifTag, word, effect, remove);
    list.append(card);
  });
}

// # 3. Конструктор: школьники сопоставляют произносимую фразу и действие.
$("rule-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const phrase = normalize($("phrase").value);
  const direction = $("direction").value;
  const steps = Number($("steps").value);
  if (!phrase || !Object.hasOwn(DIRECTIONS, direction) || ![1, 2, 3].includes(steps)) return;
  if (rules.some(rule => rule.phrase === phrase)) {
    setStatus("Такая фраза уже есть", "Удали старое правило или придумай другое название команды.", "warning");
    return;
  }
  if (rules.length >= 30) {
    setStatus("Слишком много правил", "Для этого задания достаточно 30 правил.", "warning");
    return;
  }
  rules.push({ phrase, direction, steps });
  saveRules();
  renderRules();
  $("phrase").value = "";
  $("phrase").focus();
  setStatus("Новое правило добавлено", `«${phrase}» → ${DIRECTIONS[direction].name}, ${steps} клетк(и).`, "success");
});

$("clear-rules").addEventListener("click", () => {
  if (rules.length === 0) return;
  if (!window.confirm("Удалить все правила? Движение робота сбрасываться не будет.")) return;
  rules = [];
  saveRules();
  renderRules();
  setStatus("Правила очищены", "Можно собрать новую систему команд.");
});

// # 4. Логика команд: найдём самое длинное подходящее правило.
// Например, «поверни направо» должно выигрывать у «направо».
function findRule(text) {
  const padded = ` ${normalize(text)} `;
  return [...rules]
    .sort((a, b) => b.phrase.length - a.phrase.length)
    .find(rule => padded.includes(` ${rule.phrase} `));
}

function executeCommand(spokenText) {
  const text = normalize(spokenText);
  if (!text) {
    setStatus("Пустая команда", "Напиши или произнеси фразу.", "warning");
    return;
  }
  commands++;
  if (finished) {
    renderMaze();
    setStatus("Финиш уже достигнут!", "Нажми «Начать заново», чтобы пройти лабиринт ещё раз.", "success");
    return;
  }
  const rule = findRule(text);
  if (!rule) {
    renderMaze();
    setStatus("Команда не найдена", `Распознано: «${spokenText}». Добавь для неё правило или попробуй другую фразу.`, "warning");
    return;
  }
  const direction = DIRECTIONS[rule.direction];
  let actualSteps = 0;
  for (let i = 0; i < rule.steps; i++) {
    const nextX = position.x + direction.dx;
    const nextY = position.y + direction.dy;
    if (MAP[nextY]?.[nextX] == null || MAP[nextY][nextX] === "#") break;
    position = { x: nextX, y: nextY };
    moves++;
    actualSteps++;
    if (MAP[nextY][nextX] === "F") { finished = true; break; }
  }
  renderMaze();
  if (finished) {
    setStatus("Победа! Робот нашёл финиш!", `Последняя команда: «${spokenText}». Всего ходов: ${moves}, команд: ${commands}.`, "success");
  } else if (!actualSteps) {
    setStatus("Перед роботом стена", `Фраза «${spokenText}» распознана, правило найдено. Но идти ${direction.name} нельзя.`, "warning");
  } else {
    setStatus("Команда выполнена", `Распознано: «${spokenText}». Робот прошёл ${actualSteps} клетк(и) ${direction.name}.`, "success");
  }
}

$("command-form").addEventListener("submit", (event) => {
  event.preventDefault();
  executeCommand($("command-input").value);
  $("command-input").select();
});

$("reset-maze").addEventListener("click", () => {
  position = { ...START };
  moves = 0;
  commands = 0;
  finished = false;
  renderMaze();
  setStatus("Лабиринт сброшен", "Правила остались: попробуй пройти путь заново.");
});

// # 5. Мы НЕ используем готовый speech-to-text ASR.
// Обученная на странице train.html модель выбирает одно из 4 слов.
// Браузер записывает WAV; /api/predict возвращает метку команды.
const listenButton = $("listen-btn");
const listenLabel = $("listen-label");
const voiceRecorder = new VoiceRecorder();
let busyMic = false;

function setMicButton(label, recording = false, disabled = false) {
  listenLabel.textContent = label;
  listenButton.classList.toggle("is-listening", recording);
  listenButton.disabled = disabled;
}

async function checkServer() {
  try {
    const response = await fetch("/api/health", {cache: "no-store"});
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    if (data.ready) {
      $("mic-note").textContent = "Используется твой классификатор четырёх слов. Интернет не требуется.";
      setMicButton("● Произнести слово");
    } else {
      setMicButton("Модель ещё не обучена", false, true);
      $("mic-note").textContent = data.message || "Открой этап 03 «Обучение» и обучи классификатор.";
      setStatus("Нет обученной модели", $("mic-note").textContent, "warning");
    }
  } catch(e) {
    setMicButton("Нужен локальный сервер", false, true);
    $("mic-note").textContent = "Запусти python lab_server.py и открой http://localhost:8000";
    setStatus("Не удалось соединиться с сервером", $("mic-note").textContent, "error");
  }
}

async function micToggle() {
  if (busyMic) return;
  busyMic = true;
  if (!voiceRecorder.isRecording) {
    setMicButton("Включаю микрофон…", false, true);
    try {
      await voiceRecorder.start({maxSeconds: 5, onLimit: () => { if(voiceRecorder.isRecording) void micToggle(); }});
      setMicButton("■ Остановить и распознать", true);
      setStatus("Идёт запись", "Произнеси одно из четырёх слов: вверх, вниз, влево, вправо.");
    } catch(e) {
      setStatus("Ошибка микрофона", e.message || String(e), "error");
      setMicButton("● Произнести слово");
    } finally { busyMic = false; }
    return;
  }
  setMicButton("Классифицирую…", false, true);
  try {
    const wav = await voiceRecorder.stop();
    const response = await fetch("/api/predict", {
      method: "POST", headers: {"Content-Type": "audio/wav"}, body: wav,
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || `Ошибка ${response.status}`);
    $("command-input").value = result.text;
    executeCommand(result.text);
    $("mic-note").textContent = `Модель выбрала: «${result.text}». Расстояния до ближайших записей: ${result.neighbor_distances.join(", ")}. Это не вероятности.`;
  } catch(e) {
    setStatus("Не получилось классифицировать", e.message || String(e), "error");
  } finally {
    busyMic = false;
    setMicButton("● Произнести слово");
  }
}

listenButton.addEventListener("click", () => {void micToggle();});

renderRules();
renderMaze();
void checkServer();
