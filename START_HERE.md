# Быстрый старт (без интернета на занятии)

**Подготовить заранее с интернетом**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

**В день мастер-класса — один терминал**

```bash
source .venv/bin/activate
python lab_server.py
```

**В браузере:**

1. `http://localhost:8000/collect.html` — 5+ WAV для каждой из 4 команд
2. `http://localhost:8000/visualize.html` — четыре представления выбранной записи
3. `http://localhost:8000/train.html` → «Обучить модель» → прогресс MFCC, PCA и тест новой записи
4. `http://localhost:8000/index.html` — голосовое управление лабиринтом

**Необязательно, для Colab:** скачать ZIP датасета со сборщика, открыть `training.ipynb` в Colab, загрузить ZIP, скачать `.npz`, скопировать в `models/` на своём ноутбуке. Colab требует интернета.

Подробности, тайминг занятия, Windows-команды и устранение проблем см. в `README.md`.

**Не нужно запускать Jupyter** — обучение выполняется сервером по нажатию кнопки на странице `train.html`.
