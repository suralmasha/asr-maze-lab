"""Дисковый кеш вычислений с ограничением объема и фоновой очередью."""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
import threading
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

from lab_config import CONFIG

logger = logging.getLogger(__name__)
Task = tuple[str, Callable[[], bytes]]


class DiskCache:
    """Хранит результаты на диске и последовательно исполняет фоновые задачи."""

    def __init__(self, root: Path, limit: int = CONFIG["cache_max_bytes"]) -> None:
        """
        Создает кеш и восстанавливает порядок использования файлов.

        :param root: Отдельный каталог результатов вычислений.
        :param limit: Максимальный суммарный размер результатов в байтах.
        """
        if limit <= 0:
            raise ValueError(
                "ASR-CFG-01: Invalid cache limit (Hint: fix configuration)"
            )
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.limit = limit
        self._condition = threading.Condition(threading.RLock())
        self._entries = {
            path.name: (path.stat().st_size, path.stat().st_mtime_ns)
            for path in self.root.glob("*.cache")
        }
        self._size = sum(size for size, _ in self._entries.values())
        self._generation = 0
        self._tasks: Iterator[Task] | None = None
        self._active = False
        self._owner: int | None = None
        self._depth = 0
        self._local = threading.local()
        self._background_running = False
        self._foreground_waiters = 0
        self._paused = False
        self._stopped = False
        self._error: str | None = None
        with self._condition:
            self._evict(0)
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def _evict(self, incoming: int) -> None:
        """
        Освобождает место, начиная с наиболее давно использованного результата.

        :param incoming: Размер нового результата.
        """
        while self._entries and self._size + incoming > self.limit:
            name = min(self._entries, key=lambda item: self._entries[item][1])
            (self.root / name).unlink(missing_ok=True)
            self._size -= self._entries.pop(name)[0]

    def get_or_compute(
        self, key: str, compute: Callable[[], bytes], background: bool = False
    ) -> bytes:
        """
        Возвращает сохраненный результат или вычисляет его с приоритетом запроса.

        :param key: Стабильный ключ, включающий данные и параметры вычисления.
        :param compute: Функция, возвращающая байты результата.
        :param background: Запрещает вытеснение для фонового вычисления.
        :return: Байты готового результата.
        """
        with self._condition:
            generation = getattr(self._local, "generation", self._generation)
            background = background or getattr(self._local, "background", False)
        return self._get_or_compute(key, compute, background, generation)

    def _get_or_compute(
        self,
        key: str,
        compute: Callable[[], bytes],
        background: bool,
        generation: int,
    ) -> bytes:
        """
        Вычисляет результат с поколением, зафиксированным до ожидания очереди.

        :param key: Стабильный ключ результата.
        :param compute: Функция вычисления байтов.
        :param background: Признак фонового задания.
        :param generation: Поколение кеша на момент запроса.
        :return: Байты результата.
        """
        name = hashlib.sha256(key.encode()).hexdigest() + ".cache"
        path = self.root / name
        owner = threading.get_ident()
        with self._condition:
            if name in self._entries:
                return self._read(name, background)
            if not background:
                self._foreground_waiters += 1
            try:
                self._condition.wait_for(
                    lambda: (
                        self._owner == owner
                        or (
                            not self._active
                            and (not background or self._foreground_waiters == 0)
                        )
                    )
                )
                if name in self._entries:
                    return self._read(name, background)
                self._active = True
                self._owner = owner
                self._depth += 1
                self._local.generation = generation
                self._local.background = background
            finally:
                if not background:
                    self._foreground_waiters -= 1
                self._condition.notify_all()

        try:
            value = compute()
            with self._condition:
                if generation != self._generation:
                    return value
                if background and self._size + len(value) > self.limit:
                    self._paused = True
                    return value
                if len(value) > self.limit:
                    return value
                if not background:
                    self._evict(len(value))
                descriptor, temporary = tempfile.mkstemp(dir=self.root, suffix=".tmp")
                try:
                    with os.fdopen(descriptor, "wb") as output:
                        output.write(value)
                    os.replace(temporary, path)
                finally:
                    Path(temporary).unlink(missing_ok=True)
                self._entries[name] = (len(value), path.stat().st_mtime_ns)
                self._size += len(value)
                if background and self._size >= self.limit:
                    self._paused = True
            return value
        finally:
            with self._condition:
                self._depth -= 1
                if self._depth == 0:
                    self._active = False
                    self._owner = None
                    del self._local.generation
                    del self._local.background
                self._condition.notify_all()

    def _read(self, name: str, background: bool) -> bytes:
        """
        Читает результат под блокировкой и обновляет LRU для пользователя.

        :param name: Имя файла результата.
        :param background: Признак фонового запроса.
        :return: Байты результата.
        """
        path = self.root / name
        value = path.read_bytes()
        if not background:
            path.touch()
            self._entries[name] = (len(value), path.stat().st_mtime_ns)
        return value

    def start(self, tasks: Iterable[Task]) -> None:
        """
        Заменяет фоновую очередь и разрешает ее обработку.

        :param tasks: Задания в порядке приоритета записей и настроек.
        """
        with self._condition:
            self._tasks = iter(tasks)
            self._paused = self._size >= self.limit
            self._error = None
            self._condition.notify_all()

    def _run(self) -> None:
        """Исполняет фоновые задания, уступая очередь пользовательским запросам."""
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: (
                        self._stopped
                        or (
                            self._tasks is not None
                            and not self._paused
                            and not self._foreground_waiters
                        )
                    )
                )
                if self._stopped:
                    return
                if self._tasks is None:
                    continue
                try:
                    key, compute = next(self._tasks)
                except StopIteration:
                    self._tasks = None
                    continue
                except Exception as error:
                    self._tasks = None
                    self._error = str(error)
                    logger.exception(
                        "Failed to prepare cached result", extra={"error": str(error)}
                    )
                    continue
                generation = self._generation
                self._background_running = True
            try:
                self._get_or_compute(key, compute, True, generation)
            except Exception as error:
                logger.exception(
                    "Failed to compute cached result", extra={"error": str(error)}
                )
                with self._condition:
                    if generation == self._generation:
                        self._error = str(error)
            finally:
                with self._condition:
                    self._background_running = False
                    self._condition.notify_all()

    def pause(self) -> None:
        """Отменяет очередь и запись старых вычислений, сохраняя готовый кеш."""
        with self._condition:
            self._generation += 1
            self._tasks = None
            self._paused = True
            self._condition.notify_all()

    def clear(self) -> None:
        """Очищает результаты и отменяет запись незавершенных вычислений."""
        with self._condition:
            self._generation += 1
            self._tasks = None
            self._paused = False
            self._error = None
            for name in self._entries:
                (self.root / name).unlink(missing_ok=True)
            self._entries.clear()
            self._size = 0
            self._condition.notify_all()

    def status(self) -> dict[str, Any]:
        """
        Возвращает состояние кеша и фоновой очереди.

        :return: Размер, лимит, число результатов и состояние предрасчета.
        """
        with self._condition:
            return {
                "size": self._size,
                "limit": self.limit,
                "count": len(self._entries),
                "running": self._background_running,
                "pending": int(self._tasks is not None),
                "paused": self._paused,
                "error": self._error,
            }

    def stop(self) -> None:
        """Завершает worker, отменяя очередь и запись текущего вычисления."""
        with self._condition:
            self._stopped = True
            self._generation += 1
            self._tasks = None
            self._condition.notify_all()
