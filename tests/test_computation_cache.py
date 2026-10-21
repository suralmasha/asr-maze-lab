"""Проверки дискового кеша, приоритетов и отмены предрасчета."""

from __future__ import annotations

import tempfile
import threading
import time
import unittest
from collections.abc import Callable
from pathlib import Path

from computation_cache import DiskCache


class TestDiskCache(unittest.TestCase):
    """Проверяет кеш в отдельном временном каталоге."""

    def setUp(self) -> None:
        """Создает временный каталог и кеш малого размера."""
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.cache = DiskCache(self.root, limit=8)

    def tearDown(self) -> None:
        """Останавливает фон и удаляет временные результаты."""
        self.cache.stop()
        self.cache._worker.join(timeout=2)
        self.directory.cleanup()

    def wait_for(self, condition: Callable[[], bool]) -> None:
        """
        Ожидает проверяемое состояние с ограничением времени.

        :param condition: Условие завершения ожидания.
        """
        deadline = time.monotonic() + 3
        while not condition() and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertTrue(condition())

    def test_persistence_and_lru(self) -> None:
        """Повторный запуск сохраняет результаты и вытесняет давно неиспользуемые."""
        self.cache.get_or_compute("a", lambda: b"aaaa")
        self.cache.get_or_compute("b", lambda: b"bbbb")
        self.cache.get_or_compute("a", lambda: b"wrong")
        self.cache.stop()
        self.cache._worker.join(timeout=2)
        self.cache = DiskCache(self.root, limit=8)
        self.cache.get_or_compute("c", lambda: b"cccc")
        self.assertEqual(self.cache.get_or_compute("a", lambda: b"miss"), b"aaaa")
        self.assertEqual(self.cache.get_or_compute("b", lambda: b"new"), b"new")
        self.assertLessEqual(self.cache.status()["size"], 8)

    def test_background_pauses_without_eviction(self) -> None:
        """Заполненный кеш приостанавливает фон, сохраняя готовые результаты."""
        self.cache.get_or_compute("a", lambda: b"aaaa")
        self.cache.start([("b", lambda: b"bbbbb"), ("c", lambda: b"c")])
        self.wait_for(lambda: self.cache.status()["paused"])
        self.assertEqual(self.cache.status()["count"], 1)
        self.assertEqual(self.cache.get_or_compute("a", lambda: b"miss"), b"aaaa")
        self.assertEqual(self.cache.get_or_compute("b", lambda: b"bbbbb"), b"bbbbb")
        self.assertEqual(self.cache.status()["count"], 1)
        self.assertTrue(self.cache.status()["paused"])

    def test_clear_discards_running_and_nested_results(self) -> None:
        """Очистка не позволяет старому заданию и его вложенным расчетам заполнить кеш."""
        entered = threading.Event()
        release = threading.Event()

        def compute() -> bytes:
            """
            Блокирует вычисление до очистки и вызывает вложенный расчет.

            :return: Результат вложенного вычисления.
            """
            entered.set()
            release.wait(timeout=3)
            return self.cache.get_or_compute("nested", lambda: b"old")

        self.cache.start([("old", compute), ("queued", lambda: b"old")])
        self.assertTrue(entered.wait(timeout=2))
        self.cache.clear()
        release.set()
        self.wait_for(lambda: not self.cache.status()["running"])
        self.assertEqual(self.cache.status()["count"], 0)
        self.assertEqual(list(self.root.glob("*.cache")), [])

    def test_foreground_precedes_next_background(self) -> None:
        """Пользовательский запрос выполняется перед следующим фоновым заданием."""
        entered = threading.Event()
        release = threading.Event()
        order: list[str] = []

        def first() -> bytes:
            """
            Удерживает первое фоновое вычисление.

            :return: Малый проверочный результат.
            """
            entered.set()
            release.wait(timeout=3)
            return b"a"

        def record(name: str) -> bytes:
            """
            Запоминает порядок вычислений.

            :param name: Название вычисления.
            :return: Малый проверочный результат.
            """
            order.append(name)
            return b"b"

        self.cache.start([("first", first), ("second", lambda: record("background"))])
        self.assertTrue(entered.wait(timeout=2))
        foreground = threading.Thread(
            target=lambda: self.cache.get_or_compute(
                "request", lambda: record("foreground")
            )
        )
        foreground.start()
        self.wait_for(lambda: self.cache._foreground_waiters > 0)
        release.set()
        foreground.join(timeout=2)
        self.wait_for(lambda: len(order) == 2)
        self.assertEqual(order, ["foreground", "background"])

    def test_background_hits_do_not_change_lru(self) -> None:
        """Фоновое чтение не продлевает жизнь давно неиспользуемого результата."""
        self.cache.get_or_compute("a", lambda: b"aaaa")
        self.cache.get_or_compute("b", lambda: b"bbbb")
        self.cache.get_or_compute("a", lambda: b"wrong", background=True)
        self.cache.get_or_compute("c", lambda: b"cccc")
        self.assertEqual(self.cache.get_or_compute("b", lambda: b"miss"), b"bbbb")

    def test_ready_result_is_available_during_background(self) -> None:
        """Готовый результат доступен до завершения следующего фонового расчета."""
        entered = threading.Event()
        release = threading.Event()
        received = threading.Event()
        self.cache.get_or_compute("ready", lambda: b"ready")

        def compute() -> bytes:
            """
            Имитирует длительный фоновый расчет.

            :return: Малый проверочный результат.
            """
            entered.set()
            release.wait(timeout=3)
            return b"b"

        def read() -> None:
            """Читает готовый результат и сигнализирует о завершении."""
            self.cache.get_or_compute("ready", lambda: b"wrong")
            received.set()

        self.cache.start([("slow", compute)])
        self.assertTrue(entered.wait(timeout=2))
        reader = threading.Thread(target=read)
        reader.start()
        try:
            self.assertTrue(received.wait(timeout=1))
        finally:
            release.set()
            reader.join(timeout=2)
            self.wait_for(lambda: not self.cache.status()["running"])

    def test_pause_preserves_results_and_allows_restart(self) -> None:
        """Пауза сохраняет готовые данные, отменяет старые и допускает новый фон."""
        entered = threading.Event()
        release = threading.Event()
        self.cache.get_or_compute("ready", lambda: b"ready")

        def compute() -> bytes:
            """
            Имитирует незавершенный расчет перед паузой.

            :return: Малый проверочный результат.
            """
            entered.set()
            release.wait(timeout=3)
            return b"old"

        self.cache.start([("old", compute)])
        self.assertTrue(entered.wait(timeout=2))
        self.cache.pause()
        release.set()
        self.wait_for(lambda: not self.cache.status()["running"])
        self.assertEqual(self.cache.status()["count"], 1)
        self.assertTrue(self.cache.status()["paused"])
        self.cache.start([("new", lambda: b"new")])
        self.wait_for(lambda: self.cache.status()["count"] == 2)
        self.assertEqual(self.cache.get_or_compute("ready", lambda: b"miss"), b"ready")


if __name__ == "__main__":
    unittest.main()
