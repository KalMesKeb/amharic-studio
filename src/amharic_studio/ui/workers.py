"""Background workers.

Recognition, import, export and assessment all run off the GUI thread. The pattern is
deliberately thin: a worker wraps one callable that receives a progress reporter and a
cancellation event, so :mod:`amharic_studio.core` never needs to know Qt exists and stays
testable and scriptable on its own.
"""

from __future__ import annotations

import threading
import traceback
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, QThread, Signal

#: ``(done, total, message)``
ProgressFn = Callable[[int, int, str], None]
#: A unit of background work.
TaskFn = Callable[[ProgressFn, threading.Event], Any]


class Worker(QThread):
    """Runs one callable on a background thread with progress and cancellation."""

    progress = Signal(int, int, str)
    finishedOk = Signal(object)
    failed = Signal(str)

    def __init__(self, task: TaskFn, description: str = "", parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.task = task
        self.description = description
        self.cancel_event = threading.Event()
        self._result: Any = None

    @property
    def result(self) -> Any:
        return self._result

    def cancel(self) -> None:
        self.cancel_event.set()

    @property
    def cancelled(self) -> bool:
        return self.cancel_event.is_set()

    def run(self) -> None:  # noqa: D102 - QThread entry point
        try:
            self._result = self.task(self._report, self.cancel_event)
            self.finishedOk.emit(self._result)
        except Exception as exc:  # noqa: BLE001 - surface any failure to the UI
            self.failed.emit(f"{exc}\n\n{traceback.format_exc()}")

    def _report(self, done: int, total: int, message: str = "") -> None:
        self.progress.emit(done, total, message)


class WorkerManager(QObject):
    """Keeps references to running workers so they are not garbage collected mid-run."""

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._workers: list[Worker] = []

    def start(
        self,
        task: TaskFn,
        description: str = "",
        on_progress: Callable[[int, int, str], None] | None = None,
        on_done: Callable[[Any], None] | None = None,
        on_error: Callable[[str], None] | None = None,
    ) -> Worker:
        worker = Worker(task, description)
        if on_progress:
            worker.progress.connect(on_progress)
        if on_done:
            worker.finishedOk.connect(on_done)
        if on_error:
            worker.failed.connect(on_error)

        worker.finished.connect(lambda: self._retire(worker))
        self._workers.append(worker)
        worker.start()
        return worker

    def _retire(self, worker: Worker) -> None:
        if worker in self._workers:
            self._workers.remove(worker)

    @property
    def busy(self) -> bool:
        return any(w.isRunning() for w in self._workers)

    def cancel_all(self) -> None:
        for worker in self._workers:
            worker.cancel()

    def wait_all(self, timeout_ms: int = 5000) -> None:
        for worker in list(self._workers):
            worker.wait(timeout_ms)
