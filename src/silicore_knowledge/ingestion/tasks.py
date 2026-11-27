# SPDX-FileCopyrightText: 2026 H2Lab Development Team
# SPDX-License-Identifier: Apache-2.0

"""Background indexing tasks with status tracking."""

import logging
import threading
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from enum import StrEnum, auto, unique

from pydantic import BaseModel

from ..storage.catalog import Catalog
from .pipeline import Indexer

logger = logging.getLogger("silicore_knowledge")


@unique
class TaskStatus(StrEnum):
    """Lifecycle of an indexing task."""

    QUEUED = auto()
    RUNNING = auto()
    DONE = auto()
    FAILED = auto()


class TaskInfo(BaseModel):
    """Public view of an indexing task."""

    task_id: str
    document_id: str
    status: TaskStatus
    error: str | None = None
    chunks: int | None = None
    created_at: datetime
    finished_at: datetime | None = None


class TaskRegistry:
    """Runs indexing tasks one at a time and remembers the most recent ones.

    Parameters
    ----------
    indexer : Indexer
        Pipeline used to index documents.
    catalog : Catalog
        Catalog, used to report the chunk count.
    max_tasks : int
        Number of task records kept; the oldest finished ones are dropped first.
    """

    def __init__(self, indexer: Indexer, catalog: Catalog, max_tasks: int = 200) -> None:
        self._indexer = indexer
        self._catalog = catalog
        self._max_tasks = max_tasks
        self._tasks: OrderedDict[str, TaskInfo] = OrderedDict()
        self._lock = threading.Lock()
        # A single worker serializes indexing: embeddings are memory hungry.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="indexing")

    def submit(self, document_id: str) -> TaskInfo:
        """Queue the indexing of a document and return the task record."""
        info = TaskInfo(
            task_id=uuid.uuid4().hex,
            document_id=document_id,
            status=TaskStatus.QUEUED,
            created_at=datetime.now(UTC),
        )
        with self._lock:
            self._tasks[info.task_id] = info
            self._evict()
        self._executor.submit(self._run, info.task_id)
        return info.model_copy()

    def get(self, task_id: str) -> TaskInfo | None:
        """Return a snapshot of a task, or None if unknown or evicted."""
        with self._lock:
            info = self._tasks.get(task_id)
            return None if info is None else info.model_copy()

    def close(self) -> None:
        """Stop accepting work and wait for the running task."""
        self._executor.shutdown(wait=True, cancel_futures=True)

    def _evict(self) -> None:
        for task_id in list(self._tasks):
            if len(self._tasks) <= self._max_tasks:
                return
            if self._tasks[task_id].status in (TaskStatus.DONE, TaskStatus.FAILED):
                del self._tasks[task_id]

    def _update(self, task_id: str, **changes: object) -> None:
        with self._lock:
            info = self._tasks.get(task_id)
            if info is not None:
                self._tasks[task_id] = info.model_copy(update=changes)

    def _run(self, task_id: str) -> None:
        info = self.get(task_id)
        if info is None:
            return
        self._update(task_id, status=TaskStatus.RUNNING)
        try:
            report = self._indexer.index_documents([info.document_id])
            error = report.failed.get(info.document_id)
            chunks = self._catalog.chunk_count(info.document_id)
        except Exception:  # a task must always reach a final state
            logger.exception("indexing task %s crashed", task_id)
            error, chunks = "internal error", None
        self._update(
            task_id,
            status=TaskStatus.FAILED if error else TaskStatus.DONE,
            error=error,
            chunks=chunks,
            finished_at=datetime.now(UTC),
        )
