from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, Union

from msprobe.core.common.log import logger


Payload = Union[bytes, bytearray, Iterable[Union[str, bytes]]]

_WORKER_THREADS = 2  # 写线程数
_MAX_PENDING = 1024  # 待写任务上限：满了直接丢弃，宁可丢监控数据也不拖垮引擎


class AsyncDumpWriter:
    """异步落盘：固定大小的线程池 + 有界待写队列。

    磁盘写入跟不上引擎步进时，待写任务会在内存里堆积；这里用信号量限制在途数量，
    超出即丢弃并告警 —— 监控数据允许丢，引擎进程不允许被拖垮。
    """

    def __init__(self, worker_threads: int = _WORKER_THREADS, max_pending: int = _MAX_PENDING):
        self._worker_threads = worker_threads
        self._free = threading.BoundedSemaphore(max(max_pending, 1))
        self._dropped = 0
        self._executor: ThreadPoolExecutor | None = None
        self._init_lock = threading.Lock()

    def submit(self, path: str, payload: Payload) -> None:
        executor = self._ensure_executor()
        if executor is None:
            self._write_sync(path, payload)
            return
        # 非阻塞 acquire 无法用 with 表达（pylint 的 R1732 误报）
        if not self._free.acquire(blocking=False):  # pylint: disable=consider-using-with
            self._dropped += 1
            logger.warning(f"[kv-dump-writer] pending queue full, drop: {path} (dropped={self._dropped})")
            return
        try:
            executor.submit(self._run, path, payload)
        except RuntimeError:
            # 线程池已关闭：退回同步写
            self._free.release()
            self._write_sync(path, payload)

    def _run(self, path: str, payload: Payload) -> None:
        try:
            self._safe_write(path, payload)
        finally:
            self._free.release()

    def _ensure_executor(self) -> ThreadPoolExecutor | None:
        if self._worker_threads <= 0:
            return None
        if self._executor is None:
            with self._init_lock:
                if self._executor is None:
                    self._executor = ThreadPoolExecutor(
                        max_workers=self._worker_threads,
                        thread_name_prefix="kv-dump-writer",
                    )
        return self._executor

    def _safe_write(self, path: str, payload: Payload, *, sync: bool = False) -> None:
        try:
            self._write_file(path, payload)
        except Exception as exc:
            if sync:
                logger.warning(f"[kv-dump-writer] sync write failed: {path}: {exc}")
            else:
                logger.warning(f"[kv-dump-writer] write failed: {path}: {exc}")

    def _write_sync(self, path: str, payload: Payload) -> None:
        self._safe_write(path, payload, sync=True)

    @staticmethod
    def _write_file(path: str, payload: Payload) -> None:
        with open(path, "wb") as f:
            if isinstance(payload, (bytes, bytearray)):
                f.write(payload)
                return
            for chunk in payload:
                f.write(chunk if isinstance(chunk, bytes) else chunk.encode("utf-8"))


default_writer = AsyncDumpWriter()
