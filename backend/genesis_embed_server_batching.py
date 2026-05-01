from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from .genesis_embed_server_globals import batch_history, batch_runtime_state, batch_state_lock
from .genesis_embed_server_runtime import EncodeResult, RuntimeKey, build_runtime_key


@dataclass
class EmbedBatchRequest:
    request_id: str
    model_id: str
    texts: list[str]
    runtime_key: RuntimeKey
    future: asyncio.Future[EncodeResult]
    enqueued_at: float


class EmbedBatchManager:
    def __init__(self, encoder: Callable[[str, list[str]], Awaitable[EncodeResult]]):
        self._encoder = encoder
        self._queue: asyncio.Queue[EmbedBatchRequest] = asyncio.Queue()
        self._worker_task: asyncio.Task[None] | None = None
        self._stopping = asyncio.Event()

    async def start(self) -> None:
        if self._worker_task is None or self._worker_task.done():
            self._stopping.clear()
            self._worker_task = asyncio.create_task(self._worker())
            with batch_state_lock:
                batch_runtime_state["worker_running"] = True

    async def stop(self) -> None:
        self._stopping.set()
        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        with batch_state_lock:
            batch_runtime_state["worker_running"] = False
            batch_runtime_state["queue_size"] = 0

    async def enqueue(self, model_id: str, texts: list[str], settings: dict[str, Any]) -> EncodeResult:
        runtime_key, _, _ = build_runtime_key(model_id, settings)
        loop = asyncio.get_running_loop()
        future: asyncio.Future[EncodeResult] = loop.create_future()
        request = EmbedBatchRequest(
            request_id=uuid.uuid4().hex[:12],
            model_id=runtime_key.model_id,
            texts=texts,
            runtime_key=runtime_key,
            future=future,
            enqueued_at=time.perf_counter(),
        )
        await self._queue.put(request)
        with batch_state_lock:
            batch_runtime_state["queue_size"] = self._queue.qsize()
        return await future

    def snapshot(self) -> dict[str, Any]:
        with batch_state_lock:
            payload = dict(batch_runtime_state)
            payload["recent_batches"] = list(batch_history)
            payload["queue_size"] = self._queue.qsize()
        return payload

    async def _worker(self) -> None:
        while not self._stopping.is_set():
            first = await self._queue.get()
            group = [first]
            deferred: list[EmbedBatchRequest] = []
            max_texts, max_chars, wait_seconds = self._limits()
            current_texts = len(first.texts)
            current_chars = sum(len(text) for text in first.texts)
            deadline = time.perf_counter() + wait_seconds

            while time.perf_counter() < deadline and current_texts < max_texts and current_chars < max_chars:
                timeout = max(0.0, deadline - time.perf_counter())
                try:
                    candidate = await asyncio.wait_for(self._queue.get(), timeout=timeout)
                except asyncio.TimeoutError:
                    break
                candidate_texts = len(candidate.texts)
                candidate_chars = sum(len(text) for text in candidate.texts)
                can_join = (
                    candidate.runtime_key == first.runtime_key
                    and current_texts + candidate_texts <= max_texts
                    and current_chars + candidate_chars <= max_chars
                )
                if can_join:
                    group.append(candidate)
                    current_texts += candidate_texts
                    current_chars += candidate_chars
                else:
                    deferred.append(candidate)

            for item in deferred:
                await self._queue.put(item)

            await self._run_group(group, current_texts, current_chars)

    def _limits(self) -> tuple[int, int, float]:
        from .genesis_embed_server_globals import current_settings, settings_lock

        with settings_lock:
            settings = dict(current_settings)
        return (
            max(1, int(settings.get("batch_max_texts") or 32)),
            max(256, int(settings.get("batch_max_chars") or 64000)),
            max(0, int(settings.get("batch_wait_time_ms") or 0)) / 1000,
        )

    async def _run_group(self, group: list[EmbedBatchRequest], total_texts: int, total_chars: int) -> None:
        batch_id = f"batch-{uuid.uuid4().hex[:10]}"
        started_at = datetime.now(timezone.utc).isoformat()
        started = time.perf_counter()
        with batch_state_lock:
            batch_runtime_state.update({
                "queue_size": self._queue.qsize(),
                "active_batch_id": batch_id,
                "active_batch_size": total_texts,
                "active_batch_chars": total_chars,
                "active_batch_started_at": started_at,
                "last_error": None,
            })

        all_texts: list[str] = []
        spans: list[tuple[EmbedBatchRequest, int, int]] = []
        cursor = 0
        for request in group:
            all_texts.extend(request.texts)
            next_cursor = cursor + len(request.texts)
            spans.append((request, cursor, next_cursor))
            cursor = next_cursor

        try:
            combined = await self._encoder(group[0].model_id, all_texts)
            for request, start, end in spans:
                sliced = EncodeResult(
                    model_id=combined.model_id,
                    dimension=combined.dimension,
                    vectors=combined.vectors[start:end],
                    backend=combined.backend,
                    device=combined.device,
                    device_label=combined.device_label,
                    load_duration_ms=combined.load_duration_ms,
                    encode_duration_ms=combined.encode_duration_ms,
                    total_duration_ms=combined.total_duration_ms,
                    texts_per_second=combined.texts_per_second,
                    input_count=len(request.texts),
                    input_chars=sum(len(text) for text in request.texts),
                )
                if not request.future.done():
                    request.future.set_result(sliced)
            status = "ok"
            error = None
            texts_per_second = combined.texts_per_second
        except Exception as exc:
            status = "error"
            error = str(exc)
            texts_per_second = None
            for request in group:
                if not request.future.done():
                    request.future.set_exception(exc)

        duration_ms = round((time.perf_counter() - started) * 1000, 3)
        completed_at = datetime.now(timezone.utc).isoformat()
        with batch_state_lock:
            batch_runtime_state.update({
                "queue_size": self._queue.qsize(),
                "active_batch_id": None,
                "active_batch_size": 0,
                "active_batch_chars": 0,
                "active_batch_started_at": None,
                "last_batch_completed_at": completed_at,
                "last_batch_duration_ms": duration_ms,
                "last_batch_texts_per_second": texts_per_second,
                "last_error": error,
                "total_batches_processed": batch_runtime_state["total_batches_processed"] + 1,
                "total_texts_processed": batch_runtime_state["total_texts_processed"] + total_texts,
            })
            batch_history.appendleft({
                "batch_id": batch_id,
                "timestamp": completed_at,
                "batch_size": total_texts,
                "char_count": total_chars,
                "duration_ms": duration_ms,
                "texts_per_second": texts_per_second,
                "status": status,
                "error": error,
            })
