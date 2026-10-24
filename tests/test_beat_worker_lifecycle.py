"""Submission ownership: close/terminate stop the job before returning.

A stub stands in for the long native solve. Like the real helper, it only
notices cancellation at its checkpoints (``cancel_cb``), which the real
sweep calls after every frequency and which then kills the helper process.
"""

import threading
import time

import pytest

api = pytest.importorskip("hornlab_metal_bem")
beat_worker = pytest.importorskip("hornlab_metal_bem.beat_worker")


class BlockingSolve:
    """Runs until cancelled, polling ``cancel_cb`` like a frequency loop."""

    def __init__(self):
        self.active = 0
        self.started = threading.Event()
        self.calls = 0
        self.lock = threading.Lock()

    def __call__(self, request, *, base_dir, cancel_cb):
        with self.lock:
            self.active += 1
            self.calls += 1
        self.started.set()
        try:
            for _ in range(2000):
                cancel_cb()
                time.sleep(0.005)
            raise AssertionError("blocking solve was never cancelled")
        finally:
            with self.lock:
                self.active -= 1


@pytest.fixture
def blocking(monkeypatch):
    solve = BlockingSolve()
    monkeypatch.setattr(beat_worker, "solve_compiled", solve)
    return solve


def worker():
    return beat_worker.MetalEngineWorker(availability={"available": True})


def reader(stream, sink):
    def run():
        try:
            sink.extend(stream)
        except BaseException as exc:  # noqa: BLE001  # pragma: no cover - surfaced by assert
            sink.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    return thread


def test_close_while_another_thread_reads_stops_native_work(blocking):
    w = worker()
    stream = w.submit({})
    seen = []
    thread = reader(stream, seen)
    assert blocking.started.wait(5)
    stream.close()  # must not raise "generator already executing"
    assert blocking.active == 0, "native work still running after close()"
    assert not stream.running
    thread.join(5)
    assert not thread.is_alive()
    assert all(isinstance(e, dict) for e in seen)
    assert [e["type"] for e in seen if e["type"] == "result"] == []
    stream.close()  # repeated close is a no-op


def test_terminate_stops_active_job_and_publishes_nothing_more(blocking):
    w = worker()
    stream = w.submit({})
    assert next(stream)["type"] == "status"
    assert blocking.started.wait(5)
    w.terminate()
    assert blocking.active == 0
    assert w.worker_info is None
    with pytest.raises(StopIteration):
        next(stream)


def test_replacement_starts_after_close_and_stale_close_is_ignored(blocking):
    w = worker()
    first = w.submit({})
    assert blocking.started.wait(5)
    first.close()
    blocking.started.clear()
    second = w.submit({})  # does not wait behind abandoned work
    assert blocking.started.wait(5)
    first.close()  # late close of the old stream
    time.sleep(0.05)
    assert blocking.active == 1 and second.running
    second.close()
    assert blocking.active == 0
    assert blocking.calls == 2


def test_close_before_start_and_submit_waits_for_running_job(blocking):
    w = worker()
    first = w.submit({})
    assert blocking.started.wait(5)
    submitted = threading.Event()
    holder = []

    def submit_second():
        holder.append(w.submit({}))
        submitted.set()

    thread = threading.Thread(target=submit_second)
    thread.start()
    assert not submitted.wait(0.1), "second job overlapped the first"
    first.close()
    assert submitted.wait(5)
    thread.join(5)
    holder[0].close()
    assert blocking.active == 0


def test_unsupported_operation_fails_and_releases_the_slot(monkeypatch):
    monkeypatch.setattr(
        beat_worker,
        "solve_compiled",
        lambda request, *, base_dir, cancel_cb: pytest.fail("not reached"),
    )
    w = worker()
    for _ in range(2):  # the second submit would block if the slot leaked
        events = list(w.submit({}, operation="bem_field"))
        assert [e["type"] for e in events] == ["failed"]
