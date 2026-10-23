"""The judge's untrusted side: the process that imports a submission (or SGLang's baseline) and
runs it on the parent's commands, as an unprivileged user. Started as `python worker.py` in the job's
run/ directory, which holds only this file, the submission and the problem's public files; the parent
(eval.py: the timing, the checks and their thresholds) and the run's cases stay where this process
can't read them. It reads JSON commands on CHILD_CMD_FD and answers on CHILD_REPLY_FD. CHILD_MODULE
names the code it runs: `submission` (untrusted) or `baseline` (SGLang's own path for the problem,
trusted, in a process of its own so a submission can't reach it).
"""

from __future__ import annotations

import functools
import json
import os
import sys
import time
import traceback
from typing import Any, Callable

DTYPES = ("bfloat16", "float16", "float32", "uint8")  # uint8: raw bytes from reference.collect
# About 1 µs of GPU spin in the child between GO and its start event (H200 SM clock ~2 GHz).
# Kept short on purpose: the spin sits inside the parent's window, and a child that skipped
# it would lower the parent's figure by its length (docs/TIMING_FAIRNESS.md).
SPIN_CYCLES = 2_000


# memfd seal constants. Some Python builds' fcntl lacks them (CI's 3.11 did, so the gate test
# failed there); the values are fixed by the Linux ABI (F_LINUX_SPECIFIC_BASE + 9 and the
# F_SEAL_* bits), and a Python that has them gives the same numbers.
try:
    import fcntl as _fcntl
except ImportError:  # not Linux: the gate isn't used
    _fcntl = None
_F_ADD_SEALS = getattr(_fcntl, "F_ADD_SEALS", 1033)
_F_SEAL_SEAL = getattr(_fcntl, "F_SEAL_SEAL", 0x0001)
_F_SEAL_SHRINK = getattr(_fcntl, "F_SEAL_SHRINK", 0x0002)
_F_SEAL_GROW = getattr(_fcntl, "F_SEAL_GROW", 0x0004)


class Gate:
    """The timing gate between the parent and one child: two shared host pages (memfds the
    parent creates and passes at spawn), mapped and registered with CUDA in both processes,
    so either side's GPU stream can wait on or write their words with stream memory
    operations (cuStreamWaitValue32 / cuStreamWriteValue32). CUDA's interprocess events can't
    be used: they can't be opened across users (the child runs as another user), and can't
    time.

    Words: GO (the parent's stream writes it, right after its start event: the child's
    replay may begin) and RELEASE (the parent's host writes it once both sides have enqueued
    their work, so nothing in the window waits on a host) live on the parent's page; DONE
    (the child's stream writes it after the replay: the parent's end event may be recorded)
    on the child's. Both pages are sealed against shrinking under the other side's mapping.

    The child can write the parent's page too: CUDA can only register host memory it may
    write unless the device supports read-only registration, which the H200 nodes don't
    (CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED is 0). The parent refuses a
    replay whose GO or RELEASE word it finds already set, and one whose words don't all hold
    the replay's number afterwards. A GO written in the race after the check is the same as
    not waiting for it; since the parent's stream waits for RELEASE before its flush (#184),
    such a replay either waits behind the flush like an honest one or runs before it, which
    is refused or, run partly outside the window, flagged (_gate_check)."""

    GO, RELEASE, DONE = ("ctl", 0), ("ctl", 1), ("ack", 0)
    SIZE = 4096
    _PROT_READ, _PROT_WRITE, _MAP_SHARED = 1, 2, 1
    _REGISTER = 0x3  # CU_MEMHOSTREGISTER_PORTABLE | CU_MEMHOSTREGISTER_DEVICEMAP

    def __init__(self, ctl_fd: int, ack_fd: int):
        """Maps both pages (no CUDA yet; see register)."""
        import ctypes

        self._ct = ctypes
        libc = ctypes.CDLL(None, use_errno=True)
        libc.mmap.restype = ctypes.c_void_p
        libc.mmap.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                              ctypes.c_long)
        self._host: dict[str, int] = {}
        self._words: dict[str, Any] = {}
        for page, fd in (("ctl", ctl_fd), ("ack", ack_fd)):
            addr = libc.mmap(None, self.SIZE, self._PROT_READ | self._PROT_WRITE, self._MAP_SHARED, fd, 0)
            if addr in (None, ctypes.c_void_p(-1).value):
                raise OSError(ctypes.get_errno(), f"mmap of the gate's {page} page failed")
            self._host[page] = addr
            self._words[page] = (ctypes.c_uint32 * (self.SIZE // 4)).from_address(addr)
        self._dev: dict[str, int] | None = None

    def register(self) -> None:
        """Registers both pages with CUDA (needs this process's CUDA context)."""
        if self._dev is not None:
            return
        ct = self._ct
        cuda = ct.CDLL("libcuda.so.1")
        self._cuda = cuda
        dev = {}
        for page, host in self._host.items():
            self._ck(cuda.cuMemHostRegister_v2(ct.c_void_p(host), ct.c_size_t(self.SIZE), ct.c_uint(self._REGISTER)),
                     "cuMemHostRegister")
            ptr = ct.c_uint64()
            self._ck(cuda.cuMemHostGetDevicePointer_v2(ct.byref(ptr), ct.c_void_p(host), ct.c_uint(0)),
                     "cuMemHostGetDevicePointer")
            dev[page] = ptr.value
        self._dev = dev

    @staticmethod
    def create() -> tuple[int, int]:
        """Fresh pages for one child: (the parent's page, the child's page), both sealed
        against shrinking and growing."""
        import fcntl

        fds = []
        for name in ("kc-gate-ctl", "kc-gate-ack"):
            fd = os.memfd_create(name, os.MFD_ALLOW_SEALING)
            os.ftruncate(fd, Gate.SIZE)
            fcntl.fcntl(fd, _F_ADD_SEALS, _F_SEAL_SHRINK | _F_SEAL_GROW | _F_SEAL_SEAL)
            fds.append(fd)
        return fds[0], fds[1]

    @staticmethod
    def _ck(result: int, what: str) -> None:
        if result != 0:
            raise RuntimeError(f"{what} failed (CUresult {result})")

    def __getitem__(self, word: tuple[str, int]) -> int:
        return self._words[word[0]][word[1]]

    def __setitem__(self, word: tuple[str, int], value: int) -> None:
        self._words[word[0]][word[1]] = value

    def _addr(self, word: tuple[str, int]) -> int:
        return self._dev[word[0]] + 4 * word[1]

    def wait(self, stream, word: tuple[str, int], value: int) -> None:
        """The stream waits until the word is >= value (CU_STREAM_WAIT_VALUE_GEQ)."""
        self.register()
        ct = self._ct
        self._ck(self._cuda.cuStreamWaitValue32_v2(ct.c_void_p(stream.cuda_stream), ct.c_uint64(self._addr(word)),
                                                   ct.c_uint32(value), ct.c_uint(0)), "cuStreamWaitValue32")

    def write(self, stream, word: tuple[str, int], value: int) -> None:
        self.register()
        ct = self._ct
        self._ck(self._cuda.cuStreamWriteValue32_v2(ct.c_void_p(stream.cuda_stream), ct.c_uint64(self._addr(word)),
                                                    ct.c_uint32(value), ct.c_uint(0)), "cuStreamWriteValue32")


def _busy_us(prof, is_kernel) -> float | None:
    """How long any kernel was running: the union of the kernels' intervals. Less than the sum
    of their times when kernels overlap: programmatic dependent launch (PDL), where a kernel
    starts before the one it depends on ends and its time includes the wait, or several
    streams. None if the profiler gave no intervals."""
    spans = sorted((e.time_range.start, e.time_range.end) for e in prof.events()
                   if is_kernel(e) and getattr(e, "time_range", None) is not None
                   and e.time_range.end > e.time_range.start)
    if not spans:
        return None
    busy, end = 0.0, float("-inf")
    for start, stop in spans:
        if stop > end:
            busy += stop - max(start, end)
            end = stop
    return busy


def _profile_table(prof, rows: int = 12) -> str:
    """The kernels that took the most GPU time, compactly: results must stay small."""
    def device_us(e) -> float:
        return float(getattr(e, "self_device_time_total", None) or getattr(e, "self_cuda_time_total", 0) or 0)

    def is_kernel(e) -> bool:
        # GPU kernels only: CPU-side ops (aten::*) also report the device time of the kernels
        # they launch, which would count everything twice.
        kind = getattr(e, "device_type", None)
        if kind is not None:
            return str(kind).endswith(("CUDA", "PrivateUse1"))
        return not e.key.startswith(("aten::", "cuda", "Memcpy", "Memset"))

    events = sorted((e for e in prof.key_averages() if device_us(e) > 0 and is_kernel(e)),
                    key=device_us, reverse=True)
    total = sum(device_us(e) for e in events) or 1.0
    lines = [f"{'kernel':<60} {'GPU time':>12} {'share':>6} {'calls':>6}"]
    for e in events[:rows]:
        name = e.key if len(e.key) <= 60 else e.key[:57] + "..."
        lines.append(f"{name:<60} {device_us(e):>9.1f} µs {100 * device_us(e) / total:>5.1f}% {e.count:>6}")
    if len(events) > rows:
        lines.append(f"... {len(events) - rows} more")
    lines.append(f"{'total':<60} {total:>9.1f} µs")
    try:
        busy = _busy_us(prof, is_kernel)
    except Exception:  # noqa: BLE001 - the table stands without it
        busy = None
    if busy is not None and busy < 0.99 * total:
        lines.append(f"{'GPU busy (kernels overlap: PDL or streams)':<60} {busy:>9.1f} µs")
        lines.append("an overlapping kernel's time includes any wait for the one before it, so the rows add up to more")
    return "\n".join(lines)


def _copy_into(dst, src, path: str = "data") -> None:
    """Copy fresh inputs into the tensors a captured graph reads, in place."""
    import torch

    if isinstance(dst, torch.Tensor):
        if not isinstance(src, torch.Tensor) or src.shape != dst.shape or src.dtype != dst.dtype:
            raise ValueError(f"{path}: inputs changed shape between repetitions")
        dst.copy_(src)
    elif isinstance(dst, dict):
        if not isinstance(src, dict) or dst.keys() != src.keys():
            raise ValueError(f"{path}: inputs changed structure between repetitions")
        for k in dst:
            _copy_into(dst[k], src[k], f"{path}.{k}")
    elif isinstance(dst, (list, tuple)):
        if not isinstance(src, (list, tuple)) or len(dst) != len(src):
            raise ValueError(f"{path}: inputs changed structure between repetitions")
        for i, (d, x) in enumerate(zip(dst, src)):
            _copy_into(d, x, f"{path}[{i}]")
    elif dst != src:
        raise ValueError(f"{path}: inputs changed between repetitions")


def _state_bytes(obj, seen: set | None = None) -> int:
    """The bytes of tensor storage in a prepared state that _clone_state would copy."""
    import torch

    seen = set() if seen is None else seen
    if type(obj) is torch.Tensor:
        if obj.untyped_storage().data_ptr() in seen:
            return 0
        seen.add(obj.untyped_storage().data_ptr())
        return obj.untyped_storage().nbytes()
    if type(obj) is dict:
        return sum(_state_bytes(v, seen) for v in obj.values())
    if type(obj) in (list, tuple):
        return sum(_state_bytes(v, seen) for v in obj)
    return 0


def _clone_state(obj, memo: dict):
    """A copy of a prepared state in fresh memory: every tensor in plain dicts, lists and
    tuples gets new storage with its exact size, strides and offset, and tensors that shared
    storage still do (memo). Anything else (other objects, tensor subclasses) is the same
    object as before."""
    import torch

    if type(obj) is torch.Tensor:
        old = obj.untyped_storage()
        new = memo.get(old.data_ptr())
        if new is None:
            new = memo[old.data_ptr()] = old.clone()
        return torch.empty(0, dtype=obj.dtype, device=obj.device).set_(new, obj.storage_offset(), obj.size(),
                                                                       obj.stride())
    if type(obj) is dict:
        return {k: _clone_state(v, memo) for k, v in obj.items()}
    if type(obj) in (list, tuple):
        return type(obj)(_clone_state(v, memo) for v in obj)
    return obj


def _attr_signature(prims: list[tuple[Any, str]]) -> list[tuple[Any, Any]]:
    """For each (object, attribute), the bound value itself and the owner's own __dict__ entry.
    Snapshot before a submission is imported and compared after with `_same_signature` (`is`,
    never `id()`): it changes if a timing primitive is rebound by assignment, through an alias,
    via setattr, or by mutating the type's real __dict__ with ctypes + PyType_Modified (child_main,
    _timing_intact). The snapshot holds references to the original objects, so they can't be
    freed and their address handed to a replacement (an `id()` comparison is fooled by exactly
    that). A class in the MRO without the attribute (pybind11_object has no `replay`) records it
    as absent; adding it there later changes the signature too."""
    return [(getattr(obj, attr, None), vars(obj).get(attr)) for obj, attr in prims]


def _same_signature(a: list[tuple[Any, Any]], b: list[tuple[Any, Any]]) -> bool:
    """Whether two `_attr_signature` snapshots name the very same objects (identity, not equality
    or address)."""
    return len(a) == len(b) and all(x is y and u is v for (x, u), (y, v) in zip(a, b))


def _timing_prims(Event: type, CUDAGraph: type, cuda: Any) -> list[tuple[Any, str]]:
    """Every (class, attribute) the child's timing path depends on, across each class's whole
    MRO. torch's `Event.record` / `elapsed_time` / `synchronize` / `query` and `CUDAGraph.replay`
    are thin Python wrappers that call `super().<name>()` into the C base (`_CudaEventBase`,
    `_CUDAGraph`); the actual clock readback lives on the base. Snapshotting only the subclass
    would miss a base-class swap — the same gc + __dict__ + PyType_Modified mutation that beat the
    static checks, aimed one level down, which the subclass wrapper would still call through. So
    include each base in the MRO (object excepted, which defines none of these)."""
    def across_mro(cls: type, attrs: tuple[str, ...]) -> list[tuple[Any, str]]:
        return [(c, a) for c in cls.__mro__ if c is not object for a in attrs]
    return (across_mro(Event, ("record", "elapsed_time", "synchronize", "query"))
            + across_mro(CUDAGraph, ("replay",))
            + [(cuda, "synchronize")])


def _timing_tripwire(Event: type, CUDAGraph: type, cuda: Any) -> Callable[[], bool]:
    """Snapshot every timing primitive now (child_main calls this before the submission is
    imported) and return a check that is False once any of them has been rebound since. The
    closure keeps the snapshot, and with it the original objects, alive for the child's life."""
    prims = _timing_prims(Event, CUDAGraph, cuda)
    before = _attr_signature(prims)

    def intact() -> bool:
        return _same_signature(_attr_signature(prims), before)

    return intact


def _setup_op(msg: dict, module: str, state: dict, sync: Callable[[], None], perf: Callable[[], int]) -> dict:
    """The setup op: the run's weights, handed to the module's prepare() once, then copies
    of what it returned (_clone_state)."""
    import importlib

    import torch

    from reference import generate_weights

    # The parent's random pad, allocated before the weights and prepare() and held for the
    # whole run, so whatever prepare() keeps (an nn.Module, any object _clone_state can't copy)
    # lands somewhere different each run instead of where this process's fixed allocation
    # order puts it every time (eval.py _setup).
    pads = [int(n) for n in msg.get("prepare_pad") or []]
    state["prepare_pad"] = [torch.empty(n, dtype=torch.uint8, device="cuda") for n in pads]
    prepare = importlib.import_module(module).prepare
    weights = generate_weights(msg["seed"], **(msg.get("fields") or {}))
    sync()
    t0 = perf()
    state["prep"] = prepare(weights)
    sync()
    del weights
    prepare_s = (perf() - t0) / 1e9
    # Copies of what prepare() returned, each behind the parent's pad, so the timed
    # cases run on several placements in memory (eval.py _place): as many as fit in
    # copies_bytes. Each is optional; the originals stay, so anything that points
    # at them still does.
    state["preps"], state["wpads"] = [state["prep"]], []
    size = _state_bytes(state["prep"])
    room = int(msg.get("copies_bytes", 0)) // size - 1 if size else 0
    for pad in (msg.get("copy_pads") or [])[:max(room, 0)]:
        try:
            wpad = torch.empty(int(pad), dtype=torch.uint8, device="cuda")
            prep = _clone_state(state["prep"], {})
            sync()
        except Exception:  # noqa: BLE001 - out of memory: run on fewer copies
            break
        state["preps"].append(prep)
        state["wpads"].append(wpad)
        del prep, wpad
    torch.cuda.empty_cache()
    return {"ok": True, "prepare_s": prepare_s, "copies": len(state["preps"]), "prepare_pad": sum(pads)}


def failure(e: BaseException) -> dict:
    """A failure's one line for the parent (the run card's "first fail"), and its traceback on
    stderr, which reaches the run's log (log_tail, profiles/*.log), so the line it came from is
    there too."""
    traceback.print_exception(e)
    return {"error": f"{type(e).__name__}: {e}"[:2000]}


def child_main() -> int:
    cmd = os.fdopen(int(os.environ["CHILD_CMD_FD"]), "r")
    reply = os.fdopen(int(os.environ["CHILD_REPLY_FD"]), "wb")
    module = os.environ.get("CHILD_MODULE", "submission")

    import importlib

    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    sync, perf, Event = torch.cuda.synchronize, time.perf_counter_ns, torch.cuda.Event
    # Timing primitives, bound to the original functions here — before any submission code is
    # imported — and called explicitly (`_record(ev)`, not `ev.record()`). The score is this
    # process's event time, so a submission that rebinds a timer to under-report would move the
    # leaderboard. Looking the method up on the instance at call time (`ev.elapsed_time(end)`)
    # reads whatever the class exposes *now*, which a submission can replace at import by
    # assignment, through an alias, via setattr, or by mutating the type's real __dict__ with
    # ctypes + PyType_Modified (no attribute name ever written). A local bound to the original
    # function object is immune: replacing the class binding doesn't touch the object this path
    # already holds. `_timing_intact()` is the tripwire that also *rejects* the attempt, so a
    # modified timer never just scores silently (docs/JUDGE_AUDIT.md §5).
    _record, _elapsed, _ev_sync = Event.record, Event.elapsed_time, Event.synchronize
    _replay = torch.cuda.CUDAGraph.replay
    # Across each class's whole MRO, so a swap of the C base's method (which the Python wrapper
    # the timing path holds calls through `super()`) is caught too, not only a subclass rebind.
    _timing_intact = _timing_tripwire(Event, torch.cuda.CUDAGraph, torch.cuda)

    state: dict[str, Any] = {}

    def send(header: dict, payload: bytes = b"") -> None:
        reply.write(json.dumps(header).encode() + b"\n")
        if payload:
            reply.write(payload)
        reply.flush()

    # Before any submission code is imported: the CUDA context and the gate. The first line
    # this process writes says whether that worked, so the parent can tell a broken judge
    # (an infrastructure error, rerunnable) from a broken submission.
    try:
        torch.zeros(1, device="cuda")
        # Load the spin kernel now: nothing may need CUDA's lazy module loader while the
        # stream waits on the gate (loading synchronizes the context, which would wait for
        # the gate forever).
        torch.cuda._sleep(1)
        sync()
        gate = Gate(int(os.environ["CHILD_GATE_CTL_FD"]), int(os.environ["CHILD_GATE_ACK_FD"]))
        gate.register()
    except BaseException as e:  # noqa: BLE001
        send({"hello": False, "error": f"timing gate setup failed: {type(e).__name__}: {e}"[:2000]})
        return 1
    send({"hello": True})

    def kernel():
        sub = importlib.import_module(module)
        fn = getattr(sub, "kernel_fn", None)
        if fn is None:
            # As static_checks.missing_entry_point says it, for a run whose static checks were skipped.
            renamed = " (renamed from custom_kernel: rename `def custom_kernel(` to `def kernel_fn(`)"
            raise AttributeError(f"no kernel_fn function defined{renamed if hasattr(sub, 'custom_kernel') else ''}")
        if "prep" in state:
            return functools.partial(lambda k, p, d: k(d, p), fn, state["prep"])
        return fn

    for line in cmd:
        msg = json.loads(line)
        op = msg["op"]
        try:
            if op == "setup":
                send(_setup_op(msg, module, state, sync, perf))
            elif op == "prepare":
                from reference import generate_input

                if "graph" in state:
                    # A captured case: refill the graph's inputs in place.
                    fresh = generate_input(**msg["case"])
                    _copy_into(state["data"], fresh)
                    del fresh
                else:
                    # Free the previous inputs first: a large case would otherwise briefly hold two.
                    state.pop("out", None)
                    state.pop("data", None)
                    preps = state.get("preps")
                    if preps:
                        state["prep"] = preps[int(msg.get("copy", 0)) % len(preps)]
                    if msg.get("pad"):
                        # The parent's choice of placement: allocated and held before the
                        # inputs, so they and the graph captured on them land elsewhere.
                        state["pad"] = [torch.empty(int(n), dtype=torch.uint8, device="cuda") for n in msg["pad"]]
                    state["data"] = generate_input(**msg["case"])
                sync()
                send({"ok": True})
            elif op == "release":
                # Drop the captured graph and its buffers before the next case.
                for key in ("graph", "graph_out", "out", "data", "pending", "pad"):
                    state.pop(key, None)
                torch.cuda.empty_cache()
                send({"ok": True})
            elif op == "capture":
                # Every case is timed as CUDA graph replays, as production replays decode: two
                # eager warmup calls (compilation, autotuning), then one captured call.
                try:
                    fn = kernel()
                    for _ in range(2):
                        fn(state["data"])
                    sync()
                except BaseException as e:  # noqa: BLE001 - kernel_fn's own error, not the capture's
                    send({**failure(e), "eager": True})
                    continue
                graph = torch.cuda.CUDAGraph()
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    with torch.cuda.graph(graph, stream=stream):
                        state["graph_out"] = fn(state["data"])
                torch.cuda.current_stream().wait_stream(stream)
                sync()
                # One untimed replay, so the graph's first launch (its upload) is done before
                # any replay waits behind the timing gate.
                _replay(graph)
                sync()
                state["graph"] = graph
                send({"ok": True})
            elif op == "run":
                # One untimed call (the tests): eager, or the captured graph if there is one.
                graph = state.get("graph")
                if graph is not None:
                    _replay(graph)
                    out = state["graph_out"]
                else:
                    out = kernel()(state.get("data"))
                sync()
                if msg.get("keep"):
                    state["out"] = out
                del out
                send({"ok": True})
            elif op == "timed":
                # One replay inside the parent's timing gate. Enqueued, not run: the stream
                # waits for the parent's GO, replays, and signals DONE; the parent records the
                # CUDA events around all of it and flushes L2 first (see Timer.replay). The events
                # here are this process's own view, which the parent checks against its own.
                which = msg.get("graph", "case")
                graph = state["graph"] if which == "case" else state["calib"][which]
                seq = int(msg["seq"])
                stream = torch.cuda.current_stream()
                start, end = Event(enable_timing=True), Event(enable_timing=True)
                gate.wait(stream, Gate.GO, seq)
                # A short GPU spin before the start event, the parent's choice of length
                # (SPIN_CYCLES): insurance against a slow first piece of work after the
                # context switch.
                torch.cuda._sleep(int(msg["spin"]))
                _record(start)
                _replay(graph)
                _record(end)
                gate.write(stream, Gate.DONE, seq)
                state["pending"] = (start, end, which == "case" and bool(msg.get("keep")))
                send({"ok": True})
            elif op == "finish":
                start, end, keep = state.pop("pending")
                _ev_sync(end)
                # Reject, don't score, if any timing primitive was rebound since before the
                # submission was imported — even though the measurement above went through the
                # captured originals and is already honest. A modified timer is never a valid run.
                if not _timing_intact():
                    send({"error": "a timing primitive (CUDA event or graph replay) was modified "
                                   "during the run", "integrity": True})
                    continue
                if keep:
                    state["out"] = state["graph_out"]
                send({"ok": True, "event_ns": _elapsed(start, end) * 1e6})
            elif op == "calibrate":
                # Graphs of one and of two trivial kernels, replayed through the gate like a
                # scored repetition (the parent drives them; see _calibrate). One kernel's
                # own time is (two - one), so a replay's fixed cost is one - (two - one).
                tiny = torch.zeros(1, device="cuda")
                graphs = {}
                for n in (1, 2):
                    for _ in range(2):
                        for _ in range(n):
                            tiny.add_(1)
                    sync()
                    g = torch.cuda.CUDAGraph()
                    cs = torch.cuda.Stream()
                    cs.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(cs), torch.cuda.graph(g, stream=cs):
                        for _ in range(n):
                            tiny.add_(1)
                    torch.cuda.current_stream().wait_stream(cs)
                    _replay(g)
                    sync()
                    graphs["one" if n == 1 else "two"] = g
                state["calib"] = graphs
                state["calib_buf"] = tiny
                send({"ok": True})
            elif op == "burst":
                # Back-to-back replays, timed by the parent from outside.
                graph = state["graph"]
                sync()
                for _ in range(int(msg["n"])):
                    _replay(graph)
                sync()
                send({"ok": True})
            elif op == "fetch":
                out = state.pop("out", None)
                import reference

                if hasattr(reference, "collect"):
                    # The output plus whatever else the step must write (e.g. cache entries).
                    out = reference.collect(state.get("data"), out)
                if type(out) is not torch.Tensor:  # exact type: no subclasses or lazy objects
                    send({"error": f"expected a torch.Tensor, got {type(out).__name__}"})
                    continue
                dtype = str(out.dtype).removeprefix("torch.")
                if dtype not in DTYPES:
                    send({"error": f"unsupported output dtype {dtype}"})
                    continue
                raw = out.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
                send({"ok": True, "dtype": dtype, "shape": list(out.shape), "nbytes": len(raw)}, raw)
            elif op == "profile":
                from reference import generate_input
                from torch.profiler import ProfilerActivity, profile

                data = generate_input(**msg["case"])
                fn = kernel()
                fn(data)
                sync()
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                    fn(data)
                    sync()
                send({"ok": True, "report": _profile_table(prof)})
            elif op == "exit":
                return 0
        except BaseException as e:  # noqa: BLE001 - every failure goes back as text
            send(failure(e))
    return 0



if __name__ == "__main__":
    sys.exit(child_main())
