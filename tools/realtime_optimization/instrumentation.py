"""Temporary instrumentation installed only by the standalone diagnostic pass.

No production modules import this file. Never use these process-global wrappers
inside a running app, worker, device callback, or concurrent inference session.
"""
import contextlib
import functools
import inspect
import time


class StageProfiler:
    def __init__(self, device, *, trace=False):
        self.device = str(device)
        self.trace = trace
        self.stack = []
        self.pending = []
        self.undo = []
        self.torch = None
        if self.device.startswith("cuda") or trace:
            import torch
            self.torch = torch

    @contextlib.contextmanager
    def scope(self, name):
        events = None
        if self.device.startswith("cuda"):
            events = (self.torch.cuda.Event(enable_timing=True),
                      self.torch.cuda.Event(enable_timing=True))
            events[0].record(self.torch.cuda.current_stream(self.device))
        marker = self.torch.profiler.record_function(name) if self.trace else contextlib.nullcontext()
        frame = {"stage": name, "children_ns": 0}
        self.stack.append(frame)
        begin = time.perf_counter_ns()
        try:
            with marker:
                yield
        finally:
            ns = time.perf_counter_ns() - begin
            self.stack.pop()
            if events:
                events[1].record(self.torch.cuda.current_stream(self.device))
            if self.stack:
                self.stack[-1]["children_ns"] += ns
            self.pending.append({"stage": name, "host_inclusive_ms": ns / 1e6,
                                 "host_exclusive_ms": (ns - frame["children_ns"]) / 1e6,
                                 "events": events})

    def wrap(self, obj, attr, name):
        original = getattr(obj, attr)
        local = attr in vars(obj)
        descriptor = inspect.getattr_static(obj, attr) if local else None

        @functools.wraps(original)
        def measured(*args, **kwargs):
            with self.scope(name):
                return original(*args, **kwargs)

        # Class staticmethods must stay static; module and instance functions are plain callables.
        replacement = staticmethod(measured) if isinstance(descriptor, staticmethod) else measured
        setattr(obj, attr, replacement)
        self.undo.append((obj, attr, descriptor, local))

    def finish_chunk(self, chunk):
        # Synchronize once after the whole diagnostic chunk, never inside each stage.
        # Normal service-time runs never install wrappers or execute this synchronization.
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize(self.device)
        rows = []
        for row in self.pending:
            events = row.pop("events")
            row["cuda_stream_span_ms"] = events[0].elapsed_time(events[1]) if events else None
            row["chunk"] = chunk
            rows.append(row)
        self.pending.clear()
        return rows

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        for obj, attr, original, local in reversed(self.undo):
            if local:
                setattr(obj, attr, original)
            else:
                delattr(obj, attr)
        self.undo.clear()


class _FunctionalProxy:
    """Only core.py's F reference changes, so model convolutions remain untouched."""
    def __init__(self, original, profiler):
        self.original, self.profiler = original, profiler

    def __getattr__(self, name):
        function = getattr(self.original, name)
        if name != "conv1d":
            return function

        def measured(*args, **kwargs):
            with self.profiler.scope("sola_correlation"):
                return function(*args, **kwargs)
        return measured


@contextlib.contextmanager
def instrument_voice_changer(vc, *, trace=False):
    from rvc.realtime import core, pipeline
    rt = vc.vc_model
    p = rt.pipeline
    profiler = StageProfiler(vc.device, trace=trace)
    original_f = core.F
    try:
        with profiler:
            profiler.wrap(vc, "process_audio", "chunk_total")
            profiler.wrap(rt, "inference", "realtime_frontend")
            profiler.wrap(rt.resample_in, "forward", "input_resample")
            profiler.wrap(rt.resample_out, "forward", "output_resample")
            profiler.wrap(p, "voice_conversion", "conversion")
            profiler.wrap(p, "get_f0", "f0")
            profiler.wrap(p.hubert_model, "forward", "content_encoder")
            profiler.wrap(p, "_retrieve_speaker_embeddings", "index_retrieval")
            profiler.wrap(p.vc, "inference", "generator")
            profiler.wrap(pipeline.AudioProcessorTorch, "change_rms", "volume_envelope")
            profiler.wrap(core, "circular_write", "audio_buffer_write")
            profiler.wrap(core, "phase_vocoder", "phase_vocoder")
            if rt.vad:
                profiler.wrap(rt.vad, "is_speech", "vad_cpu")
            if rt.reduced_noise:
                profiler.wrap(rt.reduced_noise, "forward", "noise_reduction")
            if rt.board is not None:
                # Pedalboard's native __call__ cannot be replaced on an instance.
                original_board = rt.board
                def board(*args, **kwargs):
                    with profiler.scope("post_processing_cpu"):
                        return original_board(*args, **kwargs)
                rt.board = board
            core.F = _FunctionalProxy(original_f, profiler)
            yield profiler
    finally:
        core.F = original_f
        if rt.board is not None and "original_board" in locals():
            rt.board = original_board
