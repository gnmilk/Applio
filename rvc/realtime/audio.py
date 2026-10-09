import os
import re
import sys
import threading
import weakref
import librosa
import traceback
import numpy as np
import sounddevice as sd
from queue import Queue, Empty
from dataclasses import dataclass

now_dir = os.getcwd()
sys.path.append(now_dir)

from rvc.realtime.core import AUDIO_SAMPLE_RATE


@dataclass
class ServerAudioDevice:
    index: int = 0
    name: str = ""
    host_api: str = ""
    max_input_channels: int = 0
    max_output_channels: int = 0
    default_samplerate: int = 0

    @property
    def label(self) -> str:
        return f"{self.name} ({self.host_api})"


class AudioDeviceError(RuntimeError):
    """An audio endpoint cannot be resolved, validated or opened safely."""


# Serialize control operations only; callbacks never acquire this lock.
_backend_lock = threading.RLock()
_audio_instances = weakref.WeakSet()


def normalize_audio_device_name(value) -> str:
    return re.sub(r"^\s*\d+\s*:\s*", "", str(value or "")).strip()


def reinitialize_audio_backend():
    """Refresh only after every Applio stream has been closed."""
    with _backend_lock:
        if any(
            getattr(audio, name, None) is not None
            for audio in _audio_instances
            for name in ("stream", "input_stream", "output_stream", "monitor")
        ):
            raise AudioDeviceError("Stop Realtime before refreshing audio devices.")
        try:
            sd._terminate()
            sd._initialize()
        except Exception as error:
            raise AudioDeviceError(f"Failed to reinitialize PortAudio: {error}") from error


def resolve_audio_device(selection, direction: str, devices=None) -> ServerAudioDevice:
    """Resolve name + host API in the current inventory, never guessing a duplicate."""
    if direction not in ("input", "output"):
        raise ValueError(f"Unsupported audio direction: {direction}")
    if devices is None:
        inputs, outputs = list_audio_device()
        devices = inputs if direction == "input" else outputs
    if isinstance(selection, ServerAudioDevice):
        selection = selection.label
    if isinstance(selection, int):
        matches = [d for d in devices if d.index == selection]
    else:
        wanted = normalize_audio_device_name(selection).casefold()
        matches = [d for d in devices if d.label.casefold() == wanted]
        # Legacy name-only values are usable only if a single endpoint matches.
        if not matches:
            matches = [d for d in devices if d.name.casefold() == wanted]
    channel_key = "max_input_channels" if direction == "input" else "max_output_channels"
    matches = [d for d in matches if getattr(d, channel_key) > 0]
    if len(matches) == 1:
        return matches[0]
    reason = "ambiguous" if matches else "unavailable"
    available = "\n".join(f"  - {d.label} [index={d.index}]" for d in devices) or "  (none)"
    raise AudioDeviceError(
        f"Selected {direction} device is {reason}: {selection!r}\n"
        f"Current {direction} devices:\n{available}\n"
        "Refresh Audio Devices and select the device and Host API again."
    )


def describe_audio_error(error):
    message = str(error)
    if "-2004287478" in message and "AUDCLNT_E_DEVICE_INVALIDATED" not in message:
        message += " [AUDCLNT_E_DEVICE_INVALIDATED / 0x8889000A]"
    return message


def _fallback_candidates(primary, direction, devices):
    """Old custom fallback: same endpoint only, never an ambiguous shortened name."""
    candidates = [primary]
    if primary.host_api != "Windows WASAPI":
        return candidates
    primary_name = " ".join(primary.name.casefold().split())
    # Do not escalate automatically to kernel-streaming drivers. The old
    # WDM-KS alias has not been verified on this installation.
    for api in ("Windows DirectSound", "MME"):
        matches = []
        for device in devices:
            if device.host_api != api:
                continue
            name = " ".join(device.name.casefold().split())
            same = name == primary_name
            if api == "MME":
                shorter, longer = sorted((name, primary_name), key=len)
                if len(shorter) >= 24 and longer.startswith(shorter):
                    sources = [d for d in devices if d.host_api == primary.host_api
                               and " ".join(d.name.casefold().split()).startswith(name)]
                    same |= len(sources) == 1 and sources[0].index == primary.index
            if same:
                matches.append(device)
        if len(matches) == 1:
            candidates.append(matches[0])
    return candidates


def check_the_device(device, type: str = "input"):
    if sys.platform == "linux" and "hw:" not in device["name"]:
        return False

    if type == "input":
        return device["max_input_channels"] > 0
    else:
        return device["max_output_channels"] > 0


def list_audio_device():
    """
    Function to query audio devices and host api.
    """
    try:
        audio_device_list = sd.query_devices()
    except Exception as e:
        print("An error occurred while querying the audio device:", e)
        audio_device_list = []
    except OSError as e:
        # This error can occur when the libportaudio2 library is missing.
        print("An error occurred while querying the audio device:", e)
        audio_device_list = []

    input_audio_device_list = [
        d
        for d in audio_device_list
        if d["max_input_channels"] > 0 and check_the_device(d, "input")
    ]
    output_audio_device_list = [
        d
        for d in audio_device_list
        if d["max_output_channels"] > 0 and check_the_device(d, "output")
    ]

    try:
        hostapis = sd.query_hostapis()
    except Exception as e:
        print("An error occurred while querying the host api:", e)
        hostapis = []
    except OSError as e:
        # This error can occur when the libportaudio2 library is missing.
        print("An error occurred while querying the host api:", e)
        hostapis = []

    audio_input_device = []
    audio_output_device = []

    for d in input_audio_device_list:
        input_audio_device = ServerAudioDevice(
            index=d["index"],
            name=d["name"],
            host_api=hostapis[d["hostapi"]]["name"],
            max_input_channels=d["max_input_channels"],
            max_output_channels=d["max_output_channels"],
            default_samplerate=d["default_samplerate"],
        )
        audio_input_device.append(input_audio_device)

    for d in output_audio_device_list:
        output_audio_device = ServerAudioDevice(
            index=d["index"],
            name=d["name"],
            host_api=hostapis[d["hostapi"]]["name"],
            max_input_channels=d["max_input_channels"],
            max_output_channels=d["max_output_channels"],
            default_samplerate=d["default_samplerate"],
        )
        audio_output_device.append(output_audio_device)

    return audio_input_device, audio_output_device


def resolve_sample_rate(
    input_device_id: int,
    asio_enabled: bool,
    audio_sample_rate: int,
) -> int:
    """Return the sample rate to use for audio streaming.

    For ASIO the user-selected rate is used as-is because ASIO drivers
    do not expose a reliable default_samplerate via PortAudio.
    For all other host APIs the input device's default_samplerate is used
    so that the stream matches the rate the OS/driver actually operates at.
    """
    if asio_enabled:
        return audio_sample_rate
    try:
        device = sd.query_devices(input_device_id)
        return int(device["default_samplerate"])
    except Exception:
        return audio_sample_rate


class Audio:
    def __init__(
        self,
        callbacks,
        f0_up_key: int = 0,
        index_rate: float = 0.5,
        protect: float = 0.5,
        volume_envelope: float = 1,
        f0_autotune: bool = False,
        f0_autotune_strength: float = 1,
        proposed_pitch: bool = False,
        proposed_pitch_threshold: float = 155.0,
        input_audio_gain: float = 1.0,
        output_audio_gain: float = 1.0,
        monitor_audio_gain: float = 1.0,
        monitor: bool = False,
    ):
        self.callbacks = callbacks
        self.mon_queue = Queue()
        self.stream = None
        self.input_stream = None
        self.output_stream = None
        self.monitor = None
        self.running = False
        self.input_audio_gain = input_audio_gain
        self.output_audio_gain = output_audio_gain
        self.monitor_audio_gain = monitor_audio_gain
        self.use_monitor = monitor
        self.f0_up_key = f0_up_key
        self.index_rate = index_rate
        self.protect = protect
        self.volume_envelope = volume_envelope
        self.f0_autotune = f0_autotune
        self.f0_autotune_strength = f0_autotune_strength
        self.proposed_pitch = proposed_pitch
        self.proposed_pitch_threshold = proposed_pitch_threshold
        _audio_instances.add(self)

    def get_input_audio_device(self, index: int):
        audioinput, _ = list_audio_device()
        serverAudioDevice = [x for x in audioinput if x.index == index]

        return serverAudioDevice[0] if len(serverAudioDevice) > 0 else None

    def get_output_audio_device(self, index: int):
        _, audiooutput = list_audio_device()
        serverAudioDevice = [x for x in audiooutput if x.index == index]

        return serverAudioDevice[0] if len(serverAudioDevice) > 0 else None

    def process_data(self, indata: np.ndarray):
        indata = indata * self.input_audio_gain
        unpacked_data = librosa.to_mono(indata.T)

        return self.callbacks.change_voice(
            unpacked_data,
            self.f0_up_key,
            self.index_rate,
            self.protect,
            self.volume_envelope,
            self.f0_autotune,
            self.f0_autotune_strength,
            self.proposed_pitch,
            self.proposed_pitch_threshold,
        )

    def process_data_with_time(self, indata: np.ndarray):
        out_wav, vol, performance_ms = self.process_data(indata)
        # print(f"real-time voice conversion performance: {performance_ms:.2f} ms")
        self.latency = performance_ms  # latency to display on the application interface
        self.volume = vol

        return out_wav

    def audio_stream_no_output_callback(
        self, indata: np.ndarray, frames, times, status
    ):
        try:
            out_wav = self.process_data_with_time(indata)

            self.mon_queue.put(out_wav)
        except Exception as error:
            print(f"An error occurred while running the audio stream: {error}")
            print(traceback.format_exc())

    def audio_stream_callback(
        self, indata: np.ndarray, outdata: np.ndarray, frames, times, status
    ):
        try:
            out_wav = self.process_data_with_time(indata)

            output_channels = outdata.shape[1]
            if self.use_monitor:
                self.mon_queue.put(out_wav)

            outdata[:] = (
                np.repeat(out_wav, output_channels).reshape(-1, output_channels)
                * self.output_audio_gain
            )
        except Exception as error:
            print(f"An error occurred while running the audio stream: {error}")
            print(traceback.format_exc())

    def audio_queue(self, outdata: np.ndarray, gain: float):
        outdata.fill(0)
        try:
            mon_wav = self.mon_queue.get_nowait()
            # Preserve the existing policy: play the oldest pending block and
            # discard its stale backlog. Bound work to the initial queue depth.
            for _ in range(self.mon_queue.qsize()):
                try:
                    self.mon_queue.get_nowait()
                except Empty:
                    break

            output_channels = outdata.shape[1]
            outdata[:] = (
                np.repeat(mon_wav, output_channels).reshape(-1, output_channels) * gain
            )
        except Empty:
            return
        except Exception as error:
            print(f"An error occurred while running the audio queue: {error}")
            print(traceback.format_exc())

    def run_audio_stream(
        self,
        block_frame: int,
        input_device_id: int,
        output_device_id: int,
        output_monitor_id: int,
        input_max_channel: int,
        output_max_channel: int,
        output_monitor_max_channel: int,
        input_extra_setting,
        output_extra_setting,
        output_monitor_extra_setting,
        audio_sample_rate: int,
    ):
        if input_device_id != output_device_id:
            # Old Windows fix: constructors open devices, so open output first.
            self.output_stream = sd.OutputStream(
                callback=lambda outdata, frames, times, status: self.audio_queue(
                    outdata, self.output_audio_gain
                ),
                latency="low",
                dtype=np.float32,
                device=output_device_id,
                blocksize=block_frame,
                samplerate=audio_sample_rate,
                channels=output_max_channel,
                extra_settings=output_extra_setting,
            )
            if self.use_monitor:
                self.monitor = sd.OutputStream(
                    callback=lambda outdata, frames, times, status: self.audio_queue(
                        outdata, self.monitor_audio_gain
                    ),
                    latency="low",
                    dtype=np.float32,
                    device=output_monitor_id,
                    blocksize=block_frame,
                    samplerate=audio_sample_rate,
                    channels=output_monitor_max_channel,
                    extra_settings=output_monitor_extra_setting,
                )
            self.input_stream = sd.InputStream(
                callback=self.audio_stream_no_output_callback,
                latency="low",
                dtype=np.float32,
                device=input_device_id,
                blocksize=block_frame,
                samplerate=audio_sample_rate,
                channels=input_max_channel,
                extra_settings=input_extra_setting,
            )
            self.input_stream.start()
            self.output_stream.start()
            if self.monitor is not None:
                self.monitor.start()
        else:
            self.stream = sd.Stream(
                callback=self.audio_stream_callback,
                latency="low",
                dtype=np.float32,
                device=(input_device_id, output_device_id),
                blocksize=block_frame,
                samplerate=audio_sample_rate,
                channels=(input_max_channel, output_max_channel),
                extra_settings=(input_extra_setting, output_extra_setting),
            )
            self.stream.start()

        if self.use_monitor and input_device_id == output_device_id:
            self.monitor = sd.OutputStream(
                callback=lambda outdata, frames, times, status: self.audio_queue(
                    outdata, self.monitor_audio_gain
                ),
                latency="low",
                dtype=np.float32,
                device=output_monitor_id,
                blocksize=block_frame,
                samplerate=audio_sample_rate,
                channels=output_monitor_max_channel,
                extra_settings=output_monitor_extra_setting,
            )
            self.monitor.start()

    def stop(self):
        with _backend_lock:
            self.running = False
            errors = []
            # Quiesce output callbacks before stopping their producer.
            for name in ("monitor", "output_stream", "stream", "input_stream"):
                stream = getattr(self, name)
                if stream is None:
                    continue
                try:
                    stream.stop()
                except Exception as error:
                    print(f"[Audio] {name}.stop: {describe_audio_error(error)}")
                try:
                    stream.close()
                except Exception as error:
                    errors.append(f"{name}: {describe_audio_error(error)}")
                else:
                    setattr(self, name, None)
            # Drain only once all callbacks have released their resources.
            if errors:
                raise AudioDeviceError("Unable to close audio streams: " + "; ".join(errors))
            while True:
                try:
                    self.mon_queue.get_nowait()
                except Empty:
                    break

    def start(
        self,
        input_device_id,
        output_device_id,
        output_monitor_id=None,
        exclusive_mode: bool = False,
        asio_input_channel: int = 0,
        asio_output_channel: int = 0,
        asio_output_monitor_channel: int = 0,
        block_frame: int = 24576,
        audio_sample_rate: int = AUDIO_SAMPLE_RATE,
        asio_output_stereo: bool = True,
    ):
        with _backend_lock:
            self.stop()
            # Bare indexes are legacy current-session inputs only. Convert them
            # to identities BEFORE refresh, then re-resolve after refresh.
            requested = [input_device_id, output_device_id, output_monitor_id]
            for i, direction in ((0, "input"), (1, "output"), (2, "output")):
                if isinstance(requested[i], int) and (i != 2 or self.use_monitor):
                    requested[i] = resolve_audio_device(requested[i], direction).label
            reinitialize_audio_backend()
            inputs, outputs = list_audio_device()
            primary_input = resolve_audio_device(requested[0], "input", inputs)
            primary_output = resolve_audio_device(requested[1], "output", outputs)
            primary_monitor = (resolve_audio_device(requested[2], "output", outputs)
                               if self.use_monitor else None)
            if block_frame <= 0 or audio_sample_rate <= 0:
                raise AudioDeviceError("Chunk frames and sample rate must be positive.")

            input_candidates = ([primary_input] if exclusive_mode else
                                _fallback_candidates(primary_input, "input", inputs))
            output_candidates = ([primary_output] if exclusive_mode else
                                 _fallback_candidates(primary_output, "output", outputs))
            monitor_candidates = ([primary_monitor] if exclusive_mode or primary_monitor is None else
                                  _fallback_candidates(primary_monitor, "output", outputs))
            combinations = [(i, o, m) for i in input_candidates
                            for o in output_candidates for m in monitor_candidates]
            # One same-identity retry, then the evidenced conservative fallbacks.
            attempts = [combinations[0], *combinations]
            errors = []
            for attempt_number, (input_device, output_device, monitor_device) in enumerate(attempts):
                try:
                    if attempt_number:
                        self.stop()
                        reinitialize_audio_backend()
                        inputs, outputs = list_audio_device()
                    input_device = resolve_audio_device(input_device.label, "input", inputs)
                    output_device = resolve_audio_device(output_device.label, "output", outputs)
                    if monitor_device is not None:
                        monitor_device = resolve_audio_device(monitor_device.label, "output", outputs)
                    # A retry refresh can add another endpoint sharing a
                    # truncated MME name. Revalidate the alias, not just its index.
                    for candidate, primary, direction, inventory in (
                        (input_device, primary_input, "input", inputs),
                        (output_device, primary_output, "output", outputs),
                        (monitor_device, primary_monitor, "output", outputs),
                    ):
                        if candidate is None or candidate.label == primary.label:
                            continue
                        current_primary = resolve_audio_device(primary.label, direction, inventory)
                        allowed = _fallback_candidates(current_primary, direction, inventory)
                        if not any(d.label == candidate.label for d in allowed):
                            raise AudioDeviceError(f"Fallback identity became ambiguous or unavailable: {candidate.label}")
                    in_channels, in_extra = self._stream_settings(input_device, "input", exclusive_mode, asio_input_channel, False)
                    out_channels, out_extra = self._stream_settings(output_device, "output", exclusive_mode, asio_output_channel, asio_output_stereo)
                    mon_channels, mon_extra = (self._stream_settings(monitor_device, "output", exclusive_mode, asio_output_monitor_channel, False)
                                              if monitor_device is not None else (None, None))
                    for direction, device, channels, extra in (
                        ("output", output_device, out_channels, out_extra),
                        ("input", input_device, in_channels, in_extra),
                        ("output", monitor_device, mon_channels, mon_extra),
                    ):
                        if device is None:
                            continue
                        print(f"[Audio] {direction}: {device.label}; index={device.index}; "
                              f"channels={channels}/{getattr(device, 'max_' + direction + '_channels')}; "
                              f"default_sr={device.default_samplerate}; stream_sr={audio_sample_rate}; "
                              f"block_frame={block_frame}; float32; attempt={attempt_number + 1}")
                        check = sd.check_input_settings if direction == "input" else sd.check_output_settings
                        check(device=device.index, channels=channels, dtype="float32",
                              extra_settings=extra, samplerate=audio_sample_rate)
                    self.run_audio_stream(block_frame, input_device.index, output_device.index,
                                          monitor_device.index if monitor_device else None,
                                          in_channels, out_channels, mon_channels,
                                          in_extra, out_extra, mon_extra, audio_sample_rate)
                    self.running = True
                    if attempt_number:
                        print(f"[Audio] Retry/fallback active: {input_device.label} -> {output_device.label}")
                    return
                except Exception as error:
                    self.stop()
                    detail = f"{input_device.label} -> {output_device.label}: {describe_audio_error(error)}"
                    errors.append(detail)
                    print(f"[Audio] Validation/open failed: {detail}")
            raise AudioDeviceError(
                "Unable to open Realtime audio devices. Stop other exclusive audio clients, "
                "refresh devices and verify the selected Host API/sample rate.\n"
                + "\n".join(errors)
            )

    @staticmethod
    def _stream_settings(device, direction, exclusive_mode, asio_channel, asio_output_stereo):
        maximum = getattr(device, "max_" + direction + "_channels")
        if maximum <= 0:
            raise AudioDeviceError(f"{device.label} has no {direction} channels.")
        channels = min(maximum, 2)
        extra = None
        if "WASAPI" in device.host_api:
            extra = sd.WasapiSettings(exclusive=exclusive_mode, auto_convert=not exclusive_mode,
                                      explicit_sample_format=False)
        elif "ASIO" in device.host_api and asio_channel != -1:
            selectors = ([asio_channel, asio_channel + 1]
                         if direction == "output" and asio_output_stereo else [asio_channel])
            if any(channel < 0 or channel >= maximum for channel in selectors):
                raise AudioDeviceError(f"Invalid {direction} ASIO channels {selectors}; {device.label} exposes {maximum} channels.")
            extra = sd.AsioSettings(channel_selectors=selectors)
            channels = len(selectors)
        return channels, extra
