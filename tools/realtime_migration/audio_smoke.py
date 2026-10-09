"""Brief shared-mode start/stop test: discard mic input and output only zeros.

No Windows configuration changes, recording, model inference or network traffic.
"""
import json
import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-hardware", action="store_true")
    options = parser.parse_args()
    if not options.allow_hardware:
        parser.error("Hardware smoke test disabled after WDF_VIOLATION. Diagnose the crash dump before explicitly enabling this test.")
    import numpy as np
    from rvc.realtime.audio import Audio

    selected = json.loads((ROOT.parent / "Applio_3.6.4_broken/assets/config.json").read_text(encoding="utf-8"))["realtime"]
    mic, output = selected["input_device"], selected["output_device"]
    calls = [0]
    def silence(input_data, *args):
        calls[0] += 1
        return np.zeros(len(input_data), dtype=np.float32), 0.0, 0.0
    audio = Audio(SimpleNamespace(change_voice=silence))
    results = []
    try:
        for cycle in range(2):
            audio.start(mic, output, block_frame=12000, audio_sample_rate=48000,
                        exclusive_mode=False, asio_input_channel=-1, asio_output_channel=-1)
            time.sleep(0.8)
            assert audio.running
            assert audio.input_stream.active and audio.output_stream.active
            audio.stop()
            assert not audio.running
            assert audio.input_stream is None and audio.output_stream is None
            assert audio.mon_queue.empty()
            results.append({"cycle": cycle + 1, "status": "PASS", "closed": True})
        assert calls[0] > 0, "No input callbacks ran"
        result = {"status": "PASS", "input": mic, "output": output,
                  "input_callbacks": calls[0], "cycles": results,
                  "kind": "physical shared-mode audio streams, silence output ONLY; no RVC/Discord listening test"}
    except Exception as error:
        result = {"status": "FAIL", "input": mic, "output": output, "error": str(error), "cycles": results}
        raise
    finally:
        audio.stop()
        (Path(__file__).parent / "evidence/audio-smoke.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
