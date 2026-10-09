"""Shared fixtures for mocked audio endpoint tests."""
from rvc.realtime import audio as backend


def device(index, name, api, inputs=0, outputs=0, rate=48000):
    return backend.ServerAudioDevice(index, name, api, inputs, outputs, rate)
