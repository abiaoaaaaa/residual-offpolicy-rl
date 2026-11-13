"""Lossless evdev listener for the dual A4Tech/"infinite" keyboard."""

from __future__ import annotations

import os
import threading
from collections import deque


class KeyboardListener:
    REQUIRED = {"KEY_A", "KEY_B", "KEY_C", "KEY_Q"}

    def __init__(self, device_path: str | None = None):
        try:
            from evdev import InputDevice, ecodes, list_devices
        except ImportError as exc:
            raise RuntimeError("Install evdev to use --keyboard") from exc
        self.ecodes, self.InputDevice = ecodes, InputDevice
        self._events: deque[str] = deque()
        self._lock = threading.Lock()
        self._ctrl_down = False
        requested = device_path or os.getenv("RESIDUAL_KEYBOARD_DEVICE")
        self.device = self._open(requested, list_devices)
        self.thread = threading.Thread(target=self._loop, daemon=True, name="residual-keyboard")
        self.thread.start()

    def _open(self, requested, list_devices):
        paths = [requested] if requested else list_devices()
        candidates = []
        for path in paths:
            try:
                dev = self.InputDevice(path)
                keys = dev.capabilities(verbose=False).get(self.ecodes.EV_KEY, [])
                names = {self.ecodes.bytype[self.ecodes.EV_KEY].get(code) for code in keys}
                if self.REQUIRED.issubset(names):
                    candidates.append(dev)
                else:
                    dev.close()
            except (OSError, PermissionError):
                continue
        preferred = [
            dev for dev in candidates
            if "a4tech" in dev.name.lower() or "hs6209" in dev.name.lower()
        ]
        if not requested and len(preferred) == 1:
            for dev in candidates:
                if dev is not preferred[0]:
                    dev.close()
            return preferred[0]
        if len(candidates) != 1:
            for dev in candidates:
                dev.close()
            raise RuntimeError(
                "Expected exactly one keyboard exposing A/B/C/Q; set "
                "RESIDUAL_KEYBOARD_DEVICE=/dev/input/by-id/..."
            )
        return candidates[0]

    def _loop(self):
        for event in self.device.read_loop():
            if event.type != self.ecodes.EV_KEY:
                continue
            name = self.ecodes.bytype[self.ecodes.EV_KEY].get(event.code)
            if isinstance(name, list):
                name = name[0]
            if name in {"KEY_LEFTCTRL", "KEY_RIGHTCTRL"}:
                self._ctrl_down = event.value != 0
                continue
            if event.value != 1:
                continue
            if name in {"KEY_1", "KEY_KP1"} and self._ctrl_down:
                with self._lock:
                    self._events.append("reset")
                continue
            if isinstance(name, str) and name.startswith("KEY_"):
                key = name[4:].lower()
                if key in {"a", "b", "c", "q"}:
                    with self._lock:
                        self._events.append(key)

    def pop(self) -> list[str]:
        with self._lock:
            events = list(self._events)
            self._events.clear()
            return events
