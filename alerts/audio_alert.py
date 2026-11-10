"""
Non-blocking audio alerts.

Warnings play a repeating alternating-tone beep in a background worker thread,
so the video loop never waits for audio. On Windows this uses winsound; the
terminal bell is a fallback if the beep device is unavailable.
"""
from __future__ import annotations

import io
import logging
import math
import struct
import wave
import platform
import queue
import shutil
import subprocess
import threading
import time
from typing import Dict, List, Optional

from config import AudioConfig

log = logging.getLogger(__name__)


class _Pyttsx3Speaker:
    name = "pyttsx3"

    def __init__(self, rate: int, volume: float) -> None:
        import pyttsx3
        self._engine = pyttsx3.init()          # must be created in the thread that uses it
        self._engine.setProperty("rate", rate)
        self._engine.setProperty("volume", volume)

    def speak(self, text: str) -> None:
        self._engine.say(text)
        self._engine.runAndWait()


class _CommandSpeaker:
    def __init__(self, command: List[str]) -> None:
        self._command = command
        self.name = command[0]

    def speak(self, text: str) -> None:
        subprocess.run(self._command + [text], check=False, timeout=20,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class _BeepSpeaker:
    name = "beep"

    def speak(self, text: str) -> None:
        try:
            import winsound

            # Route synthesized tones through the normal Windows audio device.
            sample_rate = 22050
            samples = bytearray()
            pattern = ((880, 0.24), (0, 0.10), (1320, 0.24),
                       (0, 0.10), (880, 0.24), (0, 0.22))
            for _ in range(3):
                for frequency, duration in pattern:
                    count = int(sample_rate * duration)
                    for i in range(count):
                        if frequency:
                            envelope = min(1.0, i / 220, (count - i) / 220)
                            value = int(10000 * envelope * math.sin(
                                2 * math.pi * frequency * i / sample_rate))
                        else:
                            value = 0
                        samples.extend(struct.pack("<h", value))

            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(samples)
            winsound.PlaySound(
                wav_buffer.getvalue(),
                winsound.SND_MEMORY,
            )
        except Exception as exc:
            log.warning("Windows audio playback failed (%s); using terminal bell.", exc)
            try:
                print("\a", end="", flush=True)
            except Exception as bell_exc:
                log.error("Terminal bell fallback failed: %s", bell_exc)


def _create_speaker(cfg: AudioConfig):
    system = platform.system()
    attempts = []
    if system == "Darwin":
        if shutil.which("say"):
            attempts.append(lambda: _CommandSpeaker(["say"]))
    elif system == "Linux":
        if shutil.which("spd-say"):
            attempts.append(lambda: _CommandSpeaker(["spd-say", "-w"]))
        for exe in ("espeak-ng", "espeak"):
            if shutil.which(exe):
                attempts.append(lambda exe=exe: _CommandSpeaker([exe]))
        attempts.append(lambda: _Pyttsx3Speaker(cfg.speech_rate, cfg.volume))
    else:  # Windows and anything else
        attempts.append(lambda: _Pyttsx3Speaker(cfg.speech_rate, cfg.volume))

    for make in attempts:
        try:
            return make()
        except Exception as exc:
            log.warning("Speech backend unavailable (%s); trying next option.", exc)
    log.warning("No text-to-speech backend found - falling back to beeps.")
    return _BeepSpeaker()


class AudioAlert:
    """Call request(); it returns immediately and never plays overlapping audio."""

    def __init__(self, cfg: AudioConfig) -> None:
        self._cfg = cfg
        self._queue: "queue.Queue[Optional[tuple[str, str]]]" = queue.Queue(maxsize=1)
        self._busy = threading.Event()
        self._idle = threading.Event()
        self._idle.set()
        self._muted = False
        self._last_any = float("-inf")
        self._last_by_event: Dict[str, float] = {}
        self._thread: Optional[threading.Thread] = None
        if cfg.enabled:
            self._thread = threading.Thread(target=self._worker, name="audio-alert", daemon=True)
            self._thread.start()

    # -------------------------------------------------------------- public API
    def request(self, event: str, message: str, now: float) -> bool:
        """Queue `message` unless muted, still speaking, or inside a cooldown."""
        if self._thread is None or self._muted or self._busy.is_set():
            return False
        if now - self._last_any < self._cfg.global_cooldown_s:
            return False
        if now - self._last_by_event.get(event, float("-inf")) < self._cfg.per_event_cooldown_s:
            return False

        self._busy.set()
        self._idle.clear()
        try:
            self._queue.put_nowait((event, message))
        except queue.Full:
            self._busy.clear()
            self._idle.set()
            return False
        self._last_any = now
        self._last_by_event[event] = now
        return True

    def test(self, timeout: float = 20.0) -> bool:
        """Play a short sample and wait for completion; useful without a camera event."""
        if not self.request("audio_test", "DriveGuard audio test. Sound is working.", time.monotonic()):
            return False
        return self._idle.wait(timeout)

    def toggle_mute(self) -> bool:
        self._muted = not self._muted
        return self._muted

    @property
    def muted(self) -> bool:
        return self._muted

    def close(self) -> None:
        if self._thread is None:
            return
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self._queue.put_nowait(None)      # sentinel: tells the worker to exit
        except queue.Full:
            pass
        self._thread.join(timeout=2.0)

    # ------------------------------------------------------------------ worker
    def _worker(self) -> None:
        speaker = _BeepSpeaker()
        log.info("Audio alerts using backend: %s", speaker.name)
        while True:
            message = self._queue.get()
            if message is None:
                break
            event, text = message
            try:
                speaker.speak(text)
            except Exception as exc:
                log.error("Warning beep failed for %s: %s", event, exc)
                try:
                    print("\\a", end="", flush=True)
                except Exception as bell_exc:
                    log.error("Terminal bell fallback failed: %s", bell_exc)
            finally:
                self._busy.clear()
                self._idle.set()
