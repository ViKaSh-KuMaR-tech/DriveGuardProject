"""
Non-blocking audio alerts.

Speech runs in a background worker thread, so the video loop never waits for it.
We deliberately avoid PyAudio (it needs a C toolchain / PortAudio on many
machines) and use the operating system's own text-to-speech instead:

    Windows : pyttsx3 (SAPI5)          macOS : `say`
    Linux   : spd-say / espeak-ng / espeak, else pyttsx3
    Fallback: beeps (winsound on Windows, terminal bell elsewhere)
"""
from __future__ import annotations

import logging
import platform
import queue
import shutil
import subprocess
import threading
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
            winsound.Beep(1000, 300)
            winsound.Beep(1400, 300)
        except ImportError:
            print("\a", end="", flush=True)   # terminal bell


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
        self._queue: "queue.Queue[Optional[str]]" = queue.Queue(maxsize=1)
        self._busy = threading.Event()
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
        try:
            self._queue.put_nowait(message)
        except queue.Full:
            self._busy.clear()
            return False
        self._last_any = now
        self._last_by_event[event] = now
        return True

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
        speaker = _create_speaker(self._cfg)
        log.info("Audio alerts using backend: %s", speaker.name)
        while True:
            message = self._queue.get()
            if message is None:
                break
            try:
                speaker.speak(message)
            except Exception as exc:
                log.warning("Audio alert failed: %s", exc)
            finally:
                self._busy.clear()
