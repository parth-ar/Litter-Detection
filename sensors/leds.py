"""
sensors/leds.py — Singular Orange Status LED driver for Raspberry Pi.
---------------------------------------------------------------------
Uses BCM GPIO 25 (Physical Pin 22 on the Raspberry Pi 40-pin header).

Status Indications:
  1. SOLID LIGHT        : System is functional (camera & detection ready, tracking distance).
  2. CONSTANT BLINKING  : Scanning for GNSS (boot scan or reconnection grace period).
  3. TRIPLE BLINK       : Capturing a frame from the camera on trigger.
  4. HEARTBEAT BLINK    : Error faced (camera failure, hardware fault, exception).

Gracefully degrades to dummy operations on non-Raspberry Pi / development machines.
"""

import enum
import threading
import time

from config import LED_ORANGE_PIN

# ---------------------------------------------------------------------------
# gpiozero import — gracefully degrade if not on a Pi
# ---------------------------------------------------------------------------
try:
    from gpiozero import LED as _GpioLED  # type: ignore
    _GPIO_AVAILABLE = True
except Exception:
    _GPIO_AVAILABLE = False


class _DummyLED:
    """No-op LED for non-Pi or testing environments."""
    def __init__(self, pin):
        self.pin = pin
        self.is_lit = False
    def on(self):
        self.is_lit = True
    def off(self):
        self.is_lit = False
    def close(self):
        self.is_lit = False


# ---------------------------------------------------------------------------
# LED State Enumeration
# ---------------------------------------------------------------------------
class LEDState(enum.Enum):
    OFF            = "OFF"
    SOLID          = "SOLID"            # System functional & running
    SCANNING_GNSS  = "SCANNING_GNSS"    # Constant blinking (1 Hz)
    HEARTBEAT      = "HEARTBEAT"        # Error faced (double pulse heartbeat)


# ---------------------------------------------------------------------------
# Global Driver State
# ---------------------------------------------------------------------------
_led = None
_lock = threading.Lock()
_state = LEDState.OFF
_triple_blink_active = False
_thread = None
_stop_event = threading.Event()


def _make_led(pin: int):
    if _GPIO_AVAILABLE:
        try:
            return _GpioLED(pin)
        except Exception as exc:
            print(f"[LEDS] Could not open GPIO {pin}: {exc}")
    return _DummyLED(pin)


# ---------------------------------------------------------------------------
# Background LED Blink Controller Thread
# ---------------------------------------------------------------------------
def _led_worker():
    """Background controller loop that handles blinking patterns."""
    global _triple_blink_active
    while not _stop_event.is_set():
        # Check if a one-shot triple blink is requested
        with _lock:
            do_triple = _triple_blink_active
            current_state = _state

        if do_triple:
            # Triple blink: 3 quick pulses (80ms ON / 80ms OFF)
            for _ in range(3):
                if _stop_event.is_set():
                    break
                _led.on()
                time.sleep(0.08)
                _led.off()
                time.sleep(0.08)
            with _lock:
                _triple_blink_active = False
            continue

        if current_state == LEDState.SOLID:
            _led.on()
            time.sleep(0.1)

        elif current_state == LEDState.SCANNING_GNSS:
            # Constant 1 Hz blink (500ms ON, 500ms OFF)
            _led.on()
            _stop_event.wait(0.50)
            if _stop_event.is_set():
                break
            _led.off()
            _stop_event.wait(0.50)

        elif current_state == LEDState.HEARTBEAT:
            # Heartbeat blink: pulse-pulse ... pause (120ms ON, 120ms OFF, 120ms ON, 740ms OFF)
            _led.on()
            _stop_event.wait(0.12)
            _led.off()
            _stop_event.wait(0.12)
            _led.on()
            _stop_event.wait(0.12)
            _led.off()
            _stop_event.wait(0.74)

        else:  # OFF
            _led.off()
            time.sleep(0.1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def init() -> None:
    """Initialize the Orange Status LED on GPIO 25 and start controller thread."""
    global _led, _thread, _state
    if not _GPIO_AVAILABLE:
        print("[LEDS] gpiozero not available — Orange LED control disabled (simulated).")

    _stop_event.clear()
    _led = _make_led(LED_ORANGE_PIN)
    _state = LEDState.OFF

    # Quick test flash on boot
    _led.on()
    time.sleep(0.15)
    _led.off()

    _thread = threading.Thread(target=_led_worker, daemon=True, name="orange-led-controller")
    _thread.start()
    print(f"[LEDS] Orange status LED initialized on BCM GPIO {LED_ORANGE_PIN} (Physical Pin 22).")


def set_functional() -> None:
    """Solid light: system is functional, ready, and tracking distance."""
    global _state
    with _lock:
        _state = LEDState.SOLID


def set_system_ready() -> None:
    """Alias for set_functional()."""
    set_functional()


def set_scanning_gnss() -> None:
    """Constant blinking: scanning for GNSS fix (at boot or mid-run loss)."""
    global _state
    with _lock:
        _state = LEDState.SCANNING_GNSS


def notify_capture() -> None:
    """Triple blink: indicates a frame was captured from the camera."""
    global _triple_blink_active
    with _lock:
        _triple_blink_active = True


def notify_litter_detected() -> None:
    """Alias for notify_capture()."""
    notify_capture()


def set_error(reason: str = "") -> None:
    """Heartbeat blink: indicates an error or failure condition."""
    global _state
    if reason:
        print(f"[LEDS] ERROR state triggered: {reason}")
    with _lock:
        _state = LEDState.HEARTBEAT


def update(rtc: bool = True, gnss: bool = False, gnss_connected: bool = False, gnss_fix: bool = False) -> None:
    """Compatibility helper to map status to Orange LED."""
    with _lock:
        if _state == LEDState.HEARTBEAT:
            return  # Error state holds precedence
        if not gnss_fix:
            _state = LEDState.SCANNING_GNSS
        else:
            _state = LEDState.SOLID


def close() -> None:
    """Clean up and extinguish the Orange LED."""
    global _state
    _stop_event.set()
    with _lock:
        _state = LEDState.OFF
    if _led:
        try:
            _led.off()
            _led.close()
        except Exception:
            pass
