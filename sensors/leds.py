"""
sensors/leds.py — Status LED driver for Raspberry Pi using gpiozero.

LED Hardware Mapping (BCM GPIO pins — config.py):
  RTC_GREEN  (BCM 17): RTC module status:
                       - Solid ON if RTC working & valid
                       - OFF on RTC error / offline
  IMU_GREEN  (BCM 27): Reserved / auxiliary status indicator:
                       - Solid ON when system active
  GNSS_GREEN (BCM 22): GNSS module status:
                       - Solid ON if GNSS fix is acquired
                       - Blinking (500 ms) if GNSS is looking for fix (connected, no fix yet)
                       - OFF if GNSS is disconnected / no data / module error
  YELLOW     (BCM 23): System ready & Litter Detection indicator:
                       - Solid ON when system is healthy and ready to detect litter
                       - DOUBLE BLINK (2x OFF/ON) when litter is detected, then returns to solid ON
                       - OFF if ANY module fails or program error occurs
  RED        (BCM 24): Fault indicator:
                       - Blinking (400 ms) if ANY module fails or program error is encountered
                       - OFF when all modules and system are working normally

All LEDs are active-HIGH (logic 1 = LED on) with current-limiting resistors (220-470 Ohm).
"""

import threading
import time

from config import (
    LED_RTC_GREEN, LED_IMU_GREEN, LED_GNSS_GREEN, LED_YELLOW, LED_RED,
    LED_FAULT_BLINK_INTERVAL,
)

# Blinking interval for GNSS searching for fix (500 ms = 1 Hz blink)
GNSS_SEARCH_BLINK_INTERVAL = 0.500

# ---------------------------------------------------------------------------
# gpiozero import — gracefully degrade if not on a Pi
# ---------------------------------------------------------------------------
try:
    from gpiozero import LED as _GpioLED  # type: ignore
    _GPIO_AVAILABLE = True
except Exception:
    _GPIO_AVAILABLE = False
    print("[LEDS] gpiozero not available — LED control disabled (not running on Pi?)")


class _DummyLED:
    """No-op LED for non-Pi or testing environments."""
    def __init__(self, pin):
        self.pin = pin
    def on(self): pass
    def off(self): pass
    def close(self): pass


def _make_led(pin: int):
    if _GPIO_AVAILABLE:
        try:
            return _GpioLED(pin)
        except Exception as exc:
            print(f"[LEDS] Could not open GPIO {pin}: {exc}")
    return _DummyLED(pin)


# ---------------------------------------------------------------------------
# Driver State
# ---------------------------------------------------------------------------
_leds: dict[str, object] = {}
_stop_event = threading.Event()

# Module status memories (for restoring state after blinks)
_last_rtc_ok         = False
_last_imu_ok         = True
_last_gnss_fix       = False
_last_gnss_connected = False
_program_fault       = False
_fault_reason        = ""
_system_ready        = False

# Blink ticker states
_fault_blink_state = False
_last_fault_toggle = 0.0

_gnss_blink_state = False
_last_gnss_toggle = 0.0

# Detection snap blink lock
_snap_lock     = threading.Lock()
_snap_blinking = False


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------
def init() -> None:
    """Open GPIO pins and run the power-on boot splash."""
    global _leds
    _leds = {
        "rtc_green":  _make_led(LED_RTC_GREEN),
        "imu_green":  _make_led(LED_IMU_GREEN),
        "gnss_green": _make_led(LED_GNSS_GREEN),
        "yellow":     _make_led(LED_YELLOW),
        "red":        _make_led(LED_RED),
    }
    _boot_splash()
    print("[LEDS] GPIO LED driver initialised.")


def close() -> None:
    """Release GPIO handles on shutdown."""
    global _system_ready
    _system_ready = False
    _stop_event.set()
    _all_off()
    for led in _leds.values():
        try:
            led.close()
        except Exception:
            pass


def _all_on() -> None:
    for led in _leds.values():
        led.on()


def _all_off() -> None:
    for led in _leds.values():
        led.off()


def _boot_splash() -> None:
    """3x (200 ms ON + 250 ms OFF) to verify all LED hardware connections."""
    for _ in range(3):
        _all_on()
        time.sleep(0.200)
        _all_off()
        time.sleep(0.250)
    print("[LEDS] Boot splash complete.")


# ---------------------------------------------------------------------------
# Public System Control API
# ---------------------------------------------------------------------------
def set_headless_mode(enabled: bool) -> None:
    """Kept for backward compatibility."""
    pass


def set_system_ready() -> None:
    """
    Signal that the edge application and camera loop have initialized and are
    actively ready to detect litter.
    """
    global _system_ready, _program_fault, _fault_reason
    _system_ready = True
    _program_fault = False
    _fault_reason = ""
    print("[LEDS] System READY: Camera & YOLO litter detection engine running.")


def set_fault(reason: str = "") -> None:
    """
    Signal a program or system-level fault.
    Yellow LED turns OFF immediately. Red LED starts blinking.
    """
    global _program_fault, _fault_reason
    _program_fault = True
    _fault_reason = reason
    if _leds and "yellow" in _leds:
        _leds["yellow"].off()
    if reason:
        print(f"[LEDS] System FAULT: {reason} — Yellow OFF, Red blinking.")
    else:
        print("[LEDS] System FAULT — Yellow OFF, Red blinking.")


def notify_litter_detected() -> None:
    """
    DOUBLE BLINK on Yellow LED (2 rapid blinks: OFF -> ON -> OFF -> ON)
    when litter is detected, then return to solid ON (if system is healthy).
    Non-blocking: runs in a background thread.
    """
    global _snap_blinking
    if not _leds:
        return

    def _blink_worker():
        global _snap_blinking
        with _snap_lock:
            _snap_blinking = True

        try:
            # Double yellow blink (2x: 100 ms OFF / 100 ms ON)
            for _ in range(2):
                _leds["yellow"].off()
                time.sleep(0.10)
                _leds["yellow"].on()
                time.sleep(0.10)
        finally:
            with _snap_lock:
                _snap_blinking = False

            # Restore correct steady state
            has_error = (not _last_rtc_ok) or (not _last_gnss_connected) or _program_fault
            all_healthy = _system_ready and (not has_error)
            if all_healthy:
                _leds["yellow"].on()
            else:
                _leds["yellow"].off()

    t = threading.Thread(target=_blink_worker, name="led-litter-double-blink", daemon=True)
    t.start()


# Alias for compatibility with motion detection naming
notify_motion_snap = notify_litter_detected


# ---------------------------------------------------------------------------
# State Machine Update (called regularly from detection loop)
# ---------------------------------------------------------------------------
def update(
    rtc: bool = True,
    imu: bool = True,
    gnss: bool = False,
    gnss_fix: bool | None = None,
    gnss_connected: bool | None = None,
    **kwargs,
) -> None:
    """
    Update LED indicators with current subsystem states:

    Rules:
      1. RTC Green:
         - Solid ON if RTC is working, OFF if error
      2. GNSS Green:
         - Solid ON if GNSS has fix
         - Blinking (500 ms) if GNSS is connected and looking for fix
         - OFF if GNSS module error / disconnected / no data
      3. Red LED:
         - Blinking (400 ms) if ANY required module fails (RTC/GNSS offline) OR program fault
         - OFF if all modules and program are healthy
      4. Yellow LED:
         - Solid ON if system is ready and all modules healthy
         - DOUBLE BLINK on litter detected (notify_litter_detected)
         - OFF if ANY module fails or program error occurs
    """
    global _last_rtc_ok, _last_imu_ok, _last_gnss_fix, _last_gnss_connected
    global _fault_blink_state, _last_fault_toggle
    global _gnss_blink_state, _last_gnss_toggle

    if not _leds:
        return

    has_fix = bool(gnss if gnss_fix is None else gnss_fix)
    if gnss_connected is None:
        if "gnss_data" in kwargs:
            is_connected = bool(kwargs["gnss_data"])
        elif "gnss_active" in kwargs:
            is_connected = bool(kwargs["gnss_active"])
        else:
            is_connected = True if has_fix else bool(gnss)
    else:
        is_connected = bool(gnss_connected)

    _last_rtc_ok         = rtc
    _last_imu_ok         = imu
    _last_gnss_fix       = has_fix
    _last_gnss_connected = is_connected

    now = time.monotonic()

    # Toggle fault blink tick for Red LED (400 ms)
    if (now - _last_fault_toggle) > LED_FAULT_BLINK_INTERVAL:
        _fault_blink_state = not _fault_blink_state
        _last_fault_toggle = now

    # Toggle GNSS search blink tick for GNSS Green LED (500 ms)
    if (now - _last_gnss_toggle) > GNSS_SEARCH_BLINK_INTERVAL:
        _gnss_blink_state = not _gnss_blink_state
        _last_gnss_toggle = now

    # 1. RTC Green LED
    _leds["rtc_green"].on() if rtc else _leds["rtc_green"].off()

    # 2. Auxiliary / IMU Green LED
    _leds["imu_green"].on() if imu else _leds["imu_green"].off()

    # 3. GNSS Green LED
    if has_fix:
        _leds["gnss_green"].on()                  # Solid ON: Fix acquired
    elif is_connected:
        if _gnss_blink_state:                     # Blinking: Looking for fix
            _leds["gnss_green"].on()
        else:
            _leds["gnss_green"].off()
    else:
        _leds["gnss_green"].off()                 # OFF: Disconnected / error

    # 4. Health Evaluation
    # Module failure = RTC invalid or GNSS completely disconnected
    module_failure = (not rtc) or (not is_connected)
    has_fault = module_failure or _program_fault

    # 5. Red LED (Blinks on ANY module failure or program fault)
    if has_fault:
        if _fault_blink_state:
            _leds["red"].on()
        else:
            _leds["red"].off()
    else:
        _leds["red"].off()

    # 6. Yellow LED (Solid ON when ready & healthy, OFF on any error)
    if not _snap_blinking:
        system_working_and_ready = _system_ready and (not has_fault)
        if system_working_and_ready:
            _leds["yellow"].on()
        else:
            _leds["yellow"].off()
