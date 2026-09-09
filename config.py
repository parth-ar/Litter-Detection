"""
config.py — Centralised runtime configuration for Litter Detection and SWSTP sensors.

Device identity is read from device_config.json.
Sensor and telemetry constants for GNSS (NavCast) and RTC (DS3231) are defined here.
"""

import json
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
DEVICE_CONFIG_FILE = os.path.join(_HERE, "device_config.json")
OUTPUT_DIR         = os.path.join(_HERE, "runs", "EventLogger")

# ---------------------------------------------------------------------------
# Device identity (loaded from device_config.json)
# ---------------------------------------------------------------------------
_DEFAULT_DEVICE_ID = "SWSTP-AMRMC-001"
DEFAULT_TIMEZONE   = "Asia/Kolkata"    # Indian Standard Time (IST, UTC+05:30) for Maharashtra, India
RTC_RESYNC_INTERVAL_HOURS = 1.0        # Recalibrate timing every hour using internet NTP/HTTP

def load_device_id() -> str:
    """Read device ID from device_config.json. Falls back to SWSTP-AMRMC-001."""
    try:
        with open(DEVICE_CONFIG_FILE, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        dev_id = str(cfg.get("device_id") or cfg.get("deviceId") or "").strip()
        return dev_id if dev_id else _DEFAULT_DEVICE_ID
    except FileNotFoundError:
        return _DEFAULT_DEVICE_ID
    except Exception as exc:
        print(f"[CONFIG] Warning reading {DEVICE_CONFIG_FILE}: {exc}")
        return _DEFAULT_DEVICE_ID


def load_timezone() -> str:
    """Read timezone string from device_config.json. Falls back to DEFAULT_TIMEZONE ('Asia/Kolkata')."""
    try:
        with open(DEVICE_CONFIG_FILE, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        tz_name = str(cfg.get("timezone") or "").strip()
        return tz_name if tz_name else DEFAULT_TIMEZONE
    except Exception:
        return DEFAULT_TIMEZONE


def get_timezone_obj():
    """Return a datetime.tzinfo object for the configured timezone (Asia/Kolkata / IST)."""
    import datetime
    tz_name = load_timezone()
    try:
        import zoneinfo
        return zoneinfo.ZoneInfo(tz_name)
    except Exception:
        # Robust fallback for Asia/Kolkata (IST: UTC+05:30)
        return datetime.timezone(datetime.timedelta(hours=5, minutes=30), name="IST")

# ---------------------------------------------------------------------------
# GNSS / timing thresholds & Location Trigger
# ---------------------------------------------------------------------------
GNSS_DATA_TIMEOUT_SEC        = 3.0         # Seconds before declaring data timeout
GNSS_DETAIL_INTERVAL_SEC     = 1.0         # Emit satellite detail once per second
GNSS_SATELLITE_SNAP_SEC      = 5.0         # Drop satellite detail if snapshot > 5 s old
DEFAULT_DISTANCE_INTERVAL_M  = 10.0        # Geodesic distance in meters between detection triggers
GNSS_LOST_TIMEOUT_SEC        = 15.0        # Grace/scan period in seconds on GNSS loss before clock fallback
TIME_FALLBACK_INTERVAL_SEC   = 30.0        # Interval in seconds between frames during clock fallback


# ---------------------------------------------------------------------------
# NavCast GNSS — phone app streaming NMEA over USB tethering TCP
# ---------------------------------------------------------------------------
NAVCAST_HOST        = "10.208.43.190"  # Phone USB tethering fallback IP
NAVCAST_PORT        = 10110            # NavCast TCP port
NAVCAST_AUTO_DETECT = True             # Auto-detect tethering gateway IP & port on connect

# ---------------------------------------------------------------------------
# GNSS fallback / IP-geolocation
# ---------------------------------------------------------------------------
ENABLE_GPS_FALLBACK       = False      # Do not inject fake/default coordinates; only live GNSS / NavCast is used
GNSS_FALLBACK_TIMEOUT_SEC = 20
GNSS_FALLBACK_REFRESH_SEC = 300

# ---------------------------------------------------------------------------
# Status LED Configuration (Raspberry Pi 40-Pin Header)
# ---------------------------------------------------------------------------
# Singular Orange Status LED:
#   - BCM GPIO 25 -> Physical Pin 22
#   - GND -> Physical Pin 20 (or any GND pin)
#
# Status Patterns:
#   - Solid ON           : System functional (ready, distance tracking & operating)
#   - Constant Blinking  : Scanning for GNSS (boot scan or mid-run reconnection)
#   - Triple Blink       : Capturing a frame on trigger
#   - Heartbeat Blink    : Error faced (camera failure, hardware fault)
LED_ORANGE_PIN = 25  # BCM 25 (Physical Pin 22)


