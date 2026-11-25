#!/bin/sh
set -eu

PROJECT_DIR="${TELESCOPE_OPTICAL_TRACKING_CONTROL_DIR:-${TELESCOPE_CONTROL_DIR:-$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)}}"
VENV_DIR="${TELESCOPE_OPTICAL_TRACKING_CONTROL_VENV:-${TELESCOPE_CONTROL_VENV:-$PROJECT_DIR/.venv}}"
TELESCOPE_HOST="${CELESTRON_TELESCOPE_HOST:-192.168.1.231}"
LOCAL_IP=$(/usr/bin/ip -4 route get "$TELESCOPE_HOST" 2>/dev/null | /usr/bin/awk '{for(i=1;i<=NF;i++) if($i=="src"){print $(i+1); exit}}' || true)
if [ -z "$LOCAL_IP" ]; then
    # HTTP must still start when Wi-Fi/DHCP or the telescope route is late.
    # The dashboard's route monitor replaces this fallback automatically.
    LOCAL_IP="127.0.0.1"
    echo "Telescope route is not ready; starting the dashboard with a loopback motor placeholder." >&2
fi

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR"
exec "$VENV_DIR/bin/python" "$PROJECT_DIR/tools/vision_dashboard.py" \
    --mode live-run --source picamera2 --device picamera2:0 \
    --width 640 --height 480 --fps 20 \
    --hil-host "$LOCAL_IP" --hil-port 2000 --hil-route-target "$TELESCOPE_HOST" \
    --hil-envelope-degrees 0 --hil-display-span-degrees 5 \
    --hil-dispatch-rate-hz 4 --host 0.0.0.0 --allow-lan --port 8000 \
    --log-directory "$PROJECT_DIR/logs/live-optical"

