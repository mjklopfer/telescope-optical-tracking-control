#!/bin/sh
set -eu

PROJECT_DIR="${TELESCOPE_CONTROL_DIR:-$HOME/telescope-control}"
UNIT_SOURCE="$PROJECT_DIR/packaging/systemd/user/telescope-vision-dashboard.service"
UNIT_TARGET="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"

mkdir -p "$UNIT_TARGET"
cp "$UNIT_SOURCE" "$UNIT_TARGET/"
systemctl --user daemon-reload
systemctl --user enable telescope-vision-dashboard.service
systemctl --user restart telescope-vision-dashboard.service
systemctl --user --no-pager --full status telescope-vision-dashboard.service || true
printf '%s\n' 'The dashboard starts with this user session. The companion Celestron bridge remains a separate service.'

