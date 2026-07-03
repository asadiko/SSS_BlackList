#!/bin/sh
# entrypoint.sh — start as root, fix ownership of the bind-mounted data dir
# (works for any host UID), then drop privileges and exec as `appuser`.
set -e

APP_USER=appuser
APP_UID=$(id -u "$APP_USER")
APP_GID=$(id -g "$APP_USER")
APP_HOME=$(getent passwd "$APP_USER" | cut -d: -f6)
APP_HOME=${APP_HOME:-/home/$APP_USER}

# setpriv changes uid/gid but NOT $HOME — point HOME (and lib caches) at the
# runtime user's own writable dir.
export HOME="$APP_HOME"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$APP_HOME/.config/matplotlib}"

DATA_DIR="${BL_DATA_DIR:-/data}"
mkdir -p "$DATA_DIR" "$APP_HOME/.config" 2>/dev/null || true
chown -R "$APP_UID:$APP_GID" "$DATA_DIR" "$APP_HOME/.config" 2>/dev/null || true

if [ "$(id -u)" = "0" ]; then
    exec setpriv --reuid "$APP_UID" --regid "$APP_GID" --init-groups "$@"
else
    exec "$@"
fi
