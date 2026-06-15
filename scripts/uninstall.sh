#!/usr/bin/env bash
# Auto Annotate — Uninstaller

set -euo pipefail

INSTALL_DIR="$HOME/.local/share/auto_annotate"
APP_DIR="$HOME/.local/share/applications"
ICON_DIR="$HOME/.local/share/icons/hicolor/scalable/apps"
DESKTOP_FILE="$APP_DIR/auto_annotate.desktop"
DESKTOP_LINK="$HOME/Desktop/AutoAnnotate.desktop"
WORK_DIR="$HOME/Desktop/auto_annotate"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BOLD='\033[1m'; NC='\033[0m'
info() { echo -e "${GREEN}✔${NC}  $*"; }
warn() { echo -e "${YELLOW}⚠${NC}  $*"; }

echo -e "${BOLD}"
echo "  ╔══════════════════════════════════════╗"
echo "  ║      Auto Annotate — Uninstaller     ║"
echo "  ╚══════════════════════════════════════╝"
echo -e "${NC}"

# ── Remove application files ───────────────────────────────────────────────────

if [ -d "$INSTALL_DIR" ]; then
    rm -rf "$INSTALL_DIR"
    info "Removed application directory: $INSTALL_DIR"
else
    warn "Application directory not found (already removed?): $INSTALL_DIR"
fi

# ── Remove desktop integration ─────────────────────────────────────────────────

if [ -f "$DESKTOP_FILE" ]; then
    rm -f "$DESKTOP_FILE"
    info "Removed application menu entry"
fi

if [ -f "$DESKTOP_LINK" ]; then
    rm -f "$DESKTOP_LINK"
    info "Removed desktop shortcut"
fi

if [ -f "$ICON_DIR/auto_annotate.svg" ]; then
    rm -f "$ICON_DIR/auto_annotate.svg"
    info "Removed icon"
fi

update-desktop-database "$APP_DIR" 2>/dev/null || true
gtk-update-icon-cache -f -t "$HOME/.local/share/icons/hicolor" 2>/dev/null || true

# ── Done ───────────────────────────────────────────────────────────────────────

echo ""
echo -e "${BOLD}${GREEN}  Uninstallation complete.${NC}"
echo ""
echo -e "  ${YELLOW}Your data has NOT been removed:${NC}"
echo "    $WORK_DIR/input_images/"
echo "    $WORK_DIR/datasets/"
echo ""
echo "  To delete your data too:"
echo -e "    ${BOLD}rm -rf $WORK_DIR${NC}"
echo ""
