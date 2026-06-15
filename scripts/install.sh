#!/usr/bin/env bash
# Auto Annotate — Installer
# Run from the project directory: bash scripts/install.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
INSTALL_DIR="$HOME/.local/share/auto_annotate"
APP_DIR="$HOME/.local/share/applications"
ICON_DIR="$HOME/.local/share/icons/hicolor/scalable/apps"
DESKTOP_FILE="$APP_DIR/auto_annotate.desktop"
DESKTOP_LINK="$HOME/Desktop/AutoAnnotate.desktop"
WORK_DIR="$HOME/Desktop/auto_annotate"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; BOLD='\033[1m'; NC='\033[0m'
info()  { echo -e "${GREEN}✔${NC}  $*"; }
warn()  { echo -e "${YELLOW}⚠${NC}  $*"; }
error() { echo -e "${RED}✘  ERROR:${NC} $*"; exit 1; }
step()  { echo -e "\n${BOLD}▶ $*${NC}"; }

echo -e "${BOLD}"
echo "  ╔══════════════════════════════════════╗"
echo "  ║       Auto Annotate — Installer      ║"
echo "  ╚══════════════════════════════════════╝"
echo -e "${NC}"

# ── Pre-flight checks ──────────────────────────────────────────────────────────

step "Checking requirements..."

command -v uv &>/dev/null \
    || error "uv not found.\n  Install: curl -LsSf https://astral.sh/uv/install.sh | sh"

[ -x /usr/bin/python3.10 ] \
    || error "Python 3.10 not found at /usr/bin/python3.10"

[ -d "$PROJECT_DIR/src/auto_annotator" ] \
    || error "src/auto_annotator/ not found in $PROJECT_DIR"

if [ ! -d /usr/lib/python3.10/dist-packages/tensorrt ]; then
    warn "TensorRT not found at /usr/lib/python3.10/dist-packages/tensorrt"
    warn "Inference will fail until TensorRT is installed via JetPack."
fi

info "Requirements OK"

# ── Install application files ──────────────────────────────────────────────────

step "Installing application to $INSTALL_DIR ..."

mkdir -p "$INSTALL_DIR"
cp -r "$PROJECT_DIR/src"            "$INSTALL_DIR/"
cp "$PROJECT_DIR/pyproject.toml"    "$INSTALL_DIR/"
[ -f "$PROJECT_DIR/.python-version" ] \
    && cp "$PROJECT_DIR/.python-version" "$INSTALL_DIR/" \
    || echo "3.10" > "$INSTALL_DIR/.python-version"
[ -f "$PROJECT_DIR/uv.lock" ] && cp "$PROJECT_DIR/uv.lock" "$INSTALL_DIR/"
[ -f "$PROJECT_DIR/README.md" ] && cp "$PROJECT_DIR/README.md" "$INSTALL_DIR/"

info "Application files copied"

# ── Python environment ─────────────────────────────────────────────────────────

step "Setting up Python 3.10 virtual environment..."

cd "$INSTALL_DIR"
uv venv --python /usr/bin/python3.10

# Expose system TensorRT (compiled for Python 3.10) inside the venv
TRT_SYSPATH="/usr/lib/python3.10/dist-packages"
if [ -d "$TRT_SYSPATH/tensorrt" ]; then
    echo "$TRT_SYSPATH" > "$INSTALL_DIR/.venv/lib/python3.10/site-packages/trt_system.pth"
    info "TensorRT system packages linked"
fi

step "Installing Python dependencies..."
CUDA_ROOT=/usr/local/cuda uv sync

step "Verifying imports..."
"$INSTALL_DIR/.venv/bin/python" - <<'PYCHECK'
import sys
ok = True
for mod in ["tensorrt", "pycuda.driver", "cv2", "numpy", "PIL"]:
    try:
        __import__(mod)
        print(f"  ✔  {mod}")
    except ImportError as e:
        print(f"  ✘  {mod}: {e}")
        ok = False
sys.exit(0 if ok else 1)
PYCHECK

# ── Launcher script ────────────────────────────────────────────────────────────

step "Creating launcher..."

cat > "$INSTALL_DIR/launch.sh" << LAUNCH
#!/usr/bin/env bash
cd "$INSTALL_DIR"
exec .venv/bin/python -m auto_annotator "\$@"
LAUNCH
chmod +x "$INSTALL_DIR/launch.sh"

info "Launcher created: $INSTALL_DIR/launch.sh"

# ── Application icon ───────────────────────────────────────────────────────────

step "Installing icon..."

mkdir -p "$ICON_DIR"
cp "$PROJECT_DIR/assets/auto_annotate.svg" "$ICON_DIR/auto_annotate.svg"

info "Icon installed"

# ── Desktop integration ────────────────────────────────────────────────────────

step "Registering application..."

mkdir -p "$APP_DIR"
cat > "$DESKTOP_FILE" << DESKTOP
[Desktop Entry]
Version=1.0
Type=Application
Name=Auto Annotate
GenericName=Image Annotation Tool
Comment=TensorRT-accelerated bulk image annotation — YOLO11 and RF-DETR
Exec=$INSTALL_DIR/launch.sh
Icon=$ICON_DIR/auto_annotate.svg
Terminal=false
StartupNotify=true
Categories=Graphics;Science;Education;
Keywords=annotation;yolo;detection;tensorrt;deepstream;dataset;
DESKTOP

# Make desktop shortcut
if [ -d "$HOME/Desktop" ]; then
    cp "$DESKTOP_FILE" "$DESKTOP_LINK"
    chmod +x "$DESKTOP_LINK"
    gio set "$DESKTOP_LINK" metadata::trusted true 2>/dev/null || true
    info "Desktop shortcut created: $DESKTOP_LINK"
fi

update-desktop-database "$APP_DIR" 2>/dev/null || true
gtk-update-icon-cache -f -t "$HOME/.local/share/icons/hicolor" 2>/dev/null || true

info "Application registered in system menu"

# ── Work directories ───────────────────────────────────────────────────────────

step "Creating work directories..."

mkdir -p "$WORK_DIR/input_images"
mkdir -p "$WORK_DIR/datasets"
mkdir -p "$WORK_DIR/model"

info "Work directories ready"

# ── Done ───────────────────────────────────────────────────────────────────────

echo -e "\n${BOLD}${GREEN}"
echo "  ╔══════════════════════════════════════╗"
echo "  ║      Installation Complete! ✔        ║"
echo "  ╚══════════════════════════════════════╝"
echo -e "${NC}"
echo "  Drop images into:"
echo -e "    ${BOLD}$WORK_DIR/input_images/${NC}"
echo ""
echo "  Exported datasets land in:"
echo -e "    ${BOLD}$WORK_DIR/datasets/<name>_yolo/${NC}"
echo -e "    ${BOLD}$WORK_DIR/datasets/<name>_coco/${NC}"
echo ""
echo "  Launch: double-click  Auto Annotate  on the Desktop"
echo "          or run: $INSTALL_DIR/launch.sh"
echo ""
