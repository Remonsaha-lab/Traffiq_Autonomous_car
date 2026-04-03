#!/bin/bash
# ============================================================
#  TRAFFIQ — Donkey Car Simulator Environment Setup Script
#  Run this on your PC (Windows: use Git Bash / WSL2)
# ============================================================

set -euo pipefail

echo ""
echo "=================================================="
echo "  TRAFFIQ — Environment Setup Starting..."
echo "=================================================="
echo ""

# ---------- 1. CHECK PYTHON VERSION ----------
echo "[1/7] Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required="3.8"
if [[ "$(printf '%s\n' "$required" "$python_version" | sort -V | head -n1)" == "$required" ]]; then
    echo "  ✓ Python $python_version found."
else
    echo "  ✗ Python 3.8+ required. Please install it from https://python.org"
    exit 1
fi

# ---------- 2. CREATE VIRTUAL ENVIRONMENT ----------
echo ""
echo "[2/7] Creating virtual environment 'traffiq_env'..."
python3 -m venv traffiq_env
source traffiq_env/bin/activate
echo "  ✓ Virtual environment activated."

# ---------- 3. UPGRADE PIP ----------
echo ""
echo "[3/7] Upgrading pip..."
pip install --upgrade pip --quiet
echo "  ✓ pip upgraded."

# ---------- 4. INSTALL CORE DEPENDENCIES ----------
echo ""
echo "[4/7] Installing core dependencies..."
pip install \
    tensorflow==2.13.0 \
    opencv-python==4.8.0.76 \
    numpy==1.24.3 \
    pillow==10.0.0 \
    matplotlib==3.7.2 \
    pandas==2.0.3 \
    scikit-learn==1.3.0 \
    tqdm==4.66.1 \
    pyserial \
    pyyaml \
    --quiet
echo "  ✓ Core dependencies installed."

# ---------- 5. INSTALL DONKEY CAR LIBRARY ----------
echo ""
echo "[5/7] Installing Donkey Car library (stable release)..."
# NOTE: `main` currently requires Python >=3.11. Pinning to PyPI 5.0.0 keeps
# this script compatible with Python 3.10.
pip install donkeycar==5.0.0 --quiet
echo "  ✓ Donkey Car library installed."

# ---------- 6. DOWNLOAD SIMULATOR ----------
echo ""
echo "[6/7] Downloading Donkey Car Simulator..."
OS="$(uname -s)"
SIM_DIR="$HOME/donkey_sim"
mkdir -p "$SIM_DIR"

case "$OS" in
    Linux*)
        SIM_URL="https://github.com/tawnkramer/gym-donkeycar/releases/download/v22.11.06/DonkeySimLinux.zip"
        echo "  Detected: Linux"
        ;;
    Darwin*)
        SIM_URL="https://github.com/tawnkramer/gym-donkeycar/releases/download/v22.11.06/DonkeySimMac.zip"
        echo "  Detected: macOS"
        ;;
    MINGW*|CYGWIN*|MSYS*)
        SIM_URL="https://github.com/tawnkramer/gym-donkeycar/releases/download/v22.11.06/DonkeySimWindows.zip"
        echo "  Detected: Windows"
        ;;
    *)
        echo "  ✗ Unknown OS. Please download the simulator manually from:"
        echo "    https://github.com/tawnkramer/gym-donkeycar/releases"
        exit 1
        ;;
esac

echo "  Downloading from: $SIM_URL"
curl -L "$SIM_URL" -o "$SIM_DIR/donkey_sim.zip" --progress-bar
echo "  Extracting..."
unzip -q "$SIM_DIR/donkey_sim.zip" -d "$SIM_DIR"
echo "  ✓ Simulator downloaded to: $SIM_DIR"

# ---------- 7. INSTALL GYM-DONKEYCAR ----------
echo ""
echo "[7/7] Installing gym-donkeycar (RL interface)..."
pip install gym-donkeycar --quiet
echo "  ✓ gym-donkeycar installed."

# ---------- DONE ----------
echo ""
echo "=================================================="
echo "  ✓ Setup Complete!"
echo "=================================================="
echo ""
echo "  Next steps:"
echo "  1. Activate your env:   source traffiq_env/bin/activate"
echo "  2. Open the simulator:  $SIM_DIR"
echo "  3. Collect data:        python3 scripts/collect_data.py"
echo "  4. Train your model:    python3 training/train_dave2.py"
echo ""