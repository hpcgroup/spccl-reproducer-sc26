#!/bin/bash
set -euo pipefail

module load pytorch/2.8.0

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

python -m venv --system-site-packages .venv
source .venv/bin/activate

cd axonn && pip install -e . && cd ..

pip install regex six numpy

echo "Setup complete. Activate with: source .venv/bin/activate"
