#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODE="${1:-streamlit}"

case "$MODE" in
  api)
    echo "Menjalankan FastAPI di http://localhost:8000"
    uvicorn app_fastapi:app --host 0.0.0.0 --port 8000 --reload
    ;;
  streamlit)
    echo "Menjalankan Streamlit di http://localhost:8501"
    streamlit run app_streamlit.py --server.port 8501 --server.address 0.0.0.0
    ;;
  *)
    echo "Usage: ./run.sh [streamlit|api]"
    exit 1
    ;;
esac
