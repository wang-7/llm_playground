#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}/scripts"

if [[ -f "${ROOT_DIR}/.env" ]]; then
  # Load variables from .env for local development.
  set -a
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.env"
  set +a
fi

if [[ -z "${HF_TOKEN:-}" && -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  echo "HF token missing. Set HF_TOKEN or HUGGINGFACE_HUB_TOKEN in environment (or ${ROOT_DIR}/.env)." >&2
  exit 1
fi

mkdir -p "${ROOT_DIR}/cmd_logs"
nohup python train_sft.py > "${ROOT_DIR}/cmd_logs/train_sft.log" 2>&1 &
echo "Training started. Log: ${ROOT_DIR}/cmd_logs/train_sft.log"
