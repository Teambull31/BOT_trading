#!/usr/bin/env bash
# Deploiement du trader sur une machine cible.
#
# Refuse de deployer si les tests echouent : ce systeme manipule de l'argent,
# un deploiement rouge n'a aucune raison d'exister.

set -euo pipefail

TARGET_DIR="${TARGET_DIR:-/opt/adaptive-trader}"
MODE="${1:-paper}"

echo "== Verification de l'environnement =="
command -v python3 >/dev/null || { echo "python3 introuvable"; exit 1; }

echo "== Tests =="
if ! python -m pytest tests/unit tests/integration -q; then
    echo "ECHEC : tests en erreur, deploiement annule."
    exit 1
fi

echo "== Linting =="
python -m ruff check src tests scripts

if [[ "$MODE" == "live" ]]; then
    echo "== Checklist go-live =="
    if ! python scripts/go_live_checklist.py; then
        echo "ECHEC : checklist go-live non validee, deploiement live annule."
        exit 1
    fi
fi

echo "== Installation dans ${TARGET_DIR} =="
mkdir -p "${TARGET_DIR}"
rsync -a --delete \
    --exclude ".venv" --exclude ".git" --exclude "data" \
    --exclude "logs" --exclude "artifacts" --exclude "__pycache__" \
    ./ "${TARGET_DIR}/"

cd "${TARGET_DIR}"
python3 -m venv .venv 2>/dev/null || true
./.venv/bin/pip install --quiet --upgrade pip
./.venv/bin/pip install --quiet -e .
mkdir -p data logs artifacts

echo "== Deploiement termine (mode ${MODE}) =="
echo "Demarrage    : sudo systemctl restart trader trader-watchdog"
echo "Verification : python -m trader.main status"
echo
echo "Rappel : le watchdog doit tourner dans un service SEPARE du trader."
