#!/bin/bash
# run_phases_234.sh — wait for the live Phase-1 pretrain to finish, then run the
# rest of the pipeline (SFT -> CoT -> QAT -> flux_best.pt) automatically.
#
# Detached-safe: survives the session. Aborts (does NOT run 2-4) if Phase 1 dies
# without completing, so a crashed pretrain never feeds garbage forward.

REPO="/home/redleadr/workspace/uchi"
SP="/tmp/claude-1000/-home-redleadr-workspace-uchi/dbdefb52-ad7a-4aa5-93a3-b8664e60d2ce/scratchpad"
P1LOG="$SP/phase1.log"
PID="${1:-1861584}"

cd "$REPO" || exit 1
echo "[orchestrator $(date '+%H:%M')] watching Phase 1 (pid $PID) ..."

while true; do
    if grep -q "Training complete" "$P1LOG" 2>/dev/null; then
        break
    fi
    if ! kill -0 "$PID" 2>/dev/null; then
        sleep 5
        if grep -q "Training complete" "$P1LOG" 2>/dev/null; then break; fi
        echo "[orchestrator $(date '+%H:%M')] Phase 1 pid $PID exited WITHOUT completing — aborting chain."
        exit 1
    fi
    sleep 60
done

echo "[orchestrator $(date '+%H:%M')] Phase 1 complete. Running phases 2-4 via train_all.sh ..."
bash scripts/train_all.sh
echo "[orchestrator $(date '+%H:%M')] pipeline finished (exit $?)."
