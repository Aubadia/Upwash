#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ $# -ne 2 || ! "$2" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]]; then
    printf 'Usage: bash scripts/submit_datadecide_paloma.sh /absolute/workbook.xlsx run-name\n' >&2
    exit 2
fi
workbook="$1"
root=/work/asciibase64/upwash/datadecide-paloma
run_dir="$root/runs/$2"
flock_venv="${DCLM_FLOCK_VENV:-/home/asciibase64/flock/.venv}"
python="$PWD/.venv/bin/python"
purelib='import sysconfig; print(sysconfig.get_paths()["purelib"])'
upwash_site=$("$python" -c "$purelib")
flock_site=$("$flock_venv/bin/python" -c "$purelib")

PYTHONPATH="$PWD:$PWD/src" "$root/olmo-venv/bin/python" - <<'PY'
import torch
import transformers
from hf_olmo import OLMoForCausalLM
from scripts.paloma_validation import evaluate

if torch.__version__ != "2.6.0+cu124" or transformers.__version__ != "4.50.3":
    raise RuntimeError("Unexpected DataDecide scoring environment")
PY

mkdir -p "$root/logs" "$root/locks"
exec 9>"$root/locks/$2"
flock -n 9

if [[ -f "$run_dir/manifest.json" ]]; then
    "$python" - "$run_dir/manifest.json" "$workbook" <<'PY_CHECK'
import hashlib
import json
from pathlib import Path
import sys

manifest = json.loads(Path(sys.argv[1]).read_text())
if hashlib.sha256(Path(sys.argv[2]).read_bytes()).hexdigest() != manifest["workbook_sha256"]:
    raise ValueError("This run name has a manifest for a different workbook")
PY_CHECK
else
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/src:$PWD:$upwash_site:${flock_venv%/*}:$flock_site" \
        "$python" scripts/datadecide_paloma.py prepare "$run_dir" "$workbook"
fi

submit_job() {
    local output waited=0
    until output=$(sbatch --parsable "$@" 2>&1); do
        if [[ "$output" != *QOSMaxSubmitJobPerUserLimit* ]]; then
            printf '%s\n' "$output" >&2
            return 1
        fi
        if (( waited == 0 )); then
            printf 'Waiting for an available Slurm submission slot.\n' >&2
            waited=1
        fi
        sleep 15
    done
    printf '%s\n' "${output%%;*}"
}

declare -A existing=()
if [[ -f "$run_dir/eval-job-ids.tsv" ]]; then
    while IFS=$'\t' read -r macro_rank job_id; do
        existing[$macro_rank]=$job_id
    done < "$run_dir/eval-job-ids.tsv"
fi
if [[ -f "$run_dir/jobs.json" ]]; then
    ranking=$("$python" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("ranking", ""))' "$run_dir/jobs.json")
    if [[ -n "$ranking" ]]; then
        cat "$run_dir/eval-job-ids.tsv"
        printf 'Ranking job %s\nOutput %s\n' "$ranking" "$run_dir"
        exit
    fi
fi

job_ids=()
for macro_rank in {1..10}; do
    if [[ -n "${existing[$macro_rank]:-}" ]]; then
        job_id=${existing[$macro_rank]}
    else
        job_id=$(submit_job scripts/eval_datadecide_paloma.sbatch "$run_dir" "$macro_rank")
        [[ "$job_id" =~ ^[0-9]+$ ]]
        printf '%s\t%s\n' "$macro_rank" "$job_id" >> "$run_dir/eval-job-ids.tsv"
    fi
    job_ids+=("$job_id")
    printf 'MacroRank %2d eval job %s\n' "$macro_rank" "$job_id"
done
"$python" - "$run_dir" <<'PY'
from pathlib import Path
import sys
from scripts.datadecide_paloma import atomic_json

root = Path(sys.argv[1])
entries = dict(line.split() for line in (root / "eval-job-ids.tsv").read_text().splitlines())
atomic_json(root / "jobs.json", {"evaluation": entries})
PY
dependencies=$(IFS=:; printf '%s' "${job_ids[*]}")
ranking_id=$(submit_job --dependency="afterany:$dependencies" scripts/rank_datadecide_paloma.sbatch "$run_dir")
[[ "$ranking_id" =~ ^[0-9]+$ ]]
"$python" - "$run_dir/jobs.json" "$ranking_id" <<'PY'
from pathlib import Path
import sys
from scripts.datadecide_paloma import atomic_json, read_json

path = Path(sys.argv[1])
jobs = read_json(path)
atomic_json(path, {**jobs, "ranking": sys.argv[2]})
PY
printf 'Ranking job %s\nOutput %s\n' "$ranking_id" "$run_dir"
