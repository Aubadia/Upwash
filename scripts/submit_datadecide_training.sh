#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ $# -ne 1 || ! "$1" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]]; then
    printf 'Usage: bash scripts/submit_datadecide_training.sh batch-name\n' >&2
    exit 2
fi

root=/work/asciibase64/upwash/dclm-150m
batch="$1"
manifest="$root/datadecide-top5/manifest-v1.json"
jobs="$root/datadecide-top5/$batch.tsv"
python="$PWD/.venv/bin/python"
[[ -s "$manifest" && -r .env ]]
command -v tmux >/dev/null
exec 9>"$jobs.lock"
flock -n 9

recipes=(
    falcon-and-cc-qc-orig-10p
    falcon-and-cc-qc-10p
    falcon-and-cc-qc-20p
    dclm-baseline-qc-20p
    dclm-baseline-qc-10p
)
ranks=(1 2 3 5 6)
partition=8gpus
touch "$jobs"
for index in "${!recipes[@]}"; do
    recipe="${recipes[index]}"
    rank="${ranks[index]}"
    run_id="$batch-$recipe"
    job_id=$(awk -F '\t' -v recipe="$recipe" '$2 == recipe { print $4 }' "$jobs")
    if [[ -z "$job_id" ]]; then
        submitted=$(sbatch --parsable --partition="$partition" --time=06:00:00 \
            --job-name="dd-paloma-$rank" scripts/dclm_150m.sbatch --train-datadecide \
            --manifest "$manifest" --recipe "$recipe" --run-id "$run_id")
        job_id=${submitted%%;*}
        [[ "$job_id" =~ ^[0-9]+$ ]]
        printf '%s\t%s\t%s\t%s\n' "$rank" "$recipe" "$partition" "$job_id" >> "$jobs"
    fi
    session="upwash-flock-$job_id"
    if ! tmux has-session -t "=$session" 2>/dev/null; then
        printf -v command 'cd %q && set -a && source .env && set +a && export WANDB_MODE=online PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 && exec %q scripts/sync_dclm_flock.py %q %q >> %q 2>&1' \
            "$PWD" "$python" "$root/runs/$run_id" "$job_id" "$root/logs/flock-$job_id.out"
        tmux new-session -d -s "$session" "$command" 9>&-
    fi
    printf 'Paloma rank %s  %s  %s  job %s\n' "$rank" "$recipe" "$partition" "$job_id"
done
