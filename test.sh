#!/bin/sh
set -eu

project_root="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
python="$project_root/.venv/bin/python"

if [ ! -x "$python" ]; then
    printf '%s\n' "Missing .venv/bin/python. Prepare the project's virtual environment first." >&2
    exit 1
fi

if ! "$python" -c 'import pytest' >/dev/null 2>&1; then
    printf '%s\n' "pytest is unavailable in .venv. Install pytest in the project's virtual environment first." >&2
    exit 1
fi

cd "$project_root"
export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"

if [ "$#" -eq 0 ]; then
    set -- tests/
fi

exec "$python" -m pytest -q "$@"
