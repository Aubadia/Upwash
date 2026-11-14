# 150M Sweep, S1–S3 (2026-11-14)

This is a snapshot of the overnight sweep. Results and analysis are in
[roadmap/027](../../roadmap/027-sweep-150m.md).

- `sweep_driver.py` builds the configs in `configs/sweep150m/`, submits waves
  of at most 10 jobs through `scripts/train.sbatch`, and picks the best run of
  each stage by its final `eval/loss`. The live copy runs from
  `/work/u7725908/upwash/sweep/`, which `HERE` points to; that is where it
  writes `state.json`, `RESULTS.md` and `driver.log`.
- `RESULTS.md`, `state.json` and `driver.log` are the driver's outputs,
  copied here at 09:35, after S3 finished.
- `read_wandb_history.py` reads the per-step curves from local `.wandb`
  files. The Flock server rejects `wandb.Api`, so this is the only way to
  get the curves from the command line.

This is experiment-branch code, written by the agent. It is not meant for
`main`.
