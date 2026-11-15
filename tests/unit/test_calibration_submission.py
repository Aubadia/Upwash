"""Acceptance gating must reject partial, divergent, or offline smoke summaries."""
import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('base_results',Path(__file__).parents[2]/'scripts/base_calibration_results.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

GOOD='''wandb: Syncing run base-smoke
wandb: View run at https://flock.cerulean.works/cerulean-labs/lab6-predictable-scaling/runs/abc123
wandb: eval/loss 4.5
wandb: eval/perplexity 90.0
wandb: train/tokens_per_second 1500000
wandb: train/total_tokens_seen 31457280
[12:00:00] done: /work/run
'''


def test_completed_online_summary_passes(tmp_path):
    log=tmp_path/'log';log.write_text(GOOD)
    result=module.read_summary(log,31457280)
    assert result['eval/loss']==4.5
    assert result['run_url'].endswith('/abc123')


@pytest.mark.parametrize('text',[
    GOOD.replace('31457280','131072'),
    GOOD.replace('eval/loss 4.5','eval/loss nan'),
    GOOD.replace('done:','failed:'),
    GOOD.replace('Syncing run','Offline run'),
    GOOD.replace('1500000','0'),
    GOOD+'torch._dynamo hit config.recompile_limit (8)\n',
    GOOD+'torch._inductor.exc.InductorError: CUDA driver error: file not found\n',
])
def test_invalid_smoke_cannot_launch_training(tmp_path,text):
    log=tmp_path/'log';log.write_text(text)
    with pytest.raises(ValueError):
        module.read_summary(log,31457280)
