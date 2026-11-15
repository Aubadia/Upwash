"""Read completed SDK summaries and submit only after a successful compiled smoke."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess

ROOT=Path('/work/asciibase64/upwash')


def read_summary(log_path, expected_tokens):
    text=Path(log_path).read_text(errors='replace')
    if 'hit config.recompile_limit' in text or 'InductorError:' in text or 'Traceback (most recent call last):' in text:
        raise ValueError('Smoke contains a runtime/compilation failure or eager fallback')
    values={}
    for key in ('eval/loss','eval/perplexity','train/tokens_per_second','train/total_tokens_seen'):
        matches=re.findall(rf'wandb:\s+{re.escape(key)} ([0-9.eE+-]+|nan|inf)\s*$',text,re.M)
        if not matches:
            raise ValueError(f'Missing final {key} in {log_path}')
        values[key]=float(matches[-1])
        if not math.isfinite(values[key]):
            raise ValueError(f'Nonfinite {key}: {values[key]}')
    if int(values['train/total_tokens_seen'])!=expected_tokens:
        raise ValueError(f'Token count differs from {expected_tokens}: {values}')
    if values['train/tokens_per_second']<=0:
        raise ValueError('Nonpositive throughput')
    if 'done:' not in text or 'Syncing run' not in text:
        raise ValueError('Missing successful script completion or online SDK run')
    urls=re.findall(r'https://[^\s\x1b]+/runs/[A-Za-z0-9_-]+',text)
    if not urls:
        raise ValueError('Missing online Flock run URL')
    values['run_url']=urls[-1]
    return values


def launch(log_path):
    job=os.environ['SLURM_JOB_ID']
    smoke=read_summary(log_path,240*131072)
    if not (ROOT/'validation/gpu-numerics.json').exists() or not (ROOT/'validation/data.json').exists():
        raise ValueError('Missing validation evidence')
    # Includes 50% margin plus two hours for startup, evaluation and export.
    hours=math.ceil(29999890432/smoke['train/tokens_per_second']/3600*1.5+2)
    if hours>48:
        raise ValueError(f'Projected walltime {hours}h exceeds partition maximum; do not launch')
    state_path=ROOT/'validation'/f'submissions-{job}.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else {'smoke':smoke,'wall_hours':hours,'jobs':{}}
    for multiplier in ('1.6','1.5'):
        if multiplier in state['jobs']:
            continue
        name=f'base150m-30b-seed2-am{multiplier}-{job}'
        config=f'configs/base150m/base150m-30b-seed2-am{multiplier}.json'
        env=dict(os.environ,CONFIG=config,RUN=name,WANDB_MODE='online',OUT=str(ROOT/'runs'/name))
        output=subprocess.check_output([
            'sbatch','--parsable',f'--dependency=afterok:{job}',f'--job-name={name}',
            f'--time={hours:02}:00:00','scripts/calibrate_base.sbatch',
        ],env=env,text=True).strip()
        state['jobs'][multiplier]={'job_id':output.split(';')[0],'run':name,'config':config}
        temporary=state_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(state,indent=2)+'\n');temporary.replace(state_path)
        print('CALIBRATION_SUBMITTED',multiplier,output,flush=True)


def collect(log_path):
    summary=read_summary(log_path,29999890432)
    cfg=json.loads(Path(os.environ['CONFIG']).read_text())
    summary.update(run=os.environ['RUN'],job_id=os.environ['SLURM_JOB_ID'],
                   adam_lr_multiplier=cfg['training']['optimizer']['adam_lr_multiplier'])
    output=Path(os.environ['OUT'])
    for name in ('model.safetensors','config.json','tokenizer.json'):
        if not (output/name).exists():
            raise ValueError(f'Missing export {output/name}')
    (output/'result.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('CALIBRATION_RESULT',json.dumps(summary),flush=True)
    validation_job=os.environ['RUN'].rsplit('-',1)[-1]
    pair=[ROOT/'runs'/f'base150m-30b-seed2-am{m}-{validation_job}'/'result.json' for m in ('1.6','1.5')]
    if all(path.exists() for path in pair):
        ranked=sorted((json.loads(path.read_text()) for path in pair),key=lambda item:item['eval/loss'])
        comparison={'winner':ranked[0]['run'],'criterion':'final full-dev eval/loss','results':ranked}
        destination=ROOT/'validation'/f'comparison-{validation_job}.json'
        temporary=destination.with_suffix(f'.{os.environ["SLURM_JOB_ID"]}.tmp')
        temporary.write_text(json.dumps(comparison,indent=2)+'\n');temporary.replace(destination)
        print('CALIBRATION_COMPARISON',json.dumps(comparison),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['launch','collect'])
    parser.add_argument('log')
    args=parser.parse_args()
    (launch if args.action=='launch' else collect)(args.log)
