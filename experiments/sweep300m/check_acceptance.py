"""Require numerical and actual-Trainer evidence before any 60B launch."""
import json
import math
from pathlib import Path
import sys


def check(root):
    root = Path(root)
    results = {}
    for label in ('0.002', '0.0025'):
        numeric = json.loads((root/f'numerics-{label}.json').read_text())
        assert numeric['passed']
        result = json.loads((root/f'integration-{label}.json').read_text())
        assert result['steps'] == 400 and result['tokens'] == 52428800
        assert result['export_relative_error'] < .03
        assert result['end_to_end_tokens_per_second'] > 0
        assert result['metadata']['model']['num_parameters'] == 298273792
        assert len(result['metadata']['required_optimizer_groups']) == 3
        events = result['history']
        assert [e['step'] for e in events if 'eval/loss' in e['metrics']] == [100,200,300,400]
        for event in events:
            assert all(math.isfinite(v) for v in event['metrics'].values())
        final = events[-1]['metrics']
        assert final['train/total_tokens_seen'] == 52428800
        # Trainer deliberately gives group 0 the plain, backwards-compatible key.
        for key in ('train/learning_rate','train/learning_rate/muon_down','train/learning_rate/adamw'):
            assert final[key] == 0
        results[label] = dict(passed=True,dev_loss=final['eval/loss'],
                              seconds=result['training_and_eval_seconds'],
                              peak_allocated_bytes=result['peak_allocated_bytes'])
    (root/'acceptance.json').write_text(json.dumps(dict(passed=True,results=results),indent=2)+'\n')
    print(json.dumps(results),flush=True)
    return results


if __name__ == '__main__':
    check(sys.argv[1])
