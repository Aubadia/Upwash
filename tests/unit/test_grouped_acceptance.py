"""The experiment gate must use the Trainer's existing LR metric names."""
import json


def test_acceptance_reads_first_group_from_plain_learning_rate(tmp_path):
    from experiments.sweep300m.check_acceptance import check
    for label in ('0.002', '0.0025'):
        events = [dict(step=step, metrics={'eval/loss': 6.}) for step in (100,200,300,400)]
        events[-1]['metrics'].update({
            'train/learning_rate': 0.,
            'train/learning_rate/muon_down': 0.,
            'train/learning_rate/adamw': 0.,
            'train/total_tokens_seen': 52428800})
        result = dict(steps=400,tokens=52428800,export_relative_error=.003,
                      end_to_end_tokens_per_second=1e6,training_and_eval_seconds=150.,
                      peak_allocated_bytes=20*2**30,history=events,
                      metadata=dict(model=dict(num_parameters=298273792),
                                    required_optimizer_groups=[{}, {}, {}]))
        (tmp_path/f'numerics-{label}.json').write_text(json.dumps(dict(passed=True)))
        (tmp_path/f'integration-{label}.json').write_text(json.dumps(result))
    assert set(check(tmp_path)) == {'0.002','0.0025'}
    assert json.loads((tmp_path/'acceptance.json').read_text())['passed']
