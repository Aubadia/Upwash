"""Summarize only successful completed measurements, never failed/partial runs."""
import argparse
import json
from pathlib import Path


def summarize(path):
    rows = []
    for file in sorted(Path(path).glob('*.json')):
        value = json.loads(file.read_text())
        if 'median_tokens_per_second' in value:
            rows.append(value)
    print('| Case | GPUs | tok/s median | window min–max | allocated / reserved GiB | GPU h/B |')
    print('|---|---:|---:|---:|---:|---:|')
    for r in rows:
        rates = [w['tokens_per_second'] for w in r['windows']]
        print(f"| {r['case']} | {r['world_size']} | {r['median_tokens_per_second']:,.0f} | "
              f"{min(rates):,.0f}–{max(rates):,.0f} | {r['peak_allocated_bytes']/2**30:.2f} / {r['peak_reserved_bytes']/2**30:.2f} | "
              f"{r['gpu_hours_per_billion_tokens']:.3f} |")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    summarize(parser.parse_args().directory)
