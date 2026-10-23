"""Verify seed arithmetic using only published cases and our returned job records."""
import json
from pathlib import Path


def combine(case_seed, run_offset):
    return case_seed + (case_seed + run_offset) * (case_seed + run_offset + 1) // 2


evidence = json.loads(Path(__file__).with_name('seed-comparison.json').read_text())
total = 0
for run in evidence['runs']:
    public = run['published_case_seeds']
    observed = run['returned_case_seeds']
    assert public[1] == public[0] + 1
    # For consecutive a, combine(a+1,b)-combine(a,b) = a+b+2.
    # These two observed seeds do not cross the modulus boundary.
    offset = observed[1] - observed[0] - public[0] - 2
    expected = [combine(a, offset) % evidence['modulus'] for a in public]
    assert expected == observed, run['job_id']
    total += len(public)
    print(f"{run['job_id']}: {len(public)}/{len(public)} exact matches; "
          f"{len(public)-2} cases not used to fit the offset")
print(f'Total: {total}/{total} exact matches')
