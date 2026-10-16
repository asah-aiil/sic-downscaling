"""Collect five read-only trial runs into a new, auditable server folder."""
import csv
import json
import shutil
from pathlib import Path

root = Path('local-path/evaluations/cmcc4_vs_full10_20261015')
output = root / 'aggregate_input_navigation_summary_20261016'
thresholds = ('0.15', '0.3', '0.5', '0.7', '0.8')


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, columns):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def state(truth, prediction):
    if prediction == truth:
        return 'correct_open' if truth else 'correct_closed'
    return 'false_open' if prediction else 'false_closed'


all_daily = []
all_yearly = []
all_changed = []
baseline_changes = []
copy_jobs = []
for threshold in thresholds:
    run = root / f'aggregate_input_threshold_{threshold}'
    old = root / ('seam_trial_full_threshold015' if threshold == '0.15'
                  else f'seam_trial_full_threshold_{threshold}')
    settings = json.loads((run / 'settings.json').read_text())
    assert float(settings['threshold']) == float(threshold)
    assert 'no bilinear interpolation' in settings['baseline']
    assert len(settings['route_mask_repair']['row_col']) == 24
    daily = read_csv(run / 'daily_comparison.csv')
    old_daily = read_csv(old / 'daily_comparison.csv')
    if len(daily) != 1452 or len(old_daily) != len(daily):
        raise RuntimeError(f'Incomplete daily output for {threshold}')
    historical = {(r['domain'], r['date']): r for r in old_daily}
    if len(historical) != len(daily):
        raise RuntimeError(f'Duplicate old dates for {threshold}')
    for row in daily:
        key = (row['domain'], row['date'])
        older = historical[key]
        for field in ('truth_passable', 'full10_passable', 'cmcc4_passable'):
            if row[field] != older[field]:
                raise RuntimeError(f'{threshold} {key} changed {field}; baseline-only rerun expected')
        truth = int(row['truth_passable'])
        entry = dict(threshold=threshold, year=row['year'], date=row['date'], domain=row['domain'],
                     truth_passable=truth,
                     aggregate_input_passable=int(row['aggregate_input_passable']),
                     full10_passable=int(row['full10_passable']),
                     cmcc4_passable=int(row['cmcc4_passable']))
        all_daily.append(entry)
        errors = {method: state(truth, entry[method + '_passable'])
                  for method in ('aggregate_input', 'full10', 'cmcc4')}
        if any(value.startswith('false_') for value in errors.values()):
            all_changed.append(dict(entry, **{method + '_state': value for method, value in errors.items()}))
        if row['aggregate_input_passable'] != older['bilinear_passable']:
            baseline_changes.append(dict(threshold=threshold, domain=row['domain'],
                                         date=row['date'], truth_passable=truth,
                                         aggregate_input_passable=int(row['aggregate_input_passable']),
                                         old_bilinear_passable=int(older['bilinear_passable'])))
    for year in ('2024', '2025'):
        for domain in ('shipping_mask', 'no_shipping_mask'):
            subset = [r for r in daily if r['year'] == year and r['domain'] == domain]
            if len(subset) != (364 if year == '2024' else 362):
                raise RuntimeError(f'Incomplete {threshold} {year} {domain}')
            for method in ('aggregate_input', 'full10', 'cmcc4'):
                states = [state(int(r['truth_passable']), int(r[method + '_passable'])) for r in subset]
                correct_open = states.count('correct_open')
                correct_closed = states.count('correct_closed')
                false_open = states.count('false_open')
                false_closed = states.count('false_closed')
                all_yearly.append(dict(threshold=threshold, year=year, domain=domain,
                                       method=method, evaluated_days=len(subset),
                                       truth_open_days=sum(int(r['truth_passable']) for r in subset),
                                       predicted_open_days=sum(int(r[method + '_passable']) for r in subset),
                                       correct_days=correct_open + correct_closed,
                                       wrong_days=false_open + false_closed,
                                       correct_open=correct_open, correct_closed=correct_closed,
                                       false_open=false_open, false_closed=false_closed))
    copy_jobs.extend((run / name, f'threshold_{threshold}_{name}')
                     for name in ('settings.json', 'summary.csv', 'validation.json'))

copy_jobs.append((root / 'evaluate_cmcc4_vs_full10_navigation_aggregate_input_trial.py',
                  'evaluate_aggregate_input.py'))
copy_jobs.append((Path(__file__), 'consolidate_results.py'))

if output.exists():
    raise SystemExit(f'Refusing to overwrite existing folder: {output}')
output.mkdir(parents=True)
for source, name in copy_jobs:
    shutil.copy2(source, output / name)
write_csv(output / 'daily_all_thresholds.csv', all_daily,
          ['threshold', 'year', 'date', 'domain', 'truth_passable',
           'aggregate_input_passable', 'full10_passable', 'cmcc4_passable'])
write_csv(output / 'yearly_accuracy.csv', all_yearly,
          ['threshold', 'year', 'domain', 'method', 'evaluated_days',
           'truth_open_days', 'predicted_open_days', 'correct_days', 'wrong_days',
           'correct_open', 'correct_closed', 'false_open', 'false_closed'])
write_csv(output / 'disagreement_dates.csv', all_changed,
          ['threshold', 'year', 'date', 'domain', 'truth_passable',
           'aggregate_input_passable', 'full10_passable', 'cmcc4_passable',
           'aggregate_input_state', 'full10_state', 'cmcc4_state'])
write_csv(output / 'baseline_changed_vs_old_bilinear.csv', baseline_changes,
          ['threshold', 'domain', 'date', 'truth_passable',
           'aggregate_input_passable', 'old_bilinear_passable'])
(output / 'README.md').write_text(
    '# Aggregate-input NEP navigation comparison (2024–2025)\n\n'
    'This folder consolidates threshold 0.15, 0.3, 0.5, 0.7 and 0.8 trials. '
    'Each threshold evaluates 364 dates in 2024 and 362 dates in 2025.\n\n'
    'Baseline: the cached `sic_d` input channel (25 km, 448×304), with each '
    'original value repeated to its exactly aligned 4×4 high-resolution block. '
    'No bilinear SIC interpolation is used. This block expansion is only for '
    'applying the same fine-grid route mask and snapped endpoints to every field.\n\n'
    'Truth: Bremen 6.25 km target. Models: historical Full10 and CMCC4 '
    'checkpoints. A day is navigable if a finite SIC <= threshold path connects '
    'both endpoints. Diagonals cannot cut between blocked orthogonal cells; '
    'reachability is tested with four-neighbour connected components.\n\n'
    'The original route-mask NetCDF was not edited. In memory, 24 audited '
    'non-land pixels at the 180-degree seam were added to reconnect the route '
    'mask. All trial settings are copied here. The `no_shipping_mask` domain '
    'uses different snapped endpoints and should not be interpreted as a '
    'controlled mask-only comparison.\n\n'
    '`daily_all_thresholds.csv`: all 7,260 daily rows, both domains.\n'
    '`yearly_accuracy.csv`: counts against truth by year, threshold and model.\n'
    '`disagreement_dates.csv`: dates with at least one wrong prediction.\n'
    '`baseline_changed_vs_old_bilinear.csv`: only days on which the aggregate '
    'baseline differs from the previous bilinear diagnostic. The Full10, CMCC4 '
    'and truth daily decisions were checked to be unchanged.\n'
    '`threshold_*_summary.csv` and `threshold_*_settings.json`: source trial '
    'summary and provenance. Source daily run directories remain adjacent to '
    'this folder and were not moved or overwritten. The evaluation and '
    'consolidation scripts are copied here for reproducibility.\n',
    encoding='utf-8')
print('OUTPUT', output)
print('ROWS', len(all_daily), len(all_yearly), len(all_changed), len(baseline_changes))
for row in all_yearly:
    if row['domain'] == 'shipping_mask':
        print(row['threshold'], row['year'], row['method'],
              'correct', row['correct_days'], 'false_open', row['false_open'],
              'false_closed', row['false_closed'])
