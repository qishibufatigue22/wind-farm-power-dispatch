"""Reproduce full-data metrics and compare against a general-purpose QP solver."""
import argparse
import sys
import csv
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dispatch.solver import solve_dispatch, proxy_cost

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT/'data/附件2-风电机组采集数据.mat')
    args = parser.parse_args()
    source = args.data
    if not source.is_file():
        parser.error('Missing MAT file. See data/README.md or pass --data PATH.')
    (ROOT/'results').mkdir(exist_ok=True)
    data = loadmat(source, simplify_cells=True)['data_TS_WF']
    summary = {'model': 'paper quadratic proxy; not physical fatigue or lifetime', 'farms': {}}
    for name, farm in data.items():
        inputs = np.stack([wt['inputs'] for wt in farm['WT']])
        if inputs.shape[2] != 2 or not np.all(np.isfinite(inputs)):
            raise ValueError('invalid input data')
        n, steps, _ = inputs.shape
        times, baseline, optimized, rows = [], [], [], []
        max_balance = max_bound = max_deviation = 0.0
        reference_gaps, reference_times = [], []
        cumulative_b = cumulative_o = 0.0
        with (ROOT/'results'/f'{name}_power_mw.csv').open('w',newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['time_s']+[f'WT{i+1}' for i in range(n)])
            for t in range(steps):
                wind = inputs[:,t,1].tolist()
                total = float(inputs[:,t,0].sum()/1e6)
                start = perf_counter()
                p = solve_dispatch(wind,total)
                times.append((perf_counter()-start)*1000)
                avg = total/n
                b, o = proxy_cost([avg]*n,wind), proxy_cost(p,wind)
                baseline.append(b); optimized.append(o)
                max_balance = max(max_balance,abs(sum(p)-total))
                max_bound = max(max_bound,max(max(-x,x-5,0) for x in p))
                max_deviation = max(max_deviation,max(abs(x-avg) for x in p))
                if o > b+1e-8: raise AssertionError('worse than baseline')
                writer.writerow([t+1]+p)
                cumulative_b += b
                cumulative_o += o
                rows.append([t+1,b,o,cumulative_b,cumulative_o])
                if t < 20:
                    c = np.array([0.5/(v+1)+0.5/(v+2) for v in wind])
                    start = perf_counter()
                    ref = minimize(lambda x: np.dot(c,x*x),np.full(n,avg),jac=lambda x: 2*c*x,
                        bounds=[(max(0,avg-1),min(5,avg+1))]*n,
                        constraints={'type':'eq','fun':lambda x: x.sum()-total,'jac':lambda x: np.ones(n)},
                        method='SLSQP',options={'ftol':1e-10,'maxiter':300})
                    reference_times.append((perf_counter()-start)*1000)
                    if not ref.success: raise AssertionError(ref.message)
                    reference_gaps.append(abs(o-ref.fun))
        with (ROOT/'results'/f'{name}_cost.csv').open('w',newline='') as f:
            w = csv.writer(f); w.writerow(['time_s','baseline_proxy','optimized_proxy','baseline_cumulative','optimized_cumulative']); w.writerows(rows)
        summary['farms'][name] = {
            'turbines':n,'steps':steps,'baseline_proxy_total':sum(baseline),'optimized_proxy_total':sum(optimized),
            'proxy_reduction_pct':100*(1-sum(optimized)/sum(baseline)),
            'solver_median_ms':float(np.median(times)), 'solver_p95_ms':float(np.percentile(times,95)), 'solver_max_ms':max(times),
            'max_balance_error_mw':max_balance,'max_bound_violation_mw':max_bound,'max_deviation_from_average_mw':max_deviation,
            'slsqp_comparison_steps':len(reference_gaps),'max_slsqp_objective_gap':max(reference_gaps),
            'slsqp_median_ms':float(np.median(reference_times))}
    (ROOT/'results/metrics.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__ == '__main__': main()
