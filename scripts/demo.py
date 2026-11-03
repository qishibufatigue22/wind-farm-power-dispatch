"""Run a synthetic dispatch example without external dependencies."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dispatch.solver import solve_dispatch, proxy_cost

if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    data = json.loads((root/'data/sample/dispatch.json').read_text())
    wind, total = data['wind_mps'], data['total_mw']
    power = solve_dispatch(wind,total)
    baseline = [total/len(wind)]*len(wind)
    print('Synthetic example (power: MW; wind: m/s)')
    print('Optimized power:', [round(p,6) for p in power])
    print('Total power:', sum(power))
    print('Baseline proxy cost:', proxy_cost(baseline,wind))
    print('Optimized proxy cost:', proxy_cost(power,wind))
