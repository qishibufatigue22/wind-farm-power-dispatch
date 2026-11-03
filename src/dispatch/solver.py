"""Exact convex dispatch for the paper's proxy cost; all power is in MW."""
import math


def solve_dispatch(wind, total_mw, rated_mw=5.0, deviation_mw=1.0, alpha=0.5):
    wind = list(wind)
    if not wind or any(not math.isfinite(v) or v < 0 for v in wind):
        raise ValueError('wind must contain finite nonnegative values')
    if (not all(math.isfinite(x) for x in (total_mw, rated_mw, deviation_mw, alpha))
            or rated_mw <= 0 or deviation_mw < 0 or not 0 <= alpha <= 1):
        raise ValueError('invalid parameters')
    n = len(wind)
    if not 0 <= total_mw <= n * rated_mw:
        raise ValueError('infeasible total power')
    avg = total_mw / n
    lower, upper = max(0.0, avg-deviation_mw), min(rated_mw, avg+deviation_mw)
    c = [alpha/(v+1)+(1-alpha)/(v+2) for v in wind]
    if total_mw == 0 or total_mw == n*rated_mw or lower == upper:
        return [avg]*n
    # KKT: p_i = clip(lambda / (2*c_i), lower, upper).
    lo, hi = 0.0, max(2*x*upper for x in c)
    for _ in range(64):
        mid = (lo+hi)/2
        if sum(min(upper, max(lower, mid/(2*x))) for x in c) < total_mw:
            lo = mid
        else:
            hi = mid
    return [min(upper, max(lower, (lo+hi)/(4*x))) for x in c]


def proxy_cost(power, wind):
    return sum(p*p*(0.5/(v+1)+0.5/(v+2)) for p,v in zip(power, wind))
