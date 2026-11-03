import random
import unittest
from src.dispatch.solver import solve_dispatch, proxy_cost


class DispatchTests(unittest.TestCase):
    def test_random_feasibility_and_improvement(self):
        rng = random.Random(42)
        for _ in range(100):
            wind = [rng.uniform(11.2, 25) for _ in range(100)]
            total = rng.uniform(0, 500)
            power = solve_dispatch(wind, total)
            avg = total/100
            self.assertLess(abs(sum(power)-total), 1e-9)
            self.assertTrue(all(0 <= p <= 5 and abs(p-avg) <= 1+1e-10 for p in power))
            self.assertLessEqual(proxy_cost(power, wind), proxy_cost([avg]*100, wind)+1e-9)

    def test_equal_wind_and_boundaries(self):
        for total in (0, 7, 15):
            self.assertEqual(solve_dispatch([12]*3, total), [total/3]*3)
        self.assertEqual(solve_dispatch([12, 20], 4, deviation_mw=0), [2,2])

    def test_reject_invalid_input(self):
        for wind,total in [([],0),([12],6),([float('nan')],2),([-1],2)]:
            with self.assertRaises(ValueError):
                solve_dispatch(wind,total)

    def test_two_turbine_grid_optimality(self):
        wind, total = [12,20], 8
        result = proxy_cost(solve_dispatch(wind,total),wind)
        for k in range(1001):
            p = 3+2*k/1000
            self.assertLessEqual(result,proxy_cost([p,8-p],wind)+1e-10)

if __name__ == '__main__':
    unittest.main()
