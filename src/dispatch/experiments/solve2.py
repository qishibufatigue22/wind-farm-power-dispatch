
#遗传算法
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from matplotlib import rcParams
#from pyswarm import pso  # 粒子群算法
from deap import base, creator, tools, algorithms  # 遗传算法
import random

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体显示中文
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# 主轴疲劳损伤计算函数
def calculate_fatigue_damage_shaft(P_ref, V):
    damage = (P_ref ** 2) / (V + 1)
    return damage


# 塔架疲劳损伤计算函数
def calculate_fatigue_damage_tower(P_ref, V):
    damage = (P_ref ** 2) / (V + 2)
    return damage


# 计算单个时间步的疲劳损伤
def calculate_total_damage(P_ref, V):
    # 将可能的 DEAP Individual 对象转换为 numpy 数组
    if isinstance(P_ref, list) or isinstance(P_ref, creator.Individual):
        P_ref = np.array(P_ref)

    shaft_damage = (P_ref ** 2) / (V + 1)
    tower_damage = (P_ref ** 2) / (V + 2)
    total_damage = 0.5 * np.sum(shaft_damage) + 0.5 * np.sum(tower_damage)
    return total_damage


# 累积疲劳损伤计算函数，分别计算参考功率和优化功率的损伤
def calculate_cumulative_damage(optimal_power, Pref_mw, V, previous_damage_optimal, previous_damage_ref):
    current_damage_optimal = calculate_total_damage(optimal_power, V)
    current_damage_ref = calculate_total_damage(Pref_mw, V)

    total_damage_optimal = previous_damage_optimal + current_damage_optimal
    total_damage_ref = previous_damage_ref + current_damage_ref

    return total_damage_optimal, total_damage_ref, current_damage_optimal, current_damage_ref


# 目标函数：最小化主轴+塔架 总疲劳损伤
def objective(power, Pref, V):
    total_damage = calculate_total_damage(power, V)
    return total_damage


# 使用 scipy.optimize.minimize 优化
def optimize_power_minimize(Pref, V, initial_power):
    max_power = 5
    total_pref = np.sum(Pref)

    result = minimize(
        objective, initial_power, args=(Pref, V),
        bounds=[(0, max_power)] * len(Pref),
        constraints={'type': 'eq', 'fun': power_constraints, 'args': (total_pref,)},
        options={'disp': False}
    )
    return result.x


# # 使用粒子群算法
# def pso_objective(power, *args):
#     V = args[0]
#     return calculate_total_damage(power, V)
#
#
# def optimize_power_pso(Pref, V, initial_power):
#     max_power = 5
#     total_pref = np.sum(Pref)
#
#     lb = [0] * len(Pref)
#     ub = [max_power] * len(Pref)
#
#     optimal_power, fopt = pso(pso_objective, lb, ub, args=(V,),
#                               constraints=[{'type': 'eq', 'fun': power_constraints, 'args': (total_pref,)}])
#     return optimal_power


# 遗传算法实现
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, 0, 5)  # 范围 [0, 5] MW
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=100)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def ga_objective(individual, V):
    # 确保将 Individual 转换为 numpy 数组
    power_array = np.array(individual)

    # 计算损伤
    return (calculate_total_damage(power_array, V),)


toolbox.register("evaluate", ga_objective, V=None)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)


def optimize_power_ga(Pref, V, initial_power):
    toolbox.unregister("evaluate")
    toolbox.register("evaluate", ga_objective, V=V)

    pop = toolbox.population(n=50)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=40, verbose=False)
    best_ind = tools.selBest(pop, 1)[0]
    return best_ind


# 功率约束条件
def power_constraints(power, total_pref):
    return np.sum(power) - total_pref


# 主函数
def solve_question_3(algorithm='minimize'):
    filepath = 'all_data_2.npy'
    data = np.load(filepath)

    num_wind_farms = data.shape[0]
    num_wind_turbines = data.shape[1]
    time_steps = data.shape[2]

    cumulative_damage_optimal = np.zeros((num_wind_farms, time_steps))
    cumulative_damage_ref = np.zeros((num_wind_farms, time_steps))
    optimal_power_history = np.zeros((num_wind_farms, num_wind_turbines, time_steps))

    for f in range(num_wind_farms):
        Pref = data[f, :, :, 1]
        V = data[f, :, :, 2]

        previous_damage_optimal = 0
        previous_damage_ref = 0

        for t in range(time_steps):
            Pref_mw = Pref[:, t] / 1000000

            initial_power = np.maximum(Pref[:, t] / np.max(Pref[:, t]) * 5, 0)

            # if algorithm == 'pso':
            #     optimal_power = optimize_power_pso(Pref_mw, V[:, t], initial_power)
            if algorithm == 'ga':
                optimal_power = optimize_power_ga(Pref_mw, V[:, t], initial_power)
            elif algorithm == 'minimize':
                optimal_power = optimize_power_minimize(Pref_mw, V[:, t], initial_power)
            else:
                raise ValueError("未知算法类型，支持 'pso', 'ga' 和 'minimize'")

            optimal_power_history[f, :, t] = optimal_power

            cumulative_damage_optimal[f, t], cumulative_damage_ref[
                f, t], current_damage_optimal, current_damage_ref = calculate_cumulative_damage(
                optimal_power, Pref_mw, V[:, t], previous_damage_optimal, previous_damage_ref)

            previous_damage_optimal = cumulative_damage_optimal[f, t]
            previous_damage_ref = cumulative_damage_ref[f, t]

            print(f"时间 {t + 1} 秒:")
            print(f"  当前参考功率的疲劳损伤: {current_damage_ref:.4f}")
            print(f"  当前优化功率的疲劳损伤: {current_damage_optimal:.4f}")
            print(f"  累积参考功率的疲劳损伤: {cumulative_damage_ref[f, t]:.4f}")
            print(f"  累积优化功率的疲劳损伤: {cumulative_damage_optimal[f, t]:.4f}")
            print('-' * 50)
            # 动态可视化
            if t % 100 == 0 and t > 0:
                plt.clf()

                fig = plt.figure(figsize=(16, 8))
                gs = fig.add_gridspec(2, 2, width_ratios=[3, 2], height_ratios=[2, 1], hspace=0.3)

                # 第一张子图，显示优化后的功率分配
                ax1 = fig.add_subplot(gs[0, 0])
                ax1.bar(range(num_wind_turbines), optimal_power, color='b')
                ax1.set_ylim([0, 5])
                ax1.set_title(f'风电场 {f + 1} 优化后的功率分配 at t = {t}s')
                ax1.set_xlabel('风机编号')
                ax1.set_ylabel('功率 (MW)')

                # 第二张子图，累积疲劳损伤对比
                ax2 = fig.add_subplot(gs[0, 1])
                ax2.plot(range(1, t + 1), cumulative_damage_optimal[f, :t], 'b', label='优化功率累积损伤', linewidth=2)
                ax2.plot(range(1, t + 1), cumulative_damage_ref[f, :t], 'r', label='参考功率累积损伤', linewidth=2,
                         linestyle='--')
                ax2.set_title('累积疲劳损伤对比')
                ax2.set_xlabel('时间 (s)')
                ax2.set_ylabel('累积疲劳损伤')
                ax2.legend()

                # 第三张子图，功率分配对比
                ax3 = fig.add_subplot(gs[1, :])
                ax3.plot(range(num_wind_turbines), Pref[:, t] / 1000000, 'r--', label=f'参考功率 at t={t}s')
                ax3.plot(range(num_wind_turbines), optimal_power, 'b-', label=f'优化功率 at t={t}s')
                ax3.set_title('功率分配对比')
                ax3.set_xlabel('风机编号')
                ax3.set_ylabel('功率 (MW)')
                ax3.legend(loc='upper right')

                plt.tight_layout()
                plt.pause(0.1)
    print('最终优化功率分配:')
    print(optimal_power_history)
    print('累积疲劳损伤总和（优化功率）:')
    print(np.sum(cumulative_damage_optimal))
    print('累积疲劳损伤总和（参考功率）:')
    print(np.sum(cumulative_damage_ref))


# 执行函数
solve_question_3(algorithm='ga')  # 可选择 'pso', 'ga' 或 'minimize'
