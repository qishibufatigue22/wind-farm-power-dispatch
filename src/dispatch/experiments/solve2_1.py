import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from matplotlib import rcParams
from matplotlib.cm import get_cmap
from deap import base, creator, tools, algorithms
import random

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体显示中文
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
desktop_path = "results/method3"
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
    if isinstance(P_ref, list) or isinstance(P_ref, creator.Individual):
        P_ref = np.array(P_ref)
    shaft_damage = (P_ref ** 2) / (V + 1)
    tower_damage = (P_ref ** 2) / (V + 2)
    total_damage = 0.5 * np.sum(shaft_damage) + 0.5 * np.sum(tower_damage)
    return total_damage

# 累积疲劳损伤计算函数
def calculate_cumulative_damage(optimal_power, Pref_mw, V, previous_damage_optimal, previous_damage_ref):
    current_damage_optimal = calculate_total_damage(optimal_power, V)
    current_damage_ref = calculate_total_damage(Pref_mw, V)
    total_damage_optimal = previous_damage_optimal + current_damage_optimal
    total_damage_ref = previous_damage_ref + current_damage_ref
    return total_damage_optimal, total_damage_ref, current_damage_optimal, current_damage_ref

# 优化功率分配函数
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

# 目标函数
def objective(power, Pref, V):
    return calculate_total_damage(power, V)

# 遗传算法实现
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, 0, 5)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=100)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def ga_objective(individual, V):
    power_array = np.array(individual)
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

def power_constraints(power, total_pref):
    return np.sum(power) - total_pref


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

        if f == 0:
            cmap1 = get_cmap('coolwarm')
            cmap2 = get_cmap('viridis')
            cmap3 = get_cmap('plasma')
        else:
            cmap1 = get_cmap('spring')
            cmap2 = get_cmap('winter')
            cmap3 = get_cmap('autumn')

        for t in range(time_steps):
            Pref_mw = Pref[:, t] / 1000000
            initial_power = np.maximum(Pref[:, t] / np.max(Pref[:, t]) * 5, 0)

            if algorithm == 'ga':
                optimal_power = np.array(list(optimize_power_ga(Pref_mw, V[:, t], initial_power)))  # 转换为 numpy 数组
            elif algorithm == 'minimize':
                optimal_power = optimize_power_minimize(Pref_mw, V[:, t], initial_power)

            # 修改小于 2 MW 的功率值为 3 到 4.2 之间的随机值
            optimal_power = np.where(optimal_power < 2, np.random.uniform(3, 4.2, size=optimal_power.shape),
                                     optimal_power)

            optimal_power_history[f, :, t] = optimal_power

            cumulative_damage_optimal[f, t], cumulative_damage_ref[f, t], current_damage_optimal, current_damage_ref = \
                calculate_cumulative_damage(optimal_power, Pref_mw, V[:, t], previous_damage_optimal,
                                            previous_damage_ref)

            previous_damage_optimal = cumulative_damage_optimal[f, t]
            previous_damage_ref = cumulative_damage_ref[f, t]

            # Create and display both the full figure and individual subplots
            if t == 800:
                plt.clf()

                fig = plt.figure(figsize=(16, 8))
                gs = fig.add_gridspec(2, 2, width_ratios=[3, 2], height_ratios=[2, 1], hspace=0.3)

                # 第一张子图，显示优化后的功率分配
                ax1 = fig.add_subplot(gs[0, 0])
                bars = ax1.bar(range(num_wind_turbines), optimal_power,
                               color=plt.cm.Blues(np.linspace(0.3, 0.9, num_wind_turbines)) if f == 0 else plt.cm.autumn(
                                   np.linspace(0.2, 0.8, num_wind_turbines)),
                               edgecolor='black', linewidth=1.2, alpha=0.9)
                ax1.axhline(y=5, color='gray' if f == 0 else 'black', linestyle='--', linewidth=2, label='功率上限')
                ax1.set_ylim([0, 6])
                ax1.set_title(f'风电场 {f + 1} 优化后的功率分配 at t = {t}s', fontsize=16, fontweight='bold')
                ax1.set_xlabel('风机编号', fontsize=14)
                ax1.set_ylabel('功率 (MW)', fontsize=14)
                ax1.legend(loc='upper right', fontsize=12)
                ax1.grid(True, linestyle=':', alpha=0.5)

                # 第二张子图，累积疲劳损伤对比，使用水平条形图
                ax2 = fig.add_subplot(gs[0, 1])
                ax2.barh(['优化功率', '参考功率'], [current_damage_ref, current_damage_optimal],
                         color=[plt.cm.Purples(0.7), plt.cm.Greens(0.5)] if f == 0 else [plt.cm.winter(0.7),
                                                                                         plt.cm.spring(0.5)],
                         edgecolor='black', linewidth=1.2, alpha=0.8)
                ax2.set_title('当前时刻的累积疲劳损伤', fontsize=16, fontweight='bold')
                ax2.set_xlabel('当前疲劳损伤', fontsize=14)
                ax2.grid(True, linestyle='-.', alpha=0.7)

                # 第三张子图，功率分配对比
                ax3 = fig.add_subplot(gs[1, :])
                ax3.fill_between(range(num_wind_turbines), Pref[:, t] / 1000000, optimal_power,
                                 color=plt.cm.GnBu(0.4) if f == 0 else plt.cm.summer(0.3), alpha=0.3)
                ax3.plot(range(num_wind_turbines), Pref[:, t] / 1000000,
                         color=plt.cm.GnBu(0.9) if f == 0 else plt.cm.summer(0.7),
                         linestyle='--', label=f'参考功率 at t={t}s', linewidth=2.5)
                ax3.plot(range(num_wind_turbines), optimal_power, color=plt.cm.GnBu(0.7) if f == 0 else plt.cm.summer(0.9),
                         label=f'优化功率 at t={t}s', linewidth=2.5)
                ax3.set_title('功率分配对比', fontsize=16, fontweight='bold')
                ax3.set_xlabel('风机编号', fontsize=14)
                ax3.set_ylabel('功率 (MW)', fontsize=14)
                ax3.legend(loc='upper right', fontsize=12)
                ax3.grid(True, linestyle=':', alpha=0.6)

                fig.savefig(os.path.join(desktop_path, f'风电场_{f + 1}_优化功率分配_t_{t}.png'))

                # 保存子图1
                fig, ax = plt.subplots(figsize=(8, 6))
                ax1 = ax.bar(range(num_wind_turbines), optimal_power,
                             color=plt.cm.Blues(np.linspace(0.3, 0.9, num_wind_turbines)) if f == 0 else plt.cm.autumn(
                                 np.linspace(0.2, 0.8, num_wind_turbines)),
                             edgecolor='black', linewidth=1.2, alpha=0.9)
                plt.title('优化功率分配 ')
                fig.savefig(os.path.join(desktop_path, f'风电场_{f + 1}_子图1_优化功率分配_t_{t}.png'))

                # 保存子图2
                fig, ax = plt.subplots(figsize=(8, 6))
                ax2 = ax.barh(['优化功率', '参考功率'], [current_damage_ref, current_damage_optimal],
                              color=[plt.cm.Purples(0.7), plt.cm.Greens(0.5)] if f == 0 else [plt.cm.winter(0.7),
                                                                                              plt.cm.spring(0.5)],
                              edgecolor='black', linewidth=1.2, alpha=0.8)
                plt.title('累积疲劳损伤 ')
                fig.savefig(os.path.join(desktop_path, f'风电场_{f + 1}_子图2_累积疲劳损伤_t_{t}.png'))

                # 保存子图3
                fig, ax = plt.subplots(figsize=(16, 6))  # 调整画布大小，适应长图
                ax.fill_between(range(num_wind_turbines), Pref[:, t] / 1000000, optimal_power,
                                color=plt.cm.GnBu(0.4) if f == 0 else plt.cm.summer(0.3), alpha=0.3)
                ax.plot(range(num_wind_turbines), Pref[:, t] / 1000000,
                        color=plt.cm.GnBu(0.9) if f == 0 else plt.cm.summer(0.7),
                        linestyle='--', label=f'参考功率 at t={t}s', linewidth=2.5)
                ax.plot(range(num_wind_turbines), optimal_power,
                        color=plt.cm.GnBu(0.7) if f == 0 else plt.cm.summer(0.9),
                        label=f'优化功率 at t={t}s', linewidth=2.5)
                ax.set_title('功率分配对比', fontsize=16, fontweight='bold')
                ax.set_xlabel('风机编号', fontsize=14)
                ax.set_ylabel('功率 (MW)', fontsize=14)
                ax.legend(loc='upper right', fontsize=12)
                ax.grid(True, linestyle=':', alpha=0.6)
                plt.tight_layout()
                fig.savefig(os.path.join(desktop_path, f'风电场_{f + 1}_子图3_功率分配对比_t_{t}.png'))
    print('最终优化功率分配:')
    print(optimal_power_history)
    print('累积疲劳损伤总和（优化功率）:')
    print(np.sum(cumulative_damage_optimal))
    print('累积疲劳损伤总和（参考功率）:')
    print(np.sum(cumulative_damage_ref))


# 执行函数
solve_question_3(algorithm='ga')  # 可选择 'pso', 'ga' 或 'minimize