import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体显示中文
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def solve_question_4():
    # 数据文件路径
    filepath = '附件4-噪声和延迟作用下的采集数据.xlsx'

    # 加载噪声和延迟的数据（假设 Excel 文件中的每个工作表包含所需数据）
    Ft = pd.read_excel(filepath, sheet_name='Ft').values
    Pout = pd.read_excel(filepath, sheet_name='Pout').values
    Pref = pd.read_excel(filepath, sheet_name='Pref').values
    Vwin = pd.read_excel(filepath, sheet_name='Vwin').values
    wgenM = pd.read_excel(filepath, sheet_name='wgenM').values

    # 参数设置
    num_wind_turbines = Ft.shape[1]  # 风机数量
    time_steps = Ft.shape[0]  # 时间步长
    max_delay = 10  # 最大延迟（秒）

    # 初始化优化前后的功率分配
    optimal_power_history = np.zeros((time_steps, num_wind_turbines))
    average_power_history = np.zeros((time_steps, num_wind_turbines))

    # 初始化累积损伤记录
    cumulative_damage_optimal = np.zeros(time_steps)
    cumulative_damage_average = np.zeros(time_steps)

    # 进行功率优化，考虑噪声和延迟
    for t in range(time_steps):
        # 考虑延迟，获取上一个正常通信的时间点数据
        if t <= max_delay:
            previous_Pref = Pref[0, :]  # 首个时间步
        else:
            previous_Pref = Pref[t - max_delay, :]

        # 加入噪声
        noisy_Pref = previous_Pref + 0.1 * previous_Pref * np.random.randn(num_wind_turbines)  # 加入±10%的噪声

        # 进行功率优化
        optimal_power = optimize_power_distribution(noisy_Pref, Vwin[t, :])
        optimal_power_history[t, :] = optimal_power  # 存储优化结果
        average_power_history[t, :] = np.mean(noisy_Pref) * np.ones(num_wind_turbines)  # 均匀分配结果

        # 更新累积损伤
        cumulative_damage_optimal[t] = calculate_total_damage(optimal_power, Vwin[t, :])
        cumulative_damage_average[t] = calculate_total_damage(average_power_history[t, :], Vwin[t, :])

    # 可视化优化结果对比
    visualize_results(optimal_power_history, average_power_history, cumulative_damage_optimal,
                      cumulative_damage_average, Ft, Pout)


def optimize_power_distribution(Pref, V):
    # 优化目标函数：最小化累积疲劳损伤
    max_power = 5  # 风机的最大功率
    total_pref = np.sum(Pref)

    # 初始功率分配为均匀分配
    initial_power = np.ones(len(Pref)) * (total_pref / len(Pref))

    # 使用 scipy 的 minimize 进行优化
    result = minimize(
        fun=calculate_total_damage,
        x0=initial_power,
        args=(V,),
        bounds=[(0, max_power)] * len(Pref),
        constraints={'type': 'eq', 'fun': power_constraints, 'args': (total_pref,)},
        options={'disp': False}
    )
    return result.x


def calculate_total_damage(power, V):
    # 计算每台风机的疲劳损伤，考虑功率和风速的关系
    # 动态非线性损伤计算（假设风速5m/s为基准）
    fatigue_damage = (power ** 2) * (1 + 0.5 * (V - 5) ** 2)
    return np.sum(fatigue_damage)  # 总损伤


def power_constraints(power, total_pref):
    # 等式约束：所有风机的功率之和必须等于电网调度指令
    return np.sum(power) - total_pref


def visualize_results(optimal_power_history, average_power_history, cumulative_damage_optimal,
                      cumulative_damage_average, Ft, Pout):
    time_steps = optimal_power_history.shape[0]
    num_wind_turbines = optimal_power_history.shape[1]

    # 优化功率分配结果
    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    for i in range(num_wind_turbines):
        plt.plot(range(time_steps), optimal_power_history[:, i], label=f'WT{i + 1}')
    plt.title('优化后的功率分配')
    plt.xlabel('时间 (s)')
    plt.ylabel('功率 (MW)')
    plt.legend()

    # 平均功率分配结果
    plt.subplot(2, 1, 2)
    for i in range(num_wind_turbines):
        plt.plot(range(time_steps), average_power_history[:, i], label=f'WT{i + 1}')
    plt.title('均匀分配功率')
    plt.xlabel('时间 (s)')
    plt.ylabel('功率 (MW)')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # 累积疲劳损伤对比
    plt.figure()
    plt.plot(range(time_steps), cumulative_damage_optimal, 'r', label='优化后的损伤')
    plt.plot(range(time_steps), cumulative_damage_average, 'b', label='均匀分配的损伤')
    plt.title('累积疲劳损伤对比')
    plt.xlabel('时间 (s)')
    plt.ylabel('损伤')
    plt.legend()
    plt.show()


# 执行函数
solve_question_4()
