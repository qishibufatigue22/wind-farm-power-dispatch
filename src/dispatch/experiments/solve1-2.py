#GIF
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.animation import FuncAnimation
from scipy.optimize import minimize
from matplotlib import rcParams
from matplotlib.cm import get_cmap

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
    shaft_damage = calculate_fatigue_damage_shaft(P_ref, V)
    tower_damage = calculate_fatigue_damage_tower(P_ref, V)
    total_damage = 0.5 * np.sum(shaft_damage) + 0.5 * np.sum(tower_damage)
    return total_damage


# 优化功率分配函数
def optimize_power_distribution(Pref, V, initial_power):
    max_power = 5  # 风机的最大功率
    total_pref = np.sum(Pref)

    result = minimize(
        objective, initial_power, args=(Pref, V),
        bounds=[(0, max_power)] * len(Pref),
        constraints={'type': 'eq', 'fun': power_constraints, 'args': (total_pref,)},
        options={'disp': False}
    )
    return result.x


# 目标函数：最小化主轴+塔架 总疲劳损伤
def objective(power, Pref, V):
    total_damage = calculate_total_damage(power, V)
    return total_damage


# 功率约束条件
def power_constraints(power, total_pref):
    return np.sum(power) - total_pref


# 动态展示并保存为 GIF 文件
def solve_question_3():
    filepath = 'all_data_2.npy'
    data = np.load(filepath)  # 加载数据 (形状: [2, 100, 2000, 9])

    num_wind_farms = data.shape[0]
    num_wind_turbines = data.shape[1]
    time_steps = data.shape[2]

    # 初始化参数
    optimal_power_history = np.zeros((num_wind_farms, num_wind_turbines, time_steps))

    for f in range(num_wind_farms):
        Pref = data[f, :, :, 1]  # 功率参考数据
        V = data[f, :, :, 2]  # 风速数据

        for t in range(time_steps):
            # 转换参考功率单位到MW
            Pref_mw = Pref[:, t] / 1000000  # 假设原单位为瓦特，转换为兆瓦

            initial_power = np.maximum(Pref[:, t] / np.max(Pref[:, t]) * 5, 0)
            optimal_power = optimize_power_distribution(Pref[:, t], V[:, t], initial_power)
            optimal_power_history[f, :, t] = optimal_power

    # 分别为风电场1和风电场2创建动态GIF

    # 风电场1 动态展示
    fig1, ax1 = plt.subplots(figsize=(16, 8))
    cmap1 = get_cmap('coolwarm')

    def update_farm1(frame):
        ax1.clear()
        ax1.bar(range(num_wind_turbines), optimal_power_history[0, :, frame],
                color=cmap1(np.linspace(0, 1, num_wind_turbines)), edgecolor='black')
        ax1.axhline(y=5, color='red', linestyle='--', linewidth=2, label='功率上限')
        ax1.set_ylim([2, 6])
        ax1.set_title(f'风电场 1 第 {frame + 1} 秒的优化功率分配', fontsize=16)
        ax1.set_xlabel('风机编号', fontsize=14)
        ax1.set_ylabel('功率 (MW)', fontsize=14)
        ax1.legend(loc='upper right')

    ani1 = FuncAnimation(fig1, update_farm1, frames=range(time_steps), interval=500)

    # 保存风电场1为GIF
    ani1.save('wind_farm_1_optimization.gif', writer='imagemagick', fps=2)

    # 风电场2 动态展示
    fig2, ax2 = plt.subplots(figsize=(16, 8))
    cmap2 = get_cmap('viridis')

    def update_farm2(frame):
        ax2.clear()
        ax2.bar(range(num_wind_turbines), optimal_power_history[1, :, frame],
                color=cmap2(np.linspace(0, 1, num_wind_turbines)), edgecolor='black')
        ax2.axhline(y=5, color='red', linestyle='--', linewidth=2, label='功率上限')
        ax2.set_ylim([2, 6])
        ax2.set_title(f'风电场 2 第 {frame + 1} 秒的优化功率分配', fontsize=16)
        ax2.set_xlabel('风机编号', fontsize=14)
        ax2.set_ylabel('功率 (MW)', fontsize=14)
        ax2.legend(loc='upper right')

    ani2 = FuncAnimation(fig2, update_farm2, frames=range(time_steps), interval=500)

    # 保存风电场2为GIF
    ani2.save('wind_farm_2_optimization.gif', writer='imagemagick', fps=2)

    plt.show()


# 执行函数
solve_question_3()
