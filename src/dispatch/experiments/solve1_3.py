import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体显示中文
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 主轴疲劳损伤计算函数
def calculate_fatigue_damage_shaft(Power, V):
    damage = (Power ** 2) / (V + 1)
    return damage

# 塔架疲劳损伤计算函数
def calculate_fatigue_damage_tower(Power, V):
    damage = (Power ** 2) / (V + 2)
    return damage

# 计算总疲劳损伤
def calculate_total_damage(Power, Pref, V):
    shaft_damage = calculate_fatigue_damage_shaft(Power, V)
    tower_damage = calculate_fatigue_damage_tower(Power, V)
    total_damage = 0.5 * np.sum(shaft_damage) + 0.5 * np.sum(tower_damage)
    return total_damage

# 优化功率分配函数
def optimize_power_distribution(Pref, V, initial_power):
    max_power = 5  # 风机的最大功率
    total_pref = np.sum(Pref)

    # 修改 minimize 调用，确保 calculate_total_damage 接收 Power、Pref 和 V
    result = minimize(
        lambda power: calculate_total_damage(power, Pref, V),
        initial_power,
        bounds=[(0, max_power)] * len(Pref),
        constraints={'type': 'eq', 'fun': lambda power: np.sum(power) - total_pref},
        options={'disp': False}
    )
    return result.x

# 绘制热力图
def plot_heatmap(farm1_data_1, farm1_data_2, farm2_data_1, farm2_data_2, farm_id_1, farm_id_2, time_point):
    plt.figure(figsize=(14, 10))



    # 假设 farm1_data_1, farm1_data_2, farm2_data_1, farm2_data_2 是要可视化的数据
    # 这里为风电场 1 和 2 使用不同的配色风格

    # 风电场 1 使用柔和渐变风格
    plt.subplot(2, 2, 1)
    plt.imshow(farm1_data_1, cmap='YlGnBu', aspect='auto')  # 使用黄色-绿色-蓝色的渐变色
    plt.colorbar(label='参考功率 (MW)')
    plt.title(f'风电场 {farm_id_1} 参考功率热力图 (前20设备，{time_point}秒内)', fontsize=14, fontweight='bold')
    plt.xlabel('时间 (秒)')
    plt.ylabel('设备编号')

    plt.subplot(2, 2, 2)
    plt.imshow(farm1_data_2, cmap='BuPu', aspect='auto')  # 使用蓝色-紫色的渐变色，视觉效果柔和
    plt.colorbar(label='优化功率 (MW)')
    plt.title(f'风电场 {farm_id_1} 优化功率热力图 (前20设备，{time_point}秒内)', fontsize=14, fontweight='bold')
    plt.xlabel('时间 (秒)')
    plt.ylabel('设备编号')

    # 风电场 2 使用对比鲜明的风格
    plt.subplot(2, 2, 3)
    plt.imshow(farm2_data_1, cmap='RdYlBu', aspect='auto')  # 使用红色-黄色-蓝色的渐变色，适合表示变化
    plt.colorbar(label='参考功率 (MW)')
    plt.title(f'风电场 {farm_id_2} 参考功率热力图 (前20设备，{time_point}秒内)', fontsize=14, fontweight='bold')
    plt.xlabel('时间 (秒)')
    plt.ylabel('设备编号')

    plt.subplot(2, 2, 4)
    plt.imshow(farm2_data_2, cmap='PiYG', aspect='auto')  # 使用粉色-绿色的对比色，强调不同数据
    plt.colorbar(label='优化功率 (MW)')
    plt.title(f'风电场 {farm_id_2} 优化功率热力图 (前20设备，{time_point}秒内)', fontsize=14, fontweight='bold')
    plt.xlabel('时间 (秒)')
    plt.ylabel('设备编号')
    # 布局调整使图像不重叠
    plt.tight_layout()

    # 显示图像
    plt.show()


# 主函数
def solve_question_3():
    filepath = 'all_data_2.npy'
    data = np.load(filepath)  # 加载数据 (形状: [2, 100, 2000, 9])

    num_wind_farms = data.shape[0]
    num_wind_turbines = data.shape[1]
    time_steps = data.shape[2]

    # 取出前20台设备的功率参考值和优化值在特定时间的热力图
    selected_time = 100  # 选择特定的时间点
    selected_turbines_start = 60  # 设备起始编号
    selected_turbines_end = 79  # 设备结束编号

    # 获取风电场1和风电场2的数据
    # 获取风电场1和风电场2的数据
    Pref_farm_1 = data[0, selected_turbines_start:selected_turbines_end, :, 1]
    V_farm_1 = data[0, selected_turbines_start:selected_turbines_end, :, 2]

    Pref_farm_2 = data[1, selected_turbines_start:selected_turbines_end, :, 1]
    V_farm_2 = data[1, selected_turbines_start:selected_turbines_end, :, 2]

    # 生成优化功率数据
    optimal_power_farm_1 = np.zeros_like(Pref_farm_1)
    optimal_power_farm_2 = np.zeros_like(Pref_farm_2)

    for t in range(time_steps):
        Pref_mw_1 = Pref_farm_1[:, t] / 1000000  # 假设原单位为瓦特，转换为兆瓦
        initial_power_1 = np.maximum(Pref_farm_1[:, t] / np.max(Pref_farm_1[:, t]) * 5, 0)
        optimal_power_farm_1[:, t] = optimize_power_distribution(Pref_farm_1[:, t], V_farm_1[:, t], initial_power_1)

        Pref_mw_2 = Pref_farm_2[:, t] / 1000000
        initial_power_2 = np.maximum(Pref_farm_2[:, t] / np.max(Pref_farm_2[:, t]) * 5, 0)
        optimal_power_farm_2[:, t] = optimize_power_distribution(Pref_farm_2[:, t], V_farm_2[:, t], initial_power_2)

    # 选择特定秒的数据展示热力图
    plot_heatmap(Pref_farm_1[:, :selected_time], optimal_power_farm_1[:, :selected_time],
                 Pref_farm_2[:, :selected_time], optimal_power_farm_2[:, :selected_time],
                 farm_id_1=1, farm_id_2=2, time_point=selected_time)

# 执行函数
solve_question_3()
