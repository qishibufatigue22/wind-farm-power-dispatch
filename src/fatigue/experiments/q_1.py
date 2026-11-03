import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体支持
rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

def solve_question_1():
    # 读取主轴扭矩与塔架推力数据 (假设文件已经加载到工作目录中)
    filepath = '附件1-疲劳评估数据.xls'

    # 读取前100秒的主轴扭矩数据和塔架推力数据，排除最后两行雨流计数结果
    torque_data = pd.read_excel(filepath, sheet_name='主轴扭矩').iloc[:100, 1:].to_numpy()  # 选择前100行数据
    thrust_data = pd.read_excel(filepath, sheet_name='塔架推力').iloc[:100, 1:].to_numpy()  # 选择前100行数据

    # 风机数量和时间步长
    num_wind_turbines = torque_data.shape[1]  # 100台风机
    time_steps = torque_data.shape[0]  # 100秒

    # S-N曲线参数 (假设通过实验数据获得这些参数)
    sn_curve_params = {'S0': 9.77e70, 'b': 10}  # 截距和斜率
    design_lifetime = 42565440.4361  # 风机设计寿命 (N)
    m = 10  # Wohler指数

    # Goodman 曲线参数
    fatigue_limit = 1e7  # 疲劳极限 (fatigue limit)
    uts = 5e8  # 材料抗拉强度 (ultimate tensile strength)

    # 初始化累积损伤矩阵和等效疲劳载荷
    cumulative_damage_torque = np.zeros((time_steps, num_wind_turbines))  # 主轴
    cumulative_damage_thrust = np.zeros((time_steps, num_wind_turbines))  # 塔架
    equivalent_fatigue_loads = np.zeros(num_wind_turbines)  # 等效疲劳载荷
    total_cumulative_damage = np.zeros(num_wind_turbines)  # 累积疲劳损伤

    # 遍历每个风机及其所有时间点数据
    for i in range(num_wind_turbines):
        load_sum_torque = 0  # 保存 L_i^m 的总和
        load_sum_thrust = 0
        for t in range(1, time_steps):
            # 当前风机的主轴扭矩和塔架推力
            torque_max = max(torque_data[:t, i])  # 最大应力
            torque_min = min(torque_data[:t, i])  # 最小应力
            thrust_max = max(thrust_data[:t, i])
            thrust_min = min(thrust_data[:t, i])

            # 计算应力幅值和均值应力
            torque_amplitude, torque_mean = calculate_amplitude_mean(torque_max, torque_min)
            thrust_amplitude, thrust_mean = calculate_amplitude_mean(thrust_max, thrust_min)

            # Goodman 修正
            eq_stress_torque = goodman_correction(torque_amplitude, torque_mean, fatigue_limit, uts)
            eq_stress_thrust = goodman_correction(thrust_amplitude, thrust_mean, fatigue_limit, uts)

            # 根据修正后的等效应力计算损伤
            damage_torque = calculate_damage(eq_stress_torque, sn_curve_params['S0'], sn_curve_params['b'])
            damage_thrust = calculate_damage(eq_stress_thrust, sn_curve_params['S0'], sn_curve_params['b'])

            # 累加到总损伤
            cumulative_damage_torque[t, i] = cumulative_damage_torque[t - 1, i] + damage_torque
            cumulative_damage_thrust[t, i] = cumulative_damage_thrust[t - 1, i] + damage_thrust

            # 累积 L_i^m 值
            load_sum_torque += eq_stress_torque ** m
            load_sum_thrust += eq_stress_thrust ** m

        # 计算等效疲劳载荷，开 m 次方并除以设计寿命 N
        equivalent_fatigue_loads[i] = ((load_sum_torque + load_sum_thrust) / design_lifetime) ** (1 / m)

        # 计算每台风机的累积疲劳损伤
        total_cumulative_damage[i] = cumulative_damage_torque[-1, i] + cumulative_damage_thrust[-1, i]

    # 输出累积损伤结果到Excel文件
    output_filename = '附件5-问题一答案表.xlsx'
    output_damage_to_excel(cumulative_damage_torque, cumulative_damage_thrust, total_cumulative_damage,
                           equivalent_fatigue_loads, output_filename)

    # 分别绘制主轴和塔架的累积疲劳损伤
    plot_torque_damage_growth(cumulative_damage_torque)
    plot_thrust_damage_growth(cumulative_damage_thrust)


# 计算应力幅值和均值应力
def calculate_amplitude_mean(max_stress, min_stress):
    amplitude = (max_stress - min_stress) / 2
    mean_stress = (max_stress + min_stress) / 2
    return amplitude, mean_stress


# Goodman 修正计算等效应力
def goodman_correction(amplitude, mean_stress, fatigue_limit, uts):
    return amplitude * (1 - mean_stress / uts)


# 计算损伤的辅助函数
def calculate_damage(stress, S0, b):
    # 根据 S-N 曲线的公式: N = S0 / (stress^b)
    N = S0 / (stress ** b)
    return 1 / N  # 损伤是每个周期承受的应力的反比


# 绘制主轴累积疲劳损伤增长情况
def plot_torque_damage_growth(cumulative_damage_torque):
    # 随机选择一些风机进行可视化展示
    num_turbines_to_plot = 5
    sample_turbines = np.random.choice(cumulative_damage_torque.shape[1], num_turbines_to_plot, replace=False)

    plt.figure(figsize=(10, 6))

    for i in sample_turbines:
        plt.plot(range(cumulative_damage_torque.shape[0]), cumulative_damage_torque[:, i], label=f'WT{i + 1}')
        plt.xlabel('时间 (秒)')
        plt.ylabel('累积疲劳损伤')
        plt.title('主轴累积疲劳损伤增长')
        plt.yscale('log')  # 使用对数坐标
        plt.legend()

    plt.tight_layout()
    plt.show()

# 绘制塔架累积疲劳损伤增长情况
def plot_thrust_damage_growth(cumulative_damage_thrust):
    # 随机选择一些风机进行可视化展示
    num_turbines_to_plot = 5
    sample_turbines = np.random.choice(cumulative_damage_thrust.shape[1], num_turbines_to_plot, replace=False)

    plt.figure(figsize=(10, 6))

    for i in sample_turbines:
        plt.plot(range(cumulative_damage_thrust.shape[0]), cumulative_damage_thrust[:, i], label=f'WT{i + 1}')
        plt.xlabel('时间 (秒)')
        plt.ylabel('累积疲劳损伤')
        plt.title('塔架累积疲劳损伤增长')
        plt.yscale('log')  # 使用对数坐标
        plt.legend()

    plt.tight_layout()
    plt.show()


# 输出计算结果到Excel文件
def output_damage_to_excel(cumulative_damage_torque, cumulative_damage_thrust, total_cumulative_damage,
                           equivalent_fatigue_loads, output_filename):
    # 创建时间列
    time_col = np.arange(1, cumulative_damage_torque.shape[0] + 1)

    # 创建DataFrame
    torque_df = pd.DataFrame(cumulative_damage_torque,
                             columns=[f'WT{i + 1}' for i in range(cumulative_damage_torque.shape[1])])
    thrust_df = pd.DataFrame(cumulative_damage_thrust,
                             columns=[f'WT{i + 1}' for i in range(cumulative_damage_thrust.shape[1])])

    # 插入时间列
    torque_df.insert(0, '时间', time_col)
    thrust_df.insert(0, '时间', time_col)

    # 添加等效疲劳载荷和累积疲劳损伤到表格底部
    torque_df.loc['等效疲劳载荷'] = [None] + list(equivalent_fatigue_loads)
    thrust_df.loc['等效疲劳载荷'] = [None] + list(equivalent_fatigue_loads)
    torque_df.loc['累积疲劳损伤'] = [None] + list(total_cumulative_damage)
    thrust_df.loc['累积疲劳损伤'] = [None] + list(total_cumulative_damage)

    # 写入Excel文件
    with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
        torque_df.to_excel(writer, sheet_name='主轴累积疲劳损伤', index=False)
        thrust_df.to_excel(writer, sheet_name='塔架累积疲劳损伤', index=False)


if __name__ == "__main__":
    solve_question_1()
