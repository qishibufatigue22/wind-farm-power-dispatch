#子图
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from matplotlib import rcParams
from matplotlib.cm import get_cmap

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体显示中文
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 获取桌面路径
desktop_path = 'results/method1'

# 保存图像的代码部分
# 保存图像的代码部分
def save_images(f, t, optimal_power, Pref, current_damage_optimal, current_damage_ref, num_wind_turbines):
    # 创建保存路径
    wind_farm_path = os.path.join(desktop_path, f'风电场_{f + 1}_t_{t}')
    os.makedirs(wind_farm_path, exist_ok=True)

    # 根据风电场选择不同的颜色映射
    if f == 0:
        cmap1 = get_cmap('coolwarm')
        cmap2 = get_cmap('viridis')
        cmap3 = get_cmap('plasma')
    else:
        cmap1 = get_cmap('spring')
        cmap2 = get_cmap('winter')
        cmap3 = get_cmap('autumn')

    # 保存大图
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[3, 2], height_ratios=[2, 1], hspace=0.3)

    # 第一张子图，显示优化后的功率分配
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.bar(range(num_wind_turbines), optimal_power, color=cmap1(np.linspace(0, 1, num_wind_turbines)))
    ax1.set_ylim([0, 5])
    ax1.set_title(f'风电场 {f + 1} 优化后的功率分配 at t = {t}s')
    ax1.set_xlabel('风机编号')
    ax1.set_ylabel('功率 (MW)')

    # 第二张子图，累积疲劳损伤对比
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.barh(['优化功率', '参考功率'], [current_damage_ref, current_damage_optimal],
             color=[cmap2(0.8), cmap2(0.2)], edgecolor='black', linewidth=1.5)
    ax2.set_title('累积疲劳损伤对比')
    ax2.set_xlabel('时间 (s)')
    ax2.set_ylabel('累积疲劳损伤')
    ax2.legend()

    # 第三张子图，功率分配对比
    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(range(num_wind_turbines), Pref[:, t] / 1000000, color=cmap3(0.7), linestyle='--', label=f'参考功率 at t={t}s')
    ax3.plot(range(num_wind_turbines), optimal_power, color=cmap3(0.9), label=f'优化功率 at t={t}s')
    ax3.set_title('功率分配对比')
    ax3.set_xlabel('风机编号')
    ax3.set_ylabel('功率 (MW)')
    ax3.legend(loc='upper right')

    plt.tight_layout()
    fig.savefig(os.path.join(wind_farm_path, f'风电场_{f + 1}_大图_t_{t}.png'))

    # 分别保存子图
    # 保存子图1
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(range(num_wind_turbines), optimal_power, color=cmap1(np.linspace(0, 1, num_wind_turbines)))
    ax.set_ylim([0, 5])
    ax.set_title(f'风电场 {f + 1} 优化后的功率分配 at t = {t}s')
    ax.set_xlabel('风机编号')
    ax.set_ylabel('功率 (MW)')
    fig.savefig(os.path.join(wind_farm_path, f'风电场_{f + 1}_子图1_t_{t}.png'))

    # 保存子图2
    fig, ax = plt.subplots(figsize=(8, 6))
    ax2.barh(['优化功率', '参考功率'], [current_damage_ref, current_damage_optimal],
             color=[cmap2(0.8), cmap2(0.2)], edgecolor='black', linewidth=1.5)
    ax.set_title('累积疲劳损伤对比')
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('累积疲劳损伤')
    ax.legend()
    fig.savefig(os.path.join(wind_farm_path, f'风电场_{f + 1}_子图2_t_{t}.png'))

    # 保存子图3
    fig, ax = plt.subplots(figsize=(16, 6))  # 调整画布大小，适应长图
    ax.plot(range(num_wind_turbines), Pref[:, t] / 1000000, color=cmap3(0.7), linestyle='--', label=f'参考功率 at t={t}s')
    ax.plot(range(num_wind_turbines), optimal_power, color=cmap3(0.9), label=f'优化功率 at t={t}s')
    ax.set_title('功率分配对比', fontsize=16)
    ax.set_xlabel('风机编号', fontsize=14)
    ax.set_ylabel('功率 (MW)', fontsize=14)
    ax.legend(loc='upper right')
    fig.savefig(os.path.join(wind_farm_path, f'风电场_{f + 1}_子图3_t_{t}.png'))

    plt.close('all')

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


def calculate_current_damage(optimal_power, Pref_mw, V):
    # 计算当前时刻的优化功率疲劳损伤
    current_damage_optimal = calculate_total_damage(optimal_power, V)
    # 计算当前时刻的参考功率疲劳损伤
    current_damage_ref = calculate_total_damage(Pref_mw, V)

    return current_damage_optimal, current_damage_ref


# 累积疲劳损伤计算函数，分别计算参考功率和优化功率的损伤
def calculate_cumulative_damage(optimal_power, Pref_mw, V, previous_damage_optimal, previous_damage_ref):
    # 计算当前时刻的优化功率疲劳损伤
    current_damage_optimal = calculate_total_damage(optimal_power, V)
    # 计算当前时刻的参考功率疲劳损伤（以MW为单位）
    current_damage_ref = calculate_total_damage(Pref_mw, V)

    # 累积疲劳损伤
    total_damage_optimal = previous_damage_optimal + current_damage_optimal
    total_damage_ref = previous_damage_ref + current_damage_ref

    return total_damage_optimal, total_damage_ref, current_damage_optimal, current_damage_ref


# 目标函数：最小化主轴+塔架 总疲劳损伤
def objective(power, Pref, V):
    total_damage = calculate_total_damage(power, V)
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


# 功率约束条件
def power_constraints(power, total_pref):
    return np.sum(power) - total_pref


# 主函数
def solve_question_3():
    filepath = 'all_data_2.npy'
    data = np.load(filepath)  # 加载数据 (形状: [2, 100, 2000, 9])

    num_wind_farms = data.shape[0]
    num_wind_turbines = data.shape[1]
    time_steps = data.shape[2]

    # 初始化参数
    cumulative_damage_optimal = np.zeros((num_wind_farms, time_steps))
    cumulative_damage_ref = np.zeros((num_wind_farms, time_steps))
    optimal_power_history = np.zeros((num_wind_farms, num_wind_turbines, time_steps))

    for f in range(num_wind_farms):
        Pref = data[f, :, :, 1]  # 功率参考数据
        V = data[f, :, :, 2]  # 风速数据

        previous_damage_optimal = 0
        previous_damage_ref = 0

        for t in range(time_steps):
            # 转换参考功率单位到MW
            Pref_mw = Pref[:, t] / 1000000  # 假设原单位为瓦特，转换为兆瓦

            initial_power = np.maximum(Pref[:, t] / np.max(Pref[:, t]) * 5, 0)
            optimal_power = optimize_power_distribution(Pref[:, t], V[:, t], initial_power)
            optimal_power_history[f, :, t] = optimal_power

            # 计算并累积疲劳损伤，确保使用优化功率和转换后的参考功率
            cumulative_damage_optimal[f, t], cumulative_damage_ref[
                f, t], current_damage_optimal, current_damage_ref = calculate_cumulative_damage(
                optimal_power, Pref_mw, V[:, t], previous_damage_optimal, previous_damage_ref)

            # 更新前一时刻的累积损伤
            previous_damage_optimal = cumulative_damage_optimal[f, t]
            previous_damage_ref = cumulative_damage_ref[f, t]

            # 打印每一时刻的疲劳损伤和累积疲劳损伤
            print(f"时间 {t + 1} 秒:")
            print(f"  当前参考功率的疲劳损伤: {current_damage_ref:.4f}")
            print(f"  当前优化功率的疲劳损伤: {current_damage_optimal:.4f}")
            print(f"  累积参考功率的疲劳损伤: {cumulative_damage_ref[f, t]:.4f}")
            print(f"  累积优化功率的疲劳损伤: {cumulative_damage_optimal[f, t]:.4f}")
            print('-' * 50)

            # if t == 100:
            #     save_images(f, t, optimal_power, Pref, current_damage_optimal, current_damage_ref,
            #                     num_wind_turbines)
            # if t == 400:
            #         save_images(f, t, optimal_power, Pref, current_damage_optimal, current_damage_ref,
            #                     num_wind_turbines)
            # if t == 1000:
            #             save_images(f, t, optimal_power, Pref, current_damage_optimal, current_damage_ref,
            #                         num_wind_turbines)
            # if t == 1800:
            #                 save_images(f, t, optimal_power, Pref, current_damage_optimal, current_damage_ref,
            #                             num_wind_turbines)
            # 动态可视化
            if  t > 0:
                plt.clf()

                fig = plt.figure(figsize=(16, 8))
                gs = fig.add_gridspec(2, 2, width_ratios=[3, 2], height_ratios=[2, 1], hspace=0.3)

                # 根据风电场选择不同的颜色映射
                if f == 0:
                    cmap1 = get_cmap('coolwarm')
                    cmap2 = get_cmap('viridis')
                    cmap3 = get_cmap('plasma')
                else:
                    cmap1 = get_cmap('spring')
                    cmap2 = get_cmap('winter')
                    cmap3 = get_cmap('autumn')

                # 第一张子图，显示优化后的功率分配，使用不同的颜色渐变
                ax1 = fig.add_subplot(gs[0, 0])
                bars = ax1.bar(range(num_wind_turbines), optimal_power,
                               color=cmap1(np.linspace(0, 1, num_wind_turbines)),
                               edgecolor='black', width=0.8)
                ax1.axhline(y=5, color='red', linestyle='--', linewidth=2, label='功率上限')
                ax1.set_ylim([2, 6])
                ax1.set_title(f'风电场 {f + 1} 优化后的功率分配 at t = {t}s', fontsize=14)
                ax1.set_xlabel('风机编号', fontsize=12)
                ax1.set_ylabel('功率 (MW)', fontsize=12)
                # 第二张子图，展示当前时刻的累积疲劳损伤，而不是累积之前的

                # 第二张子图，展示当前时刻的累积疲劳损伤，而不是累积之前的
                # ax2 = fig.add_subplot(gs[0, 1])
                #
                # # 调用新建的函数来计算当前时刻的疲劳损伤
                # current_damage_optimal, current_damage_ref = calculate_current_damage(optimal_power, Pref_mw, V[:, t])
                #
                # # 绘制当前秒内的损伤
                # ax2.bar(['参考功率', '优化功率'], [current_damage_optimal, current_damage_ref],
                #         color=[cmap2(0.8), cmap2(0.2)])
                # ax2.set_title('当前时刻的累积疲劳损伤', fontsize=14)
                # ax2.set_ylabel('当前疲劳损伤', fontsize=12)
                ax2 = fig.add_subplot(gs[0, 1])

                # 绘制当前秒内的损伤的水平条形图
                ax2.barh(['优化功率', '参考功率'], [current_damage_ref, current_damage_optimal],
                         color=[cmap2(0.8), cmap2(0.2)], edgecolor='black', linewidth=1.5)

                ax2.set_title('当前时刻的累积疲劳损伤', fontsize=14)
                ax2.set_xlabel('当前疲劳损伤', fontsize=12)
                ax2.grid(True, linestyle='--', alpha=0.6)

                # ax2 = fig.add_subplot(gs[0, 1])
                # ax2.plot(range(1, t + 1), cumulative_damage_optimal[f, :t], color=cmap2(0.8), label='参考功率累积损伤',
                #          linewidth=3)
                # ax2.plot(range(1, t + 1), cumulative_damage_ref[f, :t], color=cmap2(0.2), label='优化功率累积损伤',
                #          linewidth=3, linestyle='--')
                # ax2.set_title('累积疲劳损伤对比', fontsize=14)
                # ax2.set_xlabel('时间 (s)', fontsize=12)
                # ax2.set_ylabel('累积疲劳损伤', fontsize=12)

                ax2.legend()

                # 第三张子图，功率分配对比，使用不同的颜色
                ax3 = fig.add_subplot(gs[1, :])
                ax3.plot(range(num_wind_turbines), optimal_power, color=cmap3(0.7), linestyle='--',
                         label=f'参考功率 at t={t}s', linewidth=3)
                ax3.plot(range(num_wind_turbines), Pref[:, t] / 1000000, color=cmap3(0.2), label=f'优化功率 at t={t}s',
                         linewidth=3)
                ax3.set_title('功率分配对比', fontsize=14)
                ax3.set_xlabel('风机编号', fontsize=12)
                ax3.set_ylabel('功率 (MW)', fontsize=12)
                ax3.legend(loc='upper right')

                plt.tight_layout()
                plt.pause(10)

    print('最终优化功率分配:')
    print(optimal_power_history)
    print('累积疲劳损伤总和（优化功率）:')
    print(np.sum(cumulative_damage_optimal))
    print('累积疲劳损伤总和（参考功率）:')
    print(np.sum(cumulative_damage_ref))


# 执行函数
solve_question_3()
