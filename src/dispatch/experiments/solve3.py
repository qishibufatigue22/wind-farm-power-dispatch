import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体显示中文
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 神经网络模型
class PowerOptimizationNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PowerOptimizationNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# 计算总疲劳损伤
def calculate_total_damage(P_ref, V):
    shaft_damage = (P_ref ** 2) / (V + 1)
    tower_damage = (P_ref ** 2) / (V + 2)
    total_damage = 0.5 * np.sum(shaft_damage) + 0.5 * np.sum(tower_damage)
    return total_damage


# 遗传算法的适应度函数
def evaluate(individual, Pref, V):
    P_opt = np.array(individual)
    total_damage = calculate_total_damage(P_opt, V)
    return (total_damage,)


# 遗传算法优化
def genetic_algorithm_optimization(Pref, V, num_generations=100):
    num_turbines = Pref.shape[0]

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", np.random.uniform, 0, 5)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, num_turbines)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate, Pref=Pref, V=V)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=50)
    algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=num_generations, verbose=False)

    best_individual = tools.selBest(population, k=1)[0]
    return np.array(best_individual)


# 训练神经网络：同时使用参考功率和遗传算法优化结果
def train_nn_with_ga(Pref, V, optimized_power_ga, epochs=1000, lr=0.01, ref_weight=0.8, ga_weight=0.2):
    num_turbines = Pref.shape[0]
    input_size = 2  # 输入是风速和参考功率
    hidden_size = 64
    output_size = 1  # 输出是每个风机的优化功率

    # 构建神经网络模型
    model = PowerOptimizationNN(input_size, hidden_size, output_size)
    criterion = nn.MSELoss()  # 使用均方误差损失函数
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 训练神经网络
    for epoch in range(epochs):
        total_loss = 0
        for i in range(num_turbines):
            optimizer.zero_grad()

            # 输入：风速和参考功率
            inputs = torch.tensor([[Pref[i, 0], V[i, 0]]], dtype=torch.float32)
            output = model(inputs)

            # 损失函数：使用参考功率和遗传算法优化结果，按加权方式结合
            target_ref = torch.tensor([Pref[i, 0]], dtype=torch.float32)
            target_ga = torch.tensor([optimized_power_ga[i]], dtype=torch.float32)

            # 加权损失：参考功率与遗传算法结果的权重调整
            loss_ref = criterion(output, target_ref)
            loss_ga = criterion(output, target_ga)
            loss = ref_weight * loss_ref + ga_weight * loss_ga

            loss.backward()  # 反向传播
            optimizer.step()  # 更新模型参数

            total_loss += loss.item()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {total_loss:.4f}')

    return model


# 测试神经网络
def test_nn(model, Pref, V):
    num_turbines = Pref.shape[0]
    optimized_power = np.zeros(num_turbines)

    # 对每个风机进行预测
    for i in range(num_turbines):
        inputs = torch.tensor([[Pref[i, 0], V[i, 0]]], dtype=torch.float32)
        with torch.no_grad():
            optimized_power[i] = model(inputs).item()

    return optimized_power


# 主函数
# 主函数
def solve_question_3():
    filepath = 'all_data_2.npy'
    data = np.load(filepath)  # 加载数据 (形状: [2, 100, 2000, 9])

    num_wind_farms = data.shape[0]
    num_wind_turbines = data.shape[1]
    time_steps = data.shape[2]

    for f in range(num_wind_farms):
        Pref = data[f, :, :, 1]  # 功率参考数据
        V = data[f, :, :, 2]  # 风速数据

        # 使用遗传算法进行优化
        optimized_power_ga = genetic_algorithm_optimization(Pref[:, :100], V[:, :100])

        # 使用参考功率和遗传算法结果训练神经网络
        model = train_nn_with_ga(Pref[:, :100], V[:, :100], optimized_power_ga, epochs=1000, lr=0.01, ref_weight=0.5, ga_weight=0.15)

        # 累积损伤数据存储
        cumulative_damage_optimal = []
        cumulative_damage_reference = []

        # 动态可视化
        for t in range(100):  # 每一时刻更新
            plt.clf()

            # 使用神经网络预测优化功率分配
            optimized_power = test_nn(model, Pref[:, t:t + 1], V[:, t:t + 1])  # 测试当前时间步

            # 计算累积疲劳损伤（参考功率和优化功率）
            cumulative_damage_ref = calculate_total_damage(Pref[:, t], V[:, t])
            cumulative_damage_opt = calculate_total_damage(optimized_power, V[:, t])

            # 将当前时刻的损伤累积到列表中
            if t == 0:
                cumulative_damage_optimal.append(cumulative_damage_opt)
                cumulative_damage_reference.append(cumulative_damage_ref)
            else:
                cumulative_damage_optimal.append(cumulative_damage_optimal[-1] + cumulative_damage_opt)
                cumulative_damage_reference.append(cumulative_damage_reference[-1] + cumulative_damage_ref)

            # 绘图
            fig = plt.figure(figsize=(16, 8))
            gs = fig.add_gridspec(2, 2, width_ratios=[3, 2], height_ratios=[2, 1], hspace=0.3)

            # 第一张子图，显示优化后的功率分配
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.bar(range(num_wind_turbines), optimized_power, color='b')
            ax1.set_ylim([0, 5])
            ax1.set_title(f'风电场 {f + 1} 优化后的功率分配 at t = {t}s')
            ax1.set_xlabel('风机编号')
            ax1.set_ylabel('功率 (MW)')

            # 第二张子图，累积疲劳损伤对比
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.plot(range(t + 1), cumulative_damage_optimal, 'b', label='优化功率累积损伤', linewidth=2)
            ax2.plot(range(t + 1), cumulative_damage_reference, 'r', label='参考功率累积损伤', linewidth=2, linestyle='--')
            ax2.set_title('累积疲劳损伤对比')
            ax2.set_xlabel('时间 (s)')
            ax2.set_ylabel('累积疲劳损伤')
            ax2.legend()

            # 第三张子图，功率分配对比
            ax3 = fig.add_subplot(gs[1, :])
            ax3.plot(range(num_wind_turbines), Pref[:, t] , 'r--', label=f'参考功率 at t={t}s')
            ax3.plot(range(num_wind_turbines), optimized_power, 'b-', label=f'优化功率 at t={t}s')
            ax3.set_title('功率分配对比')
            ax3.set_xlabel('风机编号')
            ax3.set_ylabel('功率 (MW)')
            ax3.legend(loc='upper right')

            plt.tight_layout()
            plt.pause(100)  # 控制动画速度

    plt.show()


# 执行函数
solve_question_3()
