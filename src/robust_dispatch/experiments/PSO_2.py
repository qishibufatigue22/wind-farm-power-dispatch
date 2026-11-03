import numpy as np


# 主轴 ------>   疲劳损伤计算函数
def calculate_fatigue_damage_shaft(P_ref, V):
    damage = (P_ref ** 2) / (V + 1)
    return damage


# 塔架  ------>   疲劳损伤计算函数
def calculate_fatigue_damage_tower(P_ref, V):
    damage = (P_ref ** 2) / (V + 2)
    return damage


# 目标函数：最小化主轴+塔架 总疲劳损伤，权重各为0.5
def objective(P_ref, V):
    shaft_damage = calculate_fatigue_damage_shaft(P_ref, V)
    tower_damage = calculate_fatigue_damage_tower(P_ref, V)

    # 计算主轴和塔架疲劳损伤的平均值
    avg_shaft_damage = np.mean(shaft_damage)
    avg_tower_damage = np.mean(tower_damage)

    # 目标函数：加权平均，权重0.5
    total_damage = 0.5 * avg_shaft_damage + 0.5 * avg_tower_damage
    return total_damage


# 约束条件：(1)总功率等于调度指令、(2)功率不超过额定值、(3)与平均分配差不超过1MW
def power_constraints(P_ref, P_t, avg_power):
    constraints = [
        {'type': 'eq', 'fun': lambda P_ref: np.sum(P_ref) - P_t},  # 总功率等于电网调度指令
        {'type': 'ineq', 'fun': lambda P_ref: rated_power - P_ref},  # 每台风机的功率不超过5MW
        {'type': 'ineq', 'fun': lambda P_ref: 1 - np.abs(P_ref - avg_power)}  # 与平均分配差不超过1MW
    ]
    return constraints


# 优化功率分配  (目标+约束)
def optimize_power_distribution(P_t, P_ref_initial, V, avg_power):
    result = pso(num_particles, dim, bounds, num_iterations, w=0.5, c1=2, c2=2)
    return result.x


# 粒子群算法的实现
def pso(num_particles, dim, bounds, num_iterations, w=0.5, c1=2, c2=2):
    # 初始化粒子的位置和速度
    particles_position = np.random.uniform(bounds[0], bounds[1], (num_particles, dim))  # 粒子初始位置
    particles_velocity = np.random.uniform(-1, 1, (num_particles, dim))  # 粒子初始速度
    personal_best_position = np.copy(particles_position)  # 记录粒子各自的最优位置
    personal_best_value = np.array([objective_function(p) for p in personal_best_position])  # 最优位置对应的目标函数值

    # 初始化全局最优位置
    global_best_index = np.argmin(personal_best_value)
    global_best_position = personal_best_position[global_best_index]
    global_best_value = personal_best_value[global_best_index]

    # PSO 迭代过程
    for t in range(num_iterations):
        for i in range(num_particles):
            # 速度更新
            r1, r2 = np.random.rand(), np.random.rand()  # 随机因子
            particles_velocity[i] = (w * particles_velocity[i] +
                                     c1 * r1 * (personal_best_position[i] - particles_position[i]) +
                                     c2 * r2 * (global_best_position - particles_position[i]))

            # 位置更新
            particles_position[i] += particles_velocity[i]

            # 边界条件约束
            particles_position[i] = np.clip(particles_position[i], bounds[0], bounds[1])

            # 计算新的目标函数值
            new_value = objective_function(particles_position[i])

            # 更新个体最优位置
            if new_value < personal_best_value[i]:
                personal_best_value[i] = new_value
                personal_best_position[i] = particles_position[i]

        # 更新全局最优位置
        global_best_index = np.argmin(personal_best_value)
        global_best_position = personal_best_position[global_best_index]
        global_best_value = personal_best_value[global_best_index]

        # 输出当前迭代的全局最优值
        print(f"Iteration {t + 1}/{num_iterations}, Global Best Value: {global_best_value}")

    return global_best_position, global_best_value


# 运行 PSO 算法
num_particles = 30  # 粒子数量
dim = 2  # 搜索空间维度（如有两个参数待优化）
bounds = [-10, 10]  # 搜索空间的边界
num_iterations = 100  # 最大迭代次数

best_position, best_value = pso(num_particles, dim, bounds, num_iterations)

print(f"最优位置: {best_position}")
print(f"最优值: {best_value}")
