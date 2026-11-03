import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.linear_model import LinearRegression

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

# 加载数据
data = np.load('all_data_3.npy')

# 提取时间序列、风速、功率
time_series = data[:, :, :, 0]  # 时间序列
power_P = data[:, :, :, 1]  # 功率
wind_speed_V = data[:, :, :, 2]  # 风速

# 假设我们选择第一个风场中的第一台风机进行预测和优化
time_data = time_series[0, 0, :]
wind_speed_data = wind_speed_V[0, 0, :]
power_data = power_P[0, 0, :]


# 使用灰色模型GM(1,1)进行风速和功率预测
def gm11_model(data, predict_length):
    # 累加生成（AGO）
    cumulative_data = np.cumsum(data)
    # 创建背景值
    z = 0.5 * (cumulative_data[:-1] + cumulative_data[1:])
    # 构建方程
    B = np.vstack([-z, np.ones(len(z))]).T
    Y = data[1:]
    # 最小二乘法估计a和b
    lr = LinearRegression().fit(B, Y)
    a, b = lr.coef_[0], lr.intercept_

    # 预测函数
    def predict(n):
        return (data[0] - b / a) * np.exp(-a * n) + b / a

    # 生成预测结果，包括未来 predict_length 个时间点的预测
    predictions = [predict(i) for i in range(len(data) + predict_length)]
    return np.array(predictions)


# 使用GM(1,1)模型预测未来200秒的风速和功率
predict_length = 200
predicted_wind_speed = gm11_model(wind_speed_data, predict_length)
predicted_power = gm11_model(power_data, predict_length)

# 构建新的时间序列，包括未来200秒
extended_time_data = np.arange(len(time_data) + predict_length)

# 可视化预测结果
plt.figure(figsize=(12, 6))

# 风速预测结果可视化
plt.subplot(1, 2, 1)
plt.plot(time_data, wind_speed_data, label='实际风速', color='blue')
plt.plot(extended_time_data, predicted_wind_speed, label='预测风速', linestyle='--', color='red')
plt.xlabel('时间 (秒)')
plt.ylabel('风速 (m/s)')
plt.title('风速预测')
plt.legend()
plt.grid(True)

# 功率预测结果可视化
plt.subplot(1, 2, 2)
plt.plot(time_data, power_data, label='实际功率', color='blue')
plt.plot(extended_time_data, predicted_power, label='预测功率', linestyle='--', color='red')
plt.xlabel('时间 (秒)')
plt.ylabel('功率 (MW)')
plt.title('功率预测')
plt.legend()
plt.grid(True)

# 保存图像
plt.tight_layout()
plt.savefig('风速与功率预测结果_未来200秒.png')
plt.show()
