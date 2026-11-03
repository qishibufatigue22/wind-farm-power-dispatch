function solve_question_2()
    % 数据文件路径
    filepath = '附件2-风电机组采集数据.mat';
    data = load(filepath);  % 加载附件2的MAT文件
    
    % 初始化变量
    num_wind_turbines = 100;  % 风电场每个风场有100台风机
    time_steps = 2000;  % 每台风机数据2000秒
    WF_fields = fieldnames(data.data_TS_WF);  % 获取风电场名称（WF1, WF2）
    
    % 初始化存储结构
    all_torque_estimates = zeros(time_steps, num_wind_turbines);
    all_thrust_estimates = zeros(time_steps, num_wind_turbines);
    all_torque_error_sum = zeros(num_wind_turbines, 1);  % 扭矩差值平方和
    all_thrust_error_sum = zeros(num_wind_turbines, 1);  % 推力差值平方和
    
    % 遍历每个风电场
    for f = 1:length(WF_fields)
        wind_farm = data.data_TS_WF.(WF_fields{f}).WT;
        
        % 遍历每台风机
        for wt = 1:num_wind_turbines
            % 读取当前风机的相关数据
            wind_turbine = wind_farm{wt};  % 当前风机数据
            inputs = wind_turbine.inputs;  % 功率调度指令（Pref）和风速(V)
            outputs = wind_turbine.outputs;  % 主轴扭矩(Tshaft)，塔架推力(Ft)，实际输出功率(Pout)
            states = wind_turbine.states;  % 状态变量（桨距角、轴转速等）
            
            % 提取输入和输出数据
            Pref = inputs(:, 1);  % 功率调度指令
            V = inputs(:, 2);  % 风速 (m/s)
            Tshaft_ref = outputs(:, 1);  % 参考主轴扭矩
            Ft_ref = outputs(:, 2);  % 参考塔架推力
            
            % 初始化估算值
            Tshaft_est = zeros(time_steps, 1);  % 估算的主轴扭矩
            Ft_est = zeros(time_steps, 1);  % 估算的塔架推力
            
            % 遍历每个时间点进行估算
            for t = 1:time_steps
                % 状态变量
                pitch_angle = states(t, 1);  % 桨距角 (度)
                omega_r_rpm = states(t, 2);  % 低速轴转速 (rpm)
                
                % 估算主轴扭矩和塔架推力
                [Tshaft_est(t), Ft_est(t)] = estimate_torque_thrust(V(t), pitch_angle, omega_r_rpm);
            end
            
            % 计算扭矩和推力的差值平方和
            torque_error_squared = (Tshaft_est - Tshaft_ref).^2;  % 扭矩差值的平方
            thrust_error_squared = (Ft_est - Ft_ref).^2;  % 推力差值的平方
            
            % 计算差值平方和（所有时刻的差值平方之和）
            all_torque_error_sum(wt) = sum(torque_error_squared);  % 扭矩差值平方和
            all_thrust_error_sum(wt) = sum(thrust_error_squared);  % 推力差值平方和
            
            % 将估算值存入矩阵
            all_torque_estimates(:, wt) = Tshaft_est;
            all_thrust_estimates(:, wt) = Ft_est;
        end
    end
    
    % 输出计算结果到 Excel 文件
    output_filename = '附件6-问题二答案表.xlsx';
    output_results_to_excel(all_torque_estimates, all_thrust_estimates, all_torque_error_sum, all_thrust_error_sum, output_filename);
    
    % 可视化展示差值平方和
    plot_error_sums(all_torque_error_sum, all_thrust_error_sum);
end

% 估算主轴扭矩和塔架推力的辅助函数
function [Tshaft, Ft] = estimate_torque_thrust(V, pitch_angle, omega_r_rpm)
    % 定义常量
    air_density = 1.225;  % 空气密度，单位 kg/m^3
    rotor_radius = 40;  % 风机叶片半径，单位 m
    swept_area = pi * rotor_radius^2;  % 叶片扫过面积，单位 m^2
    
    % 将转速从 rpm 转换为 rad/s
    omega_r = omega_r_rpm * (2 * pi / 60);  % 单位 rad/s
    
    % 计算叶尖速比 lambda
    lambda = (omega_r * rotor_radius) / V;
    
    % 确保 lambda 和 pitch_angle 的值有效
    if lambda <= 0 || isnan(lambda) || isinf(lambda)
        lambda = 0.001;  % 避免除以零或出现无效值
    end
    if isnan(pitch_angle)
        pitch_angle = 0;
    end
    
    % 计算功率系数 Cp 和推力系数 Ct
    Cp = calculate_Cp(lambda, pitch_angle);
    Ct = calculate_Ct(lambda, pitch_angle);
    
    % 计算转子功率
    Protor = 0.5 * air_density * swept_area * V^3 * Cp;  % 单位 W
    
    % 计算主轴扭矩
    Tshaft = Protor / omega_r;  % 单位 Nm
    
    % 计算塔架推力
    Ft = 0.5 * air_density * swept_area * V^2 * Ct;  % 单位 N
end

% 计算功率系数 Cp 的函数
function Cp = calculate_Cp(lambda, beta)
    % 使用常用的经验公式计算 Cp
    beta = max(beta, 0);  % 桨距角不能为负
    lambda_i = 1 / (1 / (lambda + 0.08 * beta) - 0.035 / (beta^3 + 1));
    Cp = 0.22 * (116 / lambda_i - 0.4 * beta - 5) * exp(-12.5 / lambda_i);
    Cp = max(Cp, 0);  % Cp 不能为负
end

% 计算推力系数 Ct 的函数
function Ct = calculate_Ct(lambda, beta)
    % 假设 Ct 与 Cp 有一定关系
    Ct = 0.5 * calculate_Cp(lambda, beta);  % 简化假设
    Ct = max(Ct, 0);  % Ct 不能为负
end

% 输出计算结果到 Excel 文件的辅助函数
function output_results_to_excel(torque_estimates, thrust_estimates, torque_error_sum, thrust_error_sum, output_filename)
    % 创建Excel文件表格格式
    WT_headers = cell(1, size(torque_estimates, 2));
    for i = 1:length(WT_headers)
        WT_headers{i} = ['WT', num2str(i)];
    end
    time_col = (1:size(torque_estimates, 1))';
    
    % 创建表格
    torque_table = array2table(torque_estimates, 'VariableNames', WT_headers);
    thrust_table = array2table(thrust_estimates, 'VariableNames', WT_headers);
    errors_table = array2table([torque_error_sum, thrust_error_sum], 'VariableNames', {'Torque_Error_Sum', 'Thrust_Error_Sum'});
    
    % 加入时间列
    torque_table = addvars(torque_table, time_col, 'Before', 1, 'NewVariableNames', 'Time');
    thrust_table = addvars(thrust_table, time_col, 'Before', 1, 'NewVariableNames', 'Time');
    
    % 写入Excel文件
    writetable(torque_table, output_filename, 'Sheet', '主轴扭矩估算值');
    writetable(thrust_table, output_filename, 'Sheet', '塔架推力估算值');
    writetable(errors_table, output_filename, 'Sheet', '误差统计');
end

% 可视化误差平方和
function plot_error_sums(torque_error_sum, thrust_error_sum)
    figure;
    
    % 绘制主轴扭矩误差平方和
    subplot(2, 1, 1);
    bar(torque_error_sum);
    xlabel('Wind Turbine Index');
    ylabel('Torque Error Sum (Nm^2)');
    title('主轴扭矩差值平方和');
    
    % 绘制塔架推力误差平方和
    subplot(2, 1, 2);
    bar(thrust_error_sum);
    xlabel('Wind Turbine Index');
    ylabel('Thrust Error Sum (N^2)');
    title('塔架推力差值平方和');
end
