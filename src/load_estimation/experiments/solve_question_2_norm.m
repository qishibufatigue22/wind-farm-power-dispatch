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
    all_torque_errors = zeros(num_wind_turbines, 1);
    all_thrust_errors = zeros(num_wind_turbines, 1);
    
    % 遍历每个风电场
    for f = 1:length(WF_fields)
        wind_farm = data.data_TS_WF.(WF_fields{f}).WT;
        
        % 遍历每台风机
        for wt = 1:num_wind_turbines
            % 读取当前风机的相关数据
            wind_turbine = wind_farm{wt};  % 当前风机数据
            time = wind_turbine.time;  % 时间数据
            inputs = wind_turbine.inputs;  % 功率调度指令（Pref）和风速(V)
            outputs = wind_turbine.outputs;  % 主轴扭矩(Tshaft)，塔架推力(Ft)，实际输出功率(Pout)
            states = wind_turbine.states;  % 状态变量（桨距角、轴转速等）
            
            % 提取输入和输出数据
            Pref = inputs(:, 1);  % 功率调度指令
            V = inputs(:, 2);  % 风速
            Tshaft_ref = outputs(:, 1);  % 参考主轴扭矩
            Ft_ref = outputs(:, 2);  % 参考塔架推力
            
            % 初始化估算值
            Tshaft_est = zeros(time_steps, 1);  % 估算的主轴扭矩
            Ft_est = zeros(time_steps, 1);  % 估算的塔架推力
            
            % 遍历每个时间点进行估算
            for t = 1:time_steps
                % 桨距角和风速对扭矩和推力的影响模型（非线性模型）
                pitch_angle = states(t, 1);  % 桨距角
                omega_r = states(t, 2);  % 低速轴转速
                Pout = outputs(t, 3);  % 实际输出功率
                
                % 更新的非线性扭矩估算模型
                Tshaft_est(t) = calculate_torque(V(t), pitch_angle, omega_r, Pref(t));
                
                % 更新的非线性推力估算模型，加入更多影响因素
                Ft_est(t) = calculate_thrust(V(t), pitch_angle, omega_r, Pout);
            end
            
            % 将估算值存入矩阵
            all_torque_estimates(:, wt) = Tshaft_est;
            all_thrust_estimates(:, wt) = Ft_est;
            
            % 计算估算值与参照值的差的平方和
            all_torque_errors(wt) = sum((Tshaft_est - Tshaft_ref).^2);
            all_thrust_errors(wt) = sum((Ft_est - Ft_ref).^2);
        end
    end
    
    % 输出计算结果到 Excel 文件
    output_filename = '附件6-问题二答案表.xlsx';
    output_results_to_excel(all_torque_estimates, all_thrust_estimates, all_torque_errors, all_thrust_errors, output_filename);
    
    % 可视化界面
    plot_estimation_vs_reference(all_torque_estimates, all_thrust_estimates, data);
end

% 估算主轴扭矩的辅助函数（非线性）
function Tshaft = calculate_torque(V, pitch_angle, omega_r, Pref)
    % 定义常量
    air_density = 1.225;  % 空气密度，单位kg/m^3
    rotor_radius = 40;  % 风机叶片半径，单位m
    swept_area = pi * rotor_radius^2;  % 叶片扫过面积
    power_coefficient = 0.4;  % 功率系数，取一个典型值

    % 使用非线性公式估算主轴扭矩
    Tshaft = power_coefficient * air_density * swept_area * V^3 * (1 - 0.02 * pitch_angle) * omega_r;
end

% 估算塔架推力的辅助函数（非线性）
function Ft = calculate_thrust(V, pitch_angle, omega_r, Pout)
    % 定义常量
    air_density = 1.225;  % 空气密度，单位kg/m^3
    rotor_radius = 40;  % 风机叶片半径，单位m
    swept_area = pi * rotor_radius^2;  % 叶片扫过面积
    thrust_coefficient = 0.3;  % 推力系数

    % 使用非线性公式估算塔架推力
    Ft = thrust_coefficient * air_density * swept_area * V^2 * (1 - 0.02 * pitch_angle) * (1 + 0.05 * omega_r) * (1 + 0.02 * Pout);
end

% 输出计算结果到 Excel 文件的辅助函数
function output_results_to_excel(torque_estimates, thrust_estimates, torque_errors, thrust_errors, output_filename)
    % 创建Excel文件表格格式
    WT_headers = cell(1, size(torque_estimates, 2));
    for i = 1:length(WT_headers)
        WT_headers{i} = ['WT', num2str(i)];
    end
    time_col = (1:size(torque_estimates, 1))';
    
    % 创建表格
    torque_table = array2table(torque_estimates, 'VariableNames', WT_headers);
    thrust_table = array2table(thrust_estimates, 'VariableNames', WT_headers);
    errors_table = array2table([torque_errors, thrust_errors], 'VariableNames', {'Torque_Error', 'Thrust_Error'});
    
    % 加入时间列
    torque_table = addvars(torque_table, time_col, 'Before', 1, 'NewVariableNames', 'Time');
    thrust_table = addvars(thrust_table, time_col, 'Before', 1, 'NewVariableNames', 'Time');
    
    % 写入Excel文件
    writetable(torque_table, output_filename, 'Sheet', '主轴扭矩估算值');
    writetable(thrust_table, output_filename, 'Sheet', '塔架推力估算值');
    writetable(errors_table, output_filename, 'Sheet', '误差统计');
end

% 可视化估算值与参考值对比的辅助函数
function plot_estimation_vs_reference(torque_estimates, thrust_estimates, data)
    % 随机选择一些风机进行可视化展示
    num_turbines_to_plot = 5;
    sample_turbines = randi(size(torque_estimates, 2), num_turbines_to_plot, 1);
    
    figure;
    for i = 1:num_turbines_to_plot
        % 读取当前风机的参考值
        wind_farm = data.data_TS_WF.WF_1.WT;  % 假设选择WF1风电场中的风机
        Tshaft_ref = wind_farm{sample_turbines(i)}.outputs(:, 1);  % 参考主轴扭矩
        Tshaft_est = torque_estimates(:, sample_turbines(i));      % 估算主轴扭矩
        Ft_ref = wind_farm{sample_turbines(i)}.outputs(:, 2);      % 参考塔架推力
        Ft_est = thrust_estimates(:, sample_turbines(i));          % 估算塔架推力

        % 对扭矩参考值和估算值进行归一化
        Tshaft_ref_norm = (Tshaft_ref - min(Tshaft_ref)) / (max(Tshaft_ref) - min(Tshaft_ref));
        Tshaft_est_norm = (Tshaft_est - min(Tshaft_est)) / (max(Tshaft_est) - min(Tshaft_est));

        % 对推力参考值和估算值进行归一化
        Ft_ref_norm = (Ft_ref - min(Ft_ref)) / (max(Ft_ref) - min(Ft_ref));
        Ft_est_norm = (Ft_est - min(Ft_est)) / (max(Ft_est) - min(Ft_est));
        
        % 创建绘图
        figure;

        % 绘制归一化后的扭矩估算值与参考值
        subplot(3,1,1);  % 第一子图绘制扭矩
        plot(1:size(Tshaft_est_norm, 1), Tshaft_est_norm, 'r', 'DisplayName', '扭矩估算值 (归一化)');
        hold on;
        plot(1:size(Tshaft_ref_norm, 1), Tshaft_ref_norm, 'b', 'DisplayName', '扭矩参考值 (归一化)');
        xlabel('Time (s)');
        ylabel('Normalized Torque');
        title(['WT', num2str(sample_turbines(i)), ' 扭矩估算值与参考值 (归一化)']);
        legend;
        hold off;

        % 绘制归一化后的推力估算值与参考值
        subplot(3,1,2);  % 第二子图绘制推力
        plot(1:size(Ft_est_norm, 1), Ft_est_norm, 'r', 'DisplayName', '推力估算值 (归一化)');
        hold on;
        plot(1:size(Ft_ref_norm, 1), Ft_ref_norm, 'b', 'DisplayName', '推力参考值 (归一化)');
        xlabel('Time (s)');
        ylabel('Normalized Thrust');
        title(['WT', num2str(sample_turbines(i)), ' 推力估算值与参考值 (归一化)']);
        legend;
        hold off;

        % 计算并绘制差平方和
        torque_error_squared = (torque_estimates(:, sample_turbines(i)) - Tshaft_ref).^2;
        thrust_error_squared = (thrust_estimates(:, sample_turbines(i)) - Ft_ref).^2;

        % 绘制扭矩差平方和和推力差平方和
        subplot(3,1,3);
        plot(1:size(torque_error_squared, 1), torque_error_squared, 'g', 'DisplayName', '扭矩差平方');
        hold on;
        plot(1:size(thrust_error_squared, 1), thrust_error_squared, 'm', 'DisplayName', '推力差平方');
        xlabel('Time (s)');
        ylabel('Error Squared');
        title(['WT', num2str(sample_turbines(i)), ' 估算值与参考值差平方']);
        legend;
        hold off;
    end
end
