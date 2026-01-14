"""
使用 Ax (TuRBO) 对 MPCC 进行自动调参
用法: python scripts/tune_mpcc.py
"""
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from scripts.sim_4_tune import simulate 
from ax.service.ax_client import AxClient, ObjectiveProperties
import numpy as np

# 1. 定义评估函数 (Black-box function)
def evaluation_function(parameters):
    print(f"\n[AutoTune] Testing params: {parameters}")
    
    # 调用仿真 (传入当前的参数)
    result = simulate(
        config="level2.toml", 
        controller="mpcc_rotor_tune.py",  # <--- 已修改：指向您的新控制器文件
        n_runs=10,      # 每次评估跑4轮取平均，减少随机性
        render=False,  # 关闭渲染加速
        tuning_params=parameters
    )
    
    avg_time = result["avg_time"]
    success_rate = result["success_rate"]
    fail_rate = 1.0 - success_rate
    
    

    
    if success_rate < 0.5:
        return {"reward": (-500.0 - 100.0 * fail_rate, 0.0)}

    # 2. 有效时间处理 (防止全炸时 avg_time=999 导致数值异常)
    # 如果没飞完，按 20秒计算 (很慢)
    effective_time = avg_time if success_rate > 0 else 20.0

    # 3. 速度得分 (Linear Speed Bonus)
    # 设定一个慢速基准 (比如 8.0秒)
    # 每比 8.0秒 快 1秒，得 2.0 分
    # 这里的系数 2.0 代表了您对速度的渴望程度
    baseline_time = 8.0
    speed_score = (baseline_time - effective_time) * 5
    
    # 4. 风险扣分 (Quadratic Risk Penalty)
    # 使用平方项 (fail_rate^2)
    # 系数设为 20.0 (这个值经过精心设计，见下文分析)
    risk_penalty = 10.0 * (fail_rate ** 2)
    
    # 总分
    reward = speed_score - risk_penalty
    
   
        
    print(f"[AutoTune] Result: Time={avg_time:.2f}s, Success={success_rate*100:.0f}%, Reward={reward:.2f}")
    
    # Ax 需要返回 {metric_name: (mean, standard_error)}
    return {"reward": (reward, 0.0)}

# 2. 初始化 Ax Client
ax_client = AxClient(verbose_logging=False)

# 3. 定义搜索空间 (根据您要求的参数设定范围)
ax_client.create_experiment(
    name="tune_mpcc_v1",
    parameters=[
        # --- Horizon & Steps ---
        # N 必须是整数，范围给得稍大一点让它探索
        #{"name": "N", "type": "range", "bounds": [35,50], "value_type": "int"}, 
        # 预测时间长度 [s]
        #{"name": "T_HORIZON", "type": "range", "bounds": [0.5, 1.0]},
        
        # --- 轨迹追踪权重 (基础) ---
        {"name": "q_l", "type": "range", "bounds": [10.0, 250.0]}, # Lag error
        {"name": "q_c", "type": "range", "bounds": [100.0, 300.0]}, # Contour error
        
        # --- 门附近权重增强 ---
        # 通常需要比基础权重高很多，以确保穿门精度
        {"name": "q_l_gate_peak", "type": "range", "bounds": [500.0, 1000.0]},
        {"name": "q_c_gate_peak", "type": "range", "bounds": [500.0, 1000.0]},

        # --- 障碍物附近权重增强 ---
        {"name": "q_l_obst_peak", "type": "range", "bounds": [50.0, 300.0]},
        {"name": "q_c_obst_peak", "type": "range", "bounds": [50.0, 300.0]},

        # --- 速度权重 (MiU) ---
        # 决定了飞行的激进程度
        {"name": "miu", "type": "range", "bounds": [5.0, 15.0]},   
        
        # --- 减速权重 ---
        # 遇到门或障碍物时减速的倾向
        {"name": "w_v_gate", "type": "range", "bounds": [0.1, 5.0]},
        {"name": "w_v_obst", "type": "range", "bounds": [0.1, 2.0]},
    ],
    # 目标：最大化 reward
    objectives={"reward": ObjectiveProperties(minimize=False)},
)

# 4. 运行优化循环
TOTAL_TRIALS = 50  # 建议 50-100 次以获得较好结果

print(f"Starting optimization for {TOTAL_TRIALS} trials...")

for i in range(TOTAL_TRIALS):
    print(f"--- Trial {i+1}/{TOTAL_TRIALS} ---")
    
    # 获取下一组推荐参数
    parameters, trial_index = ax_client.get_next_trial()
    
    # 运行评估
    raw_data = evaluation_function(parameters)
    
    # 报告结果给 Ax
    ax_client.complete_trial(trial_index=trial_index, raw_data=raw_data)

# 5. 输出最佳结果
best_parameters, values = ax_client.get_best_parameters()
print("             TUNING COMPLETED                ")
print(f"Best Parameters: {best_parameters}")
print(f"Best Reward: {values}")
