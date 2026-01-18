"""
Using Ax (TuRBO) for parameter auto tuning
Run: python scripts/tune_mpcc.py
"""
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from scripts.sim_4_tune import simulate 
from ax.service.ax_client import AxClient, ObjectiveProperties
import numpy as np

# Evaluation function for controller performance
def evaluation_function(parameters):
    print(f"\n[AutoTune] Testing params: {parameters}")

    result = simulate(
        config="level2.toml", 
        controller="mpcc_rotor_tune.py",  # controller to be tuned
        n_runs=8,      # runs for each groups of parameters
        render=False,  
        tuning_params=parameters
    )
    
    avg_time = result["avg_time"]
    success_rate = result["success_rate"]
    fail_rate = 1.0 - success_rate
    
    # very strong penalty for success rate below 0.7
    if success_rate < 0.75:
        return {"reward": (-500.0 - 100.0 * fail_rate, 0.0)}

    # If no success at all, set time as 20 seconds
    effective_time = avg_time if success_rate > 0 else 20.0
    # every seconds faster than 8 seconds, get score of 8
    baseline_time = 8.0
    speed_score = (baseline_time - effective_time) * 5
    #Quadratic Risk Penalty
    risk_penalty = 10.0 * (fail_rate ** 2)
    reward = speed_score - risk_penalty
   
    print(f"[AutoTune] Result: Time={avg_time:.2f}s, Success={success_rate*100:.0f}%, Reward={reward:.2f}")
    return {"reward": (reward, 0.0)}

ax_client = AxClient(verbose_logging=False)
ax_client.create_experiment(
    name="tune_mpcc_v1",
    parameters=[
        #{"name": "N", "type": "range", "bounds": [35,50], "value_type": "int"}, 
        #{"name": "T_HORIZON", "type": "range", "bounds": [0.5, 1.0]},
      
        {"name": "q_l", "type": "range", "bounds": [450.0, 700.0]}, # Lag error
        {"name": "q_c", "type": "range", "bounds": [100.0, 450.0]}, # Contour error
        {"name": "q_l_gate_peak", "type": "range", "bounds": [500.0, 1000.0]},
        {"name": "q_c_gate_peak", "type": "range", "bounds": [600.0, 1000.0]},
        {"name": "q_l_obst_peak", "type": "range", "bounds": [100.0, 350.0]},
        {"name": "q_c_obst_peak", "type": "range", "bounds": [50.0, 200.0]},
        {"name": "miu", "type": "range", "bounds": [8.0, 17.0]},   
        {"name": "w_v_gate", "type": "range", "bounds": [0.1, 5.0]},
        {"name": "w_v_obst", "type": "range", "bounds": [0.1, 3.0]},
    ],
    objectives={"reward": ObjectiveProperties(minimize=False)},
)

TOTAL_TRIALS = 3000

print(f"Starting optimization for {TOTAL_TRIALS} trials...")

for i in range(TOTAL_TRIALS):
    print(f"--- Trial {i+1}/{TOTAL_TRIALS} ---")
    
    parameters, trial_index = ax_client.get_next_trial()
    raw_data = evaluation_function(parameters)
    ax_client.complete_trial(trial_index=trial_index, raw_data=raw_data)

best_parameters, values = ax_client.get_best_parameters()
print("             TUNING COMPLETED                ")
print(f"Best Parameters: {best_parameters}")
print(f"Best Reward: {values}")
