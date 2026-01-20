"""
Using BoTorch + TuRBO for parameter auto tuning
Run: python scripts/tune_mpcc_turbo.py
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from scripts.sim_4_tune import simulate 
import math
import torch
import numpy as np
from dataclasses import dataclass
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils.transforms import unnormalize, normalize
from botorch.acquisition import qExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood


PARAM_CONFIGS = [
    ("q_l",          450.0, 700.0),
    ("q_c",          100.0, 550.0),
    ("q_l_gate_peak", 300.0, 1000.0),
    ("q_c_gate_peak", 600.0, 1000.0),
    ("q_l_obst_peak", 100.0, 450.0),
    ("q_c_obst_peak", 50.0, 200.0),
    ("miu",          8.0,   17.0),
    ("w_v_gate",     0.1,   5.0),
    ("w_v_obst",     0.1,   3.0),
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.double
PARAM_NAMES = [p[0] for p in PARAM_CONFIGS]
# Tensor (2, D)
BOUNDS = torch.tensor([[p[1] for p in PARAM_CONFIGS], 
                       [p[2] for p in PARAM_CONFIGS]], device=DEVICE, dtype=torch.double)


BATCH_SIZE = 1        
NUM_INIT = 10        
TOTAL_TRIALS = 300    

def evaluation_function(parameters_dict):
    print(f"\n[TuRBO] Testing params: {parameters_dict}")

    result = simulate(
        config="level2.toml", 
        controller="mpcc_rotor_edit_tune.py", 
        n_runs=8,      
        render=False,  
        tuning_params=parameters_dict
    )
    
    avg_time = result["avg_time"]
    success_rate = result["success_rate"]
    fail_rate = 1.0 - success_rate
    
    if success_rate < 0.75:
        reward = -500.0 - 100.0 * fail_rate
    else:
        effective_time = avg_time if success_rate > 0 else 20.0
        baseline_time = 8.0
        speed_score = (baseline_time - effective_time) * 2
        risk_penalty = 20.0 * (fail_rate ** 2)
        reward = speed_score - risk_penalty
   
    print(f"[TuRBO] Result: Time={avg_time:.2f}s, Success={success_rate*100:.0f}%, Reward={reward:.2f}")
    return reward

def eval_objective(x_normalized):
    
    # [0, 1] -> [LB, UB]
    x_true = unnormalize(x_normalized, bounds=BOUNDS)
    
    results = []
    for x in x_true:
        # Tensor to Dict
        params = {name: val.item() for name, val in zip(PARAM_NAMES, x)}
        reward = evaluation_function(params)
        results.append(reward)
        
    return torch.tensor(results, device=DEVICE, dtype=DTYPE).unsqueeze(-1)

@dataclass
class TurboState:
    dim: int
    batch_size: int
    length: float = 0.8       
    length_min: float = 0.5**7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = 10 
    success_counter: int = 0
    success_tolerance: int = 3  
    best_value: float = -float("inf")
    restart_triggered: bool = False

def update_state(state, Y_next):
    if Y_next.max().item() > state.best_value + 1e-3 * math.fabs(state.best_value):
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1

    if state.success_counter == state.success_tolerance:
        state.length = min(state.length * 2.0, state.length_max)
        state.success_counter = 0
    
    elif state.failure_counter == state.failure_tolerance:
        state.length /= 2.0
        state.failure_counter = 0

    state.best_value = max(state.best_value, Y_next.max().item())
    
    if state.length < state.length_min:
        state.restart_triggered = True
    
    return state

def generate_batch(state, model, X, Y, batch_size=1):
    """在信赖域内生成下一个候选点"""
    x_center = X[Y.argmax(), :].clone()
    
    if hasattr(model.covar_module, "base_kernel"):
        lengthscale = model.covar_module.base_kernel.lengthscale
    else:
        lengthscale = model.covar_module.lengthscale
    
    weights = lengthscale.squeeze().detach()
  
    weights = weights / weights.mean()
    weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
    
    # 计算信赖域边界 (Trust Region Bounds)
    tr_lb = torch.clamp(x_center - weights * state.length / 2.0, 0.0, 1.0)
    tr_ub = torch.clamp(x_center + weights * state.length / 2.0, 0.0, 1.0)
    
    # 优化采集函数 (qEI)
    acquisition_function = qExpectedImprovement(model, best_f=Y.max())
    
    candidate, _ = optimize_acqf(
        acquisition_function,
        bounds=torch.stack([tr_lb, tr_ub]),
        q=batch_size,
        num_restarts=10,
        raw_samples=512,
    )
    return candidate


if __name__ == "__main__":
    print(f"Starting TuRBO optimization for {TOTAL_TRIALS} trials...")
    
    train_X = torch.empty(0, len(PARAM_NAMES), device=DEVICE, dtype=DTYPE)
    train_Y = torch.empty(0, 1, device=DEVICE, dtype=DTYPE)
  
    state = TurboState(dim=len(PARAM_NAMES), batch_size=BATCH_SIZE)
    
    print(f"--- Initialization ({NUM_INIT} random points) ---")
    X_init = torch.rand(NUM_INIT, len(PARAM_NAMES), device=DEVICE, dtype=DTYPE)
    Y_init = eval_objective(X_init)
    
    train_X = torch.cat((train_X, X_init))
    train_Y = torch.cat((train_Y, Y_init))
    state.best_value = train_Y.max().item()

    for i in range(TOTAL_TRIALS - NUM_INIT):
        print(f"--- Trial {NUM_INIT + i + 1}/{TOTAL_TRIALS} [TR Length: {state.length:.4f}] ---")
        

        model = SingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
      
        X_next = generate_batch(state, model, train_X, train_Y, BATCH_SIZE)
       
        Y_next = eval_objective(X_next)
    
        train_X = torch.cat((train_X, X_next))
        train_Y = torch.cat((train_Y, Y_next))
    
        state = update_state(state, Y_next)
        
        if state.restart_triggered:
            print("!!! Trust region too small. Restarting state (keeping history)... !!!")
           
            state = TurboState(dim=len(PARAM_NAMES), batch_size=BATCH_SIZE)
            state.best_value = train_Y.max().item() 

   
    best_idx = train_Y.argmax()
    best_x_norm = train_X[best_idx]
    best_reward = train_Y[best_idx].item()
   
    best_params_tensor = unnormalize(best_x_norm.unsqueeze(0), bounds=BOUNDS).squeeze()
    best_parameters = {name: val.item() for name, val in zip(PARAM_NAMES, best_params_tensor)}

    print("\n" + "="*40)
    print("             TUNING COMPLETED                ")
    print("="*40)
    print(f"Best Reward: {best_reward:.4f}")
    print("Best Parameters:")
    for k, v in best_parameters.items():
        print(f"  {k}: {v:.4f}")
