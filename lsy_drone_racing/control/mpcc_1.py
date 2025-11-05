"""
Controller Name: mpc_13.py
Author: Gemini (inspired by Romero et al., 'Model Predictive Contouring Control')
Description: 
    NMPC Trajectory Tracker with MPCC-inspired Dynamic Cost Weighting.

--- 核心思想 (基于 'mpcc.pdf' 论文) ---

本控制器基于 mpc_12.py 的 acados 框架，但融合了 mpcc.pdf 论文 
中的一个关键思想：动态权重分配。

Level 2 的挑战在于随机化。mpc_12.py 通过在 __init__ 中重新计算
全局路径来解决*门*的随机化问题（因为 sim.py 在每轮 episode 
开始时都会重新实例化控制器）。

然而，mpc_12.py 仍然是一个*静态权重*的轨迹跟踪器。

为了模拟 MPCC 的行为——即在靠近门时优先保证精度，
[cite_start]在远离门时优先保证速度 [cite: 273-275, 285]——本控制器
(mpc_13.py) 引入了动态成本函数：

1.  **定义两种权重矩阵**:
    * `self._W_far` (巡航权重): 较低的位置权重，鼓励速度。
    * `self._W_near` (穿门权重): 极高的位置权重 (X, Y, Z)，
        [cite_start]强制要求高精度，模拟 MPCC 中动态增加的 q_c [cite: 286-287]。

2.  **实时权重分配**: 
    在 `compute_control` 中，对于 MPC 预测时域 (N) 内的
    每一步 (j)，我们：
    * 计算参考点 `pos_ref[j]` 与当前目标门 `goal_pos` 的距离。
    * 根据距离动态设置 `self._solver.set(j, "W", ...)` 
        为 `_W_far` 或 `_W_near`。

这使得控制器在远离门时“不关心”微小的跟踪误差（从而更快），
但在接近门时变得“极其严格”，确保精确通过。

[FIX v1.1] 修复了 Acados 'ValueError: inconsistent dimension ny_0'。
    - W 和 W_e 矩阵现在在 __init__ 中 *之前* 定义。
    - create_ocp_solver 接受 W_default 和 W_e_default 作为参数，
      以满足初始化时的一致性检查。
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import scipy
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from casadi import MX, vertcat
from scipy.spatial.transform import Rotation as R

from drone_models.core import load_params
from drone_models.so_rpy import symbolic_dynamics_euler
from drone_models.utils.rotation import ang_vel2rpy_rates

from lsy_drone_racing.control import Controller
from collections import deque 

if TYPE_CHECKING:
    from numpy.typing import NDArray


# -----------------------------
# ACADOS model & OCP definition
# -----------------------------
def create_acados_model(parameters: dict) -> AcadosModel:
    X_dot, X, U, _ = symbolic_dynamics_euler(
        mass=parameters["mass"],
        gravity_vec=parameters["gravity_vec"],
        J=parameters["J"],
        J_inv=parameters["J_inv"],
        acc_coef=parameters["acc_coef"],
        cmd_f_coef=parameters["cmd_f_coef"],
        rpy_coef=parameters["rpy_coef"],
        rpy_rates_coef=parameters["rpy_rates_coef"],
        cmd_rpy_coef=parameters["cmd_rpy_coef"],
    )

    model = AcadosModel()
    model.name = "att_mpc_lvl13_dynW" # [MODIFIED] 新的 C-code 名称
    model.f_expl_expr = X_dot
    model.x = X
    model.u = U

    n_obs = 4
    p = MX.sym("p", n_obs * 3)
    model.p = p

    h_exprs = []
    for i in range(n_obs):
        ox, oy = p[3 * i], p[3 * i + 1]
        expr = (X[0] - ox) ** 2 + (X[1] - oy) ** 2
        h_exprs.append(expr)
    model.con_h_expr = vertcat(*h_exprs)
    return model


def create_ocp_solver(
    Tf: float, 
    N: int, 
    parameters: dict, 
    W_default: np.ndarray,      # <--- [FIX] AÑADIDO
    W_e_default: np.ndarray,    # <--- [FIX] AÑADIDO
    verbose: bool = False
):
    ocp = AcadosOcp()
    ocp.model = create_acados_model(parameters)

    ocp.dims.N = N
    n_obs = 4
    ocp.dims.nh = n_obs
    ocp.dims.nsh = n_obs 

    nx = ocp.model.x.rows() 
    nu = ocp.model.u.rows() 
    ny = nx + nu
    ny_e = nx

    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"

    # [FIX] 设置默认权重以进行初始化
    ocp.cost.W = W_default
    ocp.cost.W_e = W_e_default
    # --- FIN DE LA CORRECCIÓN ---

    Vx = np.zeros((ny, nx))
    Vx[:nx, :nx] = np.eye(nx)
    Vu = np.zeros((ny, nu))
    Vu[nx:, :] = np.eye(nu)
    ocp.cost.Vx = Vx
    ocp.cost.Vu = Vu
    ocp.cost.Vx_e = np.eye(ny_e)

    ocp.cost.yref = np.zeros(ny)
    ocp.cost.yref_e = np.zeros(ny_e)

    ocp.constraints.lbx = np.array([-0.5, -0.5, -0.5])     
    ocp.constraints.ubx = np.array([ 0.5,  0.5,  0.5])
    ocp.constraints.idxbx = np.array([3, 4, 5], dtype=int)  

    ocp.constraints.lbu = np.array([
        -0.5, -0.5, -0.5, parameters["thrust_min"] * 4.0
    ])
    ocp.constraints.ubu = np.array([
         0.5,  0.5,  0.5, parameters["thrust_max"] * 4.0
    ])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3], dtype=int)

    ocp.constraints.x0 = np.zeros(nx)

    r_safe = 0.25
    BIG = 1e9
    ocp.constraints.lh = np.ones(n_obs) * (r_safe ** 2)
    ocp.constraints.uh = np.ones(n_obs) * BIG

    ocp.constraints.idxsh = np.arange(n_obs, dtype=int)
    slack_w_lin, slack_w_quad = 5e2, 8e3
    ocp.cost.zl = slack_w_lin  * np.ones(n_obs)
    ocp.cost.zu = slack_w_lin  * np.ones(n_obs)
    ocp.cost.Zl = slack_w_quad * np.ones(n_obs)
    ocp.cost.Zu = slack_w_quad * np.ones(n_obs)

    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.tf = Tf
    ocp.parameter_values = np.zeros(n_obs * 3)

    json_name = "att_mpc_lvl13_dynW" # [MODIFIED] 新的 JSON
    solver = AcadosOcpSolver(
        ocp,
        json_file=f"c_generated_code/{json_name}.json",
        verbose=verbose,
        build=True,
        generate=True,
    )
    return solver, ocp


# -----------------------------
# Controller
# -----------------------------
class Mpc13(Controller):
    """
    Attitude NMPC with MPCC-inspired Dynamic Cost Weighting.
    """

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)

        self._N = 90
        self._dt = 1.0 / float(config.env.freq)
        self._T_HORIZON = self._N * self._dt

        self.drone_params = load_params("so_rpy", config.sim.drone_model)

        # [FIX] 预先定义维度以便创建权重矩阵
        self._nx = 12 
        self._nu = 4  
        self._ny = self._nx + self._nu
        self._ny_e = self._nx

        # --- [NEW] MPCC 动态权重参数 ---
        # [FIX] 在 create_ocp_solver 之前定义权重
        self._gate_proximity_threshold = 0.75 # (米) 切换权重的距离
        
        # 1. "巡航" 权重 (远离门时)
        Q_far = np.diag([
            100.0, 100.0, 1400.0,  # pos xyz (Z 权重较高以防下坠)
            5.0, 5.0, 5.0,       # rpy 
            1.0, 1.0, 1.0,       # vel
            5.0, 5.0, 5.0        # drpy
        ])
        R_far = np.diag([
            0.5, 0.5, 0.5,       # cmd rpy
            10.0                 # thrust (惩罚较低)
        ])
        self._W_far = scipy.linalg.block_diag(Q_far, R_far)
        
        self._W_e_far = np.diag([
            1000.0, 1000.0, 1000.0,  # pos xyz 
            3.0, 3.0, 3.0,         # rpy
            10.0, 10.0, 10.0,      # vel
            5.0, 5.0, 5.0          # drpy
        ])
        
        # 2. "穿门" 权重 (靠近门时)
        Q_near = np.diag([
            8000.0, 8000.0, 8000.0, # pos xyz (!!! 极高 !!!)
            5.0, 5.0, 5.0,         # rpy 
            1.0, 1.0, 1.0,         # vel
            5.0, 5.0, 5.0          # drpy
        ])
        R_near = np.diag([
            0.5, 0.5, 0.5,       # cmd rpy
            20.0                 # thrust (惩罚较高以求稳定)
        ])
        self._W_near = scipy.linalg.block_diag(Q_near, R_near)

        self._W_e_near = np.diag([
            10000.0, 10000.0, 10000.0, # pos xyz (!!! 极高 !!!)
            3.0, 3.0, 3.0,           # rpy
            10.0, 10.0, 10.0,        # vel
            5.0, 5.0, 5.0            # drpy
        ])
        # --- [NEW] MPCC END ---

        # [FIX] 现在调用 create_ocp_solver，传入默认权重
        self._solver, self._ocp = create_ocp_solver(
            self._T_HORIZON, 
            self._N, 
            self.drone_params,
            W_default=self._W_far,        # <--- AÑADIDO
            W_e_default=self._W_e_far   # <--- AÑADIDO
        )

        # [FIX] 从已创建的 solver 中获取*真实*维度
        self._nx = self._ocp.model.x.rows()
        self._nu = self._ocp.model.u.rows()
        self._ny = self._nx + self._nu
        self._ny_e = self._nx
        self._finished = False

        self._traj_hist = deque(maxlen=4000) 
        self._last_plan = None
        self._last_polyline = None 

        # [MODIFIED] 调整 d_pre/d_post 列表 (与 mpc_12.py 相同)
        num_gates_in_obs = len(obs["gates_pos"])
        print(f"[Mpc13] Loading MPCC Dynamic Weights (tuned for level2 4-gate track).")
        self.d_pre_list = np.array([
            0.45, 0.50, 0.45, 0.45, 
        ])
        self.d_post_list = np.array([
            0.90, 1.10, 1.20, 1.20,
        ])
        
        if len(self.d_pre_list) < num_gates_in_obs:
            self.d_pre_list = np.pad(self.d_pre_list, (0, num_gates_in_obs - len(self.d_pre_list)), 'edge')
        if len(self.d_post_list) < num_gates_in_obs:
            self.d_post_list = np.pad(self.d_post_list, (0, num_gates_in_obs - len(self.d_post_list)), 'edge')

        # [UNCHANGED] 路径构建逻辑 (在 __init__ 中运行是正确的，
        # 因为 sim.py 会在每轮 episode 重新实例化)
        self._global_waypoints = None 
        self._build_global_path(obs) 
        
        self._deque_len = 6 
        self._local_polyline_deque = deque(maxlen=self._deque_len)
        self._current_global_idx = 0
        self._consumption_dist = 0.4 
        
        self._reset_deque()

    # ---------- helpers (与 mpc_12.py 相同) ----------
    def _hover_thrust(self) -> float:
        return float(self.drone_params["mass"]) * (-float(self.drone_params["gravity_vec"][-1]))

    def _gate_forward_xy_from_quat(self, q: np.ndarray) -> np.ndarray:
        fwd = R.from_quat(q).apply(np.array([1.0, 0.0, 0.0]))  
        v = fwd[:2]
        n = np.linalg.norm(v)
        if n < 1e-12:
            return np.array([1.0, 0.0])
        return v / n

    def _build_global_path(self, initial_obs: dict):
        # (此函数与 mpc_12.py v7.5 版本完全相同)
        # (它在 __init__ 中使用 *已随机化* 的 initial_obs["gates_pos"] 
        #  来构建本轮 episode 的全局路径，这是正确的。)
        print("[Mpc13] Pre-computing global reference path (v7.5 logic)...")
        gates_pos = np.asarray(initial_obs["gates_pos"], float)
        gates_quat = np.asarray(initial_obs.get("gates_quat", None), float) \
                     if "gates_quat" in initial_obs else None
        initial_pos = np.asarray(initial_obs["pos"], float)
        
        all_waypoints = [initial_pos]
        last_post_pt = initial_pos.copy() 

        num_gates = len(gates_pos)
        gate_sequence = list(range(num_gates))
        
        for gate_idx in gate_sequence:
            g_c = gates_pos[gate_idx]
            g_quat = gates_quat[gate_idx] if gates_quat is not None else None
            
            n_xy = None
            vec_to_compare_3d = g_c - last_post_pt
            vec_to_compare_xy = vec_to_compare_3d[:2]
            norm_vec_to_compare = np.linalg.norm(vec_to_compare_xy)
            
            if norm_vec_to_compare < 1e-6:
                vec_to_compare_xy = np.array([1.0, 0.0])
                norm_vec_to_compare = 1.0
            if g_quat is not None:
                n_xy = self._gate_forward_xy_from_quat(g_quat)
                if np.dot(n_xy, vec_to_compare_xy) < 0:
                    n_xy = -n_xy
            else:
                n_xy = vec_to_compare_xy / norm_vec_to_compare
            
            pre_pt = g_c.copy()
            current_d_pre = self.d_pre_list[gate_idx]
            base_v_pre_xy = -current_d_pre * n_xy
            pre_pt[:2] += base_v_pre_xy 

            if gate_idx == 3: # G2(idx 2) 和 G3(idx 3) 之间
                post2 = last_post_pt 
                pre3 = pre_pt      
                vec_to_pre3 = pre3 - post2
                norm_vec = np.linalg.norm(vec_to_pre3)
                climb_pt = post2.copy()
                if norm_vec > 1e-6:
                    unit_vec = vec_to_pre3 / norm_vec
                    climb_pt += 0.1 * unit_vec 
                climb_pt[2] += 0.5 
                all_waypoints.append(climb_pt)
                
            post_pt = g_c.copy()
            current_d_post = self.d_post_list[gate_idx]
            base_v_post_xy = current_d_post * n_xy
            post_pt[:2] += base_v_post_xy 

            all_waypoints.append(pre_pt)
            all_waypoints.append(g_c)
            all_waypoints.append(post_pt)

            last_post_pt = post_pt.copy()
            
        self._global_waypoints = np.vstack(all_waypoints)
        self._start_idx = 1
        print(f"[Mpc13] Global path with {len(self._global_waypoints)} points built.")
    
    
    def _reset_deque(self):
        # (与 mpc_12.py 相同)
        self._local_polyline_deque.clear()
        self._current_global_idx = self._start_idx 
        
        idx = self._current_global_idx
        while len(self._local_polyline_deque) < self._deque_len:
            if idx < len(self._global_waypoints):
                self._local_polyline_deque.append(self._global_waypoints[idx])
            else:
                self._local_polyline_deque.append(self._global_waypoints[-1])
            idx += 1
        self._current_global_idx = idx

    
    def _build_local_ref(
        self,
        cur_pos: np.ndarray,
        goal_pos: np.ndarray, 
        cur_quat: np.ndarray,
        v_des: float = 0.8,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # (与 mpc_12.py 相同 - 路径采样逻辑)
        T = self._T_HORIZON 
        N = self._N 
        dt = self._dt 

        cur_pos = np.asarray(cur_pos, float).reshape(3) 

        active_waypoints = list(self._local_polyline_deque)
        base_waypoints = [cur_pos] + active_waypoints
        
        waypoints = np.vstack(base_waypoints) 
        self._last_polyline = waypoints.copy() 

        seg_vecs = waypoints[1:] - waypoints[:-1]
        seg_lens = np.linalg.norm(seg_vecs, axis=1)
        cum_lens = np.concatenate([[0.0], np.cumsum(seg_lens)])
        total_len = cum_lens[-1]

        if total_len < 1e-6:
            gate_c = np.asarray(goal_pos, float).reshape(3)
            pos_ref = np.repeat(gate_c[None, :], N + 1, axis=0)
            vel_ref = np.zeros_like(pos_ref)
            yaw_ref = np.zeros(N + 1)
            cur_yaw = R.from_quat(cur_quat).as_euler("xyz")[2] 
            yaw_ref[:] = cur_yaw 
            return pos_ref, vel_ref, yaw_ref

        s_max = v_des * T
        gamma = 0.6 
        u = np.linspace(0.0, 1.0, N + 1)
        u_ease = u ** gamma
        s_samples = np.clip(u_ease * min(s_max, total_len), 0.0, total_len)

        pos_ref = np.zeros((N + 1, 3))
        for k, s in enumerate(s_samples):
            seg_idx = np.searchsorted(cum_lens, s, side="right") - 1
            seg_idx = np.clip(seg_idx, 0, len(seg_vecs) - 1)
            s_in_seg = s - cum_lens[seg_idx]
            alpha = 0.0 if seg_lens[seg_idx] < 1e-9 else (s_in_seg / seg_lens[seg_idx])
            pos_ref[k] = waypoints[seg_idx] + alpha * seg_vecs[seg_idx]

        vel_ref = np.zeros_like(pos_ref)
        vel_ref[:-1] = (pos_ref[1:] - pos_ref[:-1]) / dt
        vel_ref[-1] = 0.0

        dpos = np.gradient(pos_ref, axis=0)
        yaw_ref = np.arctan2(dpos[:, 1], dpos[:, 0])

        unstable_yaw = np.linalg.norm(dpos[:, :2], axis=1) < 0.05
        if np.any(unstable_yaw):
            cur_yaw = R.from_quat(cur_quat).as_euler("xyz")[2]
            last_stable_yaw = cur_yaw
            for k in range(N + 1):
                if unstable_yaw[k]:
                    yaw_ref[k] = last_stable_yaw
                else:
                    last_stable_yaw = yaw_ref[k]
        
        yaw_ref = np.nan_to_num(yaw_ref, nan=0.0)
        return pos_ref, vel_ref, yaw_ref
    
    
    # ---------- debug visualization (matplotlib) ----------
    def debug_plot(self, obs: dict):
        # (此函数无需更改)
        pass
    
    
    # ---------- main MPC step ----------
    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        
        cur_pos = np.asarray(obs["pos"], float) 
        
        obs["rpy"] = R.from_quat(obs["quat"]).as_euler("xyz")
        obs["drpy"] = ang_vel2rpy_rates(obs["quat"], obs["ang_vel"])
        x0 = np.concatenate((obs["pos"], obs["rpy"], obs["vel"], obs["drpy"]))
        self._solver.set(0, "lbx", x0)
        self._solver.set(0, "ubx", x0)

        # Deque Update Logic (与 mpc_12.py 相同)
        if len(self._local_polyline_deque) > 0:
            target_pt = self._local_polyline_deque[0]
            dist_to_target_xy = np.linalg.norm(cur_pos[:2] - target_pt[:2])
            
            if dist_to_target_xy < self._consumption_dist:
                self._local_polyline_deque.popleft()
                next_idx = self._current_global_idx
                if next_idx < len(self._global_waypoints):
                    new_pt = self._global_waypoints[next_idx]
                    self._local_polyline_deque.append(new_pt)
                    self._current_global_idx += 1
                else:
                    last_pt = self._global_waypoints[-1]
                    self._local_polyline_deque.append(last_pt)

        # 障碍物参数设置 (与 mpc_12.py 相同)
        n_obs = 4 
        cur_pos_xy = np.asarray(obs["pos"], float)[:2] 
        all_obs_pos = np.asarray(obs["obstacles_pos"], float).reshape(-1, 3) 
        selected_obs_pos = np.zeros((n_obs, 3))
        if all_obs_pos.shape[0] > 0:
            distances_xy = np.linalg.norm(all_obs_pos[:, :2] - cur_pos_xy, axis=1)
            closest_indices = np.argsort(distances_xy)
            num_to_take = min(all_obs_pos.shape[0], n_obs)
            indices_to_take = closest_indices[:num_to_take]
            selected_obs_pos[:num_to_take] = all_obs_pos[indices_to_take]
            if num_to_take < n_obs:
                selected_obs_pos[num_to_take:] = np.array([1e6, 1e6, 1e6])
        else:
             selected_obs_pos[:] = np.array([1e6, 1e6, 1e6])
        p_vec = selected_obs_pos.flatten()
        for j in range(self._N + 1):
            self._solver.set(j, "p", p_vec)
        
        # 目标门参考 (与 mpc_12.py 相同)
        gate_idx = int(np.asarray(obs["target_gate"]).item()) 
        gates_pos = np.asarray(obs["gates_pos"], float) 
        
        if gate_idx < 0 or gate_idx >= len(gates_pos):
            goal_pos = gates_pos[-1] # 完成比赛，悬停在最后一个门
        else:
            goal_pos = gates_pos[gate_idx] # 当前目标门

        cur_quat = np.asarray(obs["quat"], float) 
        
        # 路径采样 (与 mpc_12.py 相同)
        pos_ref, vel_ref, yaw_ref = self._build_local_ref(
            cur_pos=cur_pos,
            goal_pos=goal_pos, 
            cur_quat=cur_quat,
            v_des=1.1, 
        )

        self._last_plan = pos_ref.copy() 
        self._traj_hist.append(cur_pos.reshape(3)) 

        # --- [MODIFIED] MPCC 动态权重设置 ---
        
        # 1. 填充 yref 矩阵
        yref_matrix = np.zeros((self._N, self._ny))
        yref_matrix[:, 0:3] = pos_ref[: self._N]    # pos
        yref_matrix[:, 5] = yaw_ref[: self._N]      # yaw
        yref_matrix[:, 6:9] = vel_ref[: self._N]    # vel
        yref_matrix[:, 15] = self._hover_thrust()   # thrust bias

        # 2. 循环设置 yref 和 动态 W
        for j in range(self._N):
            # 获取当前步的参考位置
            p_j = pos_ref[j]
            # 计算到目标门的距离
            dist_to_gate = np.linalg.norm(p_j - goal_pos)
            
            # 动态权重逻辑
            if dist_to_gate < self._gate_proximity_threshold:
                # 靠近门：使用高精度权重
                self._solver.set(j, "W", self._W_near)
            else:
                # 远离门：使用巡航权重
                self._solver.set(j, "W", self._W_far)
            
            # 设置 yref (与 mpc_12 相同)
            self._solver.set(j, "yref", yref_matrix[j]) 

        # 3. 设置 yref_e (终端) 和 动态 W_e (终端)
        yref_e = np.zeros(self._ny_e) 
        yref_e[0:3] = pos_ref[self._N] 
        yref_e[5] = yaw_ref[self._N] 
        yref_e[6:9] = vel_ref[self._N] 
        
        # 终端权重逻辑
        p_N = pos_ref[self._N]
        dist_to_gate_e = np.linalg.norm(p_N - goal_pos)
        if dist_to_gate_e < self._gate_proximity_threshold:
            self._solver.set(self._N, "W_e", self._W_e_near) # 注意: 终端权重键是 "W_e"
        else:
            self._solver.set(self._N, "W_e", self._W_e_far)
            
        self._solver.set(self._N, "y_ref", yref_e)  # 注意: 终端参考键是 "y_ref"
        # --- [MODIFIED END] ---

        # 求解
        self._solver.solve() 
        
        pred = np.zeros((self._N + 1, 3)) 
        for j in range(self._N + 1):
            xj = self._solver.get(j, "x")   
            pred[j] = xj[0:3]
        self._last_plan = pred
        u0 = self._solver.get(0, "u") 
        
        return u0
    
    # ---------- callbacks (与 mpc_12.py 相同) ----------
    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        return False

    def episode_callback(self):
        self._traj_hist.clear()
        self._last_plan = None
        print("[Mpc13] Episode reset. Resetting local deque.")
        self._reset_deque()

    def get_debug_lines(self):
        out = []
        if len(self._traj_hist) >= 2:
            traj = np.asarray(self._traj_hist, float)
            out.append((traj, np.array([0.1, 0.3, 1.0, 0.9]), 2.5, 2.5))  # 蓝：历史
        if self._last_plan is not None and self._last_plan.shape[0] >= 2:
            out.append((self._last_plan, np.array([1.0, 0.1, 0.1, 0.95]), 3.0, 3.0))  # 红：预测
        if getattr(self, "_last_polyline", None) is not None and self._last_polyline.shape[0] >= 2:
            out.append((self._last_polyline, np.array([1.0, 0.9, 0.1, 0.95]), 3.0, 3.0))  # 黄：折线
        if getattr(self, "_global_waypoints", None) is not None:
             out.append((self._global_waypoints, np.array([0.5, 0.5, 0.5, 0.7]), 2.0, 2.0)) # 灰：全局
        return out