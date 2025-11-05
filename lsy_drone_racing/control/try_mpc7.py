# trajectory aims to next gate
"""
try_mpc7.py — Attitude NMPC with soft obstacle avoidance, gate-normal pre/post points,
and MuJoCo visualization hooks.

Drop this file into lsy_drone_racing/control/ and set controller.file = "try_mpc7.py"
in your config.

[MODIFIED: 
 1. Fixed n_xy "flipping" bug by using in-vector (G_k - G_{k-1}) for sanity check.
 2. Extended lookahead polyline to (G1 -> G2 -> pre_G3) for smoother planning.
 3. [USER REQ] Changed lookahead normal logic for G2/G3 to align with 
    the previous gate's post-point (e.g., n_xy_2 uses G2 - post_pt_1).
]
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
    model.name = "att_mpc_lvl2_softobs"
    model.f_expl_expr = X_dot
    model.x = X
    model.u = U

    # parameters p: obstacles positions (we use only x,y; z ignored in constraints)
    n_obs = 4
    p = MX.sym("p", n_obs * 3)
    model.p = p

    # soft constraints h(x,p): squared XY distance to each obstacle (>= r_safe^2)
    h_exprs = []
    for i in range(n_obs):
        ox, oy = p[3 * i], p[3 * i + 1]
        expr = (X[0] - ox) ** 2 + (X[1] - oy) ** 2
        h_exprs.append(expr)
    model.con_h_expr = vertcat(*h_exprs)
    return model


def create_ocp_solver(Tf: float, N: int, parameters: dict, verbose: bool = False):
    ocp = AcadosOcp()
    ocp.model = create_acados_model(parameters)

    ocp.dims.N = N
    n_obs = 4
    ocp.dims.nh = n_obs
    ocp.dims.nsh = n_obs  # <- make soft version of all h constraints

    nx = ocp.model.x.rows()  # 12: pos(3), rpy(3), vel(3), drpy(3)
    nu = ocp.model.u.rows()  # 4 : cmd_rpy(3), thrust
    ny = nx + nu
    ny_e = nx

    # LS cost
    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"

    # tracking weights
    Q = np.diag([
        200.0, 200.0, 400.0,   # pos xyz
        3.0, 3.0, 1.5,       # rpy (roll/pitch slightly up for smoothness; yaw modest)
        10.0, 10.0, 10.0,    # vel
        5.0, 5.0, 5.0        # drpy
    ])
    Rm = np.diag([
        1.0, 1.0, 1.0,       # cmd rpy
        30.0                 # thrust (not too large, allow accel)
    ])
    ocp.cost.W = scipy.linalg.block_diag(Q, Rm)
    ocp.cost.W_e = np.diag([
        500.0, 500.0, 400.0,   # pos xyz
        3.0, 3.0, 1.5,       # rpy (roll/pitch slightly up for smoothness; yaw modest)
        10.0, 10.0, 10.0,    # vel
        5.0, 5.0, 5.0        # drpy
    ])

    Vx = np.zeros((ny, nx))
    Vx[:nx, :nx] = np.eye(nx)
    Vu = np.zeros((ny, nu))
    Vu[nx:, :] = np.eye(nu)
    ocp.cost.Vx = Vx
    ocp.cost.Vu = Vu
    ocp.cost.Vx_e = np.eye(ny_e)

    ocp.cost.yref = np.zeros(ny)
    ocp.cost.yref_e = np.zeros(ny_e)

    # box constraints on x (roll/pitch/yaw)
    ocp.constraints.lbx = np.array([-0.5, -0.5, -0.5])     # rpy bounds (rad)
    ocp.constraints.ubx = np.array([ 0.5,  0.5,  0.5])
    ocp.constraints.idxbx = np.array([3, 4, 5], dtype=int)  # state x indices for rpy

    # input bounds
    ocp.constraints.lbu = np.array([
        -0.5, -0.5, -0.5, parameters["thrust_min"] * 4.0
    ])
    ocp.constraints.ubu = np.array([
         0.5,  0.5,  0.5, parameters["thrust_max"] * 4.0
    ])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3], dtype=int)

    # initial state equality
    ocp.constraints.x0 = np.zeros(nx)

    # soft obstacle constraints: lh <= h(x,p) <= uh with slack
    r_safe = 0.35
    BIG = 1e9
    ocp.constraints.lh = np.ones(n_obs) * (r_safe ** 2)
    ocp.constraints.uh = np.ones(n_obs) * BIG

    # make all nh soft
    ocp.constraints.idxsh = np.arange(n_obs, dtype=int)
    slack_w_lin, slack_w_quad = 5e2, 8e3
    ocp.cost.zl = slack_w_lin  * np.ones(n_obs)
    ocp.cost.zu = slack_w_lin  * np.ones(n_obs)
    ocp.cost.Zl = slack_w_quad * np.ones(n_obs)
    ocp.cost.Zu = slack_w_quad * np.ones(n_obs)

    # solvers
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.tf = Tf

    # obstacle parameters (set at runtime)
    ocp.parameter_values = np.zeros(n_obs * 3)

    json_name = "att_mpc_lvl2_softobs"
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
class AttitudeMPC(Controller):
    """Attitude NMPC with soft obstacle avoidance and gate-normal pre/post points."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)

        self._N = 100
        self._dt = 1.0 / float(config.env.freq)
        self._T_HORIZON = self._N * self._dt

        self.drone_params = load_params("so_rpy", config.sim.drone_model)
        self._solver, self._ocp = create_ocp_solver(
            self._T_HORIZON, self._N, self.drone_params
        )

        self._nx = self._ocp.model.x.rows()
        self._nu = self._ocp.model.u.rows()
        self._ny = self._nx + self._nu
        self._ny_e = self._nx

        self._finished = False

        # Visualization caches
        from collections import deque
        self._traj_hist = deque(maxlen=4000)  # world-frame drone positions over episode
        self._last_plan = None                # (N+1,3) current reference positions
        self._last_polyline = None   # 存最近一次构造的折线拐点 (M,3)


    # ---------- helpers ----------
    def _hover_thrust(self) -> float:
        # gravity_vec is like [0, 0, -9.81]
        return float(self.drone_params["mass"]) * (-float(self.drone_params["gravity_vec"][-1]))

    def _gate_forward_xy_from_quat(self, q: np.ndarray) -> np.ndarray:
        """Return gate forward direction projected on XY plane (normalized)."""
        # scipy expects [x,y,z,w]
        fwd = R.from_quat(q).apply(np.array([1.0, 0.0, 0.0]))  # gate +X is "front"
        v = fwd[:2]
        n = np.linalg.norm(v)
        if n < 1e-12:
            return np.array([1.0, 0.0])
        return v / n
    
    
    
    
    def _build_local_ref(
        self,
        cur_pos: np.ndarray,
        goal_pos: np.ndarray,
        cur_quat: np.ndarray,
        all_obs_pos: np.ndarray,    # <--- [新增] 障碍物位置
        r_safe_ref: float,          # <--- [新增] 参考路径的安全半径
        gates_pos: np.ndarray | None = None,
        gate_idx: int | None = None,
        v_des: float = 0.8,
        d_pre: float = 0.8,
        d_post: float = 0.8,
        gates_quat: np.ndarray | None = None,
        next_gate_pos: np.ndarray | None = None,     # <--- [新增] 下一个门的位置
        next_gates_quat: np.ndarray | None = None, # <--- [新增] 下一个门的姿态
        
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        [鲁棒版本 v4.2]
        构造一个鲁棒的、向前看的参考路径。
        1. [V1 修复] 统一使用 (G_k - G_{k-1}) 作为法线检查标准，消除pre点翻转。
        2. 智能地移动 post_pt 以避开障碍物。
        3. [V4.1 扩展] 将路径平滑连接到下下个门的 pre-point (pre3)。
        4. [V4.2 MODIFIED] 更改 G2 和 G3 的法线逻辑，使其参考前一个门的 post-point。
        """
        T = self._T_HORIZON #
        N = self._N #
        dt = self._dt #

        gate_c = np.asarray(goal_pos, float).reshape(3)
        cur_pos = np.asarray(cur_pos, float).reshape(3) 

        # --- 1. 计算 n_xy, d_pre, d_post (G1 - 当前目标门) ---
        # (逻辑保持不变: G1的法线参考 (G1-G0) 或 (G1-Drone) )
        use_gate_normal = (gates_quat is not None) and (gate_idx is not None)
        n_xy = None
        is_last_gate = (gates_pos is None) or (gate_idx is None) or (gate_idx + 1 >= len(gates_pos)) #

        if use_gate_normal:
            gates_quat = np.asarray(gates_quat, float)
            n_xy = self._gate_forward_xy_from_quat(gates_quat[gate_idx]) #
            if gates_pos is not None:
                gates_pos = np.asarray(gates_pos, float)
                vec_to_compare = None
                
                # --- [修正逻辑 V1] ---
                # 始终使用"入口"向量 (G_curr - G_prev) 作为参考
                prev_idx = gate_idx - 1 if gate_idx - 1 >= 0 else None
                if prev_idx is not None:
                    vec_to_compare = gate_c[:2] - gates_pos[prev_idx][:2] # G_curr - G_prev
                else:
                    # 如果是第一个门, 则使用 (G_curr - Drone)
                    vec_to_compare = gate_c[:2] - cur_pos[:2] 
                # --- [修正结束] ---
                
                if np.linalg.norm(vec_to_compare) > 1e-9 and np.dot(n_xy, vec_to_compare) < 0:
                    n_xy = -n_xy
        
        if (n_xy is None) and (gates_pos is not None):
            gates_pos = np.asarray(gates_pos, float)
            if gate_idx is not None:
                prev_idx = gate_idx - 1 if gate_idx - 1 >= 0 else None
                vec_in = (gate_c - gates_pos[prev_idx]) if prev_idx is not None else (gate_c - cur_pos)
                if np.linalg.norm(vec_in) < 1e-6:
                    vec_in = np.array([1.0, 0.0, 0.0])
                n_xy = (vec_in / (np.linalg.norm(vec_in) + 1e-12))[:2]
            else:
                vec = gate_c - cur_pos
                if np.linalg.norm(vec) < 1e-6:
                    vec = np.array([1.0, 0.0, 0.0])
                n_xy = (vec / (np.linalg.norm(vec) + 1e-12))[:2]
        if n_xy is None:
            n_xy = np.array([1.0, 0.0])

        if gates_pos is not None and gate_idx is not None:
            if gate_idx - 1 >= 0:
                d_prev = np.linalg.norm(gate_c - gates_pos[gate_idx - 1])
                d_pre = min(d_pre, 0.75 * max(d_prev, 1e-6))
            if not is_last_gate: 
                d_next = np.linalg.norm(gates_pos[gate_idx + 1] - gate_c)
                d_post = min(d_post, 0.75 * max(d_next, 1e-6))
        # --- (n_xy, d_pre, d_post 计算结束) ---

        
        pre_pt = gate_c.copy()
        post_pt = gate_c.copy()
        pre_pt[:2]  -= d_pre  * n_xy
        post_pt[:2] += d_post * n_xy
        
        # --- 2. 智能移动 post_pt (逻辑保持不变) ---
        if all_obs_pos.shape[0] > 0 and not is_last_gate:
            distances_to_post_pt_xy = np.linalg.norm(all_obs_pos[:, :2] - post_pt[:2], axis=1)
            min_dist = np.min(distances_to_post_pt_xy)
            
            if min_dist < r_safe_ref:
                closest_obs = all_obs_pos[np.argmin(distances_to_post_pt_xy)]
                push_dir = gate_c[:2] - closest_obs[:2]
                push_norm = np.linalg.norm(push_dir)
                if push_norm < 1e-6:
                    push_dir_norm = -n_xy
                else:
                    push_dir_norm = push_dir / push_norm
                new_post_pt_xy = closest_obs[:2] + push_dir_norm * (r_safe_ref + 0.1)
                post_pt[:2] = new_post_pt_xy
        # --- (post_pt 移动结束) ---
        

        # --- 3. 构建“向前看”的黄线 (waypoints) [v4.2 扩展版] ---
        base_waypoints = [cur_pos, pre_pt, gate_c] # [cur, pre1, gate1]
        
        # 检查 G1 (当前目标) 是否是最后一个门
        is_g1_last_gate = (gates_pos is None) or (gate_idx is None) or (gate_idx + 1 >= len(gates_pos))

        if not is_g1_last_gate:
            # --- G1 不是最后一个门 ---
            base_waypoints.append(post_pt) # [cur, pre1, gate1, post1]
            
            # --- 规划 G2 ---
            # next_gate_pos (G2) 和 next_gates_quat (G2 quat) 已经传入
            g2_c = np.asarray(next_gate_pos, float).reshape(3)
            g2_quat = next_gates_quat
            
            # --- [MODIFIED] ---
            # 计算 n_xy_2 (G2的法线), 使用 (G2 - post_pt_1) 作为参考
            n_xy_2 = None
            if g2_quat is not None:
                n_xy_2 = self._gate_forward_xy_from_quat(g2_quat)
                # vec_to_g2 = g2_c[:2] - gate_c[:2] # (G2 - G1) <-- 旧逻辑
                vec_to_g2 = g2_c[:2] - post_pt[:2] # (G2 - post_pt_1) <-- 新逻辑
                if np.linalg.norm(vec_to_g2) > 1e-9 and np.dot(n_xy_2, vec_to_g2) < 0:
                    n_xy_2 = -n_xy_2
            else:
                # vec_in_g2 = g2_c - gate_c # (G2 - G1) <-- 旧逻辑
                vec_in_g2 = g2_c - post_pt # (G2 - post_pt_1) <-- 新逻辑
                if np.linalg.norm(vec_in_g2) < 1e-6: vec_in_g2 = np.array([1.0, 0.0, 0.0])
                n_xy_2 = (vec_in_g2 / (np.linalg.norm(vec_in_g2) + 1e-12))[:2]
            # --- [MODIFIED END] ---
            
            # 计算 G2 的 pre 和 post 点
            pre_pt_2 = g2_c.copy()
            pre_pt_2[:2] -= d_pre * n_xy_2 # (使用相同的 d_pre)
            
            post_pt_2 = g2_c.copy()
            post_pt_2[:2] += d_post * n_xy_2 # (使用相同的 d_post)

            # 添加 G2 的航点
            base_waypoints.append(pre_pt_2) # [..., pre2]
            base_waypoints.append(g2_c)      # [..., pre2, gate2]
            
            # 检查 G2 是否是最后一个门
            is_g2_last_gate = (gate_idx + 2 >= len(gates_pos))
            if not is_g2_last_gate:
                 base_waypoints.append(post_pt_2) # [..., pre2, gate2, post2]
            
                 # [额外优化] 进一步添加 G3 的 pre 点，实现最平滑的过渡
                 g3_c = gates_pos[gate_idx + 2]
                 g3_quat = gates_quat[gate_idx + 2] if gates_quat is not None else None
                 
                 # --- [MODIFIED] ---
                 # 计算 n_xy_3 (G3的法线), 使用 (G3 - post_pt_2) 作为参考
                 n_xy_3 = None
                 if g3_quat is not None:
                     n_xy_3 = self._gate_forward_xy_from_quat(g3_quat)
                     # vec_to_g3 = g3_c[:2] - g2_c[:2] # (G3 - G2) <-- 旧逻辑
                     vec_to_g3 = g3_c[:2] - post_pt_2[:2] # (G3 - post_pt_2) <-- 新逻辑
                     if np.linalg.norm(vec_to_g3) > 1e-9 and np.dot(n_xy_3, vec_to_g3) < 0:
                         n_xy_3 = -n_xy_3
                 else:
                     # vec_in_g3 = g3_c - g2_c # (G3 - G2) <-- 旧逻辑
                     vec_in_g3 = g3_c - post_pt_2 # (G3 - post_pt_2) <-- 新逻辑
                     if np.linalg.norm(vec_in_g3) < 1e-6: vec_in_g3 = np.array([1.0, 0.0, 0.0])
                     n_xy_3 = (vec_in_g3 / (np.linalg.norm(vec_in_g3) + 1e-12))[:2]
                 # --- [MODIFIED END] ---
                 
                 pre_pt_3 = g3_c.copy()
                 pre_pt_3[:2] -= d_pre * n_xy_3
                 
                 base_waypoints.append(pre_pt_3) # [..., post2, pre3]

        waypoints = np.vstack(base_waypoints) #
        # --- (黄线构建结束) ---

        self._last_polyline = waypoints.copy() #

        # --- 4. 采样与鲁棒 Yaw (逻辑保持不变) ---
        seg_vecs = waypoints[1:] - waypoints[:-1]
        seg_lens = np.linalg.norm(seg_vecs, axis=1)
        cum_lens = np.concatenate([[0.0], np.cumsum(seg_lens)])
        total_len = cum_lens[-1]

        if total_len < 1e-6:
            # 路径太短，悬停在目标门
            pos_ref = np.repeat(gate_c[None, :], N + 1, axis=0)
            vel_ref = np.zeros_like(pos_ref)
            yaw_ref = np.zeros(N + 1)
            cur_yaw = R.from_quat(cur_quat).as_euler("xyz")[2] 
            yaw_ref[:] = cur_yaw 
            return pos_ref, vel_ref, yaw_ref

        s_max = v_des * T
        gamma = 0.6 #
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

        # 鲁棒 Yaw 计算
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
        # --- (采样结束) ---
        
        return pos_ref, vel_ref, yaw_ref
    
    
    # ---------- debug visualization (matplotlib) ----------
    def debug_plot(self, obs: dict):
        import matplotlib.pyplot as plt
        cur_pos = np.asarray(obs["pos"], float)
        gates_pos = np.asarray(obs["gates_pos"], float)
        cur_quat = np.asarray(obs["quat"], float)
        gates_quat = np.asarray(obs.get("gates_quat", None), float) if "gates_quat" in obs else None
        gate_idx = int(np.asarray(obs["target_gate"]).item())
        goal_pos = gates_pos[gate_idx]
        obstacles_pos = np.asarray(obs["obstacles_pos"], float).reshape(-1, 3)

        # --- [修改] 匹配 compute_control 中的新逻辑 ---
        
        # 1. 定义用于参考路径的安全半径
        # (应大于求解器的 r_safe = 0.35)
        r_safe_for_ref = 0.40 # (确保这个值与 compute_control 中的值匹配)

        # 2. 获取下一个门的信息 (用于 "向前看" 逻辑)
        next_gate_pos = None
        next_gates_quat = None
        if gate_idx + 1 < len(gates_pos):
            next_gate_pos = gates_pos[gate_idx + 1]
            if gates_quat is not None:
                next_gates_quat = gates_quat[gate_idx + 1]
        # --- [修改结束] ---


        pos_ref, vel_ref, yaw_ref = self._build_local_ref(
            cur_pos=cur_pos, 
            goal_pos=goal_pos, 
            cur_quat=cur_quat,
            all_obs_pos=obstacles_pos, # <--- 新增
            r_safe_ref=r_safe_for_ref, # <--- 新增
            gates_pos=gates_pos, 
            gate_idx=gate_idx,
            v_des=1.2, 
            d_pre=0.8, 
            d_post=0.8,
            gates_quat=gates_quat,
            next_gate_pos=next_gate_pos,     # <--- 新增
            next_gates_quat=next_gates_quat, # <--- 新增
        )

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(pos_ref[:, 0], pos_ref[:, 1], '-', label="MPC ref path")
        ax.scatter(cur_pos[0], cur_pos[1], c='blue', marker='o', s=80, label="drone")
        ax.scatter(gates_pos[:, 0], gates_pos[:, 1], c='green', marker='s', s=40, label="gates")
        for i, gp in enumerate(gates_pos):
            ax.text(gp[0], gp[1], f"G{i}", color='green', fontsize=8)
        if obstacles_pos.size > 0:
            ax.scatter(obstacles_pos[:, 0], obstacles_pos[:, 1], c='red', marker='x', s=80, label="obstacles")
            
            # [修改] 使用求解器的 r_safe 来绘制圆圈
            r_safe_solver = 0.35 # (这匹配 create_ocp_solver)
            for (ox, oy, _oz) in obstacles_pos:
                circ = plt.Circle((ox, oy), r_safe_solver, fill=False, linestyle='--', color='red', alpha=0.5)
                ax.add_patch(circ)
        if self._last_polyline is not None and self._last_polyline.shape[0] >= 2:
            wp = self._last_polyline
            ax.plot(wp[:, 0], wp[:, 1], 'o-', linewidth=2.0, label="polyline (黄线)")
            
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.grid(True); ax.legend()
        ax.set_title(f"Local ref to gate {gate_idx}")
        plt.show()
    
    
    # ---------- main MPC step ----------
    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        # augment obs with rpy, drpy
        obs["rpy"] = R.from_quat(obs["quat"]).as_euler("xyz")
        obs["drpy"] = ang_vel2rpy_rates(obs["quat"], obs["ang_vel"])
        x0 = np.concatenate((obs["pos"], obs["rpy"], obs["vel"], obs["drpy"]))
        self._solver.set(0, "lbx", x0)
        self._solver.set(0, "ubx", x0)

        
        # [修改 4/4] 删除了旧的、不鲁棒的障碍物设置代码
        # (旧代码块:)


        # set obstacle params p for all shooting nodes [鲁棒版本]
        # (只保留这一块)
        n_obs = 4 # 求解器期望的数量
        cur_pos_xy = np.asarray(obs["pos"], float)[:2] #
        all_obs_pos = np.asarray(obs["obstacles_pos"], float).reshape(-1, 3) #
        
        selected_obs_pos = np.zeros((n_obs, 3))
        
        if all_obs_pos.shape[0] > 0:
            # (鲁棒选择逻辑保持不变)
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
        # --- 障碍物逻辑结束 ---


        # [修改 4/4] 同步 r_safe_for_ref
        # (必须大于 步骤1 中设置的 r_safe = 0.35)
        r_safe_for_ref = 0.40 # (从 0.35 增加到 0.40)

        # gate references
        gate_idx = int(np.asarray(obs["target_gate"]).item()) #
        gates_pos = np.asarray(obs["gates_pos"], float) #
        goal_pos = gates_pos[gate_idx] #
        cur_pos = np.asarray(obs["pos"], float) #
        cur_quat = np.asarray(obs["quat"], float) #
        gates_quat = np.asarray(obs.get("gates_quat", None), float) if "gates_quat" in obs else None
        
        
        # ( "向前看" 的逻辑保持不变 )
        next_gate_pos = None
        next_gates_quat = None
        if gate_idx + 1 < len(gates_pos):
            next_gate_pos = gates_pos[gate_idx + 1]
            if gates_quat is not None:
                next_gates_quat = gates_quat[gate_idx + 1]


        # local ref (pre->gate->post)
        # (调用 _build_local_ref (v4.2) 的逻辑)
        pos_ref, vel_ref, yaw_ref = self._build_local_ref(
            cur_pos=cur_pos,
            goal_pos=goal_pos,
            cur_quat=cur_quat,
            all_obs_pos=all_obs_pos,    # <---
            r_safe_ref=r_safe_for_ref,  # <---
            gates_pos=gates_pos,
            gate_idx=gate_idx,
            v_des=1.2,
            d_pre=0.5,
            d_post=1.5,
            gates_quat=gates_quat,
            next_gate_pos=next_gate_pos,        # <---
            next_gates_quat=next_gates_quat,    # <---
        )

        # ... (函数的其余部分保持不变) ...
        # store for visualization
        self._last_plan = pos_ref.copy() #
        self._traj_hist.append(cur_pos.reshape(3)) #

        # fill yrefs
        yref = np.zeros((self._N, self._ny)) #
        yref[:, 0:3] = pos_ref[: self._N] #
        yref[:, 5] = yaw_ref[: self._N]        # yaw
        yref[:, 6:9] = vel_ref[: self._N]      # vel
        yref[:, 15] = self._hover_thrust()     # thrust bias

        for j in range(self._N):
            self._solver.set(j, "yref", yref[j]) #

        yref_e = np.zeros(self._ny_e) #
        yref_e[0:3] = pos_ref[self._N] #
        yref_e[5] = yaw_ref[self._N] #
        yref_e[6:9] = vel_ref[self._N] #
        self._solver.set(self._N, "y_ref", yref_e)  # <- correct key

        # solve
        self._solver.solve() #
        pred = np.zeros((self._N + 1, 3))
        for j in range(self._N + 1):
            xj = self._solver.get(j, "x")   # x = [pos(3), rpy(3), vel(3), drpy(3)]
            pred[j] = xj[0:3]
        self._last_plan = pred
        u0 = self._solver.get(0, "u") #
        return u0
    
    # ---------- callbacks ----------
    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        # Could early-stop when all gates passed; keep simple here.
        return False

    def episode_callback(self):
        # cleanup between episodes
        self._traj_hist.clear()
        self._last_plan = None

    def get_debug_lines(self):
        """
        Returns a list of tuples: (points(N,3), rgba(4,), min_size, max_size).
        - History (blue)
        - Current plan (red)
        """
        out = []
        if len(self._traj_hist) >= 2:
            traj = np.asarray(self._traj_hist, float)
            out.append((traj, np.array([0.1, 0.3, 1.0, 0.9]), 2.5, 2.5))  # 蓝：历史
        if self._last_plan is not None and self._last_plan.shape[0] >= 2:
            out.append((self._last_plan, np.array([1.0, 0.1, 0.1, 0.95]), 3.0, 3.0))  # 红：预测
        if getattr(self, "_last_polyline", None) is not None and self._last_polyline.shape[0] >= 2:
            out.append((self._last_polyline, np.array([1.0, 0.9, 0.1, 0.95]), 3.0, 3.0))  # 黄：折线
        return out