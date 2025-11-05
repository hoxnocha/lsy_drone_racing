# -*- coding: utf-8 -*-
"""
try_mpc7.py — Attitude NMPC with soft obstacle avoidance, gate-normal pre/post points,
and MuJoCo visualization hooks.

Drop this file into lsy_drone_racing/control/ and set controller.file = "try_mpc7.py"
in your config.
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
        50.0, 50.0, 400.0,   # pos xyz
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
        100.0, 100.0, 400.0,   # pos xyz
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
    r_safe = 0.2
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

        self._N = 75
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
    
    ''' ---------- build local reference path ----------
    def _build_local_ref(
        self,
        cur_pos: np.ndarray,
        goal_pos: np.ndarray,
        cur_quat: np.ndarray,
        gates_pos: np.ndarray | None = None,
        gate_idx: int | None = None,
        v_des: float = 0.8,
        d_pre: float = 0.8,
        d_post: float = 0.8,
        gates_quat: np.ndarray | None = None,
          # <--- [新添加]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        [鲁棒版本]
        构造一个结构一致的折线参考路径：
        [cur] -> pre(gate) -> gate_center -> [post(gate)]
        
        此版本移除了不稳定的 'if/else' 逻辑，以更好地处理
        由 sensor_range 引起的 '跳变'。
        """
        T = self._T_HORIZON
        N = self._N
        dt = self._dt

        gate_c = np.asarray(goal_pos, float).reshape(3)
        cur_pos = np.asarray(cur_pos, float).reshape(3) # <--- [修改] 确保 cur_pos 也是 np.float

        # 决定放置 pre/post 点的 XY 方向 (此逻辑保持不变)
        use_gate_normal = (gates_quat is not None) and (gate_idx is not None)
        n_xy = None
        is_last_gate = (gates_pos is None) or (gate_idx is None) or (gate_idx + 1 >= len(gates_pos)) # <--- [修改] is_last_gate 定义移到这里

        if use_gate_normal:
            gates_quat = np.asarray(gates_quat, float)
            n_xy = self._gate_forward_xy_from_quat(gates_quat[gate_idx])

            # align with forward track direction if possible
            if gates_pos is not None:
                gates_pos = np.asarray(gates_pos, float)
                vec_to_compare = None # <--- [修改] 简化逻辑
                if not is_last_gate:
                    vec_to_compare = gates_pos[gate_idx + 1][:2] - gate_c[:2]
                else:
                    vec_to_compare = gate_c[:2] - cur_pos[:2] # <--- [修改] 使用 cur_pos 作为参考
                if np.linalg.norm(vec_to_compare) > 1e-9 and np.dot(n_xy, vec_to_compare) < 0:
                    n_xy = -n_xy
        if (n_xy is None) and (gates_pos is not None):
            # fallback: track direction
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

        # avoid d_pre/d_post exceeding neighbor spacing (此逻辑保持不变)
        if gates_pos is not None and gate_idx is not None:
            if gate_idx - 1 >= 0:
                d_prev = np.linalg.norm(gate_c - gates_pos[gate_idx - 1])
                d_pre = min(d_pre, 0.75 * max(d_prev, 1e-6))
            if not is_last_gate: # <--- [修改] 使用 is_last_gate
                d_next = np.linalg.norm(gates_pos[gate_idx + 1] - gate_c)
                d_post = min(d_post, 0.75 * max(d_next, 1e-6))

        pre_pt = gate_c.copy()
        post_pt = gate_c.copy()
        pre_pt[:2]  -= d_pre  * n_xy
        post_pt[:2] += d_post * n_xy
        
        # --- [核心修改] ---
        # 移除所有基于 signed_plane, use_pre, near_gate_center 的 if/else 逻辑
        #
        
        if is_last_gate:
            # 如果是最后一个门，我们不需要 post_pt
            waypoints = np.vstack([
                cur_pos,
                pre_pt,
                gate_c,
            ])
        else:
            # 对于所有其他门，始终使用完整的结构
            waypoints = np.vstack([
                cur_pos,
                pre_pt,
                gate_c,
                post_pt,
            ])

        self._last_polyline = waypoints.copy()
        # --- [核心修改结束] ---


        seg_vecs = waypoints[1:] - waypoints[:-1]
        seg_lens = np.linalg.norm(seg_vecs, axis=1)
        cum_lens = np.concatenate([[0.0], np.cumsum(seg_lens)])
        total_len = cum_lens[-1]

        if total_len < 1e-6:
            # 路径太短，悬停在目标门
            pos_ref = np.repeat(gate_c[None, :], N + 1, axis=0)
            vel_ref = np.zeros_like(pos_ref)
            yaw_ref = np.zeros(N + 1)
            # [鲁棒性修改] 保持当前偏航
            cur_yaw = R.from_quat(cur_quat).as_euler("xyz")[2] 
            yaw_ref[:] = cur_yaw 
            return pos_ref, vel_ref, yaw_ref

        s_max = v_des * T
        # s_samples = np.clip(np.linspace(0.0, s_max, N + 1), 0.0, total_len)
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

        # --- [鲁棒性修改：添加 YAW 稳定性] ---
        unstable_yaw = np.linalg.norm(dpos[:, :2], axis=1) < 0.05
        if np.any(unstable_yaw):
            # 获取当前偏航角
            cur_yaw = R.from_quat(cur_quat).as_euler("xyz")[2]
            last_stable_yaw = cur_yaw
            for k in range(N + 1):
                if unstable_yaw[k]:
                    yaw_ref[k] = last_stable_yaw
                else:
                    last_stable_yaw = yaw_ref[k]
        
        yaw_ref = np.nan_to_num(yaw_ref, nan=0.0)
        # --- [YAW 稳定性修改结束] ---
        
        return pos_ref, vel_ref, yaw_ref
        '''
        
    def _build_local_ref(
        self,
        cur_pos: np.ndarray,
        goal_pos: np.ndarray,
        cur_quat: np.ndarray,
        all_obs_pos: np.ndarray,    # <--- [修改 1/7] 新增参数：障碍物位置
        r_safe_ref: float,          # <--- [修改 2/7] 新增参数：参考路径的安全半径
        gates_pos: np.ndarray | None = None,
        gate_idx: int | None = None,
        v_des: float = 0.8,
        d_pre: float = 0.8,
        d_post: float = 0.8,
        gates_quat: np.ndarray | None = None,
        
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        [鲁棒版本 v3.0]
        ...
        [新功能] 检查 post_pt 是否在障碍物附近，如果是，则移动它。
        """
        T = self._T_HORIZON
        N = self._N
        dt = self._dt

        gate_c = np.asarray(goal_pos, float).reshape(3)
        cur_pos = np.asarray(cur_pos, float).reshape(3) 

        # ... (从第 408 行到 458 行, n_xy 和 d_pre/d_post 的计算逻辑保持不变) ...
        use_gate_normal = (gates_quat is not None) and (gate_idx is not None)
        n_xy = None
        is_last_gate = (gates_pos is None) or (gate_idx is None) or (gate_idx + 1 >= len(gates_pos)) 

        if use_gate_normal:
            gates_quat = np.asarray(gates_quat, float)
            n_xy = self._gate_forward_xy_from_quat(gates_quat[gate_idx])
            if gates_pos is not None:
                gates_pos = np.asarray(gates_pos, float)
                vec_to_compare = None 
                if not is_last_gate:
                    vec_to_compare = gates_pos[gate_idx + 1][:2] - gate_c[:2]
                else:
                    vec_to_compare = gate_c[:2] - cur_pos[:2] 
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
        # --- (以上逻辑保持不变) ---


        pre_pt = gate_c.copy()
        post_pt = gate_c.copy()
        pre_pt[:2]  -= d_pre  * n_xy
        post_pt[:2] += d_post * n_xy
        
        # --- [修改 3/7] 核心修改：检查并移动 post_pt ---
        
        if all_obs_pos.shape[0] > 0 and not is_last_gate:
            # 1. 计算 post_pt (XY平面) 到所有障碍物的距离
            distances_to_post_pt_xy = np.linalg.norm(all_obs_pos[:, :2] - post_pt[:2], axis=1)
            min_dist = np.min(distances_to_post_pt_xy)
            
            # 2. 如果任何一个障碍物太近
            if min_dist < r_safe_ref:
                
                # 3. 移动 post_pt, 而不是删除它
                closest_obs = all_obs_pos[np.argmin(distances_to_post_pt_xy)]
                
                # 4. 计算一个"推开"的方向 (从障碍物 -> 门中心, 这是一个安全的方向)
                push_dir = gate_c[:2] - closest_obs[:2]
                
                # 5. 标准化这个方向
                push_norm = np.linalg.norm(push_dir)
                if push_norm < 1e-6:
                    push_dir_norm = -n_xy # (如果重叠, 则使用门法线的反方向)
                else:
                    push_dir_norm = push_dir / push_norm
                
                # 6. 将 post_pt 移动到障碍物半径之外 (并增加 10cm 缓冲)
                new_post_pt_xy = closest_obs[:2] + push_dir_norm * (r_safe_ref + 0.1)
                
                # 7. 更新 post_pt
                post_pt[:2] = new_post_pt_xy
        
        # [修改 4/7] 
        # 因为 post_pt 现在总是安全的, 我们恢复使用原版的鲁棒逻辑
        if is_last_gate:
            waypoints = np.vstack([
                cur_pos,
                pre_pt,
                gate_c,
            ])
        else:
            # 始终使用 post_pt (因为它现在要么本来就安全, 要么已被移到安全位置)
            waypoints = np.vstack([
                cur_pos,
                pre_pt,
                gate_c,
                post_pt,
            ])
        # --- [核心修改结束] ---

        self._last_polyline = waypoints.copy() 

        # ... (函数的其余部分保持不变, 包括 gamma = 0.6 和鲁棒的 yaw 计算) ...
        #
        seg_vecs = waypoints[1:] - waypoints[:-1]
        seg_lens = np.linalg.norm(seg_vecs, axis=1)
        cum_lens = np.concatenate([[0.0], np.cumsum(seg_lens)])
        total_len = cum_lens[-1]

        if total_len < 1e-6:
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
        import matplotlib.pyplot as plt
        cur_pos = np.asarray(obs["pos"], float)
        gates_pos = np.asarray(obs["gates_pos"], float)
        cur_quat = np.asarray(obs["quat"], float)
        gates_quat = np.asarray(obs.get("gates_quat", None), float) if "gates_quat" in obs else None
        gate_idx = int(np.asarray(obs["target_gate"]).item())
        goal_pos = gates_pos[gate_idx]
        obstacles_pos = np.asarray(obs["obstacles_pos"], float).reshape(-1, 3)

        # <--- [修改] 匹配新的签名 ---
        r_safe_for_ref = 0.40 # (应与 compute_control 中的值匹配)

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
        )

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(pos_ref[:, 0], pos_ref[:, 1], '-', label="MPC ref path")
        ax.scatter(cur_pos[0], cur_pos[1], c='blue', marker='o', s=80, label="drone")
        ax.scatter(gates_pos[:, 0], gates_pos[:, 1], c='green', marker='s', s=40, label="gates")
        for i, gp in enumerate(gates_pos):
            ax.text(gp[0], gp[1], f"G{i}", color='green', fontsize=8)
        if obstacles_pos.size > 0:
            ax.scatter(obstacles_pos[:, 0], obstacles_pos[:, 1], c='red', marker='x', s=80, label="obstacles")
            r_safe = 0.30
            for (ox, oy, _oz) in obstacles_pos:
                circ = plt.Circle((ox, oy), r_safe, fill=False, linestyle='--', color='red', alpha=0.5)
                ax.add_patch(circ)
        if self._last_polyline is not None and self._last_polyline.shape[0] >= 2:
            wp = self._last_polyline
            ax.plot(wp[:, 0], wp[:, 1], 'o-', linewidth=2.0, label="polyline (cur→pre→gate→post)")
            # 可选：把 gate 中心点突出显示（假设它就是 waypoints 里间的一个点）
            ax.scatter(wp[-2, 0], wp[-2, 1], s=90, marker='*', label="gate center")
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.grid(True); ax.legend()
        ax.set_title(f"Local ref to gate {gate_idx}")
        plt.show()
    '''
    
    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        # augment obs with rpy, drpy
        obs["rpy"] = R.from_quat(obs["quat"]).as_euler("xyz")
        obs["drpy"] = ang_vel2rpy_rates(obs["quat"], obs["ang_vel"])
        x0 = np.concatenate((obs["pos"], obs["rpy"], obs["vel"], obs["drpy"]))
        self._solver.set(0, "lbx", x0)
        self._solver.set(0, "ubx", x0)

        
        n_obs = 4
        obs_pos = np.asarray(obs["obstacles_pos"], float).reshape(-1, 3)
        if obs_pos.shape[0] < n_obs:
            pad = np.zeros((n_obs - obs_pos.shape[0], 3))
            obs_pos = np.vstack([obs_pos, pad])
        p_vec = obs_pos[:n_obs].flatten()
        for j in range(self._N + 1):
            self._solver.set(j, "p", p_vec)
        

        # set obstacle params p for all shooting nodes [鲁棒版本]
        n_obs = 4 # 求解器期望的数量
        cur_pos_xy = np.asarray(obs["pos"], float)[:2] # 仅用 XY 平面距离
        all_obs_pos = np.asarray(obs["obstacles_pos"], float).reshape(-1, 3)
        
        selected_obs_pos = np.zeros((n_obs, 3))
        
        if all_obs_pos.shape[0] > 0:
            # 1. 计算到所有障碍物的 XY 距离
            distances_xy = np.linalg.norm(all_obs_pos[:, :2] - cur_pos_xy, axis=1)
            
            # 2. 获取最近的 n_obs 个障碍物的索引
            closest_indices = np.argsort(distances_xy)
            
            # 3. 填充最近的障碍物
            num_to_take = min(all_obs_pos.shape[0], n_obs)
            indices_to_take = closest_indices[:num_to_take]
            selected_obs_pos[:num_to_take] = all_obs_pos[indices_to_take]

            # 4. 如果障碍物总数 < n_obs，用很远的位置填充剩余的槽位
            #    (这比用 [0,0,0] 更安全)
            if num_to_take < n_obs:
                selected_obs_pos[num_to_take:] = np.array([1e6, 1e6, 1e6])
        else:
            # 5. 没有任何障碍物，全部填充为很远的位置
             selected_obs_pos[:] = np.array([1e6, 1e6, 1e6])

        p_vec = selected_obs_pos.flatten()
        for j in range(self._N + 1):
            self._solver.set(j, "p", p_vec)

        # gate references
        gate_idx = int(np.asarray(obs["target_gate"]).item())
        gates_pos = np.asarray(obs["gates_pos"], float)
        goal_pos = gates_pos[gate_idx]
        cur_pos = np.asarray(obs["pos"], float)
        cur_quat = np.asarray(obs["quat"], float)
        gates_quat = np.asarray(obs.get("gates_quat", None), float) if "gates_quat" in obs else None

        # local ref (pre->gate->post)
        pos_ref, vel_ref, yaw_ref = self._build_local_ref(
            cur_pos=cur_pos,
            goal_pos=goal_pos,
            cur_quat=cur_quat,
            gates_pos=gates_pos,
            gate_idx=gate_idx,
            v_des=1.2,
            d_pre=0.5,
            d_post=1.0,
            gates_quat=gates_quat,
        )

        # store for visualization
        self._last_plan = pos_ref.copy()
        self._traj_hist.append(cur_pos.reshape(3))

        # fill yrefs
        yref = np.zeros((self._N, self._ny))
        yref[:, 0:3] = pos_ref[: self._N]
        yref[:, 5] = yaw_ref[: self._N]        # yaw
        yref[:, 6:9] = vel_ref[: self._N]      # vel
        yref[:, 15] = self._hover_thrust()     # thrust bias

        for j in range(self._N):
            self._solver.set(j, "yref", yref[j])

        yref_e = np.zeros(self._ny_e)
        yref_e[0:3] = pos_ref[self._N]
        yref_e[5] = yaw_ref[self._N]
        yref_e[6:9] = vel_ref[self._N]
        self._solver.set(self._N, "y_ref", yref_e)  # <- correct key

        # solve
        self._solver.solve()
        pred = np.zeros((self._N + 1, 3))
        for j in range(self._N + 1):
            xj = self._solver.get(j, "x")   # x = [pos(3), rpy(3), vel(3), drpy(3)]
            pred[j] = xj[0:3]
        self._last_plan = pred
        u0 = self._solver.get(0, "u")
        return u0
      
      
      
      '''
    
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

        
        # [修改 2/3] 删除了旧的、不鲁棒的障碍物设置代码
        # (旧代码:)


        # set obstacle params p for all shooting nodes [鲁棒版本]
        n_obs = 4 # 求解器期望的数量
        cur_pos_xy = np.asarray(obs["pos"], float)[:2] # 仅用 XY 平面距离
        all_obs_pos = np.asarray(obs["obstacles_pos"], float).reshape(-1, 3) #
        
        selected_obs_pos = np.zeros((n_obs, 3))
        
        if all_obs_pos.shape[0] > 0:
            # 1. 计算到所有障碍物的 XY 距离
            distances_xy = np.linalg.norm(all_obs_pos[:, :2] - cur_pos_xy, axis=1)
            
            # ... (鲁棒选择逻辑保持不变) ...
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


        # <--- [修改 3/3] 定义并传递 _build_local_ref 所需的额外参数 ---
        
        # (应大于求解器的 r_safe = 0.35, 留出余量)
        r_safe_for_ref = 0.40 # (这个值必须大于您在步骤1中设置的 r_safe)

        # gate references
        gate_idx = int(np.asarray(obs["target_gate"]).item())
        gates_pos = np.asarray(obs["gates_pos"], float)
        goal_pos = gates_pos[gate_idx]
        cur_pos = np.asarray(obs["pos"], float)
        cur_quat = np.asarray(obs["quat"], float)
        gates_quat = np.asarray(obs.get("gates_quat", None), float) if "gates_quat" in obs else None

        # local ref (pre->gate->post)
        # (这是错误的调用:)
        # (替换为正确的调用)
        pos_ref, vel_ref, yaw_ref = self._build_local_ref(
            cur_pos=cur_pos,
            goal_pos=goal_pos,
            cur_quat=cur_quat,
            all_obs_pos=all_obs_pos,    # <--- 新增
            r_safe_ref=r_safe_for_ref,  # <--- 新增
            gates_pos=gates_pos,
            gate_idx=gate_idx,
            v_des=1.2,
            d_pre=0.5,
            d_post=1.0,
            gates_quat=gates_quat,
        )
        # --- 修改结束 ---

        # store for visualization
        self._last_plan = pos_ref.copy()
        self._traj_hist.append(cur_pos.reshape(3))

        # fill yrefs
        yref = np.zeros((self._N, self._ny))
        yref[:, 0:3] = pos_ref[: self._N]
        yref[:, 5] = yaw_ref[: self._N]        # yaw
        yref[:, 6:9] = vel_ref[: self._N]      # vel
        yref[:, 15] = self._hover_thrust()     # thrust bias

        for j in range(self._N):
            self._solver.set(j, "yref", yref[j])

        yref_e = np.zeros(self._ny_e)
        yref_e[0:3] = pos_ref[self._N]
        yref_e[5] = yaw_ref[self._N]
        yref_e[6:9] = vel_ref[self._N]
        self._solver.set(self._N, "y_ref", yref_e)  # <- correct key

        # solve
        self._solver.solve()
        pred = np.zeros((self._N + 1, 3))
        for j in range(self._N + 1):
            xj = self._solver.get(j, "x")   # x = [pos(3), rpy(3), vel(3), drpy(3)]
            pred[j] = xj[0:3]
        self._last_plan = pred
        u0 = self._solver.get(0, "u")
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

