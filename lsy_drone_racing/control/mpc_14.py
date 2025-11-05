# trajectory aims to next gate
"""
try_mpc9_deque_path.py — Attitude NMPC with soft obstacle avoidance, 
and a PRE-COMPUTED global reference path, tracked with a 
SMOOTHLY RECEDING (deque) polyline.

[MODIFIED: 
 1. [USER REQ] Added _build_global_path() to pre-compute all pre/post points
    at __init__ using the (G_k - post_{k-1}) logic.
 2. [USER REQ] _build_local_ref is now simplified to just *sample* from this
    fixed global path, starting from cur_pos.
 3. [USER REQ] Dynamic 'post_pt' obstacle pushing is REMOVED. All obstacle
    avoidance is now handled *only* by the MPC solver's soft constraints.
 4. [FIX] Corrected 'selfD' typo to 'self._N'.
 5. [MODIFIED v6.0] Implemented a true receding horizon reference (deque).
    - compute_control() now "consumes" waypoints from the deque
      and adds new ones from the global path.
    - _build_local_ref() is simplified to just sample from cur_pos
      along the current state of the deque.
    - 'gate_idx' is no longer used for path generation.
 6. [MODIFIED v7.5 - User Req] _build_global_path logic simplified:
    - [REMOVED] All rotational avoidance from global path. Avoidance is now
      100% handled by the MPC solver.
    - [KEPT] Added custom "climb point" between post-G2(idx 2) and pre-G3(idx 3).
 7. [MODIFIED v6.8 - User Req] Tuned MPC to fix droop:
    - Increased Q-weight for Z-position in create_ocp_solver.
 8. [FIX v6.5] Corrected typo 'pos_f'/'self_traj_hist' in compute_control.
 9. [MODIFIED v7.2] Tuned d_pre/d_post lists specifically for the 
    4-gate track in 'level2.toml'.
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
from collections import deque # [MODIFIED v6.0] Import deque
from scipy.spatial.transform import Rotation as R
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


# -----------------------------
# ACADOS model & OCP definition
# -----------------------------
from casadi import MX, vertcat, sqrt, log, exp

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
    model.name = "att_mpc_lvl2_softobs_band"  # 改名以免覆盖旧 JSON
    model.f_expl_expr = X_dot
    model.x = X
    model.u = U

    # ---- 参数布局：obs(4*3) + gate(11) + x_ref(nx) + u_ref(nu) ----
    n_obs = 4
    obs_dim = n_obs * 3
    gate_dim = 11  # [gx,gy,gz, tx,ty, nx,ny, band_lo, band_hi, band_alpha, act_s]

    nx = X.rows()
    nu = U.rows()
    p_dim = obs_dim + gate_dim + nx + nu
    p = MX.sym("p", p_dim)
    model.p = p

    # ---- 软障碍：保持你原本的 XY d^2 形式 ----
    x_pos, y_pos, z_pos = X[0], X[1], X[2]
    h_list = []
    for i in range(n_obs):
        ox = p[3*i + 0]
        oy = p[3*i + 1]
        d2 = (x_pos - ox)**2 + (y_pos - oy)**2
        h_list.append(d2)
    model.con_h_expr = vertcat(*h_list)

    # ---- 解包 gate 几何 + band 参数 ----
    off = obs_dim
    gx = p[off+0]; gy = p[off+1]; gz = p[off+2]
    tx = p[off+3]; ty = p[off+4]
    nxg = p[off+5]; nyg = p[off+6]
    band_lo    = p[off+7]
    band_hi    = p[off+8]
    band_alpha = p[off+9]
    act_s      = p[off+10]

    # ---- x_ref / u_ref ----
    xref_off = obs_dim + gate_dim
    x_ref = p[xref_off : xref_off + nx]
    u_ref = p[xref_off + nx : xref_off + nx + nu]

    # ---- 相对门几何量 ----
    dx = x_pos - gx
    dy = y_pos - gy
    dz = z_pos - gz

    s   = dx*nxg + dy*nyg                 # 法向（靠近门平面）
    lat = dx*tx  + dy*ty                  # 横向
    abslat = sqrt(lat*lat + 1e-9)
    absdz  = sqrt(dz*dz   + 1e-9)

    # alpha_s = 1.0 / (1.0 + (s/act_s)**2)  # 只在门平面附近激活
    # 以门中心的XY半径rho激活（不再仅限门平面附近）
    rho = sqrt(s*s + lat*lat + 1e-9)
    alpha_s = 1.0 / (1.0 + (rho / act_s) ** 2)


    def softplus(x):
        return log(1 + exp(x))

    def bump(r, lo, hi, k):
        sp1 = softplus(k*(r - lo)) / k   # ≈ max(0, r - lo)
        sp2 = softplus(k*(hi - r)) / k   # ≈ max(0, hi - r)
        return sp1 * sp2

    pen_lat = bump(abslat, band_lo, band_hi, band_alpha)
    pen_z   = bump(absdz , band_lo, band_hi, band_alpha)

    # ---- EXTERNAL 成本：跟踪 + 禁行带 ----
    # 复用你的思想：Z 权重大（2500），其余近似你原来的 Q/R
    Q_diag = [100, 100, 2500, 5, 5, 5, 1, 1, 1, 5, 5, 5]
    R_diag = [0.5, 0.5, 0.5, 10.0]

    dx_vec = X - x_ref
    du_vec = U - u_ref

    track_cost = 0
    for i in range(nx):
        track_cost += Q_diag[i] * dx_vec[i]*dx_vec[i]
    for i in range(nu):
        track_cost += R_diag[i] * du_vec[i]*du_vec[i]

    w_lat, w_z = 600.0, 2000.0
    # ext_cost = track_cost + alpha_s * (w_lat*pen_lat + w_z*pen_z)
    
    def bump(r, lo, hi, k):
        sp1 = log(1 + exp(k*(r - lo))) / k
        sp2 = log(1 + exp(k*(hi - r))) / k
        return sp1 * sp2

    pen_lat = bump(abslat, band_lo, band_hi, band_alpha)
    pen_z   = bump(absdz , band_lo, band_hi, band_alpha)

    ext_cost = track_cost + alpha_s * (w_lat * pen_lat + w_z * pen_z)


    model.cost_expr_ext_cost   = ext_cost
    model.cost_expr_ext_cost_0 = ext_cost

    term_cost = 0
    for i in range(nx):
        term_cost += dx_vec[i]*dx_vec[i]
    model.cost_expr_ext_cost_e = term_cost

    return model

def create_ocp_solver(Tf: float, N: int, parameters: dict, verbose: bool = False):
    ocp = AcadosOcp()
    ocp.model = create_acados_model(parameters)

    ocp.dims.N = N
    n_obs = 4

    nx = ocp.model.x.rows()
    nu = ocp.model.u.rows()

    # ---- EXTERNAL 成本 ----
    ocp.cost.cost_type   = "EXTERNAL"
    ocp.cost.cost_type_0 = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"

    # ---- 状态/输入约束（照旧）----
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

    # ---- 软障碍（照旧）----
    r_safe = 0.25
    BIG = 1e9
    ocp.dims.nh  = n_obs
    ocp.dims.nsh = n_obs
    ocp.constraints.lh = np.ones(n_obs) * (r_safe ** 2)
    ocp.constraints.uh = np.ones(n_obs) * BIG
    ocp.constraints.idxsh = np.arange(n_obs, dtype=int)
    slack_w_lin, slack_w_quad = 5e2, 8e3
    ocp.cost.zl = slack_w_lin  * np.ones(n_obs)
    ocp.cost.zu = slack_w_lin  * np.ones(n_obs)
    ocp.cost.Zl = slack_w_quad * np.ones(n_obs)
    ocp.cost.Zu = slack_w_quad * np.ones(n_obs)

    # ---- solver 选项（照旧）----
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.tf = Tf

    # ---- 参数向量长度：obs(4*3)+gate(11)+x_ref(nx)+u_ref(nu) ----
    p_dim = n_obs * 3 + 11 + nx + nu
    ocp.parameter_values = np.zeros(p_dim)

    json_name = "att_mpc_lvl2_softobs_band"
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

        self._N = 85
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
        self._traj_hist = deque(maxlen=4000)  # world-frame drone positions over episode
        self._last_plan = None                # (N+1,3) current reference positions
        self._last_polyline = None   # 存最近一次构造的折线拐点 (M,3)

        # --- [MODIFIED v6.0] Global Path & Deque Init ---
        
        # [NEW v6.7] Per-gate pre/post distances
        num_gates_in_obs = len(obs["gates_pos"])
        
        # --- [MODIFIED v7.2] 针对 'level2.toml' 中的 4 个门进行调优 ---
        print(f"[AttitudeMPC] Loading v7.5 path logic (tuned for level2.toml 4-gate track).")
        self.d_pre_list = np.array([
            0.45,  # G0 (idx 0): 正常
            0.50,  # G1 (idx 1): (G0->G1 距离 < 0.8m, 必须缩短)
            0.45,  # G2 (idx 2): 正常
            0.30,  # G3 (idx 3): (G2->G3 距离 < 1.2m, 缩短)
        ])
        self.d_post_list = np.array([
            0.80,  # G0 (idx 0): (G0->G1 距离 < 0.8m, 必须缩短)
            1.20,  # G1 (idx 1): (G1->G2 距离长)
            0.50,  # G2 (idx 2): (G2->G3 距离 < 1.2m, 缩短)
            1.20,  # G3 (idx 3): 最后一个门
        ])
        
        # 确保列表长度正确 (如果列表太短，用最后一个值填充)
        if len(self.d_pre_list) < num_gates_in_obs:
            print(f"[AttitudeMPC] WARN: d_pre_list is shorter than num_gates ({num_gates_in_obs}). Padding with last value.")
            self.d_pre_list = np.pad(self.d_pre_list, (0, num_gates_in_obs - len(self.d_pre_list)), 'edge')
        if len(self.d_post_list) < num_gates_in_obs:
            print(f"[AttitudeMPC] WARN: d_post_list is shorter than num_gates ({num_gates_in_obs}). Padding with last value.")
            self.d_post_list = np.pad(self.d_post_list, (0, num_gates_in_obs - len(self.d_post_list)), 'edge')

        
        # [MODIFIED v7.5] 移除了所有全局路径避障/旋转参数
        # _obs_safe_radius, _obs_rotate_..._deg, _gate..._rot_..._cw

        # Stores [start_pos, pre0, G0, post0, pre1, G1, post1, ...]
        self._global_waypoints = None 
        self._build_global_path(obs) # MUST be called first
        
        # This is the "active yellow line"
        self._deque_len = 6 # 黄线的航点数 (e.g., pre1, G1, post1, pre2, G2, post2)
        self._local_polyline_deque = deque(maxlen=self._deque_len)
        
        # This is the index of the *next* point to *add* from the global path
        self._current_global_idx = 0
        
        # This is the 2D distance to "consume" a point
        self._consumption_dist = 0.3 # (40 cm)
        
        # Initialize the deque
        self._reset_deque()
        # --- [MODIFIED END] ---


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
    
    # [REMOVED v7.5] _rotate_xy_vector_clockwise 不再需要

    # [MODIFIED v7.5] 替换 _build_global_path
    def _build_global_path(self, initial_obs: dict):
        """
        [MODIFIED v7.5] Pre-computes all pre/post points with custom logic:
        - Path: [0, 1, 2, 3, 4, ...] (Standard sequence)
        - [REMOVED] All rotational avoidance. Path is "naive", MPC handles all avoidance.
        - [KEPT] Added custom "climb point" between post-G2(idx 2) and pre-G3(idx 3).
        - [KEPT] Uses self.d_pre_list[idx] and self.d_post_list[idx] for lengths.
        """
        print("[AttitudeMPC] Pre-computing global reference path (v7.5 - naive path, climb point only)...")
        gates_pos = np.asarray(initial_obs["gates_pos"], float)
        gates_quat = np.asarray(initial_obs.get("gates_quat", None), float) \
                     if "gates_quat" in initial_obs else None
        initial_pos = np.asarray(initial_obs["pos"], float)
        
        # [REMOVED v7.5] 障碍物在全局路径规划中不再需要
        # obstacles_pos = ...
        # has_obstacles = ...

        all_waypoints = [initial_pos]
        last_post_pt = initial_pos.copy() # Start with drone pos for G0 logic

        num_gates = len(gates_pos)
        
        gate_sequence = list(range(num_gates))
        print(f"[AttitudeMPC] Using standard gate sequence: {gate_sequence}")
        
        for gate_idx in gate_sequence:
            g_c = gates_pos[gate_idx]
            g_quat = gates_quat[gate_idx] if gates_quat is not None else None
            
            # --- 1. Calculate n_xy using (G_k - post_{k-1}) logic (不变) ---
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
            
            # --- 2. [MODIFIED v7.5] 动态计算 Pre-Point (无避障) ---
            pre_pt = g_c.copy()
            current_d_pre = self.d_pre_list[gate_idx]
            base_v_pre_xy = -current_d_pre * n_xy
            pre_pt[:2] += base_v_pre_xy # 直接应用，不检查

            # [KEPT] 在 G2(idx 2) 和 G3(idx 3) 之间添加爬升点
            if gate_idx == 3: # 当我们处理 G4(idx 3) 时...
                print(f"[AttitudeMPC] G{gate_idx}: Adding custom climb point after post2.")
                post2 = last_post_pt # 这是 G3(idx 2) 的 post 点
                pre3 = pre_pt      # 这是 G4(idx 3) 刚计算出的 pre 点
                
                # 3D 向量 (pre3 - post2)
                vec_to_pre3 = pre3 - post2
                norm_vec = np.linalg.norm(vec_to_pre3)
                
                climb_pt = post2.copy()
                if norm_vec > 1e-6:
                    unit_vec = vec_to_pre3 / norm_vec
                    # 沿 3D 向量方向前进 0.1 米
                    climb_pt += 0.5 * unit_vec 
                
                # 在计算出的新位置上“上方1米”
                climb_pt[2] += 0.2 
                
                all_waypoints.append(climb_pt)
                
                print(f"[AttitudeMPC] ... Climb point inserted at {climb_pt}")


            # --- 3. [MODIFIED v7.5] 动态计算 Post-Point (无避障) ---
            post_pt = g_c.copy()
            current_d_post = self.d_post_list[gate_idx]
            base_v_post_xy = current_d_post * n_xy
            post_pt[:2] += base_v_post_xy # 直接应用，不检查

            # --- 4. 存储航点并更新 last_post_pt ---
            all_waypoints.append(pre_pt)
            all_waypoints.append(g_c)
            all_waypoints.append(post_pt)

            # 必须使用 *最终* 计算出的安全的 post_pt
            last_post_pt = post_pt.copy()
            if gate_idx == 3:
                extra_posts = 6           # 需要追加的 post 数量
                tail_step = 0.4           # 每个额外 post 的步长（米），可按需调 0.2~0.6
                for k in range(1, extra_posts + 1):
                    p = post_pt.copy()
                    p[:2] += k * tail_step * n_xy
                    all_waypoints.append(p)
                print(f"[AttitudeMPC] Appended {extra_posts} extra post points after last gate with step={tail_step} m.")
            
        self._global_waypoints = np.vstack(all_waypoints)
        self._start_idx = 1
        print(f"[AttitudeMPC] Global path with {len(self._global_waypoints)} points built (v7.5 - naive, tuned dist).")
    
    
    # [MODIFIED v6.0] New function to reset the deque
    def _reset_deque(self):
        """Fills the local polyline deque from the global path."""
        self._local_polyline_deque.clear()
        
        # Start targeting the first *real* waypoint (e.g., pre0)
        self._current_global_idx = self._start_idx 
        
        idx = self._current_global_idx
        while len(self._local_polyline_deque) < self._deque_len:
            if idx < len(self._global_waypoints):
                self._local_polyline_deque.append(self._global_waypoints[idx])
            else:
                # Path is shorter than deque, just append last point
                self._local_polyline_deque.append(self._global_waypoints[-1])
            idx += 1
        
        # Update the index to point to the *next* point to be added
        self._current_global_idx = idx


    
    def _build_local_ref(
        self,
        cur_pos: np.ndarray,
        goal_pos: np.ndarray, # Still needed for hover logic
        cur_quat: np.ndarray,
        v_des: float = 0.8,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        [MODIFIED v6.0 - Deque Sampling]
        Constructs a local reference by sampling from cur_pos along
        the *current state* of self._local_polyline_deque.
        
        - 'gate_idx' is no longer used.
        - Deque update logic lives in compute_control().
        """
        T = self._T_HORIZON 
        N = self._N 
        dt = self._dt 

        cur_pos = np.asarray(cur_pos, float).reshape(3) 

        # --- 1. [MODIFIED] Build waypoints from deque ---
        
        # Convert deque to a list and prepend cur_pos
        active_waypoints = list(self._local_polyline_deque)
        base_waypoints = [cur_pos] + active_waypoints
        
        waypoints = np.vstack(base_waypoints) 
        # --- (黄线构建结束) ---

        self._last_polyline = waypoints.copy() #

        # --- 2. 采样与鲁棒 Yaw (逻辑保持不变) ---
        seg_vecs = waypoints[1:] - waypoints[:-1]
        seg_lens = np.linalg.norm(seg_vecs, axis=1)
        cum_lens = np.concatenate([[0.0], np.cumsum(seg_lens)])
        total_len = cum_lens[-1]

        if total_len < 1e-6:
            # 路径太短，悬停在目标门 (use goal_pos as fallback)
            gate_c = np.asarray(goal_pos, float).reshape(3)
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
    

    def _gate_xy_dir(self, i: int, cur_pos: np.ndarray,
                    gates_pos: np.ndarray, gates_quat: np.ndarray | None) -> np.ndarray:
        """返回门 i 的 XY 法向 n_xy（指向赛道前进方向），四元数可选；否则用赛道走向。"""
        eps = 1e-12
        gi = np.asarray(gates_pos[i], float).reshape(3)

        # 赛道前进向量
        if i + 1 < len(gates_pos):
            v_prog = np.asarray(gates_pos[i + 1], float).reshape(3) - gi
        elif i - 1 >= 0:
            v_prog = gi - np.asarray(gates_pos[i - 1], float).reshape(3)
        else:
            v_prog = gi - np.asarray(cur_pos, float).reshape(3)
        v_prog_xy = v_prog[:2]
        if np.linalg.norm(v_prog_xy) < eps:
            v_prog_xy = np.array([1.0, 0.0])
        v_prog_xy /= (np.linalg.norm(v_prog_xy) + eps)

        # 优先用 gate 四元数的 +X 投影（两种排列择优）
        n_xy = None
        if gates_quat is not None:
            q = np.asarray(gates_quat[i], float).reshape(4)
            try:
                fwd1 = R.from_quat(q).apply([1.0, 0.0, 0.0])[:2]
            except Exception:
                fwd1 = np.array([1.0, 0.0])
            if np.linalg.norm(fwd1) < eps: fwd1 = np.array([1.0, 0.0])
            fwd1 /= (np.linalg.norm(fwd1) + eps)

            q_wxyz = np.array([q[1], q[2], q[3], q[0]])
            try:
                fwd2 = R.from_quat(q_wxyz).apply([1.0, 0.0, 0.0])[:2]
            except Exception:
                fwd2 = np.array([1.0, 0.0])
            if np.linalg.norm(fwd2) < eps: fwd2 = np.array([1.0, 0.0])
            fwd2 /= (np.linalg.norm(fwd2) + eps)

            n_xy = fwd1 if abs(float(np.dot(fwd1, v_prog_xy))) >= abs(float(np.dot(fwd2, v_prog_xy))) else fwd2
        if n_xy is None:
            n_xy = v_prog_xy.copy()
        if float(np.dot(n_xy, v_prog_xy)) < 0.0:
            n_xy = -n_xy
        return n_xy

    
    
    # ---------- debug visualization (matplotlib) ----------
    def debug_plot(self, obs: dict):
        import matplotlib.pyplot as plt
        cur_pos = np.asarray(obs["pos"], float)
        gates_pos = np.asarray(obs["gates_pos"], float)
        cur_quat = np.asarray(obs["quat"], float)
        # gates_quat = np.asarray(obs.get("gates_quat", None), float) if "gates_quat" in obs else None
        
        # [MODIFIED v6.0] 'gate_idx' is no longer needed for path gen,
        # but we need 'goal_pos' for the hover fallback case.
        gate_idx = int(np.asarray(obs["target_gate"]).item())
        
        # [MODIFIED v6.3] 确保 goal_pos 是一个有效的门索引
        if gate_idx < 0 or gate_idx >= len(gates_pos):
            goal_pos = gates_pos[-1]
        else:
            goal_pos = gates_pos[gate_idx] 

        obstacles_pos = np.asarray(obs["obstacles_pos"], float).reshape(-1, 3)

        # --- [MODIFIED] Call new _build_local_ref signature ---
        pos_ref, vel_ref, yaw_ref = self._build_local_ref(
            cur_pos=cur_pos, 
            goal_pos=goal_pos, # Pass fallback goal
            cur_quat=cur_quat,
            v_des=1.1 # (from user file)
        )

        fig, ax = plt.subplots(figsize=(6, 6))
        
        # [NEW] Plot the *full* global path
        if self._global_waypoints is not None:
             ax.plot(self._global_waypoints[:, 0], self._global_waypoints[:, 1], 
                     'x--', color='gray', label="Global Path (Fixed)")

        ax.plot(pos_ref[:, 0], pos_ref[:, 1], '-', label="MPC ref path (Sampled)")
        ax.scatter(cur_pos[0], cur_pos[1], c='blue', marker='o', s=80, label="drone")
        ax.scatter(gates_pos[:, 0], gates_pos[:, 1], c='green', marker='s', s=40, label="gates")
        for i, gp in enumerate(gates_pos):
            ax.text(gp[0], gp[1], f"G{i}", color='green', fontsize=8)
        if obstacles_pos.size > 0:
            ax.scatter(obstacles_pos[:, 0], obstacles_pos[:, 1], c='red', marker='x', s=80, label="obstacles")
            
            # [MODIFIED v6.2] Plot *both* safe radii
            r_safe_solver = 0.33 # (这匹配 create_ocp_solver)
            
            # [MODIFIED v7.5] _obs_safe_radius 已移除, 但我们仍然可以绘制 MPC 的半径
            # r_safe_path = self._obs_safe_radius 
            
            for (ox, oy, _oz) in obstacles_pos:
                circ1 = plt.Circle((ox, oy), r_safe_solver, fill=False, linestyle='--', color='red', alpha=0.5, label="MPC r_safe (0.33)")
                # circ2 = plt.Circle((ox, oy), r_safe_path, fill=False, linestyle=':', color='magenta', alpha=0.7, label=f"Path r_safe ({r_safe_path})")
                ax.add_patch(circ1)
                # ax.add_patch(circ2)
        
        if self._last_polyline is not None and self._last_polyline.shape[0] >= 2:
            wp = self._last_polyline
            ax.plot(wp[:, 0], wp[:, 1], 'o-', linewidth=2.0, label="polyline (黄线-当前)")
            
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.grid(True); ax.legend()
        ax.set_title(f"Local ref from Deque (targeting wp {self._current_global_idx-self._deque_len})")
        plt.show()
    
    
    # ---------- main MPC step ----------
    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        
        cur_pos = np.asarray(obs["pos"], float) # [MODIFIED v6.0] Get pos early
        
        # augment obs with rpy, drpy
        obs["rpy"] = R.from_quat(obs["quat"]).as_euler("xyz")
        obs["drpy"] = ang_vel2rpy_rates(obs["quat"], obs["ang_vel"])
        x0 = np.concatenate((obs["pos"], obs["rpy"], obs["vel"], obs["drpy"]))
        self._solver.set(0, "lbx", x0)
        self._solver.set(0, "ubx", x0)

        # --- [MODIFIED v6.0] Deque Update Logic ---
        if len(self._local_polyline_deque) > 0:
            # 1. 检查是否 "吃掉" 了黄线的第一个点
            target_pt = self._local_polyline_deque[0]
            dist_to_target_xy = np.linalg.norm(cur_pos[:2] - target_pt[:2])
            
            if dist_to_target_xy < self._consumption_dist:
                # 2. "吃掉" 点 (从左侧 pop)
                self._local_polyline_deque.popleft()
                
                # 3. "长出" 新的点 (从右侧 append)
                next_idx = self._current_global_idx
                if next_idx < len(self._global_waypoints):
                    new_pt = self._global_waypoints[next_idx]
                    self._local_polyline_deque.append(new_pt)
                    self._current_global_idx += 1
                else:
                    # 已经到了全局路径的尽头，重复添加最后一个点
                    last_pt = self._global_waypoints[-1]
                    self._local_polyline_deque.append(last_pt)
        # --- [MODIFIED END] ---


        # set obstacle params p for all shooting nodes [鲁棒版本]
        # (这部分逻辑保持不变 - MPC求解器 *仍然* 需要知道障碍物在哪里)
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

        # p_vec = selected_obs_pos.flatten()
        # for j in range(self._N + 1):
        #     self._solver.set(j, "p", p_vec)
        # --- 障碍物逻辑结束 ---


        # gate references
        # [MODIFIED v6.0] 'gate_idx' is no longer primary,
        # but we need 'goal_pos' for the hover fallback logic in _build_local_ref
        gate_idx = int(np.asarray(obs["target_gate"]).item()) 
        gates_pos = np.asarray(obs["gates_pos"], float) 
        
        # [MODIFIED v6.3] 确保 goal_pos 是一个有效的门索引
        if gate_idx < 0 or gate_idx >= len(gates_pos):
            # 如果无人机完成了比赛 (target_gate == -1)
            # 或者索引无效, 就把最后一个门作为目标
            goal_pos = gates_pos[-1]
        else:
            goal_pos = gates_pos[gate_idx] 

        cur_quat = np.asarray(obs["quat"], float) 
        
        # --- [MODIFIED] Call new _build_local_ref signature ---
        # local ref (sampled from global path)
        pos_ref, vel_ref, yaw_ref = self._build_local_ref(
            cur_pos=cur_pos,
            goal_pos=goal_pos, # Pass fallback goal
            cur_quat=cur_quat,
            v_des=0.5, # (from user file)
        )

        # ... (函数的其余部分保持不变) ...
        # store for visualization
        
        # [FIXED v6.5] 修复拼写错误 pos_f -> pos_ref
        self._last_plan = pos_ref.copy() #
        # [FIXED v6.5] 修复拼写错误 self_traj_hist -> self._traj_hist
        self._traj_hist.append(cur_pos.reshape(3)) #

        # # fill yrefs
        # yref = np.zeros((self._N, self._ny)) #
        # yref[:, 0:3] = pos_ref[: self._N] #
        # yref[:, 5] = yaw_ref[: self._N]        # yaw
        # yref[:, 6:9] = vel_ref[: self._N]      # vel
        # yref[:, 15] = self._hover_thrust()     # thrust bias

        # for j in range(self._N):
        #     self._solver.set(j, "yref", yref[j]) #

        # yref_e = np.zeros(self._ny_e) #
        # yref_e[0:3] = pos_ref[self._N] #
        # yref_e[5] = yaw_ref[self._N] #
        # yref_e[6:9] = vel_ref[self._N] #
        # self._solver.set(self._N, "y_ref", yref_e)  # <- correct key
        
        # ---------- forbidden band 参数（可放到 __init__ 作为成员） ----------
        # ---------- forbidden band 参数（与 try_mpc7 一致） ----------
        band_lo, band_hi = 0.30, 0.80
        band_alpha, act_s = 20.0, 0.50

        gate_idx = int(np.asarray(obs["target_gate"]).item())
        gates_pos = np.asarray(obs["gates_pos"], float)
        gates_quat = np.asarray(obs.get("gates_quat", None), float) if "gates_quat" in obs else None
        goal_pos = gates_pos[-1] if (gate_idx < 0 or gate_idx >= len(gates_pos)) else gates_pos[gate_idx]


        # 选“禁行带所作用的门”= 离当前飞机最近的门平面（按 |s| 最小）
        cur_xy = cur_pos[:2]

        # 1) 选择“禁行带作用的门”为：XY 平面里距离当前飞机最近的门中心
        dists = np.linalg.norm(gates_pos[:, :2] - cur_xy[None, :], axis=1)
        best_i = int(np.argmin(dists))

        # 2) 用这扇门的法向作为禁行带坐标系
        n_xy = self._gate_xy_dir(best_i, cur_pos, gates_pos, gates_quat)
        n_xy = n_xy / (np.linalg.norm(n_xy) + 1e-12)
        t_xy = np.array([-n_xy[1], n_xy[0]])
        t_xy = t_xy / (np.linalg.norm(t_xy) + 1e-12)

        g_band = gates_pos[best_i]
        gx, gy, gz = float(g_band[0]), float(g_band[1]), float(g_band[2])


        n_obs = 4
        nx, nu = self._nx, self._nu
        p_len = n_obs*3 + 11 + nx + nu

        obs_flat = selected_obs_pos.flatten()  # 用最近四障碍

        for j in range(self._N + 1):
            jj = min(j, self._N)
            xr = np.zeros(nx)
            xr[0:3] = pos_ref[jj]
            xr[5]   = yaw_ref[jj]
            xr[6:9] = vel_ref[jj]

            ur = np.zeros(nu)
            ur[3] = self._hover_thrust()

            p_full = np.zeros(p_len)
            off = 0
            p_full[off: off + n_obs*3] = obs_flat; off += n_obs*3
            p_full[off + 0: off + 3] = [gx, gy, gz]
            p_full[off + 3: off + 5] = t_xy
            p_full[off + 5: off + 7] = n_xy
            p_full[off + 7] = band_lo
            p_full[off + 8] = band_hi
            p_full[off + 9] = band_alpha
            p_full[off +10] = act_s
            off += 11
            p_full[off: off + nx] = xr; off += nx
            p_full[off: off + nu] = ur; off += nu

            self._solver.set(j, "p", p_full)




        # solve
        self._solver.solve() #
        
        # --- [FIXED] ---
        pred = np.zeros((self._N + 1, 3)) # Was 'selfD'
        # --- [FIXED] ---
        
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
        
        # [MODIFIED v6.0] Reset the deque and global index
        print("[AttitudeMPC] Episode reset. Resetting local deque.")
        self._reset_deque()
        
        # We assume the global path (gates) does not change between episodes.
        # If it did, we would also call self._build_global_path(obs) here.


    def get_debug_lines(self):
        """
        Returns a list of tuples: (points(N,3), rgba(4,), min_size, max_size).
        - History (blue)
        - Current plan (red)
        - Current polyline (yellow)
        - [NEW] Global path (gray)
        """
        out = []
        if len(self._traj_hist) >= 2:
            traj = np.asarray(self._traj_hist, float)
            out.append((traj, np.array([0.1, 0.3, 1.0, 0.9]), 2.5, 2.5))  # 蓝：历史
        if self._last_plan is not None and self._last_plan.shape[0] >= 2:
            out.append((self._last_plan, np.array([1.0, 0.1, 0.1, 0.95]), 3.0, 3.0))  # 红：预测
        if getattr(self, "_last_polyline", None) is not None and self._last_polyline.shape[0] >= 2:
            out.append((self._last_polyline, np.array([1.0, 0.9, 0.1, 0.95]), 3.0, 3.0))  # 黄：折线
        
        # [NEW] Add global path visualization
        if getattr(self, "_global_waypoints", None) is not None:
             out.append((self._global_waypoints, np.array([0.5, 0.5, 0.5, 0.7]), 2.0, 2.0)) # 灰：全局
        return out