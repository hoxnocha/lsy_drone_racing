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
 6. [MODIFIED v7.7 - User Req]
    - [NEW] Added dynamic replanning: _build_global_path is re-called if
      new gate/obstacle data is observed (using obs["..._visited"]).
    - [NEW] Added dynamic speed: v_des is now reduced from 1.1m/s to 0.6m/s
      when approaching the target gate.
    - [NEW] Refactored compute_control for readability (_set_acados_parameters).
    - [FIX] _gate_xy_dir now STRICTLY follows the gate's specified (quaternion)
      direction, removing the "auto-flip" (np.dot) logic.
 7. [MODIFIED v6.8 - User Req] Tuned MPC to fix droop:
    - Increased Q-weight for Z-position in create_acados_model (2500.0).
 8. [MODIFIED v7.2] Tuned d_pre/d_post lists specifically for the 
    4-gate track in 'level2.toml'.
 9. [MODIFIED v7.5] Global path is "naive" (no self-avoidance), 
    all avoidance handled by MPC external cost and soft constraints.
]
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import scipy
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from casadi import MX, vertcat, sqrt, log, exp
from scipy.spatial.transform import Rotation as R

from drone_models.core import load_params
from drone_models.so_rpy import symbolic_dynamics_euler
from drone_models.utils.rotation import ang_vel2rpy_rates

from lsy_drone_racing.control import Controller
from collections import deque # [MODIFIED v6.0] Import deque

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
    model.name = "att_mpc_lvl2_softobs_band"  # 改名以免覆盖旧 JSON
    model.f_expl_expr = X_dot
    model.x = X
    model.u = U

    # ---- 参数布局：obs(4*3) + gate(11) + x_ref(nx) + u_ref(nu) ----
    # [COMMENT] 
    # 'p' 是一个巨大的参数向量，我们在 compute_control 中为每个 MPC 步骤设置它。
    # 它包含 MPC 在该步骤需要知道的 *所有* 动态信息。
    n_obs = 4
    obs_dim = n_obs * 3
    gate_dim = 11  # [gx,gy,gz, tx,ty, nx,ny, band_lo, band_hi, band_alpha, act_s]

    nx = X.rows()
    nu = U.rows()
    p_dim = obs_dim + gate_dim + nx + nu
    p = MX.sym("p", p_dim)
    model.p = p

    # ---- 软障碍：保持你原本的 XY d^2 形式 ----
    # [COMMENT] 这是对 'obstacles_pos' 的标准软约束。
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
    gx = p[off+0]; gy = p[off+1]; gz = p[off+2] # 门中心
    tx = p[off+3]; ty = p[off+4]               # 门横向 t_xy
    nxg = p[off+5]; nyg = p[off+6]             # 门法向 n_xy
    band_lo    = p[off+7] # 禁行带内径 (e.g., 0.30m)
    band_hi    = p[off+8] # 禁行带外径 (e.g., 0.80m)
    band_alpha = p[off+9] # "bump" 函数的陡峭度
    act_s      = p[off+10]# 激活距离 (e.g., 0.50m)

    # ---- x_ref / u_ref (来自全局路径规划器) ----
    xref_off = obs_dim + gate_dim
    x_ref = p[xref_off : xref_off + nx]
    u_ref = p[xref_off + nx : xref_off + nx + nu]

    # ---- 相对门几何量 ----
    dx = x_pos - gx
    dy = y_pos - gy
    dz = z_pos - gz

    s   = dx*nxg + dy*nyg                 # 法向距离 (s=0 在门平面上)
    lat = dx*tx  + dy*ty                  # 横向距离 (lat=0 在门中心线上)
    abslat = sqrt(lat*lat + 1e-9)         # |横向|
    absdz  = sqrt(dz*dz   + 1e-9)         # |垂向|

    # [COMMENT] 激活函数：只在门中心 XY 半径 act_s (0.5m) 内激活
    rho = sqrt(s*s + lat*lat + 1e-9)
    alpha_s = 1.0 / (1.0 + (rho / act_s) ** 2)

    # [COMMENT] "Bump" 函数：在 [lo, hi] 区域内产生一个 "驼峰" 惩罚
    # 当 r 位于 lo 和 hi 之间时，pen > 0
    # 当 r 在该区域之外时，pen ≈ 0
    def bump(r, lo, hi, k):
        sp1 = log(1 + exp(k*(r - lo))) / k   # ≈ max(0, r - lo)
        sp2 = log(1 + exp(k*(hi - r))) / k   # ≈ max(0, hi - r)
        return sp1 * sp2

    # [COMMENT] 计算横向和垂向的门框惩罚
    pen_lat = bump(abslat, band_lo, band_hi, band_alpha)
    pen_z   = bump(absdz , band_lo, band_hi, band_alpha)

    # ---- EXTERNAL 成本：跟踪 + 禁行带 ----
    
    # [COMMENT] 1. 跟踪成本 (来自 v6.8 的调优，Z 权重=2500)
    Q_diag = [100, 100, 2500, 5, 5, 5, 1, 1, 1, 5, 5, 5]
    R_diag = [0.5, 0.5, 0.5, 10.0]

    dx_vec = X - x_ref
    du_vec = U - u_ref

    track_cost = 0
    for i in range(nx):
        track_cost += Q_diag[i] * dx_vec[i]*dx_vec[i]
    for i in range(nu):
        track_cost += R_diag[i] * du_vec[i]*du_vec[i]

    # [COMMENT] 2. 门框（禁行带）惩罚成本
    w_lat, w_z = 600.0, 2000.0 # 惩罚权重
    
    # [COMMENT] 总成本 = 跟踪成本 + (激活权重 * 门框惩罚)
    ext_cost = track_cost + alpha_s * (w_lat * pen_lat + w_z * pen_z)

    model.cost_expr_ext_cost   = ext_cost
    model.cost_expr_ext_cost_0 = ext_cost # 起始步也使用此成本

    # [COMMENT] 终端成本仅为位置跟踪
    term_cost = 0
    for i in range(nx): # 仅跟踪状态
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
    # [COMMENT] 这是对障碍物的软约束，与门框的 EXTERNAL 成本分离
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
    p_dim = n_obs*3 + 11 + nx + nu
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
    """
    Attitude NMPC v7.7:
    - EXTERNAL cost for gate frame avoidance ("forbidden band").
    - Soft constraints for obstacle avoidance.
    - Naive global path (deque) with custom climb point.
    - Dynamic replanning based on sensor observations.
    - Dynamic speed control for gate approach.
    - [FIX] Gate direction STRICTLY follows quaternion (no auto-flip).
    """

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

        self._finished = False

        # Visualization caches
        self._traj_hist = deque(maxlen=4000)  # world-frame drone positions over episode
        self._last_plan = None                # (N+1,3) current reference positions
        self._last_polyline = None   # 存最近一次构造的折线拐点 (M,3)

        # --- [NEW v7.6] 动态速度参数 ---
        self._v_des_base = 1.1          # 巡航速度 (m/s)
        self._v_des_gate = 0.6          # 进门速度 (m/s)
        self._v_des_slowdown_dist = 1.5 # 开始减速的距离 (m)

        # --- [NEW v7.6] 动态地图/重规划状态 ---
        self._initial_obs = obs.copy() # 保存初始 obs 以获取 drones[0].pos
        num_gates = len(obs["gates_pos"])
        num_obstacles = len(obs["obstacles_pos"])
        
        self._gate_visited = np.zeros(num_gates, dtype=bool)
        self._obstacle_visited = np.zeros(num_obstacles, dtype=bool)
        
        self._known_gates_pos = np.asarray(obs["gates_pos"], float).copy()
        self._known_obstacles_pos = np.asarray(obs["obstacles_pos"], float).reshape(-1, 3).copy()
        
        self._replan_requested = False 

        # --- [MODIFIED v7.2] 针对 'level2.toml' 中的 4 个门进行调优 ---
        print(f"[AttitudeMPC] Loading v7.7 path logic (tuned for level2.toml 4-gate track).")
        self.d_pre_list = np.array([
            0.45,  # G0 (idx 0): 正常
            0.30,  # G1 (idx 1): (G0->G1 距离 < 0.8m, 必须缩短)
            0.45,  # G2 (idx 2): 正常
            0.45,  # G3 (idx 3): (G2->G3 距离 < 1.2m, 缩短)
        ])
        self.d_post_list = np.array([
            1.20,  # G0 (idx 0): 
            1.20,  # G1 (idx 1): (G1->G2 距离长)
            0.50,  # G2 (idx 2): (G2->G3 距离 < 1.2m, 缩短)
            1.20,  # G3 (idx 3): 最后一个门
        ])
        
        # 确保列表长度正确 (如果列表太短，用最后一个值填充)
        if len(self.d_pre_list) < num_gates:
            print(f"[AttitudeMPC] WARN: d_pre_list is shorter than num_gates ({num_gates}). Padding with last value.")
            self.d_pre_list = np.pad(self.d_pre_list, (0, num_gates - len(self.d_pre_list)), 'edge')
        if len(self.d_post_list) < num_gates:
            print(f"[AttitudeMPC] WARN: d_post_list is shorter than num_gates ({num_gates}). Padding with last value.")
            self.d_post_list = np.pad(self.d_post_list, (0, num_gates - len(self.d_post_list)), 'edge')

        # [REMOVED v7.5] 移除了所有全局路径避障/旋转参数

        # Stores [start_pos, pre0, G0, post0, pre1, G1, post1, ...]
        self._global_waypoints = None 
        self._build_global_path() # [MODIFIED v7.6] 不再需要 obs
        
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
        """计算悬停推力 (标量)。"""
        return float(self.drone_params["mass"]) * (-float(self.drone_params["gravity_vec"][-1]))

    # [FIXED v7.6] _build_global_path 需要这个辅助函数
    def _gate_forward_xy_from_quat(self, q: np.ndarray) -> np.ndarray:
        """Return gate forward direction projected on XY plane (normalized)."""
        # scipy expects [x,y,z,w]
        fwd = R.from_quat(q).apply(np.array([1.0, 0.0, 0.0]))  # gate +X is "front"
        v = fwd[:2]
        n = np.linalg.norm(v)
        if n < 1e-12:
            return np.array([1.0, 0.0])
        return v / n

    # [NEW v7.6] 检查传感器数据并请求重规划
    def _check_and_request_replan(self, obs: dict):
        """
        检查新观测到的物体，更新已知地图，并请求重规划。
        """
        new_gate_obs = np.asarray(obs.get("gate_visited", []), dtype=bool)
        new_obs_obs = np.asarray(obs.get("obstacle_visited", []), dtype=bool)

        # 1. 检查新看到的门
        for i in range(len(new_gate_obs)):
            if new_gate_obs[i] and not self._gate_visited[i]:
                self._gate_visited[i] = True
                self._known_gates_pos[i] = obs["gates_pos"][i]
                self._replan_requested = True
                print(f"[AttitudeMPC] Sensor observed new gate {i} at {self._known_gates_pos[i]}")

        # 2. 检查新看到的障碍物
        for i in range(len(new_obs_obs)):
            if new_obs_obs[i] and not self._obstacle_visited[i]:
                self._obstacle_visited[i] = True
                self._known_obstacles_pos[i] = obs["obstacles_pos"][i]
                self._replan_requested = True
                print(f"[AttitudeMPC] Sensor observed new obstacle {i} at {self._known_obstacles_pos[i]}")

    # [NEW v7.6] 动态速度辅助函数
    def _get_dynamic_v_des(self, cur_pos: np.ndarray, target_gate_idx: int) -> float:
        """
        根据到下一个门的距离计算期望速度。
        """
        # 如果比赛完成或索引无效，使用基础速度
        if target_gate_idx < 0 or target_gate_idx >= len(self._known_gates_pos):
            return self._v_des_base

        target_gate_pos = self._known_gates_pos[target_gate_idx]
        dist_to_gate = np.linalg.norm(cur_pos[:2] - target_gate_pos[:2])

        d = self._v_des_slowdown_dist
        v_base = self._v_des_base
        v_gate = self._v_des_gate

        if dist_to_gate >= d:
            return v_base
        
        # 线性插值: v = v_gate at dist=0, v=v_base at dist=d
        alpha = dist_to_gate / d
        v_des = v_gate + alpha * (v_base - v_gate)
        return max(v_gate, v_des) # 确保速度不低于门速度

    # [MODIFIED v7.6] 替换 _build_global_path
    def _build_global_path(self):
        """
        [MODIFIED v7.7] 
        - 使用 self._known_gates_pos (可变) 和 self._initial_obs (固定)
        - [REMOVED] 移除了所有全局路径避障
        - [KEPT] 保留了 G2/G3 之间的爬升点
        - [FIX] _gate_xy_dir (v7.7) 严格遵循 quat
        """
        print("[AttitudeMPC] Re-building global reference path (v7.7)...")
        
        # [MODIFIED v7.6] 使用成员变量
        gates_pos = self._known_gates_pos
        # Quat 和 Start Pos 总是使用初始观测值
        gates_quat = np.asarray(self._initial_obs.get("gates_quat", None), float) \
                     if "gates_quat" in self._initial_obs else None
        initial_pos = np.asarray(self._initial_obs["pos"], float)
        
        all_waypoints = [initial_pos]
        last_post_pt = initial_pos.copy() # Start with drone pos for G0 logic

        num_gates = len(gates_pos)
        
        gate_sequence = list(range(num_gates))
        print(f"[AttitudeMPC] Using standard gate sequence: {gate_sequence}")
        
        for gate_idx in gate_sequence:
            g_c = gates_pos[gate_idx]
            
            # --- 1. Calculate n_xy (严格遵循 quat) ---
            # [MODIFIED v7.7] 使用 self._gate_xy_dir
            n_xy = self._gate_xy_dir(gate_idx, last_post_pt, gates_pos, gates_quat)
            
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
            
            # [MODIFIED v7.6] 确保 G3 之后的尾巴使用正确的 n_xy
            if gate_idx == 3:
                # [FIX v7.7] n_xy 必须是 G3 的法向
                n_xy_g3 = self._gate_xy_dir(gate_idx, last_post_pt, gates_pos, gates_quat)
                
                extra_posts = 6           # 需要追加的 post 数量
                tail_step = 0.4           # 每个额外 post 的步长（米），可按需调 0.2~0.6
                for k in range(1, extra_posts + 1):
                    p = post_pt.copy()
                    p[:2] += k * tail_step * n_xy_g3
                    all_waypoints.append(p)
                print(f"[AttitudeMPC] Appended {extra_posts} extra post points after last gate with step={tail_step} m.")
            
        self._global_waypoints = np.vstack(all_waypoints)
        self._start_idx = 1
        print(f"[AttitudeMPC] Global path with {len(self._global_waypoints)} points built (v7.7).")
    
    
    # [MODIFIED v6.0] New function to reset the deque
    def _reset_deque(self):
        """Fills the local polyline deque from the global path."""
        print("[AttitudeMPC] Resetting local polyline deque.")
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
        - [MODIFIED v7.6] v_des is now dynamic.
        """
        T = self._T_HORIZON 
        N = self._N 
        dt = self._dt 

        cur_pos = np.asarray(cur_pos, float).reshape(3) 

        # --- 1. [MODIFIED] Build waypoints from deque ---
        active_waypoints = list(self._local_polyline_deque)
        base_waypoints = [cur_pos] + active_waypoints
        waypoints = np.vstack(base_waypoints) 
        self._last_polyline = waypoints.copy()

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

        # [MODIFIED v7.6] v_des (lookahead speed) is now dynamic
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
        vel_ref[-1] = 0.0 # 最后一个速度为 0 (如果 v_des 足够小)

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
    
    # [MODIFIED v7.7] 严格遵循 Quat
    def _gate_xy_dir(self, i: int, cur_pos: np.ndarray,
                    gates_pos: np.ndarray, gates_quat: np.ndarray | None) -> np.ndarray:
        """
        返回门 i 的 XY 法向 n_xy（指向赛道前进方向）。
        [V7.7] 严格优先使用 quat，即使它与赛道流向相反。
        """
        eps = 1e-12
        gi = np.asarray(gates_pos[i], float).reshape(3)

        # 1. 计算赛道前进向量 (用于消歧和后备)
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

        # 2. 尝试从 Quaternion 获取方向
        if gates_quat is not None and i < len(gates_quat):
            q = np.asarray(gates_quat[i], float).reshape(4)
            
            fwd1 = np.array([1.0, 0.0]) # 默认
            try: # [x, y, z, w]
                fwd1_raw = R.from_quat(q).apply([1.0, 0.0, 0.0])[:2]
                if np.linalg.norm(fwd1_raw) > eps:
                    fwd1 = fwd1_raw / np.linalg.norm(fwd1_raw)
            except Exception:
                pass # 保持默认

            fwd2 = np.array([1.0, 0.0]) # 默认
            try: # [w, x, y, z]
                q_wxyz = np.array([q[3], q[0], q[1], q[2]]) 
                fwd2_raw = R.from_quat(q_wxyz).apply([1.0, 0.0, 0.0])[:2]
                if np.linalg.norm(fwd2_raw) > eps:
                    fwd2 = fwd2_raw / np.linalg.norm(fwd2_raw)
            except Exception:
                pass # 保持默认

            # 消歧：选择与赛道流向 *最* 对齐的四元数解释
            if abs(float(np.dot(fwd1, v_prog_xy))) >= abs(float(np.dot(fwd2, v_prog_xy))):
                n_xy = fwd1
            else:
                n_xy = fwd2
            
            # [NEW v7.7] 严格信任 quat。不翻转。
            return n_xy

        # 3. [FALLBACK] 如果没有 quat，使用赛道流向
        # print(f"[AttitudeMPC] G{i} using FALLBACK (progression) direction.")
        return v_prog_xy.copy()

    
    
    # ---------- debug visualization (matplotlib) ----------
    def debug_plot(self, obs: dict):
        import matplotlib.pyplot as plt
        cur_pos = np.asarray(obs["pos"], float)
        # [MODIFIED v7.6] 使用已知的门/障碍物
        gates_pos = self._known_gates_pos
        obstacles_pos = self._known_obstacles_pos
        cur_quat = np.asarray(obs["quat"], float)
        
        gate_idx = int(np.asarray(obs["target_gate"]).item())
        if gate_idx < 0 or gate_idx >= len(gates_pos):
            goal_pos = gates_pos[-1]
        else:
            goal_pos = gates_pos[gate_idx] 

        # [MODIFIED v7.6] 使用动态 V_des
        v_des = self._get_dynamic_v_des(cur_pos, gate_idx)

        pos_ref, vel_ref, yaw_ref = self._build_local_ref(
            cur_pos=cur_pos, 
            goal_pos=goal_pos, # Pass fallback goal
            cur_quat=cur_quat,
            v_des=v_des
        )

        fig, ax = plt.subplots(figsize=(6, 6))
        
        if self._global_waypoints is not None:
             ax.plot(self._global_waypoints[:, 0], self._global_waypoints[:, 1], 
                     'x--', color='gray', label="Global Path (Fixed)")

        ax.plot(pos_ref[:, 0], pos_ref[:, 1], '-', label=f"MPC ref path (v_des={v_des:.2f})")
        ax.scatter(cur_pos[0], cur_pos[1], c='blue', marker='o', s=80, label="drone")
        ax.scatter(gates_pos[:, 0], gates_pos[:, 1], c='green', marker='s', s=40, label="gates")
        for i, gp in enumerate(gates_pos):
            ax.text(gp[0], gp[1], f"G{i}", color='green', fontsize=8)
        if obstacles_pos.size > 0:
            ax.scatter(obstacles_pos[:, 0], obstacles_pos[:, 1], c='red', marker='x', s=80, label="obstacles")
            
            r_safe_solver = 0.33 # (这匹配 create_ocp_solver)
            for (ox, oy, _oz) in obstacles_pos:
                circ1 = plt.Circle((ox, oy), r_safe_solver, fill=False, linestyle='--', color='red', alpha=0.5, label="MPC r_safe (0.33)")
                ax.add_patch(circ1)
        
        if self._last_polyline is not None and self._last_polyline.shape[0] >= 2:
            wp = self._last_polyline
            ax.plot(wp[:, 0], wp[:, 1], 'o-', linewidth=2.0, label="polyline (黄线-当前)")
            
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.grid(True); ax.legend()
        ax.set_title(f"Local ref from Deque (targeting wp {self._current_global_idx-self._deque_len})")
        plt.show()
    
    
    # [NEW v7.6] 辅助函数，用于设置 acados 参数 p
    def _set_acados_parameters(self, cur_pos, pos_ref, vel_ref, yaw_ref):
        """
        为 EXTERNAL cost 设置 acados 参数向量 'p'。
        'p' 包含：[obs_pos, gate_geom, x_ref, u_ref]
        """
        
        # ---------- 1. 障碍物参数 (p[0:12]) ----------
        n_obs = 4 
        cur_pos_xy = cur_pos[:2]
        # [MODIFIED v7.6] 使用 self._known_obstacles_pos
        all_obs_pos = self._known_obstacles_pos
        
        selected_obs_pos = np.zeros((n_obs, 3))
        
        if all_obs_pos.shape[0] > 0:
            distances_xy = np.linalg.norm(all_obs_pos[:, :2] - cur_pos_xy, axis=1)
            closest_indices = np.argsort(distances_xy)
            num_to_take = min(all_obs_pos.shape[0], n_obs)
            indices_to_take = closest_indices[:num_to_take]
            selected_obs_pos[:num_to_take] = all_obs_pos[indices_to_take]
            # [FIX] 填充远处的值，以防 obs < n_obs
            if num_to_take < n_obs:
                selected_obs_pos[num_to_take:] = np.array([1e6, 1e6, 1e6])
        else:
             selected_obs_pos[:] = np.array([1e6, 1e6, 1e6])
        
        obs_flat = selected_obs_pos.flatten()

        # ---------- 2. 禁行带参数 (p[12:23]) ----------
        band_lo, band_hi = 0.30, 0.80
        band_alpha, act_s = 20.0, 0.50

        # [MODIFIED v7.6] 使用 self._known_gates_pos
        gates_pos = self._known_gates_pos
        gates_quat = np.asarray(self._initial_obs.get("gates_quat", None), float) \
                     if "gates_quat" in self._initial_obs else None

        # 选择“禁行带作用的门”= XY 平面里距离当前飞机最近的门中心
        dists = np.linalg.norm(gates_pos[:, :2] - cur_pos_xy[None, :], axis=1)
        best_i = int(np.argmin(dists))

        # [MODIFIED v7.7] 用这扇门的 *指定* 法向作为禁行带坐标系
        n_xy = self._gate_xy_dir(best_i, cur_pos, gates_pos, gates_quat)
        # n_xy 已经被归一化
        t_xy = np.array([-n_xy[1], n_xy[0]])
        # t_xy 已经被归一化 (因为 n_xy 是)

        g_band = gates_pos[best_i]
        gx, gy, gz = float(g_band[0]), float(g_band[1]), float(g_band[2])


        # ---------- 3. 组装并设置 p 向量 (p[0:end]) ----------
        nx, nu = self._nx, self._nu
        p_len = n_obs*3 + 11 + nx + nu # 总长度

        for j in range(self._N + 1):
            # 3.1 获取 x_ref, u_ref
            jj = min(j, self._N) # 终端成本使用 N
            xr = np.zeros(nx)
            xr[0:3] = pos_ref[jj]
            xr[5]   = yaw_ref[jj]
            xr[6:9] = vel_ref[jj]

            ur = np.zeros(nu)
            ur[3] = self._hover_thrust()

            # 3.2 组装完整的 p 向量
            p_full = np.zeros(p_len)
            off = 0
            # 障碍物 (12)
            p_full[off: off + n_obs*3] = obs_flat; off += n_obs*3 
            # 门几何 (11)
            p_full[off + 0: off + 3] = [gx, gy, gz]
            p_full[off + 3: off + 5] = t_xy
            p_full[off + 5: off + 7] = n_xy
            p_full[off + 7] = band_lo
            p_full[off + 8] = band_hi
            p_full[off + 9] = band_alpha
            p_full[off +10] = act_s
            off += 11
            # 参考 (nx + nu)
            p_full[off: off + nx] = xr; off += nx
            p_full[off: off + nu] = ur; off += nu

            # 3.3 设置参数
            self._solver.set(j, "p", p_full)


    # ---------- main MPC step ----------
    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        
        cur_pos = np.asarray(obs["pos"], float) # [MODIFIED v6.0] Get pos early
        
        # --- [NEW v7.6] 1. 检查传感器数据并请求重规划 ---
        self._check_and_request_replan(obs)

        # --- [NEW v7.6] 2. 如果需要，执行重规划 ---
        if self._replan_requested:
            print("[AttitudeMPC] Replanning global path with new sensor data.")
            self._build_global_path()
            self._reset_deque()
            self._replan_requested = False
        
        # --- 3. 设置初始状态 ---
        obs["rpy"] = R.from_quat(obs["quat"]).as_euler("xyz")
        obs["drpy"] = ang_vel2rpy_rates(obs["quat"], obs["ang_vel"])
        x0 = np.concatenate((obs["pos"], obs["rpy"], obs["vel"], obs["drpy"]))
        self._solver.set(0, "lbx", x0)
        self._solver.set(0, "ubx", x0)

        # --- 4. 更新 Deque (吃/长) ---
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

        # --- 5. 构建局部参考轨迹 ---
        gate_idx = int(np.asarray(obs["target_gate"]).item()) 
        gates_pos = self._known_gates_pos # [MODIFIED v7.6] 使用已知门
        
        if gate_idx < 0 or gate_idx >= len(gates_pos):
            goal_pos = gates_pos[-1]
        else:
            goal_pos = gates_pos[gate_idx] 

        cur_quat = np.asarray(obs["quat"], float)
        
        # [NEW v7.6] 获取动态速度
        v_des = self._get_dynamic_v_des(cur_pos, gate_idx)
        
        pos_ref, vel_ref, yaw_ref = self._build_local_ref(
            cur_pos=cur_pos,
            goal_pos=goal_pos, # Pass fallback goal
            cur_quat=cur_quat,
            v_des=v_des, 
        )

        # 缓存可视化
        self._last_plan = pos_ref.copy()
        self._traj_hist.append(cur_pos.reshape(3))

        # --- 6. [NEW v7.6] 设置 Acados 参数 p ---
        # (此辅助函数包含障碍物、门禁行带和 x_ref/u_ref 的所有逻辑)
        self._set_acados_parameters(cur_pos, pos_ref, vel_ref, yaw_ref)

        # --- 7. 求解 MPC ---
        self._solver.solve()
        
        # --- 8. 提取结果 ---
        pred = np.zeros((self._N + 1, 3))
        for j in range(self._N + 1):
            xj = self._solver.get(j, "x")
            pred[j] = xj[0:3]
        self._last_plan = pred # 用实际预测覆盖参考
        
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
        
        # [MODIFIED v7.6] 
        # 重置已访问状态，并使用 *初始* (标称) 数据重建路径
        print("[AttitudeMPC] Episode reset.")
        
        num_gates = len(self._initial_obs["gates_pos"])
        num_obstacles = len(self._initial_obs["obstacles_pos"])
        
        self._gate_visited = np.zeros(num_gates, dtype=bool)
        self._obstacle_visited = np.zeros(num_obstacles, dtype=bool)
        
        self._known_gates_pos = np.asarray(self._initial_obs["gates_pos"], float).copy()
        self._known_obstacles_pos = np.asarray(self._initial_obs["obstacles_pos"], float).reshape(-1, 3).copy()
        
        self._replan_requested = False
        
        # 用标称路径重建
        self._build_global_path()
        self._reset_deque()


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