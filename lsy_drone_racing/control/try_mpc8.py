# trajectory aims to next gate
"""
try_mpc8_global_path.py — Attitude NMPC with soft obstacle avoidance, 
and a PRE-COMPUTED global reference path.

[MODIFIED: 
 1. [USER REQ] Added _build_global_path() to pre-compute all pre/post points
    at __init__ using the (G_k - post_{k-1}) logic.
 2. [USER REQ] _build_local_ref is now simplified to just *sample* from this
    fixed global path, starting from cur_pos.
 3. [USER REQ] Dynamic 'post_pt' obstacle pushing is REMOVED. All obstacle
    avoidance is now handled *only* by the MPC solver's soft constraints.

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
        400.0, 400.0, 800.0,   # pos xyz
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
        750.0, 750.0, 600.0,   # pos xyz
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

        # --- [NEW] Global Path Pre-computation ---
        self.d_pre = 0.5  # Store globally
        self.d_post = 1.5 # Store globally
        # Stores [start_pos, pre0, G0, post0, pre1, G1, post1, ...]
        self._global_waypoints = None 
        self._build_global_path(obs)
        # --- [NEW END] ---


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
    

    def _build_global_path(self, initial_obs: dict):
        """
        [NEW] Pre-computes all pre/post points at init time based on user logic.
        - G0 normal uses (G0 - initial_pos)
        - Gk normal uses (Gk - post_{k-1})
        """
        print("[AttitudeMPC] Pre-computing global reference path...")
        gates_pos = np.asarray(initial_obs["gates_pos"], float)
        gates_quat = np.asarray(initial_obs.get("gates_quat", None), float) \
                     if "gates_quat" in initial_obs else None
        initial_pos = np.asarray(initial_obs["pos"], float)

        all_waypoints = [initial_pos]
        last_post_pt = initial_pos.copy() # Start with drone pos for G0 logic

        num_gates = len(gates_pos)
        
        for i in range(num_gates):
            g_c = gates_pos[i]
            g_quat = gates_quat[i] if gates_quat is not None else None
            
            # --- 1. Calculate n_xy using (G_k - post_{k-1}) logic ---
            n_xy = None
            
            # Vector from last post-point (or drone) to current gate
            vec_to_compare_3d = g_c - last_post_pt
            vec_to_compare_xy = vec_to_compare_3d[:2]
            norm_vec_to_compare = np.linalg.norm(vec_to_compare_xy)
            
            if norm_vec_to_compare < 1e-6:
                # Fallback if points are overlapping
                vec_to_compare_xy = np.array([1.0, 0.0])
                norm_vec_to_compare = 1.0

            if g_quat is not None:
                n_xy = self._gate_forward_xy_from_quat(g_quat)
                # Check alignment
                if np.dot(n_xy, vec_to_compare_xy) < 0:
                    n_xy = -n_xy
            else:
                # No quat, just use the direction vector
                n_xy = vec_to_compare_xy / norm_vec_to_compare
            
            # --- 2. Calculate and store points ---
            pre_pt = g_c.copy()
            pre_pt[:2] -= self.d_pre * n_xy
            
            post_pt = g_c.copy()
            post_pt[:2] += self.d_post * n_xy

            all_waypoints.append(pre_pt)
            all_waypoints.append(g_c)
            all_waypoints.append(post_pt)

            # --- 3. Update last_post_pt for next iteration ---
            last_post_pt = post_pt.copy()
            
        self._global_waypoints = np.vstack(all_waypoints)
        print(f"[AttitudeMPC] Global path with {len(self._global_waypoints)} points built.")

    
    
    def _build_local_ref(
        self,
        cur_pos: np.ndarray,
        goal_pos: np.ndarray, # Still needed for hover logic
        cur_quat: np.ndarray,
        gates_pos: np.ndarray, # Needed to get num_gates
        gate_idx: int,
        v_des: float = 0.8,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        [MODIFIED v5.0 - Global Path]
        Constructs a *local* reference by sampling from the *pre-computed*
        global path.
        
        - All pre/post points are FIXED from self._global_waypoints.
        - Dynamic obstacle logic (pushing post_pt) is REMOVED.
        - Path is [cur, pre1, G1, post1, pre2, G2]
        """
        T = self._T_HORIZON 
        N = self._N 
        dt = self._dt 

        gate_c = np.asarray(goal_pos, float).reshape(3) # G1
        cur_pos = np.asarray(cur_pos, float).reshape(3) 

        # --- 1. [MODIFIED] Build waypoints from GLOBAL path ---
        # We find the relevant segment of the pre-computed path
        # _global_waypoints = [start, pre0, G0, post0, pre1, G1, post1, ...]
        # Index for pre_Gk = 1 + 3*k
        # Index for Gk     = 2 + 3*k
        # Index for post_Gk  = 3 + 3*k
        
        base_waypoints = [cur_pos]
        
        try:
            # Get G1 points
            pre1_idx = 1 + 3 * gate_idx
            gate1_idx = 2 + 3 * gate_idx
            
            base_waypoints.append(self._global_waypoints[pre1_idx])
            base_waypoints.append(self._global_waypoints[gate1_idx]) # goal_pos == G1

            # Check if G1 is the last gate
            is_g1_last_gate = (gate_idx + 1 >= len(gates_pos))

            if not is_g1_last_gate:
                # Get G1-post and G2-pre, G2-gate
                post1_idx = 3 + 3 * gate_idx
                pre2_idx  = 1 + 3 * (gate_idx + 1)
                gate2_idx = 2 + 3 * (gate_idx + 1)
                
                base_waypoints.append(self._global_waypoints[post1_idx])
                base_waypoints.append(self._global_waypoints[pre2_idx])
                base_waypoints.append(self._global_waypoints[gate2_idx])
        
        except IndexError:
            # This might happen if gate_idx is out of sync
            # Fallback: just use the points we have
            print(f"[AttAttitudeMPC] Warning: IndexError in _build_local_ref. "
                  f"gate_idx {gate_idx} may be out of bounds for global path.")
            if len(base_waypoints) == 1: # Only cur_pos
                base_waypoints.append(gate_c) # Add goal
        
        waypoints = np.vstack(base_waypoints) 
        # --- (黄线构建结束) ---

        self._last_polyline = waypoints.copy() #

        # --- 2. 采样与鲁棒 Yaw (逻辑保持不变) ---
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
        # gates_quat = np.asarray(obs.get("gates_quat", None), float) if "gates_quat" in obs else None
        gate_idx = int(np.asarray(obs["target_gate"]).item())
        goal_pos = gates_pos[gate_idx]
        obstacles_pos = np.asarray(obs["obstacles_pos"], float).reshape(-1, 3)

        # --- [MODIFIED] Call new _build_local_ref signature ---
        pos_ref, vel_ref, yaw_ref = self._build_local_ref(
            cur_pos=cur_pos, 
            goal_pos=goal_pos, 
            cur_quat=cur_quat,
            gates_pos=gates_pos, 
            gate_idx=gate_idx,
            v_des=1.1
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
            
            r_safe_solver = 0.35 # (这匹配 create_ocp_solver)
            for (ox, oy, _oz) in obstacles_pos:
                circ = plt.Circle((ox, oy), r_safe_solver, fill=False, linestyle='--', color='red', alpha=0.5)
                ax.add_patch(circ)
        
        if self._last_polyline is not None and self._last_polyline.shape[0] >= 2:
            wp = self._last_polyline
            ax.plot(wp[:, 0], wp[:, 1], 'o-', linewidth=2.0, label="polyline (黄线-当前)")
            
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.grid(True); ax.legend()
        ax.set_title(f"Local ref to gate {gate_idx} (from Global Path)")
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

        p_vec = selected_obs_pos.flatten()
        for j in range(self._N + 1):
            self._solver.set(j, "p", p_vec)
        # --- 障碍物逻辑结束 ---


        # gate references
        gate_idx = int(np.asarray(obs["target_gate"]).item()) 
        gates_pos = np.asarray(obs["gates_pos"], float) 
        goal_pos = gates_pos[gate_idx] 
        cur_pos = np.asarray(obs["pos"], float) 
        cur_quat = np.asarray(obs["quat"], float) 
        
        # (不再需要 'next_gate_pos' 等, 因为它们已在 _global_waypoints 中)

        # --- [MODIFIED] Call new _build_local_ref signature ---
        # local ref (sampled from global path)
        pos_ref, vel_ref, yaw_ref = self._build_local_ref(
            cur_pos=cur_pos,
            goal_pos=goal_pos,
            cur_quat=cur_quat,
            gates_pos=gates_pos, #
            gate_idx=gate_idx,   #
            v_des=1.1,
            # d_pre, d_post, all_obs_pos, r_safe_ref, etc. are no longer needed here
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
        # [NEW] 我们不需要重建全局路径，因为它对于这个赛道是不变的
        # 除非赛道布局在 episode 之间发生变化，否则 self._global_waypoints 保持有效
        
        # 如果赛道在 episode 之间会变，你需要在这里重新构建它：
        # if obs is provided to episode_callback:
        #     self._build_global_path(obs) 
        pass


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