from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from collections import deque
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from casadi import MX, vertcat, sqrt, log, exp
from scipy.spatial.transform import Rotation as R

from drone_models.core import load_params
from drone_models.so_rpy import symbolic_dynamics_euler
from drone_models.utils.rotation import ang_vel2rpy_rates
from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray

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
    model.name = "att_mpc_lvl2_softobs_band"
    model.f_expl_expr = X_dot
    model.x = X
    model.u = U

    n_obs = 4
    obs_dim = n_obs * 3
    gate_dim = 11

    nx = X.rows()
    nu = U.rows()
    p_dim = obs_dim + gate_dim + nx + nu
    p = MX.sym("p", p_dim)
    model.p = p

    x_pos, y_pos, z_pos = X[0], X[1], X[2]
    h_list = []
    for i in range(n_obs):
        ox = p[3*i + 0]
        oy = p[3*i + 1]
        d2 = (x_pos - ox)**2 + (y_pos - oy)**2
        h_list.append(d2)
    model.con_h_expr = vertcat(*h_list)

    off = obs_dim
    gx = p[off+0]; gy = p[off+1]; gz = p[off+2]
    tx = p[off+3]; ty = p[off+4]
    nxg = p[off+5]; nyg = p[off+6]
    band_lo    = p[off+7]
    band_hi    = p[off+8]
    band_alpha = p[off+9]
    act_s      = p[off+10]

    xref_off = obs_dim + gate_dim
    x_ref = p[xref_off : xref_off + nx]
    u_ref = p[xref_off + nx : xref_off + nx + nu]

    dx = x_pos - gx
    dy = y_pos - gy
    dz = z_pos - gz
    s   = dx*nxg + dy*nyg
    lat = dx*tx  + dy*ty
    abslat = sqrt(lat*lat + 1e-9)
    absdz  = sqrt(dz*dz   + 1e-9)

    rho = sqrt(s*s + lat*lat + 1e-9)
    alpha_s = 1.0 / (1.0 + (rho / act_s) ** 2)

    def bump(r, lo, hi, k):
        sp1 = log(1 + exp(k*(r - lo))) / k
        sp2 = log(1 + exp(k*(hi - r))) / k
        return sp1 * sp2

    pen_lat = bump(abslat, band_lo, band_hi, band_alpha)
    pen_z   = bump(absdz , band_lo, band_hi, band_alpha)

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

    ocp.cost.cost_type   = "EXTERNAL"
    ocp.cost.cost_type_0 = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"

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

    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.tf = Tf

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


class AttitudeMPC(Controller):
    _CONSUME_DIST = 0.3  

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

        self._traj_hist = deque(maxlen=4000)
        self._last_plan = None
        self._last_polyline = None

        print("[AttitudeMPC] v15q: Online refresh using gates_quat + direction memory.")
        num_gates = len(obs["gates_pos"])
        self.d_pre_list  = np.array([0.30, 0.50, 0.30, 0.30], dtype=float)
        self.d_post_list = np.array([0.50, 0.50, 0.30, 1.20], dtype=float)
        if len(self.d_pre_list) < num_gates:
            self.d_pre_list  = np.pad(self.d_pre_list,  (0, num_gates-len(self.d_pre_list)),  'edge')
        if len(self.d_post_list) < num_gates:
            self.d_post_list = np.pad(self.d_post_list, (0, num_gates-len(self.d_post_list)), 'edge')

        self._gates_quat = np.asarray(obs.get("gates_quat", None), float) if "gates_quat" in obs else None
        self._gates_working = np.asarray(obs["gates_pos"], float).copy()
        self._gates_visited = np.array(obs.get("gates_visited", np.zeros(num_gates, dtype=bool)), dtype=bool).copy()

        self._nxy_mem = np.full((num_gates, 2), np.nan, dtype=float)

        self._v_nominal = 0.5
        self._v_slow = 0.25
        self._slow_gate = None  
        self._slow_active = False

        self._global_waypoints = None
        self._wp_idx_pre  = None
        self._wp_idx_gate = None
        self._wp_idx_post = None

        self._deque_len = 6
        self._local_polyline_deque = deque(maxlen=self._deque_len)

        cur_pos = np.asarray(obs["pos"], float)
        self._build_global_path(cur_pos)
        tg = int(np.asarray(obs["target_gate"]).item())
        tg = max(0, min(tg, num_gates-1))
        self._current_global_idx = int(self._wp_idx_pre[tg])
        self._fill_deque_from(self._current_global_idx)

    def _gate_forward_xy_from_quat(self, q: np.ndarray) -> np.ndarray:
        fwd = R.from_quat(q).apply(np.array([1.0, 0.0, 0.0]))
        v = fwd[:2]
        n = np.linalg.norm(v)
        return np.array([1.0, 0.0]) if n < 1e-12 else (v / n)

    def _nxy_for_gate(self, i: int) -> np.ndarray:
        """Return stabilized n_xy for gate i: from quat, then aligned with memory to avoid flips."""
        if self._gates_quat is None:
            mem = self._nxy_mem[i]
            if np.all(np.isfinite(mem)):
                return mem
            return np.array([1.0, 0.0], dtype=float)

        n_xy = self._gate_forward_xy_from_quat(self._gates_quat[i])

        mem = self._nxy_mem[i]
        if np.all(np.isfinite(mem)):
            if float(np.dot(n_xy, mem)) < 0.0:
                n_xy = -n_xy  
        self._nxy_mem[i] = n_xy
        return n_xy

    def _build_global_path(self, initial_pos: np.ndarray):
        """Path = [start, pre0, G0, post0, pre1, G1, post1, ...], n_xy ONLY from quat (+ memory)."""
        gates_pos = self._gates_working
        all_wp = [initial_pos.copy()]

        ng = len(gates_pos)
        wp_idx_pre  = np.full(ng, -1, dtype=int)
        wp_idx_gate = np.full(ng, -1, dtype=int)
        wp_idx_post = np.full(ng, -1, dtype=int)

        for i in range(ng):
            g = gates_pos[i].copy()
            n_xy = self._nxy_for_gate(i) 

            pre  = g.copy(); pre [:2] += -float(self.d_pre_list[i])  * n_xy
            post = g.copy(); post[:2] +=  float(self.d_post_list[i]) * n_xy

            if i == 2:
                pre[2]  -= 0.05
                g[2]  -= 0.05
                post[2] -= 0.05

            idx_pre  = len(all_wp); all_wp.append(pre)
            idx_gate = len(all_wp); all_wp.append(g)
            idx_post = len(all_wp); all_wp.append(post)

            wp_idx_pre [i] = idx_pre
            wp_idx_gate[i] = idx_gate
            wp_idx_post[i] = idx_post
            
            if i == 2 and (i + 1) < ng:
                all_wp.append(pre.copy())  

            if i == ng - 1:
                extra_posts, tail_step = 6, 0.4
                for k in range(1, extra_posts + 1):
                    p = post.copy()  
                    p[:2] += k * tail_step * n_xy
                    all_wp.append(p)

        self._global_waypoints = np.vstack(all_wp)
        self._wp_idx_pre  = wp_idx_pre
        self._wp_idx_gate = wp_idx_gate
        self._wp_idx_post = wp_idx_post
        print(f"[AttitudeMPC] Global path rebuilt: {len(self._global_waypoints)} points.")


    def _fill_deque_from(self, start_idx: int):
        self._local_polyline_deque.clear()
        idx = int(start_idx)
        for _ in range(self._deque_len):
            if idx < len(self._global_waypoints):
                self._local_polyline_deque.append(self._global_waypoints[idx])
            else:
                self._local_polyline_deque.append(self._global_waypoints[-1])
            idx += 1
        self._current_global_idx = idx

    def _maybe_refresh_on_gate_truth(self, obs: dict):
        gv = np.array(obs.get("gates_visited", np.zeros_like(self._gates_visited)), dtype=bool)
        if gv.shape != self._gates_visited.shape:
            return
        if np.array_equal(gv, self._gates_visited):
            return

        newly_true = np.where(np.logical_and(gv, np.logical_not(self._gates_visited)))[0]
        if newly_true.size > 0:
            gates_now = np.asarray(obs["gates_pos"], float)
            self._gates_working[newly_true] = gates_now[newly_true]
            tg = int(np.asarray(obs["target_gate"]).item())
            if tg in newly_true:
                self._slow_gate = tg
                self._slow_active = True
            print(f"[AttitudeMPC] Gates turned True: {newly_true.tolist()} -> updated to truth. Slow gate = {self._slow_gate}")

        self._gates_visited = gv.copy()

        cur_pos = np.asarray(obs["pos"], float)
        self._build_global_path(cur_pos)

        tg = int(np.asarray(obs["target_gate"]).item())
        tg = max(0, min(tg, len(self._gates_working) - 1))
        anchor_idx = int(self._wp_idx_pre[tg])
        self._fill_deque_from(anchor_idx)
        print(f"[AttitudeMPC] Re-anchored deque to pre[target_gate={tg}] at global idx {anchor_idx}.")

    def _build_local_ref(self, cur_pos: np.ndarray, goal_pos: np.ndarray, cur_quat: np.ndarray, v_des: float):
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
            cur_yaw = R.from_quat(cur_quat).as_euler("xyz")[2]
            yaw_ref = np.full(N + 1, cur_yaw)
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

    def _hover_thrust(self) -> float:
        return float(self.drone_params["mass"]) * (-float(self.drone_params["gravity_vec"][-1]))

    def compute_control(self, obs: dict[str, NDArray[np.floating]], info: dict | None = None) -> NDArray[np.floating]:

        self._maybe_refresh_on_gate_truth(obs)

        obs["rpy"] = R.from_quat(obs["quat"]).as_euler("xyz")
        obs["drpy"] = ang_vel2rpy_rates(obs["quat"], obs["ang_vel"])
        x0 = np.concatenate((obs["pos"], obs["rpy"], obs["vel"], obs["drpy"]))
        self._solver.set(0, "lbx", x0)
        self._solver.set(0, "ubx", x0)

        cur_pos = np.asarray(obs["pos"], float)
        if len(self._local_polyline_deque) > 0:
            target_pt = self._local_polyline_deque[0]
            if np.linalg.norm(cur_pos[:2] - target_pt[:2]) < self._CONSUME_DIST:
                self._local_polyline_deque.popleft()
                if self._current_global_idx < len(self._global_waypoints):
                    self._local_polyline_deque.append(self._global_waypoints[self._current_global_idx])
                    self._current_global_idx += 1
                else:
                    self._local_polyline_deque.append(self._global_waypoints[-1])

        gates_pos = np.asarray(obs["gates_pos"], float)
        gate_idx = int(np.asarray(obs["target_gate"]).item())
        gate_idx = max(0, min(gate_idx, len(gates_pos)-1))
        goal_pos = gates_pos[gate_idx]

        v_des = self._v_nominal
        if self._slow_active and (self._slow_gate is not None):
            i = int(self._slow_gate)
            g_xy = np.asarray(self._gates_working[i][:2], float)
            n_xy = self._nxy_for_gate(i)  
            s_cur = float(np.dot(cur_pos[:2] - g_xy, n_xy))
            if s_cur < 0.0:  
                v_des = self._v_slow
            else:
                self._slow_active = False  

        cur_quat = np.asarray(obs["quat"], float)
        pos_ref, vel_ref, yaw_ref = self._build_local_ref(cur_pos, goal_pos, cur_quat, v_des=v_des)
        self._last_plan = pos_ref.copy()
        self._traj_hist.append(cur_pos.reshape(3))

        n_obs = 4
        cur_xy = cur_pos[:2]
        all_obs_pos = np.asarray(obs["obstacles_pos"], float).reshape(-1, 3)
        selected_obs_pos = np.zeros((n_obs, 3))
        if all_obs_pos.shape[0] > 0:
            distances_xy = np.linalg.norm(all_obs_pos[:, :2] - cur_xy, axis=1)
            idxs = np.argsort(distances_xy)[:min(n_obs, len(all_obs_pos))]
            selected_obs_pos[:len(idxs)] = all_obs_pos[idxs]
            if len(idxs) < n_obs:
                selected_obs_pos[len(idxs):] = np.array([1e6, 1e6, 1e6])
        else:
            selected_obs_pos[:] = np.array([1e6, 1e6, 1e6])

        band_lo, band_hi = 0.30, 0.80
        band_alpha, act_s = 20.0, 0.50

        n_xy_band = self._nxy_for_gate(gate_idx)
        t_xy = np.array([-n_xy_band[1], n_xy_band[0]])
        t_xy /= (np.linalg.norm(t_xy) + 1e-12)
        gx, gy, gz = [float(x) for x in self._gates_working[gate_idx]]

        nx, nu = self._nx, self._nu
        p_len = n_obs*3 + 11 + nx + nu
        obs_flat = selected_obs_pos.flatten()

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
            p_full[off + 5: off + 7] = n_xy_band
            p_full[off + 7] = band_lo
            p_full[off + 8] = band_hi
            p_full[off + 9] = band_alpha
            p_full[off +10] = act_s
            off += 11
            p_full[off: off + nx] = xr; off += nx
            p_full[off: off + nu] = ur; off += nu
            self._solver.set(j, "p", p_full)

        self._solver.solve()

        pred = np.zeros((self._N + 1, 3))
        for j in range(self._N + 1):
            xj = self._solver.get(j, "x")
            pred[j] = xj[0:3]
        self._last_plan = pred

        u0 = self._solver.get(0, "u")
        return u0

    # ---------- debug ----------
    def debug_plot(self, obs: dict):
        import matplotlib.pyplot as plt
        cur_pos = np.asarray(obs["pos"], float)
        gates_pos = np.asarray(obs["gates_pos"], float)
        cur_quat = np.asarray(obs["quat"], float)

        gate_idx = int(np.asarray(obs["target_gate"]).item())
        goal_pos = gates_pos[-1] if (gate_idx < 0 or gate_idx >= len(gates_pos)) else gates_pos[gate_idx]
        obstacles_pos = np.asarray(obs["obstacles_pos"], float).reshape(-1, 3)

        v_des = self._v_slow if (self._slow_active and self._slow_gate == gate_idx) else self._v_nominal
        pos_ref, _, _ = self._build_local_ref(cur_pos, goal_pos, cur_quat, v_des=v_des)

        fig, ax = plt.subplots(figsize=(6, 6))
        if self._global_waypoints is not None:
            ax.plot(self._global_waypoints[:, 0], self._global_waypoints[:, 1], 'x--', color='gray', label="Global Path (refreshed)")
        ax.plot(pos_ref[:, 0], pos_ref[:, 1], '-', label="MPC ref path (sampled)")
        ax.scatter(cur_pos[0], cur_pos[1], c='blue', marker='o', s=80, label="drone")
        ax.scatter(gates_pos[:, 0], gates_pos[:, 1], c='green', marker='s', s=40, label="gates")
        for i, gp in enumerate(gates_pos):
            ax.text(gp[0], gp[1], f"G{i}", color='green', fontsize=8)
        if obstacles_pos.size > 0:
            ax.scatter(obstacles_pos[:, 0], obstacles_pos[:, 1], c='red', marker='x', s=80, label="obstacles")
            r_safe_solver = 0.33
            for (ox, oy, _oz) in obstacles_pos:
                circ1 = plt.Circle((ox, oy), r_safe_solver, fill=False, linestyle='--', color='red', alpha=0.5, label="MPC r_safe (0.33)")
                ax.add_patch(circ1)
        if self._last_polyline is not None and self._last_polyline.shape[0] >= 2:
            wp = self._last_polyline
            ax.plot(wp[:, 0], wp[:, 1], 'o-', linewidth=2.0, label="polyline (yellow deque)")
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.grid(True); ax.legend()
        ax.set_title(f"Deque ref (start idx {self._current_global_idx-self._deque_len})")
        plt.show()

    def get_debug_lines(self):
        out = []
        if len(self._traj_hist) >= 2:
            traj = np.asarray(self._traj_hist, float)
            out.append((traj, np.array([0.1, 0.3, 1.0, 0.9]), 2.5, 2.5))
        if getattr(self, "_last_plan", None) is not None and self._last_plan.shape[0] >= 2:
            out.append((self._last_plan, np.array([1.0, 0.1, 0.1, 0.95]), 3.0, 3.0))
        if getattr(self, "_last_polyline", None) is not None and self._last_polyline.shape[0] >= 2:
            out.append((self._last_polyline, np.array([1.0, 0.9, 0.1, 0.95]), 3.0, 3.0))
        if getattr(self, "_global_waypoints", None) is not None and self._global_waypoints.shape[0] >= 2:
            out.append((self._global_waypoints, np.array([0.5, 0.5, 0.5, 0.7]), 2.0, 2.0))
        return out

    def step_callback(self, *args, **kwargs) -> bool:
        return False

    def episode_callback(self):
        self._traj_hist.clear()
        self._last_plan = None
        self._last_polyline = None
        print("[AttitudeMPC] Episode reset. Deque will be realigned on first step.")
