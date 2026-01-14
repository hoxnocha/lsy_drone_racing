from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple, Optional
import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from casadi import MX, vertcat, dot, DM, norm_2, floor, if_else, substitute
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize

from drone_models.core import load_params
from drone_models.so_rpy import symbolic_dynamics_euler
from drone_models.utils.rotation import ang_vel2rpy_rates

from lsy_drone_racing.control import Controller
from enum import IntEnum

if TYPE_CHECKING:
    from numpy.typing import NDArray

# ==============================================================================
#  MODULE 1: Utilities & Enums
# ==============================================================================

class ObstacleType(IntEnum):
    CYLINDER_2D = 0 
    CAPSULE_3D  = 2 

class FrameUtils:
    @staticmethod
    def quat_to_axis(quat: NDArray[np.floating], axis_index: int = 1) -> NDArray[np.floating]:
        rot = R.from_quat(quat)
        mats = np.asarray(rot.as_matrix())
        if mats.ndim == 3: return mats[:, :, axis_index]
        return mats[:, axis_index]

class PathTools:
    def build_gate_waypoints(self, start_pos, gates_positions, gates_normals):
        # 简单连接：起点 -> 门1中心 -> 门2中心 ...
        waypoints = [start_pos]
        for i in range(len(gates_positions)):
            # 略微偏向门后一点点，保证穿过
            waypoints.append(gates_positions[i]) 
        return np.array(waypoints)

    def reparametrize_by_arclength(self, trajectory, arc_step=0.05, epsilon=1e-5):
        if hasattr(trajectory, 'x'): total_range = trajectory.x[-1] - trajectory.x[0]
        else: total_range = trajectory.ts_cumulative[-1]

        for _ in range(20): # 限制迭代次数防止卡死
            n_seg = max(2, int(total_range / arc_step))
            t = np.linspace(0, total_range, n_seg)
            pts = trajectory(t)
            dist = np.concatenate(([0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))))
            
            # 去重
            valid = np.concatenate(([True], np.diff(dist) > 1e-6))
            dist = dist[valid]; pts = pts[valid]
            
            if len(dist) < 2: return trajectory
            total_range = dist[-1]
            trajectory = CubicSpline(dist, pts)
            if np.std(np.diff(dist)) <= epsilon: return trajectory
        return CubicSpline(dist, pts)

    def extend_spline_tail(self, trajectory, extend_length=1.0):
        t_end = trajectory.x[-1]
        p_end = trajectory(t_end)
        v_end = trajectory(t_end, 1)
        v_dir = v_end / (np.linalg.norm(v_end) + 1e-6)
        
        knots = np.arange(t_end + 0.1, t_end + extend_length, 0.1)
        pts = np.array([p_end + v_dir * (t - t_end) for t in knots])
        
        new_x = np.concatenate([trajectory.x, knots])
        new_y = np.vstack([trajectory(trajectory.x), pts])
        return CubicSpline(new_x, new_y, axis=0)

# ==============================================================================
#  MODULE 2: Trajectory Optimization (FIXED Math Core)
# ==============================================================================

class GCOPTER_Lite:
    """极速插值内核 (修复了边界速度为0的BUG)"""
    def __init__(self, waypoints, avg_speed=6.0):
        self.waypoints = np.array(waypoints)
        self.n_seg = len(waypoints) - 1
        self.dim = waypoints.shape[1]
        
        # [Fix 1]: 动态时间分配，不再依赖外部传入的固定 duration
        dists = np.linalg.norm(np.diff(self.waypoints, axis=0), axis=1)
        self.T = dists / avg_speed
        self.T = np.maximum(self.T, 0.1) # 最小每段 0.1s
        
        self.coeffs = self._solve_min_jerk_fast(self.waypoints, self.T)
        self.ts_cumulative = np.concatenate(([0], np.cumsum(self.T)))
        self.x = self.ts_cumulative

    def _solve_min_jerk_fast(self, Q, T):
        n = self.n_seg; dim = self.dim; coeffs = np.zeros((n, dim, 6))
        for d in range(dim):
            qs = Q[:, d]
            vs = np.zeros(n + 1); as_ = np.zeros(n + 1)
            
            # [Fix 2]: 边界速度初始化 (不再默认为0)
            # 使用线性速度作为边界条件，保证起步顺滑
            vs[0] = (qs[1] - qs[0]) / T[0]
            vs[n] = (qs[n] - qs[n-1]) / T[n-1]
            
            # 中间点速度 (平均法)
            for i in range(1, n):
                v_prev = (qs[i] - qs[i-1]) / T[i-1]
                v_next = (qs[i+1] - qs[i]) / T[i]
                vs[i] = 0.5 * (v_prev + v_next)
            
            # 加速度估算
            for i in range(1, n):
                a_prev = (vs[i] - vs[i-1]) / T[i-1]
                a_next = (vs[i+1] - vs[i]) / T[i]
                as_[i] = 0.5 * (a_prev + a_next)
                
            # 求解五次多项式系数
            for i in range(n):
                p0, p1 = qs[i], qs[i+1]; v0, v1 = vs[i], vs[i+1]; a0, a1 = as_[i], as_[i+1]; t = T[i]
                k_c3 = (20*(p1-(p0+v0*t+0.5*a0*t**2)) - 8*(v1-(v0+a0*t))*t + (a1-a0)*t**2) / (2*t**3)
                k_c4 = (-30*(p1-(p0+v0*t+0.5*a0*t**2)) + 14*(v1-(v0+a0*t))*t - 2*(a1-a0)*t**2) / (2*t**4)
                k_c5 = (12*(p1-(p0+v0*t+0.5*a0*t**2)) - 6*(v1-(v0+a0*t))*t + (a1-a0)*t**2) / (2*t**5)
                coeffs[i, d, :] = [p0, v0, 0.5*a0, k_c3, k_c4, k_c5]
        return coeffs

    def __call__(self, t_in, derivative=0):
        t_in = np.atleast_1d(t_in)
        res = np.zeros((len(t_in), self.dim))
        total_T = self.ts_cumulative[-1]
        t_in = np.clip(t_in, 0, total_T - 1e-6)
        indices = np.searchsorted(self.ts_cumulative, t_in, side='right') - 1
        indices = np.clip(indices, 0, self.n_seg - 1)
        t_start = self.ts_cumulative[indices]; dt = t_in - t_start; c = self.coeffs[indices]
        for d in range(self.dim):
            cd = c[:, d, :]
            if derivative == 0: res[:, d] = cd[:,0] + cd[:,1]*dt + cd[:,2]*dt**2 + cd[:,3]*dt**3 + cd[:,4]*dt**4 + cd[:,5]*dt**5
            elif derivative == 1: res[:, d] = cd[:,1] + 2*cd[:,2]*dt + 3*cd[:,3]*dt**2 + 4*cd[:,4]*dt**3 + 5*cd[:,5]*dt**4
            elif derivative == 2: res[:, d] = 2*cd[:,2] + 6*cd[:,3]*dt + 12*cd[:,4]*dt**2 + 20*cd[:,5]*dt**3
        if len(res) == 1: return res[0]
        return res
    
    def derivative(self, n=1): return lambda t: self.__call__(t, derivative=n)

class GCOPTER_Optimizer:
    """基于势场优化的轨迹生成器 (Z轴硬约束版)"""
    def __init__(self, start_p, end_p, nominal_waypoints, obstacles, avg_speed=6.0):
        self.start = np.array(start_p, dtype=float)
        self.end = np.array(end_p, dtype=float)
        self.nominal_mid = np.array(nominal_waypoints, dtype=float)
        self.obstacles = obstacles
        self.avg_speed = float(avg_speed)

        self.z_min = 0.10
        self.K_per_seg = 50   # 硬约束采样密度：越大越不容易漏穿地板
        self.slsqp_maxiter = 120

        init_guess_points = self.nominal_mid.copy()
        init_guess_points[:, 2] = np.maximum(init_guess_points[:, 2], self.z_min + 0.10)
        self.init_guess = init_guess_points.flatten()

        n_points = len(self.nominal_mid)
        bounds = []
        for _ in range(n_points):
            bounds.append((None, None))        # x
            bounds.append((None, None))        # y
            bounds.append((self.z_min, None))  # waypoint 的硬墙（仍保留）

        # ---- 关键：离散硬约束（trajectory 采样点 z >= z_min）----
        def floor_ineq(q_flat):
            q_mid = q_flat.reshape(-1, 3)
            waypoints = np.vstack([self.start, q_mid, self.end])
            traj = GCOPTER_Lite(waypoints, avg_speed=self.avg_speed)

            totalT = float(traj.ts_cumulative[-1])
            N = max(10, traj.n_seg * self.K_per_seg)
            ts = np.linspace(0.0, max(1e-6, totalT - 1e-6), N)
            pts = traj(ts)            # (N,3)
            z = pts[:, 2]
            return z - self.z_min     # 要求每个分量都 >= 0

        constraints = [{'type': 'ineq', 'fun': floor_ineq}]

        res = minimize(
            self._cost_function,
            self.init_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': self.slsqp_maxiter, 'ftol': 1e-4, 'disp': False}
        )

        opt_mid = res.x.reshape(-1, 3)
        self.full_waypoints = np.vstack([self.start, opt_mid, self.end])
        self.traj_gen = GCOPTER_Lite(self.full_waypoints, avg_speed=self.avg_speed)



    def _cost_function(self, q_flat):
        q_mid = q_flat.reshape(-1, 3)
        waypoints = np.vstack([self.start, q_mid, self.end])

        # 1) Smoothness (discrete)
        acc = np.diff(waypoints, n=2, axis=0)
        j_smooth = np.sum(acc**2) * 5.0

        # 2) Fidelity to nominal mid
        j_pos = np.sum((q_mid - self.nominal_mid)**2) * 1000.0

        # 3) Obstacle potential field
      
        def dist_to_obstacle(pt, obs):
            pt = np.asarray(pt, dtype=float)
            otype = obs['type']
            opos = np.asarray(obs['pos'], dtype=float)

            if otype == ObstacleType.CYLINDER_2D:
                return np.linalg.norm(pt[:2] - opos[:2])

            # CAPSULE_3D: point-to-segment distance
            vec = np.asarray(obs.get('vec', np.array([1.0, 0.0, 0.0])), dtype=float)
            half_len = float(obs.get('half_len', 0.0))
            vnorm = np.linalg.norm(vec) + 1e-9
            v = vec / vnorm
            proj = np.dot(pt - opos, v)
            proj = np.clip(proj, -half_len, half_len)
            closest = opos + proj * v
            return np.linalg.norm(pt - closest)

        j_obs = 0.0
        if self.obstacles:
            for pt in q_mid:
                for obs in self.obstacles:
                    dist = dist_to_obstacle(pt, obs)
                    margin = float(obs['margin']) * 0.9
                    if dist < margin:
                        j_obs += (margin - dist)**2 * 5000.0


        return j_smooth + j_pos + j_obs 


    @property
    def ts_cumulative(self): return self.traj_gen.ts_cumulative
    @property
    def x(self): return self.traj_gen.x
    def __call__(self, t, d=0): 
        if d != 0: return self.traj_gen(t, derivative=d)
        return self.traj_gen(t)
    def derivative(self, n=1): return lambda t: self.traj_gen(t, derivative=n)


# ==============================================================================
#  MODULE 3: Path Planner
# ==============================================================================

class PathPlanner:
    def __init__(self, initial_pos, initial_gates_pos):
        self.trajectory = None
        self.arc_trajectory = None
        # [Fix]: 移除 self.duration 状态，改为每次动态计算
        
        self.cached_gates_pos = initial_gates_pos
        self.cached_obstacles_pos = np.array([])
        self._last_gate_flags = None
        self._last_obst_flags = None
        self.tools = PathTools()

    def check_env_update(self, obs) -> bool:
        curr_gates = np.array(obs.get("gates_visited", []), dtype=bool)
        curr_obst = np.array(obs.get("obstacles_visited", []), dtype=bool)
        if self._last_gate_flags is None:
            self._last_gate_flags = curr_gates; self._last_obst_flags = curr_obst; return True 
        changed = np.any((~self._last_gate_flags) & curr_gates) or np.any((~self._last_obst_flags) & curr_obst)
        self._last_gate_flags = curr_gates; self._last_obst_flags = curr_obst
        return bool(changed)

    def replan(self, current_pos, obs, model_traj_length):
        print(f"[Planner] Replanning from {current_pos}...")

        gate_pos = obs["gates_pos"]
        obst_pos = obs["obstacles_pos"]
        gate_quat = obs["gates_quat"]

        self.cached_gates_pos = gate_pos
        self.cached_obstacles_pos = obst_pos

        z_min = 0.10

        # 1) 基础路径 (Current -> Gates)
        gate_normals = FrameUtils.quat_to_axis(gate_quat, 0)
        base_waypoints = self.tools.build_gate_waypoints(current_pos, gate_pos, gate_normals)

        # 基础点也抬高（避免起点或门点在地面下）
        base_waypoints = np.array(base_waypoints, dtype=float)
        base_waypoints[:, 2] = np.maximum(base_waypoints[:, 2], z_min)

        # 2) 收集障碍物
        all_obstacles = self._collect_obstacles(gate_pos, gate_quat, obst_pos)

        # 3) 运行优化
        p_mid = base_waypoints[1:]

        if len(base_waypoints) < 3:
            self.trajectory = GCOPTER_Lite(base_waypoints, avg_speed=6.0)
        else:
            self.trajectory = GCOPTER_Optimizer(
                start_p=base_waypoints[0],
                end_p=base_waypoints[-1],
                nominal_waypoints=p_mid[:-1],
                obstacles=all_obstacles,
                avg_speed=6.0
            )

        # 3.5) 兜底：采样检查轨迹最低 z，如下穿则整体抬高再生成一次
        try:
            totalT = self.trajectory.ts_cumulative[-1]
            ts = np.linspace(0.0, max(1e-6, totalT - 1e-6), 200)
            pts = self.trajectory(ts)
            min_z = float(np.min(pts[:, 2]))
            if min_z < z_min:
                dz = (z_min - min_z) + 0.02

                # 获取轨迹对应的 waypoints
                if hasattr(self.trajectory, "waypoints"):
                    wps = np.array(self.trajectory.waypoints, dtype=float)
                elif hasattr(self.trajectory, "full_waypoints"):
                    wps = np.array(self.trajectory.full_waypoints, dtype=float)
                else:
                    wps = base_waypoints.copy()

                wps[:, 2] += dz
                self.trajectory = GCOPTER_Lite(wps, avg_speed=6.0)
        except Exception:
            pass

        # 4) 尾部延长 + 弧长参数化
        extended = self.tools.extend_spline_tail(self.trajectory, model_traj_length)
        self.arc_trajectory = self.tools.reparametrize_by_arclength(extended)
        return self.arc_trajectory

    def _collect_obstacles(self, gate_pos, gate_quat, real_obst_pos):
        obstacles = []

        # Real obstacles: treat as vertical cylinders in XY (as you did)
        for pos in real_obst_pos:
            obstacles.append({
                'type': ObstacleType.CYLINDER_2D,
                'pos': np.array(pos, dtype=float),
                'margin': 0.35
            })

        y_axes = FrameUtils.quat_to_axis(gate_quat, 1)
        z_axes = FrameUtils.quat_to_axis(gate_quat, 2)

        w, h = 0.7, 0.7
        half_w = w / 2.0
        half_h = h / 2.0

        for i, c in enumerate(gate_pos):
            c = np.array(c, dtype=float)
            y = np.array(y_axes[i], dtype=float)
            z = np.array(z_axes[i], dtype=float)

            # Side posts (2D cylinders)
            for s in [1, -1]:
                obstacles.append({
                    'type': ObstacleType.CYLINDER_2D,
                    'pos': c + s * half_w * y,
                    'margin': 0.20
                })

            # Bars (3D capsules): axis along y, centered at top/bottom bar
            for s in [1, -1]:
                obstacles.append({
                    'type': ObstacleType.CAPSULE_3D,
                    'pos': c + s * half_h * z,     # capsule center
                    'vec': y,                      # capsule axis direction (unit-ish)
                    'half_len': half_w,            # half segment length
                    'margin': 0.20
                })

        return obstacles

# ==============================================================================
#  MODULE 4: MPCC Controller
# ==============================================================================

class MPCC(Controller):
    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)
        self._ctrl_freq = config.env.freq
        self._step_count = 0
        self._cfg = config
        
        self.tuning_params = getattr(config, "tuning", {})
        
        self._dyn_params = load_params("so_rpy_rotor_drag", config.sim.drone_model)
        self.hover_thrust = float(self._dyn_params["mass"]) * -float(self._dyn_params["gravity_vec"][-1])
        
        self.planner = PathPlanner(obs["pos"], obs["gates_pos"])
        
        self.N = 35
        self.T_HORIZON = 0.7
        self.model_arc_length = 0.05
        self.model_traj_length = 12.0
        
        self.planner.check_env_update(obs)
        self.planner.replan(obs["pos"], obs, self.model_traj_length)
        
        self.acados_ocp_solver, self.ocp = self._build_ocp_and_solver(self.T_HORIZON, self.N)
        
        self.last_theta = 0.0; self.last_f_cmd = self.hover_thrust; self.last_rpy_cmd = np.zeros(3)
        self.finished = False; self._x_warm = None

    def compute_control(self, obs: dict[str, NDArray[np.floating]], info: dict | None = None) -> NDArray[np.floating]:
        self._current_obs_pos = obs["pos"]
        
        if self.planner.check_env_update(obs):
            self.planner.replan(obs["pos"], obs, self.model_traj_length)
            param_vec = self._encode_traj_params(self.planner.arc_trajectory)
            for k in range(self.N + 1): self.acados_ocp_solver.set(k, "p", param_vec)
            print("       -> Solver params updated.")

        quat = obs["quat"]; r_obj = R.from_quat(quat); roll_pitch_yaw = r_obj.as_euler("xyz", degrees=False)
        drpy = ang_vel2rpy_rates(quat, obs["ang_vel"]) if "ang_vel" in obs else np.zeros(3)
        
        x_now = np.concatenate((obs["pos"], roll_pitch_yaw, obs["vel"], drpy, self.last_rpy_cmd, [self.last_f_cmd], [self.last_theta]))

        if self._x_warm is None:
            self._x_warm = [x_now.copy() for _ in range(self.N + 1)]; self._u_warm = [np.zeros(5) for _ in range(self.N)]
        else:
            self._x_warm = self._x_warm[1:] + [self._x_warm[-1]]; self._u_warm = self._u_warm[1:] + [self._u_warm[-1]]

        for i in range(self.N):
            self.acados_ocp_solver.set(i, "x", self._x_warm[i]); self.acados_ocp_solver.set(i, "u", self._u_warm[i])
        self.acados_ocp_solver.set(self.N, "x", self._x_warm[self.N])
        self.acados_ocp_solver.set(0, "lbx", x_now); self.acados_ocp_solver.set(0, "ubx", x_now)

        if self.last_theta >= self.planner.arc_trajectory.x[-1]: self.finished = True
        
        self.acados_ocp_solver.solve()
        
        self._x_warm = [self.acados_ocp_solver.get(i, "x") for i in range(self.N + 1)]
        self._u_warm = [self.acados_ocp_solver.get(i, "u") for i in range(self.N)]
        x_next = self.acados_ocp_solver.get(1, "x")
        self.last_rpy_cmd = x_next[12:15]; self.last_f_cmd = x_next[15]; self.last_theta = x_next[16]
        
        self._step_count += 1
        return np.array([*self.last_rpy_cmd, self.last_f_cmd], dtype=float)

    def _encode_traj_params(self, trajectory):
        theta = np.arange(0.0, self.model_traj_length, self.model_arc_length)
        pd = trajectory(theta); tp = trajectory(theta, 1)
        qc_gate = np.zeros_like(theta); qc_obst = np.zeros_like(theta)
        
        for gc in self.planner.cached_gates_pos:
            qc_gate = np.maximum(qc_gate, np.exp(-2.0 * np.linalg.norm(pd - gc, axis=1)**2))
        for oc in self.planner.cached_obstacles_pos:
            qc_obst = np.maximum(qc_obst, 1.0 * np.exp(-0.5 * np.linalg.norm(pd[:,:2] - oc[:2], axis=1)**2))
            
        return np.concatenate([pd.ravel(), tp.ravel(), qc_gate, qc_obst])

    def _export_dynamics_model(self) -> AcadosModel:
        """
        使用 drone_models.so_rpy.symbolic_dynamics_euler 的真实动力学：

        X_phys: [px,py,pz, roll,pitch,yaw, vx,vy,vz, dr,dp,dy]   (12)
        U_phys: [r_cmd, p_cmd, y_cmd, f_cmd]                     (4)

        在外面再加 4 个“命令状态” + 1 个 theta:
        X = [X_phys, r_cmd_state, p_cmd_state, y_cmd_state, f_cmd_state, theta] (17)
        U = [df_cmd, dr_cmd, dp_cmd, dy_cmd, v_theta_cmd]

        其中真实动力学的控制输入 U_phys = [r_cmd_state, p_cmd_state, y_cmd_state, f_cmd_state]。
        """

        model_name = "lsy_example_mpc_real"

        params = self._dyn_params

        # 真实动力学（仅 12 状态 + 4 控制）
        X_dot_phys, X_phys, U_phys, _ = symbolic_dynamics_euler(
            mass=params["mass"],
            gravity_vec=params["gravity_vec"],
            J=params["J"],
            J_inv=params["J_inv"],
            acc_coef=params["acc_coef"],
            cmd_f_coef=params["cmd_f_coef"],
            rpy_coef=params["rpy_coef"],
            rpy_rates_coef=params["rpy_rates_coef"],
            cmd_rpy_coef=params["cmd_rpy_coef"],
        )

        # 物理状态别名
        self.px = X_phys[0]
        self.py = X_phys[1]
        self.pz = X_phys[2]
        self.roll = X_phys[3]
        self.pitch = X_phys[4]
        self.yaw = X_phys[5]
        self.vx = X_phys[6]
        self.vy = X_phys[7]
        self.vz = X_phys[8]
        self.dr = X_phys[9]
        self.dp = X_phys[10]
        self.dy = X_phys[11]

        # 命令状态（将作为真实动力学的输入）
        self.r_cmd_state = MX.sym("r_cmd_state")
        self.p_cmd_state = MX.sym("p_cmd_state")
        self.y_cmd_state = MX.sym("y_cmd_state")
        self.f_cmd_state = MX.sym("f_cmd_state")

        # 路径进度 theta
        self.theta = MX.sym("theta")

     
        self.df_cmd = MX.sym("df_cmd")
        self.dr_cmd = MX.sym("dr_cmd")
        self.dp_cmd = MX.sym("dp_cmd")
        self.dy_cmd = MX.sym("dy_cmd")
        self.v_theta_cmd = MX.sym("v_theta_cmd")

        states = vertcat(
            X_phys,
            self.r_cmd_state,
            self.p_cmd_state,
            self.y_cmd_state,
            self.f_cmd_state,
            self.theta,
        )
        inputs = vertcat(
            self.df_cmd,
            self.dr_cmd,
            self.dp_cmd,
            self.dy_cmd,
            self.v_theta_cmd,
        )

        # 真实动力学的控制输入由命令状态给出
        U_phys_full = vertcat(
            self.r_cmd_state,
            self.p_cmd_state,
            self.y_cmd_state,
            self.f_cmd_state,
        )

        # 用 casadi.substitute 把原本的 U_phys 换成 U_phys_full
        f_dyn_phys = substitute(X_dot_phys, U_phys, U_phys_full)

        # 命令状态一阶积分
        r_cmd_dot = self.dr_cmd
        p_cmd_dot = self.dp_cmd
        y_cmd_dot = self.dy_cmd
        f_cmd_dot = self.df_cmd

        theta_dot = self.v_theta_cmd

        f_dyn = vertcat(
            f_dyn_phys,
            r_cmd_dot,
            p_cmd_dot,
            y_cmd_dot,
            f_cmd_dot,
            theta_dot,
        )

        # 轨迹参数
        n_samples = int(self.model_traj_length / self.model_arc_length)
        self.pd_list = MX.sym("pd_list", 3 * n_samples)
        self.tp_list = MX.sym("tp_list", 3 * n_samples)

        # 拆成 gate / obstacle 两类“权重”曲线
        self.qc_gate = MX.sym("qc_gate", 1 * n_samples)
        self.qc_obst = MX.sym("qc_obst", 1 * n_samples)

        params_sym = vertcat(self.pd_list, self.tp_list, self.qc_gate, self.qc_obst)

        model = AcadosModel()
        model.name = model_name
        model.f_expl_expr = f_dyn
        model.x = states
        model.u = inputs
        model.p = params_sym
        return model


    def _piecewise_linear_interp(self, theta, theta_vec, flattened_points, dim: int = 3):
        # 保持你原有的 _piecewise_linear_interp 不变
        M = len(theta_vec); idx_float = (theta - theta_vec[0]) / (theta_vec[-1] - theta_vec[0]) * (M - 1)
        idx_low = floor(idx_float); idx_high = idx_low + 1; alpha = idx_float - idx_low
        idx_low = if_else(idx_low < 0, 0, idx_low); idx_high = if_else(idx_high >= M, M - 1, idx_high)
        p_low = vertcat(*[flattened_points[dim * idx_low + i] for i in range(dim)])
        p_high = vertcat(*[flattened_points[dim * idx_high + i] for i in range(dim)])
        return (1.0 - alpha) * p_low + alpha * p_high

    def _stage_cost_expression(self):
        # 保持你原有的 _stage_cost_expression 不变
        position_vec = vertcat(self.px, self.py, self.pz)
        att_vec = vertcat(self.roll, self.pitch, self.yaw)
        ctrl_vec = vertcat(self.df_cmd, self.dr_cmd, self.dp_cmd, self.dy_cmd)
        theta_grid = np.arange(0.0, self.model_traj_length, self.model_arc_length)
        pd_theta = self._piecewise_linear_interp(self.theta, theta_grid, self.pd_list)
        tp_theta = self._piecewise_linear_interp(self.theta, theta_grid, self.tp_list)
        qc_gate_theta = self._piecewise_linear_interp(self.theta, theta_grid, self.qc_gate, dim=1)
        qc_obst_theta = self._piecewise_linear_interp(self.theta, theta_grid, self.qc_obst, dim=1)
        tp_unit = tp_theta / (norm_2(tp_theta) + 1e-6)
        e_theta = position_vec - pd_theta
        e_lag = dot(tp_unit, e_theta) * tp_unit; e_contour = e_theta - e_lag
        track_cost = ((self.q_l + self.q_l_gate_peak * qc_gate_theta + self.q_l_obst_peak * qc_obst_theta) * dot(e_lag, e_lag) +
                      (self.q_c + self.q_c_gate_peak * qc_gate_theta + self.q_c_obst_peak * qc_obst_theta) * dot(e_contour, e_contour) +
                      att_vec.T @ self.Q_w @ att_vec)
        smooth_cost = ctrl_vec.T @ self.R_df @ ctrl_vec
        speed_cost = (- self.miu * self.v_theta_cmd + self.w_v_gate * qc_gate_theta * (self.v_theta_cmd ** 2) +
                      self.w_v_obst * qc_obst_theta * (self.v_theta_cmd ** 2))
        return track_cost + smooth_cost + speed_cost

    def _build_ocp_and_solver(self, Tf, N_horizon, verbose=False):
        ocp = AcadosOcp()
        model = self._export_dynamics_model()
        ocp.model = model
        self.nx = model.x.rows()
        self.nu = model.u.rows()
        ocp.solver_options.N_horizon = N_horizon
        ocp.cost.cost_type = "EXTERNAL"
        
        self.q_l = self.tuning_params.get("q_l", 200.0)
        self.q_c = self.tuning_params.get("q_c", 100.0)
        
        # 2. 姿态稳定权重 (简化为调节整体倍率)
        w_att_scale = self.tuning_params.get("w_att_scale", 1.0)
        self.Q_w = w_att_scale * DM(np.eye(3))

        # 3. 门与障碍物附近的增强权重
        self.q_l_gate_peak = self.tuning_params.get("q_l_gate", 640.0)
        self.q_c_gate_peak = self.tuning_params.get("q_c_gate", 800.0)
        self.q_l_obst_peak = self.tuning_params.get("q_l_obst", 100.0)
        self.q_c_obst_peak = self.tuning_params.get("q_c_obst", 50.0)

        # 4. 输入平滑权重 (R_df)
        #    把推力(f)和姿态(rpy)的平滑程度分开调
        r_thrust = self.tuning_params.get("r_thrust", 0.1)
        r_rpy    = self.tuning_params.get("r_rpy", 0.5)
        self.R_df = DM(np.diag([r_thrust, r_rpy, r_rpy, r_rpy]))

        # 5. 进度奖励 (飞多快)
        self.miu = self.tuning_params.get("miu", 8.0)
        
        # 6. 减速惩罚 (过门时减速多少)
        self.w_v_gate = self.tuning_params.get("w_v_gate", 4.0)
        self.w_v_obst = self.tuning_params.get("w_v_obst", 1.0)
        
        
        ocp.model.cost_expr_ext_cost = self._stage_cost_expression()
        thrust_min = float(self._dyn_params["thrust_min"]) * 4.0
        thrust_max = float(self._dyn_params["thrust_max"]) * 4.0
        ocp.constraints.lbx = np.array([thrust_min, -1.57, -1.57, -1.57])
        ocp.constraints.ubx = np.array([thrust_max, 1.57, 1.57, 1.57])
        ocp.constraints.idxbx = np.array([15, 12, 13, 14])
        ocp.constraints.lbu = np.array([-10.0, -10.0, -10.0, -10.0, 0.0])
        ocp.constraints.ubu = np.array([10.0, 10.0, 10.0, 10.0, 4.0])
        ocp.constraints.idxbu = np.array([0, 1, 2, 3, 4])
        ocp.constraints.x0 = np.zeros(self.nx)
        ocp.parameter_values = np.zeros(3 * int(12.0/0.05) * 2 + 2 * int(12.0/0.05))
        ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.nlp_solver_type = "SQP_RTI"
        ocp.solver_options.tf = Tf
        solver = AcadosOcpSolver(ocp, json_file="mpcc_prescripted_real_dyn.json", verbose=verbose)
        return solver, ocp

    def step_callback(self, action, obs, reward, terminated, truncated, info): 
        return self.finished
    
    def episode_callback(self):
        print("[MPCC] Episode reset.")
        self._step_count = 0; self.finished = False; self._x_warm = None
        self.last_theta = 0.0; self.last_f_cmd = self.hover_thrust; self.last_rpy_cmd = np.zeros(3)
        self.planner._last_gate_flags = None

    def get_debug_lines(self):
        debug_lines = []
        if self.planner.arc_trajectory is not None:
            try:
                t_end = self.planner.arc_trajectory.x[-1]
                if t_end > 0.1:
                    pts = self.planner.arc_trajectory(np.linspace(0, t_end, 100))
                    debug_lines.append((pts, np.array([0.0, 1.0, 0.0, 1.0]), 2.0, 2.0))
            except: pass
        if self._x_warm is not None:
            try:
                pts = np.asarray([x[:3] for x in self._x_warm])
                debug_lines.append((pts, np.array([1.0, 0.0, 0.0, 1.0]), 3.0, 3.0))
            except: pass
        return debug_lines