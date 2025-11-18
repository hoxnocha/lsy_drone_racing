
from __future__ import annotations  

from typing import TYPE_CHECKING, List

import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from casadi import MX, cos, sin, vertcat, dot, DM, norm_2, floor, if_else
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import CubicSpline

from lsy_drone_racing.control import Controller  
from drone_models.core import load_params 

if TYPE_CHECKING:
    from numpy.typing import NDArray

class TransformTool:
    @staticmethod
    def quad_to_norm(quad: NDArray[np.floating], axis : int= 1) -> NDArray[np.floating]:
        '''
        Return the normal vector of gates(x-axis, y-axis, z-axis)
        '''
        rotates = R.from_quat(quad)
        rot_matrices = np.array(rotates.as_matrix())
        if len(rot_matrices.shape) == 3:
            return np.array(rot_matrices[:,:,axis]) 
        elif len(rot_matrices.shape) == 2:
            return np.array(rot_matrices[:,axis])
        else:
            return None
    
    @staticmethod
    def vector_to_quaternion_z_to_v(v: np.ndarray) -> NDArray[np.floating]:
        v = v / np.linalg.norm(v)
        z = np.array([0.0, 0.0, 1.0])

        if np.allclose(v, z):
            quat = [0, 0, 0, 1] # identity rotation
        elif np.allclose(v, -z):
            quat = R.from_rotvec(np.pi * np.array([1, 0, 0])).as_quat()
        else:
            rot_axis = np.cross(z, v)
            rot_axis /= np.linalg.norm(rot_axis)
            angle = np.arccos(np.clip(np.dot(z, v), -1.0, 1.0))
            quat = R.from_rotvec(angle * rot_axis).as_quat() # [x, y, z, w]
        return quat
    
class LinAlgTool:
    @staticmethod
    def normalize(vec: NDArray[np.floating]) -> NDArray[np.floating]:
        norm = np.linalg.norm(vec)
        if norm < 1e-6:
            return vec
        return vec / norm
    
    @staticmethod
    def dot_safe(a: NDArray[np.floating], b: NDArray[np.floating]) -> float:
        return float(np.clip(np.dot(a, b), -1.0, 1.0))

class PolynomialTool:
    @staticmethod
    def cubic_solve_real(a : np.floating, b : np.floating, c : np.floating, d: np.floating) -> List[np.float64]:
        roots = np.roots(np.array([a,b,c,d], dtype = np.float64))
        return [r.real for r in roots if np.isclose(r.imag, 0)]
    
    @staticmethod
    def quartic_solve_real(a : np.floating, b : np.floating, c : np.floating, d: np.floating, e : np.floating) -> List[np.float64]:
        roots = np.roots(np.array([a,b,c,d,e], dtype = np.float64))
        return [r.real for r in roots if np.isclose(r.imag, 0)]

class TwoTraj:
    trajectory_1 : CubicSpline
    trajectory_2 : CubicSpline
    offset : np.floating
    x : NDArray[np.floating]

    def __init__(self, spline_1 : CubicSpline, spline_2 : CubicSpline, offset : np.floating):
        self.trajectory_1 = spline_1
        self.trajectory_2 = spline_2
        self.offset = offset
        self.x = np.concatenate([spline_1.x, spline_2.x + offset])

    def __call__(self, t):
        if np.isscalar(t):
            if t < self.offset:
                return self.trajectory_1(t)
            else:
                return self.trajectory_2(t - self.offset)
        else:
            return np.array([self(tau) for tau in t])
    
    def derivative(self, k):
        derivative_1 = self.trajectory_1.derivative(k)
        derivative_2 = self.trajectory_2.derivative(k)
        return TwoTraj(derivative_1, derivative_2, offset = self.offset)
        

class TrajectoryTool:

    def compute_3d_curvature_from_vector_spline(self, spline: CubicSpline, t_vals: np.ndarray, eps : np.ndarray = 1e-8, positive : bool= True) -> np.ndarray:
        r_dot = spline(t_vals, 1)    # shape: (len(t_vals), 3)
        r_ddot = spline(t_vals, 2)  # shape: (len(t_vals), 3)
        cross = np.cross(r_dot, r_ddot)
        cross_norm = np.linalg.norm(cross, axis=1)
        r_dot_norm = np.linalg.norm(r_dot, axis=1)
        curvature = cross_norm / (r_dot_norm ** 3 + eps)
        return np.abs(curvature) if positive else curvature
    
    def compute_3d_turning_radius_from_vector_spline(self, spline: CubicSpline, t_vals: np.ndarray, eps : np.ndarray = 1e-8, positive : bool = True) -> np.ndarray:
        r_dot = spline(t_vals, 1)    # shape: (len(t_vals), 3)
        r_ddot = spline(t_vals, 2)  # shape: (len(t_vals), 3)
        cross = np.cross(r_dot, r_ddot)
        cross_norm = np.linalg.norm(cross, axis=1)
        r_dot_norm = np.linalg.norm(r_dot, axis=1)
        radius = (r_dot_norm ** 3)/ (cross_norm + eps)
        return np.abs(radius) if positive else radius

    def calc_waypoints(
            self, drone_init_pos: NDArray[np.floating], gates_pos: NDArray[np.floating], gates_norm: NDArray[np.floating], distance: float = 0.5, num_int_pnts: int = 5,
    ) -> NDArray[np.floating]:
        """Compute waypoints interpolated between gates."""
        num_gates = gates_pos.shape[0]
        wp = np.concatenate([gates_pos - distance * gates_norm + i/(num_int_pnts-1) * 2 * distance * gates_norm for i in range(num_int_pnts)], axis=1).reshape(num_gates, num_int_pnts, 3).reshape(-1,3)
        wp = np.concatenate([np.array([drone_init_pos]), wp], axis=0)
        return wp
    
    def trajectory_generate(
        self, t_total: float, waypoints: NDArray[np.floating],
    ) -> CubicSpline:
        """Generate a cubic spline trajectory from waypoints."""
        diffs = np.diff(waypoints, axis=0)
        segment_length = np.linalg.norm(diffs, axis=1)
        arc_cum_length = np.concatenate([[0], np.cumsum(segment_length)])
        t = arc_cum_length / (arc_cum_length[-1] + 1e-6) * t_total # 增加了 1e-6 避免除零
        return CubicSpline(t, waypoints)
    
    def arclength_reparameterize(
            self, trajectory: CubicSpline, arc_length:float = 0.05, epsilon:float = 1e-5
        ):
        """reparameterize trajectory by arc length
        return a CubicSpline object with parameter t in [0, total_length] and is uniform in arc_length
        """
        # initialize total_length by t_total
        total_length = trajectory.x[-1] - trajectory.x[0]
        for _ in range(99):
            # sample total_length/0.1 waypoints
            t_sample = np.linspace(0, total_length, int(total_length / arc_length))
            wp_sample = trajectory(t_sample)
            # measure linear distances
            diffs = np.diff(wp_sample, axis=0)
            segment_length = np.linalg.norm(diffs, axis=1)
            arc_cum_length = np.concatenate([[0], np.cumsum(segment_length)])
            t_reallocate = arc_cum_length
            total_length = arc_cum_length[-1]
            # regenerate spline function
            trajectory = CubicSpline(t_reallocate, wp_sample)
            # terminal condition
            if np.std(segment_length) <= epsilon:
                return CubicSpline(t_reallocate, wp_sample)
        return CubicSpline(t_reallocate, wp_sample)
            
    def extend_trajectory(self, trajectory: CubicSpline, extend_length:float = 1):
        """takes an arbirary 3D trajectory and extend it in the direction of terminal derivative."""
        theta_original = trajectory.x
        delta_theta = min(theta_original[1] - theta_original[0], 0.2)
        # calc last derivative
        p_end = trajectory(theta_original[-1])
        dp_end = trajectory.derivative(1)(theta_original[-1])
        dp_end_normalized = dp_end / (np.linalg.norm(dp_end) + 1e-6) # 增加了 1e-6
        # calc extended theta list
        theta_extend = np.arange(theta_original[-1] + delta_theta, theta_original[-1] + extend_length, delta_theta)
        p_extend = np.array([p_end + dp_end_normalized * (s - theta_original[-1]) for s in theta_extend])
        # cat original traj and extended traj
        theta_new = np.concatenate([theta_original, theta_extend])
        p_new = np.vstack([trajectory(theta_original), p_extend])

        extended_trajectory = CubicSpline(theta_new, p_new, axis=0)
        return extended_trajectory

    def traj_preprocessing(self, t, pos):
        # 1. find 3rd gate turning point
        idx_max = 20 + np.argmax(np.array(pos)[20:,1])
        # 2. split waypoints
        t = np.array(t)
        t1 = t[:idx_max+1]
        pos1 = pos[:idx_max+1]
        t2 = t[idx_max:] - t[idx_max]
        pos2 = pos[idx_max:]
        # 3. generate splines separately
        trajectory1 = CubicSpline(t1, pos1)
        trajectory2 = CubicSpline(t2, pos2)
        # 4. arc length reparameterize
        arc_trajectory1 = self.arclength_reparameterize(trajectory1)
        arc_trajectory2 = self.arclength_reparameterize(trajectory2)
        arc_trajectory1_cut = CubicSpline(arc_trajectory1.x[:-1], arc_trajectory1(arc_trajectory1.x[:-1]))
        return TwoTraj(arc_trajectory1_cut, arc_trajectory2, arc_trajectory1.x[-1])
            
    def find_nearest_waypoint(
            self, trajectory: CubicSpline, pos: NDArray[np.floating], total_length: float = None, sample_interval:float = 0.05
            ):
        """find nearest waypoint to given position on a trajectory
        return index and 3D waypoint
        """
        if total_length is None:
            total_length = trajectory.x[-1]
        # sample waypoints
        # t_sample = np.linspace(0, total_length, int(total_length / sample_interval))
        t_sample = np.arange(0, total_length, sample_interval)
        if len(t_sample) == 0: # 增加保护
             return 0.0, trajectory(0.0)
        wp_sample = trajectory(t_sample)
        # find nearest waypoint
        distances = np.linalg.norm(wp_sample - pos, axis=1)
        idx_nearest = np.argmin(distances)
        return idx_nearest * sample_interval, wp_sample[idx_nearest]
    
    def find_gate_waypoint(
            self, trajectory: CubicSpline, gates_pos: NDArray[np.floating], total_length: float = None, sample_interval:float = 0.05
        ):
        """find waypoints of gates center, mainly corresponding indices
        """
        if total_length is None:
            total_length = trajectory.x[-1]
        indices = []
        gates_wp = []
        for pos in gates_pos:
            idx, wp = self.find_nearest_waypoint(trajectory, pos, total_length, sample_interval)
            indices.append(idx)
            gates_wp.append(wp)
        
        # NOTE: 你的代码中有一个关于第3个门的特殊处理
        # 这里为了通用性，我们先去掉它。如果需要，可以再加回来。
        # total_length = trajectory.x[-1]
        # t_sample = np.arange(0, total_length, sample_interval)
        # wp_sample = trajectory(t_sample)
        # start_idx2 = int((indices[2]-1.0)//sample_interval)
        # idx2 = start_idx2 + np.argmax(wp_sample[start_idx2:, 1]-wp_sample[start_idx2:, 2])
        # gates_wp[2] = wp_sample[idx2]
        # indices[2] = idx2 * sample_interval
        
        return np.array(indices), np.array(gates_wp)

class GeometryTool:
    @staticmethod
    def trilinear_interpolation(grid: np.ndarray, idx_f: np.ndarray) -> float:
        x, y, z = idx_f
        x0, y0, z0 = np.floor([x, y, z]).astype(int)
        dx, dy, dz = x - x0, y - y0, z - z0

        def get(ix, iy, iz):
            if 0 <= ix < grid.shape[0] and 0 <= iy < grid.shape[1] and 0 <= iz < grid.shape[2]:
                return grid[ix, iy, iz]
            else:
                return 0.0

        c000 = get(x0, y0, z0)
        c001 = get(x0, y0, z0 + 1)
        c010 = get(x0, y0 + 1, z0)
        c011 = get(x0, y0 + 1, z0 + 1)
        c100 = get(x0 + 1, y0, z0)
        c101 = get(x0 + 1, y0, z0 + 1)
        c110 = get(x0 + 1, y0 + 1, z0)
        c111 = get(x0 + 1, y0 + 1, z0 + 1)

        c00 = c000 * (1 - dx) + c100 * dx
        c01 = c001 * (1 - dx) + c101 * dx
        c10 = c010 * (1 - dx) + c110 * dx
        c11 = c011 * (1 - dx) + c111 * dx

        c0 = c00 * (1 - dy) + c10 * dy
        c1 = c01 * (1 - dy) + c11 * dy

        return c0 * (1 - dz) + c1 * dz



class MPCC(Controller):  # 继承自 Controller
    """Implementation of MPCC using the collective thrust and attitude interface."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the attitude controller.
        """
        super().__init__(obs, info, config)  # 调用 Controller 的 __init__
        self.freq = config.env.freq
        self._tick = 0
        self.config = config

        # 存储初始位置
        self.init_pos = obs['pos']
        # 手动存储门的位置 (替代 EasyController 的 self.gates)
        self.gates_pos_store = obs["gates_pos"] 
        # 设置轨迹的默认总时间 (会被 avoid_collision 覆盖)
        self.t_total = 30 
        
        
        self.traj_tool = TrajectoryTool()

        # 执行初始轨迹规划
        self._plan_trajectory(obs)
        # self.trajectory 是现在由 _plan_trajectory 生成 (CubicSpline)

        # global params
        self.N = 50
        self.T_HORIZON = 0.6
        self.dt = self.T_HORIZON / self.N
        self.model_arc_length = 0.05 # the segment interval for trajectory to be input to the model
        self.model_traj_length = 12 # maximum trajectory length the param can take

        
        self.arc_trajectory = self.traj_tool.arclength_reparameterize(
            self.traj_tool.extend_trajectory(self.trajectory, extend_length=self.model_traj_length) # 确保延长足够长
        )
        
        # build model & create solver
        self.acados_ocp_solver, self.ocp = self.create_ocp_solver(
            self.T_HORIZON, self.N, self.arc_trajectory
        )

        # initialize
        self.pos_bound = [np.array([-2.6, 2.6]), np.array([-2.0, 1.8]), np.array([-0.1, 2.0])]
        self.velocity_bound = [-1.0, 4.0]

        self.last_theta = 0.0
        self.last_v_theta = 0.0 # TODO: replan?
        self.last_f_collective = 0.3
        self.last_rpy_cmd = np.zeros(3)
        self.last_f_cmd = 0.3
        self.finished = False
        
    

    def export_quadrotor_ode_model(self) -> AcadosModel:
        """Symbolic Quadrotor Model."""
        model_name = "lsy_example_mpc"
        GRAVITY = 9.806
        params_pitch_rate = [-6.003842038081178, 6.213752925707588]
        params_roll_rate = [-3.960889336015948, 4.078293254657104]
        params_yaw_rate = [-0.005347588299390372, 0.0]
        params_acc = [20.907574256269616, 3.653687545690674]

        self.px = MX.sym("px")
        self.py = MX.sym("py")
        self.pz = MX.sym("pz")
        self.vx = MX.sym("vx")
        self.vy = MX.sym("vy")
        self.vz = MX.sym("vz")
        self.roll = MX.sym("r")
        self.pitch = MX.sym("p")
        self.yaw = MX.sym("y")
        self.f_collective = MX.sym("f_collective")
        self.f_collective_cmd = MX.sym("f_collective_cmd")
        self.r_cmd = MX.sym("r_cmd")
        self.p_cmd = MX.sym("p_cmd")
        self.y_cmd = MX.sym("y_cmd")
        self.df_cmd = MX.sym("df_cmd")
        self.dr_cmd = MX.sym("dr_cmd")
        self.dp_cmd = MX.sym("dp_cmd")
        self.dy_cmd = MX.sym("dy_cmd")
        self.theta = MX.sym("theta")
        self.v_theta_cmd = MX.sym("v_theta_cmd")

        states = vertcat(
            self.px, self.py, self.pz, self.vx, self.vy, self.vz,
            self.roll, self.pitch, self.yaw, self.f_collective,
            self.f_collective_cmd, self.r_cmd, self.p_cmd, self.y_cmd,
            self.theta
        )
        inputs = vertcat(
            self.df_cmd, self.dr_cmd, self.dp_cmd, self.dy_cmd, 
            self.v_theta_cmd
        )

        f = vertcat(
            self.vx,
            self.vy,
            self.vz,
            (params_acc[0] * self.f_collective + params_acc[1])
            * (cos(self.roll) * sin(self.pitch) * cos(self.yaw) + sin(self.roll) * sin(self.yaw)),
            (params_acc[0] * self.f_collective + params_acc[1])
            * (cos(self.roll) * sin(self.pitch) * sin(self.yaw) - sin(self.roll) * cos(self.yaw)),
            (params_acc[0] * self.f_collective + params_acc[1]) * cos(self.roll) * cos(self.pitch) - GRAVITY,
            params_roll_rate[0] * self.roll + params_roll_rate[1] * self.r_cmd,
            params_pitch_rate[0] * self.pitch + params_pitch_rate[1] * self.p_cmd,
            params_yaw_rate[0] * self.yaw + params_yaw_rate[1] * self.y_cmd,
            10.0 * (self.f_collective_cmd - self.f_collective),
            self.df_cmd, self.dr_cmd, self.dp_cmd, self.dy_cmd,
            self.v_theta_cmd,
        )

        self.pd_list = MX.sym("pd_list", 3*int(self.model_traj_length/self.model_arc_length))
        self.tp_list = MX.sym("tp_list", 3*int(self.model_traj_length/self.model_arc_length))
        self.qc_dyn = MX.sym("qc_dyn", 1*int(self.model_traj_length/self.model_arc_length))
        params = vertcat(self.pd_list, self.tp_list, self.qc_dyn)

        model = AcadosModel()
        model.name = model_name
        model.f_expl_expr = f
        model.x = states
        model.u = inputs
        model.p = params
        return model
    
    
    def casadi_linear_interp(self, theta, theta_list, p_flat, dim=3):
        M = len(theta_list)
        idx_float = (theta - theta_list[0]) / (theta_list[-1] - theta_list[0]) * (M - 1)
        idx_lower = floor(idx_float)
        idx_upper = idx_lower + 1
        alpha = idx_float - idx_lower
        idx_lower = if_else(idx_lower < 0, 0, idx_lower)
        idx_upper = if_else(idx_upper >= M, M-1, idx_upper)
        p_lower = vertcat(*[p_flat[dim * idx_lower + d] for d in range(dim)])
        p_upper = vertcat(*[p_flat[dim * idx_upper + d] for d in range(dim)])
        p_interp = (1 - alpha) * p_lower + alpha * p_upper
        return p_interp
    
    def get_updated_traj_param(self, trajectory: CubicSpline):
        """get updated trajectory parameters upon replaning"""
        theta_list = np.arange(0, self.model_traj_length, self.model_arc_length)
        pd_list = trajectory(theta_list)
        tp_list = trajectory.derivative(1)(theta_list)
        qc_dyn_list = np.zeros_like(theta_list)
        
        if hasattr(self, "gates_pos_store"):
            for gate_pos in self.gates_pos_store:
                distances = np.linalg.norm(pd_list - gate_pos, axis=-1)
                qc_dyn_gate = np.exp(-5 * distances**2) # gaussian
                qc_dyn_list = np.maximum(qc_dyn_gate, qc_dyn_list)
                
        p_vals = np.concatenate([pd_list.flatten(), tp_list.flatten(), qc_dyn_list])
        return p_vals

    def mpcc_cost(self):
        """calculate mpcc cost function"""
        pos = vertcat(self.px, self.py, self.pz)
        ang = vertcat(self.roll, self.pitch, self.yaw)
        control_input = vertcat(self.f_collective_cmd, self.dr_cmd, self.dp_cmd, self.dy_cmd)
        
        theta_list = np.arange(0, self.model_traj_length, self.model_arc_length)
        pd_theta = self.casadi_linear_interp(self.theta, theta_list, self.pd_list)
        tp_theta = self.casadi_linear_interp(self.theta, theta_list, self.tp_list)
        qc_dyn_theta = self.casadi_linear_interp(self.theta, theta_list, self.qc_dyn, dim=1)
        tp_theta_norm = tp_theta / (norm_2(tp_theta) + 1e-6) # 增加了 1e-6
        e_theta = pos - pd_theta
        e_l = dot(tp_theta_norm, e_theta) * tp_theta_norm
        e_c = e_theta - e_l

        mpcc_cost = (self.q_l + self.q_l_peak * qc_dyn_theta) * dot(e_l, e_l) + \
                    (self.q_c  + self.q_c_peak * qc_dyn_theta) * dot(e_c, e_c) + \
                    (ang.T @ self.Q_w @ ang) + \
                    (control_input.T @ self.R_df @ control_input) + \
                    (-self.miu) * self.v_theta_cmd
        return mpcc_cost

    def create_ocp_solver(
        self, Tf: float, N: int, trajectory: CubicSpline,  verbose: bool = False
    ) -> tuple[AcadosOcpSolver, AcadosOcp]:
        """Creates an acados Optimal Control Problem and Solver."""
        ocp = AcadosOcp()
        model = self.export_quadrotor_ode_model()
        ocp.model = model
        self.nx = model.x.rows()
        self.nu = model.u.rows()
        ocp.solver_options.N_horizon = N
        ocp.cost.cost_type = "EXTERNAL"

        self.q_l = 160
        self.q_l_peak = 640
        self.q_c =  80
        self.q_c_peak = 800
        self.Q_w = 1 * DM(np.eye(3))
        self.R_df = DM(np.diag([1,0.5,0.5,0.5]))
        self.miu = 8
        
        ocp.model.cost_expr_ext_cost = self.mpcc_cost()

        ocp.constraints.lbx = np.array([0.1, 0.1, -1.57, -1.57, -1.57]) 
        ocp.constraints.ubx = np.array([0.55, 0.55, 1.57, 1.57, 1.57])
        ocp.constraints.idxbx = np.array([9, 10, 11, 12, 13])

        ocp.constraints.lbu = np.array([-10.0, -10.0, -10.0, -10.0, 0.0])
        ocp.constraints.ubu = np.array([10.0, 10.0, 10.0, 10.0, 4.0])
        ocp.constraints.idxbu = np.array([0, 1, 2, 3, 4])

        ocp.constraints.x0 = np.zeros((self.nx))
        p_vals = self.get_updated_traj_param(self.arc_trajectory)
        ocp.parameter_values = p_vals

        ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.nlp_solver_type = "SQP_RTI"
        ocp.solver_options.tol = 1e-5
        ocp.solver_options.qp_solver_cond_N = N
        ocp.solver_options.qp_solver_warm_start = 1
        ocp.solver_options.qp_solver_iter_max = 20
        ocp.solver_options.nlp_solver_max_iter = 50
        ocp.solver_options.tf = Tf

        acados_ocp_solver = AcadosOcpSolver(ocp, json_file="mpcc_prescripted.json", verbose=verbose)
        return acados_ocp_solver, ocp

   

    def _plan_trajectory(self, obs: dict[str, NDArray[np.floating]]):
        """
        根据当前的 obs 生成或重新生成一条完整的轨迹。
        """
        print(f"T={self._tick / self.freq:.2f}: (Re)planning trajectory...")

        current_gates_pos = obs['gates_pos']
        current_obstacles_pos = obs['obstacles_pos']
        current_gates_quat = obs['gates_quat']
        
        self.gates_pos_store = current_gates_pos
        
        # [MODIFIED] 使用 TransformTool (静态方法)
        gates_norm = TransformTool.quad_to_norm(current_gates_quat, axis=0)
        gate_y_axes = TransformTool.quad_to_norm(current_gates_quat, axis=1)
        gate_z_axes = TransformTool.quad_to_norm(current_gates_quat, axis=2)
        
        # [MODIFIED] 使用 self.traj_tool 的方法
        waypoints = self.traj_tool.calc_waypoints(self.init_pos, current_gates_pos, gates_norm)
        
        waypoints = self._add_detour_waypoints( # 这个仍在 MPCC 类中
            waypoints, current_gates_pos, gates_norm,
            gate_y_axes, gate_z_axes
        )
        
        t, waypoints_avoided = self.avoid_collision(waypoints, current_obstacles_pos, 0.3) # 这个也在 MPCC 类中

        if len(t) < 2:
            print("警告: _plan_trajectory 中 avoid_collision 返回点少于2个。")
            self.trajectory = self.traj_tool.trajectory_generate(self.t_total, waypoints)
        else:
            self.trajectory = CubicSpline(t, waypoints_avoided)
            self.t_total = self.trajectory.x[-1] # 更新总时间
    
    
    def avoid_collision(
        self, waypoints: NDArray[np.floating], obstacles_pos: NDArray[np.floating], safe_dist: float,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """修改路径点以避免与障碍物碰撞 (来自 level2_1.py)。"""
        # [MODIFIED] 使用 self.traj_tool.trajectory_generate
        pre_trajectory = self.traj_tool.trajectory_generate(self.t_total, waypoints)
        
        num_steps = int(self.freq * self.t_total)
        if num_steps <= 0: num_steps = 1
        t_axis = np.linspace(0, self.t_total, num_steps)
        wp = pre_trajectory(t_axis)

        for obst_idx, obst in enumerate(obstacles_pos):
            flag = False
            t_results = []
            wp_results = []
            
            for i in range(wp.shape[0]):
                point = wp[i]
                if np.linalg.norm(obst[:2] - point[:2]) < safe_dist and not flag: 
                    flag = True
                    in_idx = i
                elif np.linalg.norm(obst[:2] - point[:2]) >= safe_dist and flag:    
                    out_idx = i
                    flag = False
                    direction = wp[in_idx][:2] - obst[:2] + wp[out_idx][:2] - obst[:2]
                    direction = direction / (np.linalg.norm(direction) + 1e-6)
                    new_point_xy = obst[:2] + direction * safe_dist
                    new_point_z = (wp[in_idx][2] + wp[out_idx][2])/2
                    new_point = np.concatenate([new_point_xy, [new_point_z]])
                    t_results.append((t_axis[in_idx] + t_axis[out_idx])/2)
                    wp_results.append(new_point)
                elif np.linalg.norm(obst[:2] - point[:2]) >= safe_dist:   
                    t_results.append(t_axis[i])
                    wp_results.append(point)
            
            if flag:
                t_results.append(t_axis[-1])
                wp_results.append(wp[-1])

            t_axis = np.array(t_results)
            wp = np.array(wp_results)

        if len(t_axis) > 0:
            unique_indices = np.unique(t_axis, return_index=True)[1]
            t_axis = t_axis[unique_indices]
            wp = wp[unique_indices]

        if len(t_axis) < 2:
            print("Avoid_collision: 过滤后点不足，返回原始路径点。")
            t_axis_fallback = self.traj_tool.trajectory_generate(self.t_total, waypoints).x
            wp_fallback = waypoints
            return t_axis_fallback, wp_fallback

        return t_axis, wp
    
    def pos_change_detect(self, obs: dict[str, NDArray[np.bool_]]) -> bool:
        """
        检测 'visited' 标志是否从 False 变为 True。
        """
        if not hasattr(self, 'last_gate_flags'):
            self.last_gate_flags = np.array(obs.get('gates_visited', []), dtype=bool)
            self.last_obst_flags = np.array(obs.get('obstacles_visited', []), dtype=bool)
            return False

        curr_gate_flags = np.array(obs.get('gates_visited', []), dtype=bool)
        curr_obst_flags = np.array(obs.get('obstacles_visited', []), dtype=bool)
        
        # 增加长度检查，以防 obs 结构变化
        if curr_gate_flags.shape != self.last_gate_flags.shape:
            self.last_gate_flags = curr_gate_flags
            return False # 形状不匹配，跳过一次
        if curr_obst_flags.shape != self.last_obst_flags.shape:
            self.last_obst_flags = curr_obst_flags
            return False # 形状不匹配，跳过一次
            
        gate_triggered = np.any((~self.last_gate_flags) & curr_gate_flags)
        obst_triggered = np.any((~self.last_obst_flags) & curr_obst_flags)
        
        self.last_gate_flags = curr_gate_flags
        self.last_obst_flags = curr_obst_flags
        return gate_triggered or obst_triggered
    
    def _extract_gate_coordinate_frames(
        self, 
        gates_quaternions: NDArray[np.floating]
    ) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
        """[MODIFIED] 使用 TransformTool"""
        normals = TransformTool.quad_to_norm(gates_quaternions, axis=0) # X 轴
        y_axes = TransformTool.quad_to_norm(gates_quaternions, axis=1)  # Y 轴
        z_axes = TransformTool.quad_to_norm(gates_quaternions, axis=2)  # Z 轴
        return normals, y_axes, z_axes

    def _add_detour_waypoints(
        self,
        waypoints: NDArray[np.floating],
        gate_positions: NDArray[np.floating],
        gate_normals: NDArray[np.floating],
        gate_y_axes: NDArray[np.floating],
        gate_z_axes: NDArray[np.floating],
        num_intermediate_points: int = 5,
        angle_threshold: float = 120.0,
        detour_distance: float = 0.65
    ) -> NDArray[np.floating]:
        """(与 level2_1.py 相同，保留在 MPCC 类中)"""
        num_gates = gate_positions.shape[0]
        waypoints_list = list(waypoints)  
        inserted_count = 0
        
        for i in range(num_gates - 1):
            last_idx_gate_i = 1 + (i + 1) * num_intermediate_points - 1 + inserted_count
            first_idx_gate_i_plus_1 = 1 + (i + 1) * num_intermediate_points + inserted_count
            
            # 增加边界检查
            if last_idx_gate_i >= len(waypoints_list) or first_idx_gate_i_plus_1 >= len(waypoints_list):
                break # 索引超出范围，停止添加
                
            p1 = waypoints_list[last_idx_gate_i]
            p2 = waypoints_list[first_idx_gate_i_plus_1]
            v = p2 - p1
            v_norm = np.linalg.norm(v)
            if v_norm < 1e-6: continue
            
            normal_i = gate_normals[i]
            cos_angle = np.dot(v, normal_i) / v_norm
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle_deg = np.arccos(cos_angle) * 180 / np.pi
            
            if angle_deg > angle_threshold:
                gate_center = gate_positions[i]
                y_axis = gate_y_axes[i]
                z_axis = gate_z_axes[i]
                v_proj = v - np.dot(v, normal_i) * normal_i
                v_proj_norm = np.linalg.norm(v_proj)
                
                if v_proj_norm < 1e-6:
                    detour_direction_vector = y_axis
                else:
                    v_proj_y = np.dot(v_proj, y_axis)
                    v_proj_z = np.dot(v_proj, z_axis)
                    proj_angle_deg = np.arctan2(v_proj_z, v_proj_y) * 180 / np.pi
                    
                    if -90 <= proj_angle_deg < 45:
                        detour_direction_vector = y_axis
                    elif 45 <= proj_angle_deg < 135:
                        detour_direction_vector = z_axis
                    else:
                        detour_direction_vector = -y_axis
                
                detour_waypoint = gate_center + detour_distance * detour_direction_vector
                insert_position = last_idx_gate_i + 1
                waypoints_list.insert(insert_position, detour_waypoint)
                inserted_count += 1
        
        return np.array(waypoints_list)

    

    def position_out_of_bound(self, pos : NDArray[np.floating]):
        if self.pos_bound is None: return False
        for i in range(3):
            if pos[i] < self.pos_bound[i][0] or pos[i] > self.pos_bound[i][1]:
                return True
        return False

    def velocity_out_of_bound(self, vel : NDArray[np.floating]):
        if self.velocity_bound is None: return False
        velocity = np.linalg.norm(vel)
        return not(self.velocity_bound[0] < velocity < self.velocity_bound[1])

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired attitude and thrust command."""
        
        self._current_obs_pos = obs["pos"] # 存储当前位置用于调试

        # ---== L2 RE-PLANNING LOGIC ==---
        if self.pos_change_detect(obs):
            print(f"T={self._tick / self.freq:.2f}: MPCC detecting object, replanning...")
            self._plan_trajectory(obs) # 重新规划
            
            # [FIXED] 重新进行弧长参数化
            self.arc_trajectory = self.traj_tool.arclength_reparameterize(
                self.traj_tool.extend_trajectory(self.trajectory, extend_length=self.model_traj_length)
            )
            # [FIXED] 更新求解器中的轨迹参数
            p_vals = self.get_updated_traj_param(self.arc_trajectory)
            for i in range(self.N + 1):
                self.acados_ocp_solver.set(i, "p", p_vals)
        # ---== END L2 LOGIC ==---

        q = obs["quat"]
        r = R.from_quat(q)
        rpy = r.as_euler("xyz", degrees=False)

        xcurrent = np.concatenate(
            (
                obs["pos"], obs["vel"], rpy,
                np.array([self.last_f_collective, self.last_f_cmd]),
                self.last_rpy_cmd,
                np.array([self.last_theta])
            )
        )
        
        if not hasattr(self, "x_guess"):
            self.x_guess = [xcurrent for _ in range(self.N + 1)]
            self.u_guess = [np.zeros(self.nu) for _ in range(self.N)]
        else:
            self.x_guess = self.x_guess[1:] + [self.x_guess[-1]]
            self.u_guess = self.u_guess[1:] + [self.u_guess[-1]]

        for i in range(self.N):
            self.acados_ocp_solver.set(i, "x", self.x_guess[i])
            self.acados_ocp_solver.set(i, "u", self.u_guess[i])
        self.acados_ocp_solver.set(self.N, "x", self.x_guess[self.N])

        self.acados_ocp_solver.set(0, "lbx", xcurrent)
        self.acados_ocp_solver.set(0, "ubx", xcurrent)

        # Safety checks
        if self.last_theta >= self.arc_trajectory.x[-1]: # 检查是否到达轨迹末端
            self.finished = True
            print("Quit-finished")
        if self.position_out_of_bound(obs["pos"]):
            self.finished = True
            print("Quit-flying out of safe area")
        if self.velocity_out_of_bound(obs["vel"]):
            self.finished = True
            print("Quit-out of safe velocity range")
        
        status = self.acados_ocp_solver.solve()
        
        self.x_guess = [self.acados_ocp_solver.get(i, "x") for i in range(self.N + 1)]
        self.u_guess = [self.acados_ocp_solver.get(i, "u") for i in range(self.N)]

        x1 = self.acados_ocp_solver.get(1, "x")
        w = 1 / self.config.env.freq / self.dt
        self.last_f_collective = self.last_f_collective * (1 - w) + x1[9] * w
        self.last_theta = self.last_theta * (1 - w) + x1[14] * w
        self.last_f_cmd = self.last_f_cmd * (1-w) + x1[10] * w
        self.last_rpy_cmd = self.last_rpy_cmd * (1-w) + x1[11:14] * w

        cmd = np.concatenate(
            (
                np.array([self.last_f_cmd]),
                self.last_rpy_cmd
            )
        )
        return cmd

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Increment the tick counter."""
        self._tick += 1
        return self.finished

    def episode_callback(self):
        """
        Reset the controller's internal state.
        (逻辑从 episode_reset 移到这里)
        """
        print("[MPCC] Episode reset.")
        self._tick = 0
        self.finished = False
        
        # 重置 'visited' 标志的跟踪器
        if hasattr(self, 'last_gate_flags'):
            delattr(self, 'last_gate_flags')
        if hasattr(self, 'last_obst_flags'):
            delattr(self, 'last_obst_flags')
            
        # 重置 MPCC 内部状态
        self.last_theta = 0.0
        self.last_f_collective = 0.3
        self.last_rpy_cmd = np.zeros(3)
        self.last_f_cmd = 0.3
        if hasattr(self, "x_guess"):
            delattr(self, "x_guess")
        if hasattr(self, "u_guess"):
            delattr(self, "u_guess")
        if hasattr(self, "_current_obs_pos"):
            delattr(self, "_current_obs_pos")

    # [NEW] 调试函数，适配 mpc3_5.py 的格式
    def get_debug_lines(self):
        """返回用于 PyBullet 可视化的调试线条列表。"""
        out = []
        
        # 1. 绘制完整的全局轨迹
        if hasattr(self, "arc_trajectory"):
            try:
                # 绘制整条弧长参数化轨迹
                full_path_points = self.arc_trajectory(self.arc_trajectory.x)
                out.append((full_path_points, np.array([0.5, 0.5, 0.5, 0.7]), 2.0, 2.0))
            except Exception:
                pass # 轨迹可能尚未完全初始化
        
        # 2. 绘制 MPC 预测的视界
        if hasattr(self, "x_guess"): # x_guess 持有最后的解
            pred_traj = np.array([x[:3] for x in self.x_guess])
            out.append((pred_traj, np.array([1.0, 0.1, 0.1, 0.95]), 3.0, 3.0))
        
        # 3. 绘制从无人机到当前路径目标点的连线
        if hasattr(self, "last_theta") and hasattr(self, "arc_trajectory") and hasattr(self, "_current_obs_pos"):
            try:
                target_point_on_path = self.arc_trajectory(self.last_theta)
                line_to_target = np.stack([self._current_obs_pos, target_point_on_path])
                out.append((line_to_target, np.array([0.0, 0.0, 1.0, 1.0]), 1.0, 1.0))
            except Exception:
                pass

        return out