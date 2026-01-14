"""Simulate the competition as in the IROS 2022 Safe Robot Learning competition.

Run as:
    $ python scripts/sim.py --config level0.toml
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING
from datetime import datetime
import fire
import gymnasium
import jax.numpy as jp
import numpy as np
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy

from lsy_drone_racing.utils import load_config, load_controller
import matplotlib.pyplot as plt
# noinspection PyUnresolvedReferences
from mpl_toolkits.mplot3d import Axes3D

if TYPE_CHECKING:
    from ml_collections import ConfigDict
    from lsy_drone_racing.control.controller import Controller
    from lsy_drone_racing.envs.drone_race import DroneRaceEnv


logger = logging.getLogger(__name__)

def visualize_trajectory(
    controller: Controller,
    gates_pos: np.ndarray,
    obstacles_pos: np.ndarray,
    output_path: str = "trajectory_viz.png"
):
    """仅可视化规划轨迹 (Planned Trajectory)."""
    
    fig = plt.figure(figsize=(16, 12))
    
    # 1. 提取规划轨迹 (Planned Path & Speed)
    # -------------------------------------------------------------------------
    planned_pos = None
    planned_vel = None
    t_vals = None
    
    # 适配 MPCC 结构
    if hasattr(controller, 'planner') and hasattr(controller.planner, 'arc_trajectory'):
        traj = controller.planner.arc_trajectory
        if traj is not None:
            t_end = traj.x[-1]
            t_vals = np.linspace(0, t_end, 2000)
            planned_pos = traj(t_vals)
            # 计算规划的切向速度 (用于第4个子图)
            vel_vecs = traj.derivative(1)(t_vals)
            planned_vel = np.linalg.norm(vel_vecs, axis=1)
            
    # 兼容旧控制器
    elif hasattr(controller, '_waypoints_pos') and controller._waypoints_pos is not None:
        planned_pos = controller._waypoints_pos
    # -------------------------------------------------------------------------

    # ===== 1. 3D视图 =====
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    
    if planned_pos is not None:
        ax1.plot(planned_pos[:, 0], planned_pos[:, 1], planned_pos[:, 2],
                'b-', linewidth=2, alpha=0.8, label='Planned Spline')
        ax1.scatter(planned_pos[0, 0], planned_pos[0, 1], planned_pos[0, 2],
                   c='green', s=100, marker='o', label='Start')

    # 绘制环境
    for i, gate_pos in enumerate(gates_pos):
        ax1.scatter(gate_pos[0], gate_pos[1], gate_pos[2],
                   c='orange', s=200, marker='s', edgecolors='black', linewidths=2)
        ax1.text(gate_pos[0], gate_pos[1], gate_pos[2] + 0.1, f'G{i+1}', fontsize=10)
    
    for i, obs_pos in enumerate(obstacles_pos):
        ax1.scatter(obs_pos[0], obs_pos[1], obs_pos[2],
                   c='red', s=300, marker='x', linewidths=3)

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D View: Planned Path Only')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # ===== 2. XY平面视图 (Top View) =====
    ax2 = fig.add_subplot(2, 2, 2)
    
    if planned_pos is not None:
        ax2.plot(planned_pos[:, 0], planned_pos[:, 1],
                'b-', linewidth=2, alpha=0.8, label='Planned')
        
    # 绘制障碍物范围 (检查避障)
    for i, obs_pos in enumerate(obstacles_pos):
        # 假设半径 0.8 (对应 PathPlanner 中的设置)
        circle = plt.Circle((obs_pos[0], obs_pos[1]), 0.8, color='red', alpha=0.15)
        ax2.add_patch(circle)
        ax2.scatter(obs_pos[0], obs_pos[1], c='red', marker='x')

    for i, gate_pos in enumerate(gates_pos):
        ax2.scatter(gate_pos[0], gate_pos[1], c='orange', marker='s', edgecolors='k', s=100)

    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Top View (Obstacle Check)')
    ax2.axis('equal')
    ax2.grid(True, alpha=0.3)
    
    # ===== 3. XZ平面视图 (Side View) =====
    ax3 = fig.add_subplot(2, 2, 3)
    
    if planned_pos is not None:
        ax3.plot(planned_pos[:, 0], planned_pos[:, 2],
                'b-', linewidth=2, label='Planned')
        
    for gate_pos in gates_pos:
        ax3.scatter(gate_pos[0], gate_pos[2], c='orange', marker='s', edgecolors='k', s=100)
        # 画个简单的地面线
        ax3.axhline(0, color='k', linestyle='-', linewidth=0.5)
        
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Z (m)')
    ax3.set_title('Side View (Height Check)')
    ax3.grid(True, alpha=0.3)

    # ===== 4. 规划速度曲线 (Planned Speed Profile) =====
    ax4 = fig.add_subplot(2, 2, 4)
    
    if planned_vel is not None and t_vals is not None:
        ax4.plot(t_vals, planned_vel, 'b-', linewidth=2, label='Planned Speed')
        ax4.fill_between(t_vals, 0, planned_vel, alpha=0.2, color='blue')
        ax4.set_xlabel('Path Parameter (Time/Arc)')
        ax4.set_ylabel('Speed (m/s)')
        ax4.set_title('Planned Speed Profile')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No Velocity Data Available', 
                ha='center', va='center', transform=ax4.transAxes)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ 规划轨迹可视化已保存到: {output_path}")
    plt.close()


def simulate(
    config: str = "level0.toml",
    controller: str | None = None,
    n_runs: int = 1,
    render: bool | None = None,
    save_trajectory: bool = True,
) -> list[float]:
    """Evaluate the drone controller over multiple episodes."""
    config = load_config(Path(__file__).parents[1] / "config" / config)
    if render is None:
        render = config.sim.render
    else:
        config.sim.render = render

    control_path = Path(__file__).parents[1] / "lsy_drone_racing/control"
    controller_path = control_path / (controller or config.controller.file)
    controller_cls = load_controller(controller_path)

    env: DroneRaceEnv = gymnasium.make(
        config.env.id,
        freq=config.env.freq,
        sim_config=config.sim,
        sensor_range=config.env.sensor_range,
        control_mode=config.env.control_mode,
        track=config.env.track,
        disturbances=config.env.get("disturbances"),
        randomizations=config.env.get("randomizations"),
        seed=config.env.seed,
    )

    env = JaxToNumpy(env)
    ep_times = []

    for run_idx in range(n_runs):
        obs, info = env.reset()
        controller_instance: Controller = controller_cls(obs, info, config)
        
        # [修改] 移除了 actual_trajectory 的初始化
        
        i = 0
        fps = 60

        while True:
            curr_time = i / config.env.freq
            action = controller_instance.compute_control(obs, info)
            action = np.asarray(jp.asarray(action), copy=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            # [修改] 移除了 actual_trajectory.append(...)

            controller_finished = controller_instance.step_callback(
                action, obs, reward, terminated, truncated, info
            )

            if terminated or truncated or controller_finished:
                break

            if config.sim.render:
                if ((i * fps) % config.env.freq) < fps:
                    env.render()

            i += 1

        controller_instance.episode_callback()
        log_episode_stats(obs, info, config, curr_time)
        
        # 保存轨迹
        if save_trajectory:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"trajectory_run{run_idx+1}_{timestamp}.png"
            visualize_trajectory(
                controller_instance,
                obs['gates_pos'],
                obs['obstacles_pos'],
                output_path=output_file # [修改] 不再传递 actual_trajectory
            )
        
        controller_instance.episode_reset()
        ep_times.append(curr_time if obs["target_gate"] == -1 else None)

    env.close()
    return ep_times


def log_episode_stats(obs: dict, info: dict, config: ConfigDict, curr_time: float):
    """Log the statistics of a single episode."""
    gates_passed = obs["target_gate"]
    if gates_passed == -1:
        gates_passed = len(config.env.track.gates)
    finished = gates_passed == len(config.env.track.gates)
    logger.info(
        f"Flight time (s): {curr_time}\nFinished: {finished}\nGates passed: {gates_passed}\n"
    )


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger("lsy_drone_racing").setLevel(logging.INFO)
    logger.setLevel(logging.INFO)
    fire.Fire(simulate, serialize=lambda _: None)