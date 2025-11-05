"""Simulate the competition as in the IROS 2022 Safe Robot Learning competition.

Run as:

    $ python scripts/sim.py --config level0.toml

Look for instructions in `README.md` and in the official documentation.
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
from lsy_drone_racing.utils import draw_line  
import matplotlib.pyplot as plt
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
    actual_trajectory: np.ndarray,
    output_path: str = "trajectory_viz.png"
):
    """可视化规划轨迹、实际轨迹、门和障碍物."""
    
    fig = plt.figure(figsize=(16, 12))
    
    # ===== 3D视图 =====
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    
    # 绘制规划轨迹
    if hasattr(controller, '_waypoints_pos') and controller._waypoints_pos is not None:
        planned_traj = controller._waypoints_pos
        ax1.plot(planned_traj[:, 0], planned_traj[:, 1], planned_traj[:, 2],
                'b-', linewidth=2, alpha=0.6, label='Planned Trajectory')
        ax1.scatter(planned_traj[0, 0], planned_traj[0, 1], planned_traj[0, 2],
                   c='green', s=100, marker='o', label='Start')
    
    # 绘制实际轨迹
    if len(actual_trajectory) > 0:
        actual_traj = np.array(actual_trajectory)
        ax1.plot(actual_traj[:, 0], actual_traj[:, 1], actual_traj[:, 2],
                'r-', linewidth=1.5, alpha=0.8, label='Actual Trajectory')
    
    # 绘制门
    for i, gate_pos in enumerate(gates_pos):
        ax1.scatter(gate_pos[0], gate_pos[1], gate_pos[2],
                   c='orange', s=200, marker='s', edgecolors='black', linewidths=2)
        ax1.text(gate_pos[0], gate_pos[1], gate_pos[2] + 0.1,
                f'Gate {i+1}', fontsize=10, weight='bold')
    
    # 绘制障碍物
    for i, obs_pos in enumerate(obstacles_pos):
        ax1.scatter(obs_pos[0], obs_pos[1], obs_pos[2],
                   c='red', s=300, marker='x', linewidths=3)
        ax1.text(obs_pos[0], obs_pos[1], obs_pos[2] + 0.1,
                f'Obs {i+1}', fontsize=9)
    
    ax1.set_xlabel('X (m)', fontsize=12)
    ax1.set_ylabel('Y (m)', fontsize=12)
    ax1.set_zlabel('Z (m)', fontsize=12)
    ax1.set_title('3D Trajectory View', fontsize=14, weight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # ===== XY平面视图 =====
    ax2 = fig.add_subplot(2, 2, 2)
    
    if hasattr(controller, '_waypoints_pos') and controller._waypoints_pos is not None:
        planned_traj = controller._waypoints_pos
        ax2.plot(planned_traj[:, 0], planned_traj[:, 1],
                'b-', linewidth=2, alpha=0.6, label='Planned')
        ax2.scatter(planned_traj[0, 0], planned_traj[0, 1],
                   c='green', s=100, marker='o', zorder=5)
    
    if len(actual_trajectory) > 0:
        actual_traj = np.array(actual_trajectory)
        ax2.plot(actual_traj[:, 0], actual_traj[:, 1],
                'r-', linewidth=1.5, alpha=0.8, label='Actual')
    
    for i, gate_pos in enumerate(gates_pos):
        ax2.scatter(gate_pos[0], gate_pos[1], c='orange', s=200,
                   marker='s', edgecolors='black', linewidths=2)
        ax2.text(gate_pos[0] + 0.05, gate_pos[1] + 0.05, f'G{i+1}',
                fontsize=10, weight='bold')
    
    for i, obs_pos in enumerate(obstacles_pos):
        circle = plt.Circle((obs_pos[0], obs_pos[1]), 0.35, color='red',
                           alpha=0.3, label='Obstacle' if i == 0 else '')
        ax2.add_patch(circle)
        ax2.scatter(obs_pos[0], obs_pos[1], c='red', s=100, marker='x', linewidths=3)
    
    ax2.set_xlabel('X (m)', fontsize=12)
    ax2.set_ylabel('Y (m)', fontsize=12)
    ax2.set_title('Top View (XY Plane)', fontsize=14, weight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # ===== XZ平面视图 =====
    ax3 = fig.add_subplot(2, 2, 3)
    
    if hasattr(controller, '_waypoints_pos') and controller._waypoints_pos is not None:
        planned_traj = controller._waypoints_pos
        ax3.plot(planned_traj[:, 0], planned_traj[:, 2],
                'b-', linewidth=2, alpha=0.6, label='Planned')
    
    if len(actual_trajectory) > 0:
        actual_traj = np.array(actual_trajectory)
        ax3.plot(actual_traj[:, 0], actual_traj[:, 2],
                'r-', linewidth=1.5, alpha=0.8, label='Actual')
    
    for i, gate_pos in enumerate(gates_pos):
        ax3.scatter(gate_pos[0], gate_pos[2], c='orange', s=200,
                   marker='s', edgecolors='black', linewidths=2)
        ax3.axhline(y=gate_pos[2], color='orange', linestyle='--', alpha=0.3)
    
    ax3.set_xlabel('X (m)', fontsize=12)
    ax3.set_ylabel('Z (m)', fontsize=12)
    ax3.set_title('Side View (XZ Plane)', fontsize=14, weight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # ===== 速度曲线 =====
    ax4 = fig.add_subplot(2, 2, 4)
    
    if len(actual_trajectory) > 0:
        actual_traj = np.array(actual_trajectory)
        velocities = np.sqrt(np.sum(np.diff(actual_traj, axis=0)**2, axis=1)) * 50  # 50Hz
        time_steps = np.arange(len(velocities)) / 50.0
        
        ax4.plot(time_steps, velocities, 'g-', linewidth=2, label='Speed')
        ax4.fill_between(time_steps, 0, velocities, alpha=0.3, color='green')
        ax4.axhline(y=np.mean(velocities), color='r', linestyle='--',
                   label=f'Avg: {np.mean(velocities):.2f} m/s')
    
    ax4.set_xlabel('Time (s)', fontsize=12)
    ax4.set_ylabel('Speed (m/s)', fontsize=12)
    ax4.set_title('Speed Profile', fontsize=14, weight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ 轨迹可视化已保存到: {output_path}")
    plt.close()


def simulate(
    config: str = "level0.toml",
    controller: str | None = None,
    n_runs: int = 1,
    render: bool | None = None,
    save_trajectory: bool = True,  # 新增：是否保存轨迹可视化
) -> list[float]:
    """Evaluate the drone controller over multiple episodes.
    
    Args:
        config: The path to the configuration file. Assumes the file is in `config/`.
        controller: The name of the controller file in `lsy_drone_racing/control/` or None.
        n_runs: The number of episodes.
        render: Enable/disable rendering the simulation.
        save_trajectory: Whether to save trajectory visualization.
        
    Returns:
        A list of episode times.
    """
    # Load configuration and check if firmare should be used.
    config = load_config(Path(__file__).parents[1] / "config" / config)
    if render is None:
        render = config.sim.render
    else:
        config.sim.render = render

    # Load the controller module
    control_path = Path(__file__).parents[1] / "lsy_drone_racing/control"
    controller_path = control_path / (controller or config.controller.file)
    controller_cls = load_controller(controller_path)

    # Create the racing environment
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
        
        # 记录实际飞行轨迹
        actual_trajectory = [obs['pos'].copy()]
        
        i = 0
        fps = 60

        while True:
            curr_time = i / config.env.freq
            action = controller_instance.compute_control(obs, info)
            action = np.asarray(jp.asarray(action), copy=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            # 记录位置
            actual_trajectory.append(obs['pos'].copy())

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
        
        # 保存轨迹可视化
        if save_trajectory:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"trajectory_run{run_idx+1}_{timestamp}.png"
            visualize_trajectory(
                controller_instance,
                obs['gates_pos'],
                obs['obstacles_pos'],
                actual_trajectory,
                output_file
            )
        
        controller_instance.episode_reset()
        ep_times.append(curr_time if obs["target_gate"] == -1 else None)

    env.close()
    return ep_times


def log_episode_stats(obs: dict, info: dict, config: ConfigDict, curr_time: float):
    """Log the statistics of a single episode."""
    gates_passed = obs["target_gate"]
    if gates_passed == -1:  # The drone has passed the final gate
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