"""Simulate the competition as in the IROS 2022 Safe Robot Learning competition.

Run as:

    $ python scripts/sim.py --config level0.toml

Look for instructions in `README.md` and in the official documentation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import fire
import gymnasium
import jax.numpy as jp
import numpy as np
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy

from lsy_drone_racing.utils import load_config, load_controller
from lsy_drone_racing.utils import draw_line  

if TYPE_CHECKING:
    from ml_collections import ConfigDict

    from lsy_drone_racing.control.controller import Controller
    from lsy_drone_racing.envs.drone_race import DroneRaceEnv


logger = logging.getLogger(__name__)


# ... (前面的 import 保持不变) ...

def simulate(
    config: str = "level2.toml",
    controller: str | None = None,
    n_runs: int = 3,  # [修改] 默认跑3次取平均，节省时间
    render: bool | None = False, # [修改] 默认不渲染
    tuning_params: dict | None = None, # [新增] 接收调参字典
) -> dict: # [修改] 返回字典而不是 list
    """Evaluate the drone controller for Auto-Tuning."""
    
    # 1. 加载 Config
    config_obj = load_config(Path(__file__).parents[1] / "config" / config)
    
    # [新增] 注入调参参数
    if tuning_params is not None:
        # 动态给 config 对象添加属性，供 Controller 读取
        config_obj.tuning = tuning_params
    
    if render is None:
        render = config_obj.sim.render
    else:
        config_obj.sim.render = render

    # 2. 加载控制器类
    control_path = Path(__file__).parents[1] / "lsy_drone_racing/control"
    controller_path = control_path / (controller or config_obj.controller.file)
    controller_cls = load_controller(controller_path)

    # 3. 创建环境
    env: DroneRaceEnv = gymnasium.make(
        config_obj.env.id,
        freq=config_obj.env.freq,
        sim_config=config_obj.sim,
        sensor_range=config_obj.env.sensor_range,
        control_mode=config_obj.env.control_mode,
        track=config_obj.env.track,
        disturbances=config_obj.env.get("disturbances"),
        randomizations=config_obj.env.get("randomizations"),
        seed=config_obj.env.seed,
    )
    env = JaxToNumpy(env)

    ep_times = []
    solver_failures = 0
    crashes = 0

    for _ in range(n_runs):
        obs, info = env.reset()
        
        # [保护] 控制器初始化可能会因为参数极值而报错
        try:
            controller_instance = controller_cls(obs, info, config_obj)
        except Exception as e:
            print(f"[AutoTune] Controller init failed: {e}")
            env.close()
            # 返回极差的结果
            return {"avg_time": 999.0, "success_rate": 0.0, "crashes": n_runs}

        i = 0
        fps = 60
        episode_success = False

        while True:
            curr_time = i / config_obj.env.freq

            try:
                action = controller_instance.compute_control(obs, info)
            except Exception as e:
                print(f"[AutoTune] Solver crashed at step {i}: {e}")
                solver_failures += 1
                break

            action = np.asarray(jp.asarray(action), copy=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            # [修改] 打印精简，避免刷屏
            # print(f"[step {i}] ...") 

            controller_finished = controller_instance.step_callback(
                action, obs, reward, terminated, truncated, info
            )

            # 检查是否成功完成 (target_gate == -1)
            if obs["target_gate"] == -1:
                episode_success = True

            if terminated or truncated or controller_finished:
                # 检查是否是碰撞导致的终止
                if "collisions" in info and info["collisions"] > 0:
                    pass # 确实撞了
                break

            if config_obj.sim.render:
                if ((i * fps) % config_obj.env.freq) < fps:
                    env.render()
            i += 1

        controller_instance.episode_callback()
        controller_instance.episode_reset()
        
        if episode_success:
            ep_times.append(curr_time)
        else:
            crashes += 1

    env.close()
    
    # 4. 计算统计结果
    success_count = len(ep_times)
    success_rate = success_count / max(1, n_runs)
    
    if success_count > 0:
        avg_time = float(np.mean(ep_times))
    else:
        avg_time = 999.0 # 惩罚值

    return {
        "avg_time": avg_time,
        "success_rate": success_rate,
        "crashes": crashes,
        "solver_failures": solver_failures
    }

# ... (main 部分保持不变) ...
