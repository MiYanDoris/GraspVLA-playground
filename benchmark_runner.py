#!/usr/bin/env python3
"""
Common Benchmark Runner

Author: Mi Yan
Created: 2025-07-10
License: CC-BY-NC 4.0
"""

import sys
sys.path.insert(0, 'third_party/robosuite')

import random
import tqdm
import numpy as np
import logging
from typing import Any, Dict

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from agent import RemoteAgent
from misc.logger import VideoLogger


def configure_logging():
    """Configure logging to suppress verbose library logs."""
    logging.getLogger('curobo').setLevel(logging.ERROR)
    logging.getLogger('robomimic').setLevel(logging.ERROR)
    logging.getLogger('robosuite').setLevel(logging.ERROR)


def create_environment(bddl_file_path: str, seed: int = 0, scene_properties: dict = None) -> OffScreenRenderEnv:
    """
    Create and configure the OffScreenRenderEnv environment.
    
    Args:
        bddl_file_path: Path to the BDDL file
        seed: Random seed for the environment
        scene_properties: Scene properties for the environment
        
    Returns:
        Configured OffScreenRenderEnv instance
    """
    env_args = {
        "bddl_file_name": bddl_file_path,
        "camera_names": ["front_view", "side_view"],
        "camera_heights": 256,
        "camera_widths": 256,
        "control_freq": 5,
        "controller": "IK_POSE",
        "scene_properties": scene_properties
    }
    
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    env.reset()
    return env


def stabilize_scene(env: OffScreenRenderEnv, agent: RemoteAgent, obs: Any, steps: int = 10) -> Any:
    """
    Since the obejcts are dropped from the air, we need to stabilize the scene by executing zero action.
    """
    for _ in range(steps):
        action = agent.get_current_proprio(obs)
        action[-1] = -1  # Keep gripper opened
        obs, _, _, _ = env.step(action)
    return obs


def run_episode(
    env: OffScreenRenderEnv,
    agent: RemoteAgent,
    video_logger: VideoLogger,
    initial_obs: Any,
    max_steps: int = 300,
    stabilize_steps: int = 10,
    debug: bool = False
) -> None:
    """
    Run a single episode with the given environment and agent.
    
    Args:
        env: The environment instance
        agent: The agent instance
        video_logger: Logger for recording videos
        initial_obs: Initial observation after environment reset
        max_steps: Maximum number of steps in the episode
        stabilize_steps: Number of stabilization steps
    """
    # Stabilize the scene
    obs = stabilize_scene(env, agent, initial_obs, stabilize_steps)
    
    # Main episode loop
    for step in tqdm.tqdm(range(max_steps)):
        action, bbox = agent.step(obs, debug=debug)
        obs, reward, done, info = env.step(action)
        video_logger.log_frame(obs, bbox)
        
        if done:
            video_logger.stop_recording(success=True)
            return
    
    video_logger.stop_recording(success=False)
    return


def setup_benchmark_paths() -> Dict[str, str]:
    """
    Set up common benchmark paths.
    
    Returns:
        Dictionary containing benchmark paths
    """
    return {
        "benchmark_root": get_libero_path("benchmark_root"),
        "init_states": get_libero_path("init_states"),
        "datasets": get_libero_path("datasets"),
        "bddl_files": get_libero_path("bddl_files"),
    }


def get_benchmark_dict() -> Dict[str, Any]:
    """
    Get the benchmark dictionary.
    
    Returns:
        Dictionary of available benchmarks
    """
    return benchmark.get_benchmark_dict()


def set_random_seeds(seed: int):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)


def process_initial_state(state: np.ndarray, task_name: str, test_set: str, object_num: int) -> np.ndarray:
    """
    Since the basket occludes the side view severely, we need to remove it.
    
    Args:
        state: Initial state array
        task_name: Name of the task
        test_set: Test set name
        object_num: Number of objects in the scene
        
    Returns:
        Processed state array
    """
    if test_set == 'libero_object':
        # remove basket, which is the second object
        num = len(state)
        new_state = np.zeros(num-13)
        new_state[:17] = state[:17]  # time 1 + robot state 9 + first object 7
        new_state[17:] = state[24:-6]  # velocity are all zero
    elif ('LIVING_ROOM_SCENE1' in task_name) or ('LIVING_ROOM_SCENE2' in task_name):
        # remove basket, which is the last object
        num = len(state)
        new_state = np.zeros(num-13)
        new_state[:1+9+7*object_num] = state[:1+9+7*object_num]
    else:
        new_state = state
    return new_state