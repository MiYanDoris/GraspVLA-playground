#!/usr/bin/env python3
"""
Evaluate Libero Tasks

This script evaluates the Libero tasks, including libero_goal, libero_object, libero_10 suites.

Usage:
    python evaluate_libero_tasks.py name=libero_tasks trial_num=50

Author: Mi Yan
Created: 2025-07-10
License: CC-BY-NC 4.0
"""

import sys
sys.path.insert(0, 'third_party/robosuite')

import os
from termcolor import colored
import hydra
from omegaconf import DictConfig

from benchmark_runner import (
    configure_logging,
    create_environment,
    run_episode,
    process_initial_state,
    setup_benchmark_paths,
    get_benchmark_dict,
)
from agent import RemoteAgent
from misc.logger import VideoLogger

# Set up paths
paths = setup_benchmark_paths()
bddl_files_default_path = paths["bddl_files"]
benchmark_dict = get_benchmark_dict()


def test(task, task_id, seed, test_set, video_logger, cfg, benchmark_instance):
    # Create environment
    bddl_file_path = os.path.join(bddl_files_default_path, task.problem_folder, task.bddl_file)
    env = create_environment(bddl_file_path, seed=0)
    
    # Get and process initial state
    init_states = benchmark_instance.get_task_init_states(task_id)
    processed_state = process_initial_state(init_states[seed], task.bddl_file, test_set, len(env.env.objects_dict))
    assert len(env.get_sim_state()) == len(processed_state), f"env state {len(env.get_sim_state())} != processed state {len(processed_state)}"
    obs = env.set_init_state(processed_state)

    # Start video recording
    print('Instruction:', colored(env.language_instruction, 'green'))
    if env.language_instruction == 'invalid':
        print("Invalid task, skipped")
        return
    object_name = '_'.join(env.language_instruction.split(' ')[2:])
    video_logger.start_recording(test_set, task_id, object_name, seed)
    agent = RemoteAgent(env.language_instruction, cfg.port)
    run_episode(env, agent, video_logger, obs, debug=cfg.debug)
    return


def get_seed_list(cfg):
    """Get list of seeds for evaluation."""
    import json
    
    # First, try to load from seed_list_file if provided
    if hasattr(cfg, 'seed_list_file') and cfg.seed_list_file is not None:
        seed_list = json.load(open(cfg.seed_list_file, 'r'))
        print(colored(f"Loaded {len(seed_list)} seeds from {cfg.seed_list_file}", 'cyan'))
    else: # If no seed_list_file, generate sequential seeds
        seed_list = list(range(cfg.trial_num))
        print(colored(f"Generated {cfg.trial_num} sequential seeds", 'cyan'))
    
    return seed_list


@hydra.main(version_base=None, config_path="config", config_name="evaluate_libero_tasks")
def main(cfg: DictConfig) -> None:
    """Main function to run the benchmark tests."""
    configure_logging()
    
    print(colored("Start Testing Libero Benchmarks", "yellow"))
    video_logger = VideoLogger(cfg.video_save_dir)
    seed_list = get_seed_list(cfg)
    for benchmark_name in cfg.benchmarks:
        print("Testing", colored(benchmark_name, "red"))
        benchmark_instance = benchmark_dict[benchmark_name]()
        number_of_tasks = min(cfg.max_tasks_per_benchmark, len(benchmark_instance.tasks))
        for task_id in range(number_of_tasks):
            for seed in seed_list:
                task = benchmark_instance.get_task(task_id)
                test(task, task_id, seed, benchmark_name, video_logger, cfg, benchmark_instance)


if __name__ == "__main__":
    main()