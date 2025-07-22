#!/usr/bin/env python3
"""
Playground Test Runner

This script randomly samples objects from playground_assets and creates a BDDL file, then runs the model on this scene.

Usage:
    python playground.py name=playground_test trial_num=5
    python playground.py name=playground_test seed_list_file=config/example_seeds.json

Author: Mi Yan
Created: 2025-07-10
License: CC-BY-NC 4.0
"""

import sys
sys.path.insert(0, 'third_party/robosuite')

import os
import random
import numpy as np
from termcolor import colored
import hydra
from omegaconf import DictConfig
import json

from libero.libero import get_libero_path
from benchmark_runner import (
    configure_logging,
    create_environment,
    run_episode,
    get_benchmark_dict,
    set_random_seeds
)
from agent import RemoteAgent
from misc.export_as_bddl import export_as_bddl_file
from misc.logger import VideoLogger
from misc.sampling import sample_init_state, sample_objects, sample_background


def test(benchmark_instance, seed, video_logger: VideoLogger, object_root_dir, bddl_file_name, object_num, port, debug: bool = False) -> None:
    """
    Run a single test episode with the given parameters.
    """
    set_random_seeds(seed)
    
    complete_object_names = sample_objects(object_num=object_num, object_root_dir=object_root_dir)
    object_names = ['_'.join(c.split('_')[:-1]) for c in complete_object_names]
    print('objects in the scene:', colored(object_names, 'green'))
    
    target_object_name = random.choice(object_names).replace('_', ' ')
    init_state = sample_init_state(
        complete_object_names, 
        offset=np.array([-0.6, 0.0, 0.0]),
        object_root_dir=object_root_dir
    )
    export_as_bddl_file(object_names, target_object_name, bddl_file_name)
    
    task = benchmark_instance.get_task(i=10)
    print('target object:', colored(target_object_name, 'green'))

    bddl_files_default_path = get_libero_path("bddl_files")
    bddl_file_path = os.path.join(bddl_files_default_path, task.problem_folder, task.bddl_file)
    scene_properties = sample_background()
    env = create_environment(bddl_file_path, seed=seed, scene_properties=scene_properties)
    
    assert len(env.get_sim_state()) == len(init_state), \
        f"env state {len(env.get_sim_state())} != processed state {len(init_state)}"
    obs = env.set_init_state(init_state)

    agent = RemoteAgent('pick up ' + target_object_name, port)
    video_logger.start_recording('', '', target_object_name, seed)
    run_episode(env, agent, video_logger, obs, debug=debug)
    return


def get_seed_list(cfg):
    # Handle seed generation and loading
    seed_list = None
    
    # First, try to load from seed_list_file if provided
    if cfg.seed_list_file is not None:
        seed_list = json.load(open(cfg.seed_list_file, 'r'))
        print(colored(f"Loaded {len(seed_list)} seeds from {cfg.seed_list_file}", 'cyan'))
    else: # If no seed_list_file, generate random seeds
        seed_list = [random.randint(0, 10000000) for _ in range(cfg.trial_num)]
        print(colored(f"Generated {cfg.trial_num} random seeds", 'cyan'))
    
    # Save seed list to json file
    seed_list_json_path = os.path.join(cfg.data_dir, 'seed_list.json')
    with open(seed_list_json_path, 'w') as f:
        json.dump(seed_list, f)
    print(colored(f"Saved seed list to {seed_list_json_path}", 'green'))
    return seed_list


@hydra.main(version_base=None, config_path="config", config_name="playground")
def main(cfg: DictConfig) -> None:
    """Main function to run the benchmark tests."""
    configure_logging()
    
    # Initialize components
    video_logger = VideoLogger(cfg.video_save_dir)
    benchmark_dict = get_benchmark_dict()
    benchmark_instance = benchmark_dict["libero_object"]()
    
    seed_list = get_seed_list(cfg)

    # Run tests for each seed
    for seed in seed_list:
        print(colored(f"\n============== Running test with seed {seed} ==============", 'yellow'))
        test(benchmark_instance, seed, video_logger, cfg.object_root_dir, cfg.bddl_file_name, cfg.object_num, cfg.port, debug=cfg.debug)


if __name__ == "__main__":
    main()