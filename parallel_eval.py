#!/usr/bin/env python3
"""
Parallel Evaluation for Libero Tasks

This script runs parallel evaluation of Libero tasks by splitting trials across multiple processes.

Usage:
    python parallel_eval.py exp_name=parallel_eval_libero_tasks parallel_env_num=5 trial_num=50

Author: Mi Yan
Created: 2025-07-10
License: CC-BY-NC 4.0
"""

import os
import hydra
from omegaconf import DictConfig
from termcolor import colored
import json


@hydra.main(version_base=None, config_path="config", config_name="parallel_eval")
def main(cfg: DictConfig) -> None:
    """Main function to run parallel evaluation."""
    
    print(colored(f"Starting parallel evaluation for {cfg.exp_name} with {cfg.parallel_env_num} parallel environments", "yellow"))
    print(f"Running {cfg.trial_num} trials per task")
    
    # Create output directory
    output_dir = f'data/{cfg.exp_name}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate shell commands
    commands = 'eval "$(conda shell.bash hook)" \n'
    commands += 'conda activate playground \n'
    
    # Split trials across parallel processes
    for i in range(cfg.parallel_env_num):
        seed_list = list(range(cfg.trial_num)[i::cfg.parallel_env_num])
        
        # Create seed file for this process
        seed_file_path = f'{output_dir}/seeds_{i}.json'
        with open(seed_file_path, 'w') as f:
            json.dump(seed_list, f)
        
        # Create temporary config file for this process
        temp_config_path = f'{output_dir}/temp_config_{i}.yaml'
        with open(temp_config_path, 'w') as f:
            f.write(f"""# Temporary config for parallel process {i}
defaults:
  - _self_
  
name: {cfg.exp_name}
debug: false
seed_list_file: {seed_file_path}
port: {cfg.port}
video_save_dir: data/{cfg.exp_name}/videos
data_dir: data/{cfg.exp_name}
""")
            
            # Add benchmarks if specified
            if hasattr(cfg, 'benchmarks') and cfg.benchmarks:
                f.write("benchmarks:\n")
                for benchmark in cfg.benchmarks:
                    f.write(f"  - {benchmark}\n")
            
            # Add other config overrides
            if hasattr(cfg, 'max_tasks_per_benchmark'):
                f.write(f"max_tasks_per_benchmark: {cfg.max_tasks_per_benchmark}\n")
        
        # Generate hydra command using the temporary config
        abs_config_path = os.path.abspath(output_dir)
        cmd = f'python evaluate_libero_tasks.py --config-path={abs_config_path} --config-name=temp_config_{i} '
        cmd += '& \n'
        
        commands += cmd
    
    # Wait for all processes to complete
    commands += 'wait $(jobs -p) \n'
    commands += f'python misc/get_statistics.py name={cfg.exp_name} \n'
    
    # Write and execute commands
    script_path = f'{output_dir}/commands.sh'
    with open(script_path, 'w') as f:
        f.write(commands)
    
    print(colored(f"Generated command script: {script_path}", "green"))
    print(colored("Executing parallel evaluation...", "blue"))
    
    os.system(f'bash {script_path}')
    print(colored("Parallel evaluation completed!", "green"))


if __name__ == "__main__":
    main() 