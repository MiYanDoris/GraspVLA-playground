#!/usr/bin/env python3
"""
Get Statistics for Libero Evaluation

This script processes video files from Libero evaluation runs and computes success rates
for different benchmark suites (libero_10, libero_goal, libero_object).

Usage:
    python misc/get_statistics.py name=parallel_eval_libero_tasks

Author: Mi Yan
Created: 2025-07-10
License: CC-BY-NC 4.0
"""

import os
import hydra
from omegaconf import DictConfig
import numpy as np
from termcolor import colored


@hydra.main(version_base=None, config_path="../config", config_name="get_statistics")
def main(cfg: DictConfig) -> None:
    """Main function to compute statistics."""
    
    print(colored(f"Computing statistics for {cfg.name}", "yellow"))
    
    video_dir = os.path.join('data', cfg.name, 'videos')
    
    # Check if video directory exists
    if not os.path.exists(video_dir):
        print(colored(f"Video directory {video_dir} does not exist", "red"))
        return
    
    all_videos = os.listdir(video_dir)
    
    # Handle multiple video directories if they exist (from parallel runs)
    for i in range(10):  # Check up to 10 parallel runs
        parallel_video_dir = os.path.join('data', cfg.name, f'videos_{i}')
        if os.path.exists(parallel_video_dir):
            parallel_videos = os.listdir(parallel_video_dir)
            all_videos.extend(parallel_videos)
    
    prefix = ['libero_10', 'libero_goal', 'libero_object']
    
    log_file = os.path.join('data', cfg.name, 'statistics.txt')
    
    print(colored("Statistics Summary:", "green"))
    print("=" * 50)
    
    for prefix_name in prefix:
        prefix_videos = [video for video in all_videos if video.startswith(prefix_name)]
        num = len(prefix_videos)
        if num == 0:
            continue
        success_num = np.sum([1 for video in prefix_videos if 'success' in video])
        fail_num = np.sum([1 for video in prefix_videos if 'fail' in video])
        
        success_rate = success_num / num if num > 0 else 0.0
        result = f"{prefix_name}: {success_num}/{num} = {success_rate:.2f}"
        
        print(colored(result, "cyan"))
        
        with open(log_file, 'a') as f:
            f.write(f"{result}\n")
    
    print("=" * 50)
    print(colored(f"Statistics saved to: {log_file}", "green"))


if __name__ == "__main__":
    main()