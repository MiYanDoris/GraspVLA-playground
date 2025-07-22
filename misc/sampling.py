import os
import json
import random
import numpy as np
import trimesh
import transforms3d
from libero.libero.envs.arenas.style import FLOOR_STYLE, WALL_STYLE


FRANKA_INIT_POSE = [-1.85879334e-02, -1.49071300e-01, -2.69055127e-02, -2.47824178e+00,  9.87954057e-03,  2.23193464e+00,  7.75406363e-01, 2.08330000e-02, -2.08330000e-02]


def sample_background():
    floor_style = random.choice(list(FLOOR_STYLE.keys()))
    wall_style = random.choice(list(WALL_STYLE.keys()))
    scene_properties = {
        "floor_style": floor_style,
        "wall_style": wall_style,
    }
    return scene_properties


def sample_init_state(object_names, offset, object_root_dir='playground_assets'):
    """
    Sample initial state for objects in the scene.
    
    Args:
        object_names: List of object names to place in the scene
        offset: Numpy array offset to apply to all object positions
        object_root_dir: Root directory containing object assets
    
    Returns:
        numpy array representing the initial state
    """
    new_state = np.zeros(19 + 13 * len(object_names))
    new_state[1:10] = FRANKA_INIT_POSE
    collision_manager = trimesh.collision.CollisionManager()
    
    for i, object_name in enumerate(object_names):
        table_pose_path = os.path.join(object_root_dir, object_name, 'table_pose.json')
        table_poses = json.load(open(table_pose_path))
        object_mesh = trimesh.load_mesh(f'{object_root_dir}/{object_name}/simplified.obj')

        cnt = 0
        success = False
        while cnt < 100:
            stable_pose = random.choice(table_poses)
            rand_pos = np.array([random.uniform(0.35, 0.7), random.uniform(-0.2, 0.2), stable_pose[2]]) # target should be within the reach of the robot
            stable_ori_mat = transforms3d.quaternions.quat2mat(stable_pose[3:]) # 3x3
            rand_ori_mat = transforms3d.euler.euler2mat(0.0, 0.0, random.uniform(0.0, 2.0 * np.pi))
            rand_ori_mat = rand_ori_mat @ stable_ori_mat
            transformation = np.eye(4)
            transformation[:3, :3] = rand_ori_mat
            transformation[:3, 3] = rand_pos
            
            # Handle different return types from min_distance_single
            distance_result = collision_manager.min_distance_single(
                mesh=object_mesh, transform=transformation
            )
            # Extract distance value - it might be a tuple or float
            if isinstance(distance_result, tuple):
                distance = distance_result[0]
            else:
                distance = distance_result
                
            if distance > 0.02:
                collision_manager.add_object(name=f'{object_name}', mesh=object_mesh, transform=transformation)
                rand_quat = transforms3d.quaternions.mat2quat(rand_ori_mat)
                success = True
                break
            cnt += 1
            
        if not success:
            raise ValueError(f"Failed to sample init state for {object_name}")
            
        new_state[10 + i * 7:10 + i * 7 + 3] = rand_pos + offset
        new_state[10 + i * 7 + 3:10 + i * 7 + 7] = rand_quat
        
    return new_state


def sample_objects(object_num=6, object_root_dir='playground_assets'):
    """
    Sample random objects from the available object assets.
    
    Args:
        object_num: Number of objects to sample
        object_root_dir: Root directory containing object assets
    
    Returns:
        List of sampled object names
    """
    candidates = os.listdir(object_root_dir)
    object_names = random.sample(candidates, object_num)
    return object_names 