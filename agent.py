#!/usr/bin/env python3

"""
Remote Robot Control Agent

This module implements a remote robot control agent that communicates with a model server
via ZMQ to execute robot manipulation tasks. The agent processes visual observations,
maintains proprioceptive history, and converts model predictions into robot actions.

Author: Mi Yan
License: CC-BY-NC 4.0
Created: 2025-07-10
"""

import numpy as np
import transforms3d as t3d
import zmq
from collections import deque
from curobo.wrap.reacher.ik_solver import IKSolver
import curobo.util_file
from curobo.types.base import TensorDeviceType
import torch
from typing import Dict, Tuple, Optional, Any
import numpy.typing as npt


class RemoteAgent():
    # Constants
    PROPRIO_HISTORY_SIZE = 4
    GRIPPER_OPEN = 1.0
    GRIP_TRANSITION_ACTIONS = 4
    ROBOT_CONFIG_PATH = 'assets/franka_with_extended_finger/franka.yml'

    def __init__(self, instruction: str, port: int) -> None:
        """Initialize the RemoteAgent."""
        self._validate_inputs(instruction, port)
        self._setup_zmq_connection(port)
        self._setup_robot_config()
        self._initialize_state(instruction)
        
    def _validate_inputs(self, instruction: str, port: int) -> None:
        """Validate initialization parameters."""
        if not instruction.strip():
            raise ValueError("Instruction cannot be empty")
        if not (1 <= port <= 65535):
            raise ValueError(f"Port must be between 1-65535, got {port}")
    
    def _setup_zmq_connection(self, port: int) -> None:
        """Set up ZMQ connection to model server."""
        try:
            self.zmq_context = zmq.Context()
            self.socket = self.zmq_context.socket(zmq.REQ)
            self.socket.connect(f"tcp://127.0.0.1:{port}")
            # Set socket timeout to prevent hanging
            self.socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5 second timeout
        except Exception as e:
            raise ConnectionError(f"Failed to establish ZMQ connection: {e}")
    
    def _setup_robot_config(self) -> None:
        """Initialize robot configuration and IK solver."""
        self.robot_cfg = curobo.util_file.load_yaml(self.ROBOT_CONFIG_PATH)["robot_cfg"]
        self.ik_solver = IKSolver(IKSolver.load_from_robot_config(
            self.robot_cfg,
            None,
            TensorDeviceType(),
        ))
    
    def _initialize_state(self, instruction: str) -> None:
        """Initialize agent state variables."""
        self.proprio_history = deque(maxlen=self.PROPRIO_HISTORY_SIZE)
        self.instruction = instruction
        self.pred_actions = deque()
        self.finger_state = self.GRIPPER_OPEN

    def get_current_proprio(self, obs: Dict[str, Any]) -> npt.NDArray[np.float64]:
        """Get the current proprioceptive state of the robot including position, orientation and gripper state.

        # We use absolute end-effector pose in the robot frame as the proprioceptive state.
        # However, LIBERO provides proprioception in the world frame and the world-to-robot transform varies across scenes.
        # So we use forward kinematics to convert the joint positions to eef pose in the robot frame.
        
        Args:
            obs (dict): Observation dictionary
            
        Returns:
            current_proprio: Array of shape (7,) containing [x, y, z, rx, ry, rz, gripper_state].
        """
        current_joint_pos = obs['robot0_joint_pos']
        current_joint_states = torch.tensor(current_joint_pos).reshape(1, 7).cuda().float()
        current_eef_states = self.ik_solver.fk(current_joint_states)

        current_proprio = np.concatenate([
            current_eef_states.ee_position[0].cpu().numpy(),
            t3d.euler.quat2euler(current_eef_states.ee_quaternion[0].cpu().numpy(), axes='sxyz'),
            np.array([self.finger_state])
            ])
        return current_proprio
    
    
    def step(self, obs: Dict[str, Any], debug: bool = False) -> Tuple[npt.NDArray[np.float64], Optional[Any]]:
        """Execute one step of the robot control loop.
        
        This method processes the current observation, maintains proprioceptive history,
        and either executes a cached action or requests new actions from the model server.
        
        Args:
            obs: Observation dictionary
                
        Returns:
            Tuple containing:
                - action: Robot action array (7,) [x, y, z, rx, ry, rz, gripper]
                - bbox: Bounding box information for visualization (optional)
        """
        self._process_proprio(obs)

        # if the last action chunk is all executed
        if len(self.pred_actions) == 0:
            self._post_and_get(obs, debug=debug)
        action, bbox = self.pred_actions.popleft()

        # Our model uses -1 to denote gripper close, 1 to denote open.
        # while in robosuite, -1 denotes open, 1 denotes close.
        # So we need to flip the sign of the gripper state.
        action[6] = -action[6]
        return action, bbox


    def _process_proprio(self, obs):
        """Process and update the proprioceptive history buffer.
        
        Maintains a rolling history of the robot's proprioception. If the history is shorter than maxlen, it pads with
        copies of the most recent observation.
        
        Args:
            obs (dict): Observation dictionary
        """
        current_proprio = self.get_current_proprio(obs)
        self.proprio_history.append(current_proprio)
        
        while len(self.proprio_history) < self.proprio_history.maxlen:
            self.proprio_history.append(self.proprio_history[-1])


    def _post_and_get(self, obs: Dict[str, Any], debug: bool = False) -> None:
        """Send observation data to the model server and turn the received delta action chunk into absolute actions."""
        data = {
            'front_view_image': [obs["front_view_image"][::-1]],
            'side_view_image': [obs["side_view_image"][::-1]],
            'proprio_array': [np.copy(proprio) for proprio in self.proprio_history],
            'text': self.instruction,
        }
        self.socket.send_pyobj(data)
        response = self.socket.recv_pyobj()
        bbox = response['debug']['bbox']

        last_finger_state = self.finger_state
        current_pose = self.proprio_history[-1][:6]
        for delta_action in response['result']:
            abs_action = self._delta_to_abs(delta_action, current_pose)
            current_pose = abs_action[:6]
            if abs_action[6] == 0 or abs_action[6] == last_finger_state:
                # our model outputs 0 when the gripper state doesn't change
                # when the gripper state isn't changing
                self.pred_actions.append([abs_action, bbox])
            else:
                # when the gripper state is changing, we split the action 
                # into arm movement and multiple grip actions to 
                # ensure precise gripper state transition.
                arm_action = np.copy(abs_action)
                arm_action[6] = 0
                self.pred_actions.append([arm_action, bbox])
                for _ in range(self.GRIP_TRANSITION_ACTIONS):
                    self.pred_actions.append([np.copy(abs_action), bbox])
                self.finger_state = abs_action[6]
            last_finger_state = abs_action[6] if abs_action[6] != 0 else last_finger_state
        if debug:
            # log_data = {
            #     'input': data,
            #     'output': response
            # }
            print('-' * 40)
            print('proprio transition', [round(p, 4) for p in data['proprio_array'][-4][:3]], [round(p, 4) for p in data['proprio_array'][-1][:3]])
            print('proprio gripper', data['proprio_array'][-4][-1], data['proprio_array'][-1][-1])

            actions = response['result'][1::2]
            z_actions = [round(action[2], 4) for action in actions]
            gripper_actions = [action[6] for action in actions]
            print('z actions', z_actions)
            print('gripper actions', gripper_actions)
            # np.save(f'data/debug/log_{obs["step"]}.npy', log_data)
            

    def _delta_to_abs(self, delta_action: np.ndarray, 
                     current_pose: np.ndarray) -> np.ndarray:
        """Convert delta action to absolute action.
        
        Args:
            delta_action (np.ndarray): Array of shape (7,) representing delta pose 
                                     [x, y, z, rx, ry, rz, gripper_state]
            current_pose (np.ndarray): Array of shape (6,) representing current absolute pose 
                                     [x, y, z, rx, ry, rz]
        """
        current_rot = t3d.euler.euler2mat(*current_pose[3:6])
        next_rot = t3d.euler.euler2mat(*delta_action[3:6]) @ current_rot
        current_trans = current_pose[:3]
        next_trans = current_trans + delta_action[:3]
        new_action = np.concatenate([
          next_trans,
          t3d.euler.mat2euler(next_rot),
          [delta_action[6]]
        ])
        return new_action