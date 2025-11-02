import numpy as np
import torch

from typing import Sequence, Dict, Tuple
from abc import ABC, abstractmethod
import random

from models import WorldModel, Pendulum, MountainCar

from StarV.set.star import Star

class Verifier(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def verify_single_cell(self, cell: Dict, num_steps: int = 20) -> bool:
        pass

# ============ Pendulum Neural Network Verifier ============
class PendulumVerifier(Verifier):
    """StarV-based Neural Network Verification for Pendulum System"""

    def __init__(self, weights_path: str, goal_angle_threshold = 0.15):
        # Create the controller and decoder
        self.worldmodel = WorldModel()
        self.pendulum = Pendulum()
        self.goal_angle_threshold = goal_angle_threshold

        weights = torch.load(weights_path, 'cpu', weights_only=True)
        self.worldmodel.load_state_dict(weights)

    def verify_single_step(self, theta_bound: Sequence[float], omega_bound: Sequence[float]) -> Dict:
        """
        Perform single-step verification with pendulum dynamics
        theta' = theta + 0.05 * omega + 0.0075 * u + 0.0375 * sin(theta)
        omega' = omega + 0.15 * u + 0.75 * sin(theta), clipped to [-8, 8]
        """
        # Create initial Star set
        theta_min, theta_max = theta_bound
        omega_min, omega_max = omega_bound
        
        lb = np.array([theta_min, omega_min], dtype=np.float32)
        ub = np.array([theta_max, omega_max], dtype=np.float32)
        input_star = Star(lb, ub)
        # Verify the model
        output_star = self.worldmodel.reach(input_star)

        d = output_star.dim
        A = 2.0 * np.eye(d, dtype=np.float32)
        b = np.zeros(d, dtype=np.float32)
        action = output_star.affineMap(A, b)
        action_bound = np.concatenate(action.getRanges(lp_solver='gurobi'))

        # Verfity the pendulum dynamic system
        next_theta_bound, next_omega_bound = self.pendulum.reach(theta_bound, omega_bound, action_bound)

        return next_theta_bound, next_omega_bound

    def split_merge_bounds(self, theta_bounds_list, omega_bounds_list):
        splited_theta_bounds_list = []
        splited_omega_bounds_list = []
        for theta_bound, omega_bound in zip(theta_bounds_list, omega_bounds_list):
            theta_min, theta_max = theta_bound
            if theta_min > theta_max:
                splited_theta_bounds_list += [[theta_min, np.pi], [-np.pi, theta_max]]
                splited_omega_bounds_list += [omega_bound, omega_bound]
            else:
                splited_omega_bounds_list.append(omega_bound)
                splited_theta_bounds_list.append(theta_bound)

        splited_theta_bounds = np.array(splited_theta_bounds_list, dtype=np.float32)
        splited_omega_bounds = np.array(splited_omega_bounds_list, dtype=np.float32)
        return splited_theta_bounds, splited_omega_bounds

    def verify_single_cell(self, cell: Dict, num_steps: int = 20) -> bool:
        """Verify a single cell for the specified number of steps (early-stop when |theta| <= 0.15).
        Only keep the final step's bounds to save memory.
        """
        # Current state bounds
        theta_bounds = np.array([cell['theta']])
        omega_bounds = np.array([cell['omega']])

        # Safety tracking (reach goal if BOTH bounds are within ±0.15)
        reached_goal = False

        # Perform verification steps with early-stop
        for step in range(1, num_steps + 1):
            theta_bound_list = []
            omega_bound_list = []

            for theta_bound, omega_bound in zip(theta_bounds, omega_bounds): 
                
                    new_theta_bound, new_omega_bound = self.verify_single_step(theta_bound, omega_bound)
                    theta_bound_list.append(new_theta_bound)
                    omega_bound_list.append(new_omega_bound)
            
            # Update current state (we only keep the latest bounds)
            theta_bounds, omega_bounds = self.split_merge_bounds(theta_bound_list, omega_bound_list)
            
            # EARLY STOP: if BOTH |theta_min| and |theta_max| <= 0.15, treat as SAFE and stop.
            if np.max(np.abs(theta_bounds)) <= self.goal_angle_threshold:
                reached_goal = True
                break

        return reached_goal

class MountainCarVerifier(Verifier):
    """StarV-based Neural Network Verification for Mountain Car System"""

    def __init__(self, weights_path: str, goal_position_threshold = 0.6):
        # Create the controller and decoder
        self.worldmodel = WorldModel()
        self.mountain_car = MountainCar()

        # Safety condition: BOTH min and max position >= 0.6
        self.goal_position_threshold = goal_position_threshold

        weights = torch.load(weights_path, 'cpu', weights_only=True)
        self.worldmodel.load_state_dict(weights)


    def verify_single_step(self, pos_bound: Sequence[float], vel_bound: Sequence[float]) -> Dict:
        """Perform single-step verification: state -> action -> next_state"""
        pos_min, pos_max = pos_bound
        vel_min, vel_max = vel_bound
        # Create the initial star set
        lb = np.array([pos_min, vel_min], dtype=np.float32)
        ub = np.array([pos_max, vel_max], dtype=np.float32)

        input_star = Star(lb, ub)

        # Verify the model
        output_star = self.worldmodel.reach(input_star)
        action_bound = np.concatenate(output_star.getRanges('gurobi'))

        # Step 3: Compute next state bounds
        next_pos_bound, next_vel_bound = self.mountain_car.reach(pos_bound, vel_bound, action_bound)

        return next_pos_bound, next_vel_bound
    
    def verify_single_cell(self, cell: Dict, num_steps: int = 30) -> Dict:
        """Verify a single cell for the specified number of steps"""
        # Current state bounds
        current_pos_bound = cell['pos']
        current_vel_bound = cell['vel']

        # Safety tracking
        reached_goal = False

        # Perform verification steps
        for step in range(1, num_steps + 1):
            # Single step verification
            next_pos_bound, next_vel_bound = self.verify_single_step(current_pos_bound, current_vel_bound)

            # Check safety condition: BOTH min and max position must be >= threshold
            if min(current_pos_bound) >= self.goal_position_threshold:
                reached_goal = True
                break

            current_pos_bound = next_pos_bound
            current_vel_bound = next_vel_bound

        return reached_goal
    
class Cartpole(Verifier):
    def __init__(self, decoder_path: str, controller_path: str):
        pass

    def verify_single_cell(self, cell, num_steps = 20):
        pos_lb, pos_ub = cell['pos']
        angle_lb, angle_ub = cell['angle']

        reach_goal = False

        # Note: You don't have to handle exception here

        return reach_goal

class _Test(Verifier):
    def __init__(self, raise_error = True):
        self.raise_error = raise_error

    def verify_single_cell(self, cell, num_steps = 20):
        if self.raise_error:
            raise RuntimeError("Error raised to test the program")
        
        return random.choice([True, False])
        
