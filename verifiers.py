import numpy as np
import torch

from typing import Sequence, Dict, Tuple
from abc import ABC, abstractmethod
import random

from models import NNModel, Pendulum, MountainCar

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
        self.nnmodel = NNModel('tanh', 2.0)
        self.pendulum = Pendulum()
        self.goal_angle_threshold = goal_angle_threshold

        weights = torch.load(weights_path, 'cpu', weights_only=True)
        self.nnmodel.load_state_dict(weights)

    def verify_single_step(self, bound: np.ndarray) -> Dict:
        """
        Perform single-step verification with pendulum dynamics
        theta' = theta + 0.05 * omega + 0.0075 * u + 0.0375 * sin(theta)
        omega' = omega + 0.15 * u + 0.75 * sin(theta), clipped to [-8, 8]
        """
        # bound: theta, omega
        state_star = Star(bound[0], bound[1])
        # Verify the model
        action_star = self.nnmodel.reach(state_star)
        action_bound = np.array(action_star.getRanges(lp_solver='gurobi'))
        bound = np.concatenate([bound, action_bound], axis=1) # theta omega action

        # Verfity the pendulum dynamic system
        new_bound = self.pendulum.reach(bound) # theta omega

        return new_bound
    def split_merge_bounds(self, bounds):
        splited_bounds = []
        for bound in bounds:
            theta_bound = bound[:,0]
            omega_bound = bound[:,1]
            theta_min, theta_max = theta_bound
            omega_min, omega_max = omega_bound
            if theta_min > theta_max:
                # If the theta range is wrapped around, break the bound into two bounds
                splited_bounds.append(np.array([[theta_min, omega_min], [np.pi, omega_max]]))
                splited_bounds.append(np.array([[-np.pi, omega_min], [theta_max, omega_max]]))
            else:
                splited_bounds.append(bound)

        return splited_bounds

    def verify_single_cell(self, cell: Dict, num_steps: int = 20) -> bool:
        """Verify a single cell for the specified number of steps (early-stop when |theta| <= 0.15).
        Only keep the final step's bounds to save memory.
        """
        # Current state bounds
        current_bounds = [np.array([cell['theta'], cell['omega']]).T]

        # Safety tracking (reach goal if BOTH bounds are within ±0.15)
        reached_goal = False

        # Perform verification steps with early-stop
        for step in range(1, num_steps + 1):
            new_bounds = []
            for bound in current_bounds: 
                
                    new_bound = self.verify_single_step(bound)
                    new_bounds.append(new_bound)
            
            # Update current state (we only keep the latest bounds)
            current_bounds = self.split_merge_bounds(new_bounds)
            
            # EARLY STOP: if BOTH |theta_min| and |theta_max| <= 0.15, treat as SAFE and stop. 
            if np.max(np.abs(np.array(current_bounds)[:,0,:])) <= self.goal_angle_threshold:
                reached_goal = True
                break

        return reached_goal

class MountainCarVerifier(Verifier):
    """StarV-based Neural Network Verification for Mountain Car System"""

    def __init__(self, weights_path: str, goal_position_threshold = 0.6):
        # Create the controller and decoder
        self.nnmodel = NNModel('tanh', 1.0)
        self.mountain_car = MountainCar()

        # Safety condition: BOTH min and max position >= 0.6
        self.goal_position_threshold = goal_position_threshold

        weights = torch.load(weights_path, 'cpu', weights_only=True)
        self.nnmodel.load_state_dict(weights)


    def verify_single_step(self, bound: np.ndarray) -> Dict:
        """Perform single-step verification: state -> action -> next_state"""

        input_star = Star(bound[0], bound[1])

        # Verify the model
        output_star = self.nnmodel.reach(input_star)
        action_bound = np.array(output_star.getRanges('gurobi'))
        bound = np.concatenate([bound, action_bound], axis=1)
        # Step 3: Compute next state bounds
        new_bound = self.mountain_car.reach(bound)

        return new_bound
    
    def verify_single_cell(self, cell: Dict, num_steps: int = 30) -> Dict:
        """Verify a single cell for the specified number of steps"""
        # Current state bounds
        current_bound = np.array([cell['pos'], cell['vel']]).T

        # Safety tracking
        reached_goal = False

        # Perform verification steps
        for step in range(1, num_steps + 1):
            # Single step verification
            current_bound = self.verify_single_step(current_bound)

            # Check safety condition: BOTH min and max position must be >= threshold
            if np.min(current_bound[:,0]) >= self.goal_position_threshold:
                reached_goal = True
                break

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
        
