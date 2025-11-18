import numpy as np

from typing import Dict, List
from abc import ABC, abstractmethod
import random

from models import Pendulum, MountainCar, Cartpole, FullModel

from colorama import Fore, Style

class Verifier(ABC):
    def __init__(self, save_history = False, num_steps = 20, early_stop = True):
        self.save_history = save_history
        self.num_steps = num_steps
        self.early_stop = early_stop

    def verify_single_cell(self, model: FullModel, cell: Dict):
        inital_bound = cell['bounds'][0].T
            
        # Verify one cell
        current_bounds = [inital_bound]

        for step in range(1, self.num_steps + 1):
            current_bounds = self.verify_single_step(model, current_bounds)
            # Save history if needed
            if self.save_history:
                cell['bounds'].append(np.concatenate(current_bounds).T)

            result = self.criteria(current_bounds)
            if result and self.early_stop:
                break

        if step < self.num_steps:
            print(Fore.YELLOW + f"early stop at step {step}" + Style.RESET_ALL)
        
        # If history wasn't saved, save the last bound
        if not self.save_history:
            cell['bounds'].append(np.concatenate(current_bounds).T)

        cell['result'] = result
    
    def split_merge_bounds(self, bounds: List[np.ndarray]) -> List[np.ndarray]:
        return bounds
    
    @abstractmethod
    def criteria(self, bounds: List[np.ndarray]) -> bool:
        pass

    @abstractmethod
    def dynamic_step(self, combined_bound: np.ndarray) -> np.ndarray:
        pass

    def verify_single_bound(self, model: FullModel, state_bound: np.ndarray) -> np.ndarray:
        action_bound = model.reach(state_bound)
        combined_bound = np.concatenate([state_bound, action_bound], axis=1)
        return self.dynamic_step(combined_bound)

    def verify_single_step(self, model: FullModel, bounds: List[np.ndarray]) -> List[np.ndarray]:
        new_bounds = []
        for bound in bounds: 

            new_bound = self.verify_single_bound(model, bound)
            new_bounds.append(new_bound)
        
        new_bounds = self.split_merge_bounds(new_bounds)
        return new_bounds

# ============ Pendulum Neural Network Verifier ============
class PendulumVerifier(Verifier):
    """StarV-based Neural Network Verification for Pendulum System"""

    def __init__(self, goal_angle_threshold = 0.15, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.goal_angle_threshold = goal_angle_threshold
        self.dynamic = Pendulum()

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
    
    def dynamic_step(self, bound: np.ndarray):
        return self.dynamic.reach(bound)
    
    def criteria(self, bounds: List[np.ndarray]) -> bool:
        # if BOTH |theta_min| and |theta_max| <= 0.15, treat as SAFE.
        theta_bounds = np.array(bounds)[:,:,0]
        return True if np.all(np.abs(theta_bounds) <= self.goal_angle_threshold) else False


class MountainCarVerifier(Verifier):
    """StarV-based Neural Network Verification for Mountain Car System"""

    def __init__(self, goal_position_threshold = 0.6, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.goal_position_threshold = goal_position_threshold
        self.dynamic = MountainCar()

    def dynamic_step(self, bound: np.ndarray):
        return self.dynamic.reach(bound)

    def criteria(self, bounds: List[np.ndarray]) -> bool:
        # Check safety condition: BOTH min and max position must be >= threshold
        pos_bound = np.array(bounds)[:,:,0]
        return True if np.all(pos_bound >= self.goal_position_threshold) else False
    
class CartpoleVerifier(Verifier):
    def __init__(self, goal_angle_threshold = 12, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.goal_angle_threshold = goal_angle_threshold
        self.dynamic = Cartpole()

    def verify_single_bound(self, model: FullModel, state_bound: np.ndarray) -> np.ndarray:
        action_bound = model.reach(state_bound[:,(0,2)])
        combined_bound = np.concatenate([state_bound, action_bound], axis=1)
        return self.dynamic_step(combined_bound)

    def dynamic_step(self, bound: np.ndarray):
        return self.dynamic.reach(bound)

    def criteria(self, bounds: List[np.ndarray]) -> bool:
        # Check safety condition
        angle_bound = np.array(bounds)[:,:,2]
        return True if np.all(np.abs(angle_bound) <= self.goal_angle_threshold) else False

# class _Test(Verifier):
    # def __init__(self, raise_error = True):
    #     self.raise_error = raise_error

    # def verify_single_cell(self, cell):
    #     if self.raise_error:
    #         raise RuntimeError("Error raised to test the program")
        
    #     while True:
    #         yield random.choice([True, False])
        
