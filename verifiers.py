import numpy as np

from typing import Dict, List
from abc import ABC, abstractmethod
import random

from models import FullModel


class Verifier(ABC):
    def __init__(self):
        pass

    def verify_single_cell(self, cell: Dict):
        current_bounds = [np.array(list(cell['init_bound'].values())).T]

        # Perform verification steps
        while True:
            current_bounds = self.verify_single_step(current_bounds)

            yield self.criteria(current_bounds)
    
    def split_merge_bounds(self, bounds: List[np.ndarray]) -> List[np.ndarray]:
        return bounds
    
    @abstractmethod
    def criteria(self, bounds: List[np.ndarray]) -> bool:
        pass

    @abstractmethod
    def verify_single_bound(self, bound: np.ndarray) -> np.ndarray:
        pass

    def verify_single_step(self, bounds: List[np.ndarray]) -> List[np.ndarray]:
        new_bounds = []
        for bound in bounds: 
            
            new_bound = self.verify_single_bound(bound)
            new_bounds.append(new_bound)
        
        new_bounds = self.split_merge_bounds(new_bounds)
        return new_bounds

# ============ Pendulum Neural Network Verifier ============
class PendulumVerifier(Verifier):
    """StarV-based Neural Network Verification for Pendulum System"""

    def __init__(self, goal_angle_threshold = 0.15):
        # Create the controller and decoder
        self.goal_angle_threshold = goal_angle_threshold
        self.fullmodel = FullModel('pendulum')

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
    
    def verify_single_bound(self, bound: np.ndarray):
        return self.fullmodel.reach(bound)
    
    def criteria(self, bounds: List[np.ndarray]) -> bool:
        # if BOTH |theta_min| and |theta_max| <= 0.15, treat as SAFE.
        theta_bounds = np.array(bounds)[:,:,0]
        return True if np.all(np.abs(theta_bounds) <= self.goal_angle_threshold) else False


class MountainCarVerifier(Verifier):
    """StarV-based Neural Network Verification for Mountain Car System"""

    def __init__(self, goal_position_threshold = 0.6):
        # Safety condition: BOTH min and max position >= 0.6
        self.goal_position_threshold = goal_position_threshold
        self.fullmodel = FullModel('mountain_car')

    def verify_single_bound(self, bound: np.ndarray):
        return self.fullmodel.reach(bound)

    def criteria(self, bounds: List[np.ndarray]) -> bool:
        # Check safety condition: BOTH min and max position must be >= threshold
        pos_bound = bounds[0][:,0]
        return True if np.all(pos_bound >= self.goal_position_threshold) else False
    
class CartpoleVerifier(Verifier):
    def __init__(self, goal_angle_threshold = 12):
        self.goal_angle_threshold = goal_angle_threshold
        self.fullmodel = FullModel('cartpole')

    def verify_single_bound(self, bound: np.ndarray):
        return self.fullmodel.reach(bound)

    def criteria(self, bounds: List[np.ndarray]) -> bool:
        # Check safety condition
        angle_bound = bounds[0][:,1]
        return True if np.all(np.abs(angle_bound) <= self.goal_angle_threshold) else False

class _Test(Verifier):
    def __init__(self, raise_error = True):
        self.raise_error = raise_error

    def verify_single_cell(self, cell):
        if self.raise_error:
            raise RuntimeError("Error raised to test the program")
        
        while True:
            yield random.choice([True, False])
        
