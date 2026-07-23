import numpy as np

from typing import Dict, List
from abc import ABC, abstractmethod
import random

from starv_verification.model import FullModel
from starv_verification.dynamic import Pendulum, MountainCar, CartPole, Brake

from colorama import Fore, Style

class Verifier(ABC):
    def __init__(self, save_history = True, num_steps = 20, early_stop = False):
        self.save_history = save_history
        self.num_steps = num_steps
        self.early_stop = early_stop

    def verify_single_cell(self, model: FullModel, cell: Dict):
        inital_bound = cell['bounds'][0].T
            
        # Verify one cell
        current_bounds = [inital_bound]
        result = True

        for step in range(1, self.num_steps + 1):
            current_bounds = self.verify_single_step(model, current_bounds)
            # Save history if needed
            if self.save_history:
                cell['bounds'].append(np.concatenate(current_bounds).T)

            if self.is_unsafe(current_bounds):
                result = False
                if self.early_stop:
                    break
            else:
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

    def is_unsafe(self, bounds: List[np.ndarray]) -> bool:
        # Reach-avoid verifiers (e.g. BrakeVerifier) override this to flag
        # cells whose reachable set already touched the unsafe region; a
        # flagged cell stays unsafe even if a later step satisfies criteria.
        return False

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

    def __init__(self, goal_angle_threshold = 0.15, lp_solver = 'gurobi',
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.goal_angle_threshold = goal_angle_threshold
        self.dynamic = Pendulum(lp_solver=lp_solver)

    def split_merge_bounds(self, bounds):
        splited_bounds = []
        for bound in bounds:
            theta_bound = bound[:,0]
            omega_bound = bound[:,1]
            theta_min, theta_max = theta_bound
            omega_min, omega_max = omega_bound
            two_pi = 2 * np.pi
            if theta_max - theta_min >= two_pi:
                splited_bounds.append(
                    np.array([[-np.pi, omega_min], [np.pi, omega_max]])
                )
                continue

            start_period = np.floor((theta_min + np.pi) / two_pi)
            end_period = np.floor((theta_max + np.pi) / two_pi)
            wrapped_min = self.dynamic.angle_normalize(theta_min)
            wrapped_max = self.dynamic.angle_normalize(theta_max)

            if start_period == end_period:
                splited_bounds.append(
                    np.array([[wrapped_min, omega_min], [wrapped_max, omega_max]])
                )
            else:
                splited_bounds.append(
                    np.array([[wrapped_min, omega_min], [np.pi, omega_max]])
                )
                if wrapped_max > -np.pi:
                    splited_bounds.append(
                        np.array([[-np.pi, omega_min], [wrapped_max, omega_max]])
                    )

        return splited_bounds
    
    def dynamic_step(self, bound: np.ndarray):
        return self.dynamic.step(bound)
    
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
        return self.dynamic.step(bound)

    def criteria(self, bounds: List[np.ndarray]) -> bool:
        # Check safety condition: BOTH min and max position must be >= threshold
        pos_bound = np.array(bounds)[:,:,0]
        return True if np.all(pos_bound >= self.goal_position_threshold) else False
    
class CartpoleVerifier(Verifier):
    def __init__(self, goal_angle_threshold = 12, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.goal_angle_threshold = goal_angle_threshold
        self.dynamic = CartPole()

    def verify_single_bound(self, model: FullModel, state_bound: np.ndarray) -> np.ndarray:
        action_bound = model.reach(state_bound[:,(0,2)])
        combined_bound = np.concatenate([state_bound, action_bound], axis=1)
        return self.dynamic_step(combined_bound)

    def dynamic_step(self, bound: np.ndarray):
        return self.dynamic.step(bound)

    def criteria(self, bounds: List[np.ndarray]) -> bool:
        # Check safety condition
        angle_bound = np.array(bounds)[:,:,2]
        return True if np.all(np.abs(angle_bound) <= self.goal_angle_threshold) else False

class BrakeVerifier(Verifier):
    """StarV-based Neural Network Verification for the AEBS brake system"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dynamic = Brake()

    def dynamic_step(self, bound: np.ndarray):
        return self.dynamic.step(bound)

    def is_unsafe(self, bounds: List[np.ndarray]) -> bool:
        # Collision: distance lower bound reaching 0 at any step
        dis_lb = np.array(bounds)[:, 0, 0]
        return bool(np.any(dis_lb <= 0.0))

    def criteria(self, bounds: List[np.ndarray]) -> bool:
        dis_bound = np.array(bounds)[:, :, 0]
        return bool(np.all(dis_bound > 0.0))

# class _Test(Verifier):
    # def __init__(self, raise_error = True):
    #     self.raise_error = raise_error

    # def verify_single_cell(self, cell):
    #     if self.raise_error:
    #         raise RuntimeError("Error raised to test the program")
        
    #     while True:
    #         yield random.choice([True, False])
        
