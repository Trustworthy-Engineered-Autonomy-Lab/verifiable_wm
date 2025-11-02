from StarV.layer.FullyConnectedLayer import FullyConnectedLayer
from StarV.layer.ConvTranspose2DLayer import ConvTranspose2DLayer
from StarV.layer.ReLULayer import ReLULayer
from StarV.layer.Conv2DLayer import Conv2DLayer
from StarV.layer.SatLinLayer import SatLinLayer
from StarV.layer.TanSigLayer import TanSigLayer
from StarV.set.star import Star
from StarV.set.imagestar import ImageStar
from StarV.dynamic.Sine import SinLayer

import torch

import numpy as np
import math

from typing import Sequence, Dict, Tuple
from abc import ABC, abstractmethod
import random

class Verifier(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def verify_single_cell(self, cell: Dict, num_steps: int = 20) -> bool:
        pass

class Module(ABC):
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        return self.reach(*args, **kwargs)

    @abstractmethod
    def reach(self, *args, **kwargs):
        pass

class StarVModule(Module):
    def __init__(self, weights_path):
        super().__init__()
        self._weights = torch.load(weights_path, map_location="cpu", weights_only=True)

class Controller(StarVModule):
    def __init__(self, weights_path):
        super().__init__(weights_path)

    def reach(self, image_star: ImageStar) -> Star:
        # Conv1 + ReLU
        w_conv1 = self._weights['conv1.weight'].cpu().numpy()
        b_conv1 = self._weights['conv1.bias'].cpu().numpy()
        w_conv1 = np.transpose(w_conv1, (2, 3, 1, 0))
        L_conv1 = Conv2DLayer([w_conv1, b_conv1], [2, 2], [1, 1, 1, 1], [1, 1])
        R_conv1 = L_conv1.reach([image_star])
        IM_conv1 = R_conv1[0]

        IM_conv1_star = IM_conv1.toStar()
        L_relu_conv1 = ReLULayer()
        R_relu_conv1 = L_relu_conv1.reach(IM_conv1_star, method='approx')
        IM_relu_conv1 = R_relu_conv1.toImageStar(image_shape=(48, 48, 4))

        # Conv2 + ReLU
        w_conv2 = self._weights['conv2.weight'].cpu().numpy()
        b_conv2 = self._weights['conv2.bias'].cpu().numpy()
        w_conv2 = np.transpose(w_conv2, (2, 3, 1, 0))
        L_conv2 = Conv2DLayer([w_conv2, b_conv2], [2, 2], [1, 1, 1, 1], [1, 1])
        R_conv2 = L_conv2.reach([IM_relu_conv1])
        IM_conv2 = R_conv2[0]

        IM_conv2_star = IM_conv2.toStar()
        L_relu_conv2 = ReLULayer()
        R_relu_conv2 = L_relu_conv2.reach(IM_conv2_star, method='approx')

        # FC1 + ReLU
        Wc1 = self._weights["fc1.weight"].cpu().numpy()
        bc1 = self._weights["fc1.bias"].cpu().numpy()
        L_fc_c1 = FullyConnectedLayer([Wc1, bc1])
        R_fc_c1 = L_fc_c1.reach([R_relu_conv2])
        star_fc_c1 = R_fc_c1[0]

        L_relu_fc1 = ReLULayer()
        R_relu_fc1 = L_relu_fc1.reach(star_fc_c1, method='approx')

        # FC2
        Wc2 = self._weights["fc2.weight"].cpu().numpy()
        bc2 = self._weights["fc2.bias"].cpu().numpy()
        L_fc_c2 = FullyConnectedLayer([Wc2, bc2])
        R_fc_c2 = L_fc_c2.reach([R_relu_fc1])
        star_fc_c2 = R_fc_c2[0]

        # Tanh activation
        L_tanh = TanSigLayer()
        IM_tanh = L_tanh.reach(star_fc_c2, method='approx', RF=0.0)

        return IM_tanh
    
class Decoder(StarVModule):
    def __init__(self, weights_path):
        super().__init__(weights_path)

    def reach(self, input_star: Star) -> ImageStar:
        # FC1 + ReLU
        W1 = self._weights["fc1.weight"].cpu().numpy()
        b1 = self._weights["fc1.bias"].cpu().numpy()
        L_fc_1 = FullyConnectedLayer([W1, b1])
        R_fc_1 = L_fc_1.reach([input_star])
        L_relu_1 = ReLULayer()
        R_relu_1 = L_relu_1.reach(R_fc_1[0], method='approx')

        # FC2 + ReLU
        W2 = self._weights["fc2.weight"].cpu().numpy()
        b2 = self._weights["fc2.bias"].cpu().numpy()
        L_fc_2 = FullyConnectedLayer([W2, b2])
        R_fc_2 = L_fc_2.reach([R_relu_1])
        L_relu_2 = ReLULayer()
        R_relu_2 = L_relu_2.reach(R_fc_2[0], method='approx')

        # FC3 + ReLU
        W3 = self._weights["fc3.weight"].cpu().numpy()
        b3 = self._weights["fc3.bias"].cpu().numpy()
        L_fc_3 = FullyConnectedLayer([W3, b3])
        R_fc_3 = L_fc_3.reach([R_relu_2])
        L_relu_3 = ReLULayer()
        R_relu_3 = L_relu_3.reach(R_fc_3[0], method='approx')

        # Convert to ImageStar
        nP = R_relu_3.nVars
        V4 = R_relu_3.V.reshape(3, 12, 12, nP + 1).transpose(1, 2, 0, 3)
        IM = ImageStar(V4, R_relu_3.C, R_relu_3.d, R_relu_3.pred_lb, R_relu_3.pred_ub)

        # ConvTranspose1 + ReLU
        w_dec_conv1 = self._weights["dec_conv1.weight"].cpu().numpy()
        b_dec_conv1 = self._weights["dec_conv1.bias"].cpu().numpy()
        w_dec_conv1 = np.transpose(w_dec_conv1, (2, 3, 1, 0))
        L_convt_1 = ConvTranspose2DLayer([w_dec_conv1, b_dec_conv1], [2, 2], [1, 1, 1, 1], [1, 1])
        R_convt_1 = L_convt_1.reach(IM, method='approx')
        R_star_convt_1 = R_convt_1.toStar()
        L_relu_convt_1 = ReLULayer()
        R_star_relu_convt_1 = L_relu_convt_1.reach(R_star_convt_1, method='approx')
        R_convt_1 = R_star_relu_convt_1.toImageStar(image_shape=(24, 24, 4))

        # ConvTranspose2 + ReLU
        w_dec_conv2 = self._weights["dec_conv2.weight"].cpu().numpy()
        b_dec_conv2 = self._weights["dec_conv2.bias"].cpu().numpy()
        w_dec_conv2 = np.transpose(w_dec_conv2, (2, 3, 1, 0))
        L_convt_2 = ConvTranspose2DLayer([w_dec_conv2, b_dec_conv2], [2, 2], [1, 1, 1, 1], [1, 1])
        R_convt_2 = L_convt_2.reach(R_convt_1, method='approx')
        R_star_convt_2 = R_convt_2.toStar()
        L_relu_convt_2 = ReLULayer()
        R_star_relu_convt_2 = L_relu_convt_2.reach(R_star_convt_2, method='approx')
        R_convt_2 = R_star_relu_convt_2.toImageStar(image_shape=(48, 48, 8))

        # ConvTranspose3
        w_dec_conv3 = self._weights["dec_conv3.weight"].cpu().numpy()
        b_dec_conv3 = self._weights["dec_conv3.bias"].cpu().numpy()
        w_dec_conv3 = np.transpose(w_dec_conv3, (2, 3, 1, 0))
        L_convt_3 = ConvTranspose2DLayer([w_dec_conv3, b_dec_conv3], [2, 2], [1, 1, 1, 1], [1, 1])
        R_convt_3 = L_convt_3.reach(R_convt_2, method='approx')

        # SatLin
        S1 = R_convt_3.toStar()
        L_satlin = SatLinLayer()
        IM_satlin_list = L_satlin.reach(S1, method='approx', lp_solver='gurobi')
        IM_satlin = IM_satlin_list.toImageStar(image_shape=(96, 96, 1))

        return IM_satlin
    
class Pendulum(Module):
    def __init__(self):
        super().__init__()

    def angle_normalize(self, x):
        """Normalize angle to [-π, π] range"""
        return ((x + np.pi) % (2 * np.pi)) - np.pi

    def reach(self, theta_bound: Sequence[float], omega_bound: Sequence[float], action_bound: Sequence[float]):
        # Compute sin(theta) using SinLayer
        theta_min, theta_max = theta_bound
        omega_min, omega_max = omega_bound
        action_min, action_max = action_bound

        L_sin = SinLayer()
        lb_theta = np.array([theta_min], dtype=np.float32)
        ub_theta = np.array([theta_max], dtype=np.float32)
        S_theta = Star(lb_theta, ub_theta)

        IM_sin = L_sin.reach(S_theta, method='approx', lp_solver='gurobi', RF=0.0)
        z_min, z_max = np.concatenate(IM_sin.getRanges(lp_solver='gurobi'))

        lb_full = np.array([theta_min, omega_min, action_min, z_min], dtype=np.float32)
        ub_full = np.array([theta_max, omega_max, action_max, z_max], dtype=np.float32)

        S_full = Star(lb_full, ub_full)

        # Apply dynamics
        M = np.array([[1.0, 0.05, 0.0075, 0.0375],
                      [0.0, 1.0, 0.15, 0.75]], dtype=np.float32)
        b_dyn = np.zeros(2, dtype=np.float32)
        S_next = S_full.affineMap(M, b_dyn)

        # Step 7: Get bounds BEFORE clipping
        lb_next, ub_next = S_next.getRanges('gurobi', RF=0.0)
        bound = np.array([lb_next, ub_next]).T

        # Clip omega' to [-8, 8]
        next_omega_bound = np.clip(bound[1], -8.0, 8.0)

        # Normalize theta to [-π, π]
        next_theta_bound = self.angle_normalize(bound[0])

        return list(next_theta_bound), list(next_omega_bound)
    
class MountainCar(Module):
    def __init__(self):
        super().__init__()

        # Mountain Car dynamics constants
        self.MIN_POS = -1.2
        self.MAX_POS = 0.6
        self.MAX_SPEED = 0.07
        self.MIN_SPEED = -0.07
        self.POWER = 0.0015

    def reach(self, pos_bound: Sequence[float], vel_bound: Sequence[float], action_bound: Sequence[float]):
        # Cosine term bounds: cos(3*pos)
        pos_min, pos_max = pos_bound
        vel_min, vel_max = vel_bound
        action_min, action_max = action_bound

        cos_3p_min = math.cos(3.0 * pos_min)
        cos_3p_max = math.cos(3.0 * pos_max)
        cos_min = min(cos_3p_min, cos_3p_max)
        cos_max = max(cos_3p_min, cos_3p_max)

        # Velocity update: v' = v + action*POWER - 0.0025*cos(3*pos)
        vel_next_min = vel_min + action_min * self.POWER - 0.0025 * cos_max
        vel_next_max = vel_max + action_max * self.POWER - 0.0025 * cos_min

        # Velocity clipping: both bounds must respect speed limits
        vel_next_min = max(self.MIN_SPEED, min(self.MAX_SPEED, vel_next_min))
        vel_next_max = max(self.MIN_SPEED, min(self.MAX_SPEED, vel_next_max))

        # Ensure velocity bounds are valid (min <= max)
        if vel_next_min > vel_next_max:
            vel_next_min, vel_next_max = vel_next_max, vel_next_min

        # Position update: pos' = pos + v'
        pos_next_min = pos_min + vel_next_min
        pos_next_max = pos_max + vel_next_max

        # Ensure position bounds are valid (min <= max)
        # if pos_next_min > pos_next_max:
        #     pos_next_min, pos_next_max = pos_next_max, pos_next_min

        return [pos_next_min, pos_next_max], [vel_next_min, vel_next_max]

# ============ Pendulum Neural Network Verifier ============
class PendulumVerifier(Verifier):
    """StarV-based Neural Network Verification for Pendulum System"""

    def __init__(self, decoder_path: str, controller_path: str, goal_angle_threshold = 0.15):
        # Create the controller and decoder
        self.controller = Controller(controller_path)
        self.decoder = Decoder(decoder_path)
        self.pendulum = Pendulum()
        self.goal_angle_threshold = goal_angle_threshold
    

    def verify_single_step(self, theta_bound: Sequence[float], omega_bound: Sequence[float]) -> Dict:
        """
        Perform single-step verification with pendulum dynamics
        theta' = theta + 0.05 * omega + 0.0075 * u + 0.0375 * sin(theta)
        omega' = omega + 0.15 * u + 0.75 * sin(theta), clipped to [-8, 8]
        """
        # Create initial Star set
        theta_min, theta_max = theta_bound
        omega_min, omega_max = omega_bound
        # Verify decoder
        lb = np.array([theta_min, omega_min], dtype=np.float32)
        ub = np.array([theta_max, omega_max], dtype=np.float32)
        input_star = Star(lb, ub)
        image_star = self.decoder(input_star)
        # Verify controller and get action Star
        output_star = self.controller(image_star)

        d = output_star.dim
        A = 2.0 * np.eye(d, dtype=np.float32)
        b = np.zeros(d, dtype=np.float32)
        action = output_star.affineMap(A, b)
        action_bound = np.concatenate(action.getRanges(lp_solver='gurobi'))

        # Verfity the pendulum dynamic system
        next_theta_bound, next_omega_bound = self.pendulum(theta_bound, omega_bound, action_bound)

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

    def __init__(self, decoder_path: str, controller_path: str, goal_position_threshold = 0.6):
        # Create the controller and decoder
        self.controller = Controller(controller_path)
        self.decoder = Decoder(decoder_path)
        self.mountain_car = MountainCar()

        # Safety condition: BOTH min and max position >= 0.6
        self.goal_position_threshold = goal_position_threshold

    def verify_single_step(self, pos_bound: Sequence[float], vel_bound: Sequence[float]) -> Dict:
        """Perform single-step verification: state -> action -> next_state"""
        pos_min, pos_max = pos_bound
        vel_min, vel_max = vel_bound

        lb = np.array([pos_min, vel_min], dtype=np.float32)
        ub = np.array([pos_max, vel_max], dtype=np.float32)

        input_star = Star(lb, ub)

        # Step 1: Verify decoder
        image_star = self.decoder(input_star)

        # Step 2: Verify controller
        output_star = self.controller(image_star)

        action_bound = np.concatenate(output_star.getRanges('gurobi'))

        # Step 3: Compute next state bounds
        next_pos_bound, next_vel_bound = self.mountain_car(pos_bound, vel_bound, action_bound)

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
        
