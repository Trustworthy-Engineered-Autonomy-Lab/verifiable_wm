from models import PenController, PenDecoder
from models import MCController, MCDecoder

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

class Verifer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def verify_single_cell(self, cell: Dict, num_steps: int = 20) -> bool:
        pass

# ============ Pendulum Neural Network Verifier ============
class Pendulum(Verifer):
    """StarV-based Neural Network Verification for Pendulum System"""

    def __init__(self, decoder_path: str, controller_path: str, goal_angle_threshold = 0.15):

        self.decoder_weights = torch.load(decoder_path, map_location="cpu", weights_only=True)
        self.controller_weights = torch.load(controller_path, map_location="cpu", weights_only=True)

        self.goal_angle_threshold = goal_angle_threshold

    def angle_normalize(self, x):
        """Normalize angle to [-π, π] range"""
        return ((x + np.pi) % (2 * np.pi)) - np.pi

    def verify_decoder(self, theta_min: float, theta_max: float,
                       omega_min: float, omega_max: float) -> ImageStar:
        """Verify decoder network using StarV framework"""
        lb = np.array([theta_min, omega_min], dtype=np.float32)
        ub = np.array([theta_max, omega_max], dtype=np.float32)
        input_star = Star(lb, ub)

        # FC1 + ReLU
        W1 = self.decoder_weights["fc1.weight"].cpu().numpy()
        b1 = self.decoder_weights["fc1.bias"].cpu().numpy()
        L_fc_1 = FullyConnectedLayer([W1, b1])
        R_fc_1 = L_fc_1.reach([input_star])
        L_relu_1 = ReLULayer()
        R_relu_1 = L_relu_1.reach(R_fc_1[0], method='approx')

        # FC2 + ReLU
        W2 = self.decoder_weights["fc2.weight"].cpu().numpy()
        b2 = self.decoder_weights["fc2.bias"].cpu().numpy()
        L_fc_2 = FullyConnectedLayer([W2, b2])
        R_fc_2 = L_fc_2.reach([R_relu_1])
        L_relu_2 = ReLULayer()
        R_relu_2 = L_relu_2.reach(R_fc_2[0], method='approx')

        # FC3 + ReLU
        W3 = self.decoder_weights["fc3.weight"].cpu().numpy()
        b3 = self.decoder_weights["fc3.bias"].cpu().numpy()
        L_fc_3 = FullyConnectedLayer([W3, b3])
        R_fc_3 = L_fc_3.reach([R_relu_2])
        L_relu_3 = ReLULayer()
        R_relu_3 = L_relu_3.reach(R_fc_3[0], method='approx')

        # Convert to ImageStar
        nP = R_relu_3.nVars
        V4 = R_relu_3.V.reshape(3, 12, 12, nP + 1).transpose(1, 2, 0, 3)
        IM = ImageStar(V4, R_relu_3.C, R_relu_3.d, R_relu_3.pred_lb, R_relu_3.pred_ub)

        # ConvTranspose1 + ReLU
        w_dec_conv1 = self.decoder_weights["dec_conv1.weight"].cpu().numpy()
        b_dec_conv1 = self.decoder_weights["dec_conv1.bias"].cpu().numpy()
        w_dec_conv1 = np.transpose(w_dec_conv1, (2, 3, 1, 0))
        L_convt_1 = ConvTranspose2DLayer([w_dec_conv1, b_dec_conv1], [2, 2], [1, 1, 1, 1], [1, 1])
        R_convt_1 = L_convt_1.reach(IM, method='approx')
        R_star_convt_1 = R_convt_1.toStar()
        L_relu_convt_1 = ReLULayer()
        R_star_relu_convt_1 = L_relu_convt_1.reach(R_star_convt_1, method='approx')
        R_convt_1 = R_star_relu_convt_1.toImageStar(image_shape=(24, 24, 4))

        # ConvTranspose2 + ReLU
        w_dec_conv2 = self.decoder_weights["dec_conv2.weight"].cpu().numpy()
        b_dec_conv2 = self.decoder_weights["dec_conv2.bias"].cpu().numpy()
        w_dec_conv2 = np.transpose(w_dec_conv2, (2, 3, 1, 0))
        L_convt_2 = ConvTranspose2DLayer([w_dec_conv2, b_dec_conv2], [2, 2], [1, 1, 1, 1], [1, 1])
        R_convt_2 = L_convt_2.reach(R_convt_1, method='approx')
        R_star_convt_2 = R_convt_2.toStar()
        L_relu_convt_2 = ReLULayer()
        R_star_relu_convt_2 = L_relu_convt_2.reach(R_star_convt_2, method='approx')
        R_convt_2 = R_star_relu_convt_2.toImageStar(image_shape=(48, 48, 8))

        # ConvTranspose3
        w_dec_conv3 = self.decoder_weights["dec_conv3.weight"].cpu().numpy()
        b_dec_conv3 = self.decoder_weights["dec_conv3.bias"].cpu().numpy()
        w_dec_conv3 = np.transpose(w_dec_conv3, (2, 3, 1, 0))
        L_convt_3 = ConvTranspose2DLayer([w_dec_conv3, b_dec_conv3], [2, 2], [1, 1, 1, 1], [1, 1])
        R_convt_3 = L_convt_3.reach(R_convt_2, method='approx')

        # SatLin
        S1 = R_convt_3.toStar()
        L_satlin = SatLinLayer()
        IM_satlin_list = L_satlin.reach(S1, method='approx', lp_solver='gurobi')
        IM_satlin = IM_satlin_list.toImageStar(image_shape=(96, 96, 1))

        return IM_satlin

    def verify_controller(self, image_star: ImageStar) -> Star:
        """Verify controller network and return action Star"""
        # Conv1 + ReLU
        w_conv1 = self.controller_weights['conv1.weight'].cpu().numpy()
        b_conv1 = self.controller_weights['conv1.bias'].cpu().numpy()
        w_conv1 = np.transpose(w_conv1, (2, 3, 1, 0))
        L_conv1 = Conv2DLayer([w_conv1, b_conv1], [2, 2], [1, 1, 1, 1], [1, 1])
        R_conv1 = L_conv1.reach([image_star])
        IM_conv1 = R_conv1[0]

        IM_conv1_star = IM_conv1.toStar()
        L_relu_conv1 = ReLULayer()
        R_relu_conv1 = L_relu_conv1.reach(IM_conv1_star, method='approx')
        IM_relu_conv1 = R_relu_conv1.toImageStar(image_shape=(48, 48, 4))

        # Conv2 + ReLU
        w_conv2 = self.controller_weights['conv2.weight'].cpu().numpy()
        b_conv2 = self.controller_weights['conv2.bias'].cpu().numpy()
        w_conv2 = np.transpose(w_conv2, (2, 3, 1, 0))
        L_conv2 = Conv2DLayer([w_conv2, b_conv2], [2, 2], [1, 1, 1, 1], [1, 1])
        R_conv2 = L_conv2.reach([IM_relu_conv1])
        IM_conv2 = R_conv2[0]

        IM_conv2_star = IM_conv2.toStar()
        L_relu_conv2 = ReLULayer()
        R_relu_conv2 = L_relu_conv2.reach(IM_conv2_star, method='approx')

        # FC1 + ReLU
        Wc1 = self.controller_weights["fc1.weight"].cpu().numpy()
        bc1 = self.controller_weights["fc1.bias"].cpu().numpy()
        L_fc_c1 = FullyConnectedLayer([Wc1, bc1])
        R_fc_c1 = L_fc_c1.reach([R_relu_conv2])
        star_fc_c1 = R_fc_c1[0]

        L_relu_fc1 = ReLULayer()
        R_relu_fc1 = L_relu_fc1.reach(star_fc_c1, method='approx')

        # FC2
        Wc2 = self.controller_weights["fc2.weight"].cpu().numpy()
        bc2 = self.controller_weights["fc2.bias"].cpu().numpy()
        L_fc_c2 = FullyConnectedLayer([Wc2, bc2])
        R_fc_c2 = L_fc_c2.reach([R_relu_fc1])
        star_fc_c2 = R_fc_c2[0]

        # Tanh activation
        L_tanh = TanSigLayer()
        IM_tanh = L_tanh.reach(star_fc_c2, method='approx', RF=0.0)

        # Scale action by 2
        d = IM_tanh.dim
        A = 2.0 * np.eye(d, dtype=np.float32)
        b = np.zeros(d, dtype=np.float32)
        IM_action = IM_tanh.affineMap(A, b)

        return IM_action

    def verify_single_step(self, theta_bound: Sequence[float], omega_bound: Sequence[float]) -> Dict:
        """
        Perform single-step verification with pendulum dynamics
        theta' = theta + 0.05 * omega + 0.0075 * u + 0.0375 * sin(theta)
        omega' = omega + 0.15 * u + 0.75 * sin(theta), clipped to [-8, 8]
        """
        # Create initial Star set
        # bounds = np.stack([theta_bound, omega_bound]).T
        # S_state = Star(bounds[0], bounds[1])
        theta_min, theta_max = theta_bound
        omega_min, omega_max = omega_bound
        # Verify decoder
        image_star = self.verify_decoder(*theta_bound, *omega_bound)

        # Verify controller and get action Star
        IM_action = self.verify_controller(image_star)
        x_min, x_max = IM_action.getRanges(lp_solver='gurobi')

        # Compute sin(theta) using SinLayer
        L_sin = SinLayer()
        lb_theta = np.array([theta_min], dtype=np.float32)
        ub_theta = np.array([theta_max], dtype=np.float32)
        S_theta = Star(lb_theta, ub_theta)

        IM_sin = L_sin.reach(S_theta, method='approx', lp_solver='gurobi', RF=0.0)
        z_min, z_max = IM_sin.getRanges(lp_solver='gurobi')
        
        theta_lb, theta_ub = float(theta_min), float(theta_max)
        omega_lb, omega_ub = float(omega_min), float(omega_max)

        u_lb, u_ub = float(x_min), float(x_max)
        s_lb, s_ub = float(z_min), float(z_max)

        lb_full = np.array([theta_lb, omega_lb, u_lb, s_lb], dtype=np.float32)
        ub_full = np.array([theta_ub, omega_ub, u_ub, s_ub], dtype=np.float32)

        S_full = Star(lb_full, ub_full)

        # Apply dynamics
        M = np.array([[1.0, 0.05, 0.0075, 0.0375],
                      [0.0, 1.0, 0.15, 0.75]], dtype=np.float32)
        b_dyn = np.zeros(2, dtype=np.float32)
        S_next = S_full.affineMap(M, b_dyn)

        # Step 7: Get bounds BEFORE clipping
        lb_next, ub_next = S_next.getRanges('gurobi', RF=0.0)

        # Step 8: Clip omega' to [-8, 8]
        theta_next_min = float(lb_next[0])
        theta_next_max = float(ub_next[0])
        omega_next_min = float(lb_next[1])
        omega_next_max = float(ub_next[1])

        omega_next_min_clipped = np.clip(omega_next_min, -8.0, 8.0)
        omega_next_max_clipped = np.clip(omega_next_max, -8.0, 8.0)

        # Step 9: Normalize theta to [-π, π]
        theta_min_norm = self.angle_normalize(theta_next_min)
        theta_max_norm = self.angle_normalize(theta_next_max)

        # Step 10: Create new state Star
        # lb_state_norm = np.array([theta_min_norm, omega_next_min_clipped], dtype=np.float32)
        # ub_state_norm = np.array([theta_max_norm, omega_next_max_clipped], dtype=np.float32)

        return [theta_min_norm, theta_max_norm], [omega_next_min_clipped, omega_next_max_clipped]

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
    
# ============ StarV Neural Network Verifier ============
class MountainCar(Verifer):
    """StarV-based Neural Network Verification for Mountain Car System"""

    def __init__(self, decoder_path: str, controller_path: str, goal_position_threshold = 0.6):

        # Load pre-trained weights
        self.decoder_weights = torch.load(decoder_path, map_location="cpu", weights_only=True)
        self.controller_weights = torch.load(controller_path, map_location="cpu", weights_only=True)

        # Safety condition: BOTH min and max position >= 0.6
        self.goal_position_threshold = goal_position_threshold

        # Mountain Car dynamics constants
        self.MIN_POS = -1.2
        self.MAX_POS = 0.6
        self.MAX_SPEED = 0.07
        self.MIN_SPEED = -0.07
        self.POWER = 0.0015

    def verify_decoder(self, pos_min: float, pos_max: float,
                       vel_min: float, vel_max: float) -> ImageStar:
        """Verify decoder network using StarV framework"""
        # Create input Star set
        lb = np.array([pos_min, vel_min], dtype=np.float32)
        ub = np.array([pos_max, vel_max], dtype=np.float32)
        input_star = Star(lb, ub)

        # ---- Decoder Verification Pipeline ----

        # FC1 + ReLU
        W1 = self.decoder_weights["fc1.weight"].cpu().numpy()
        b1 = self.decoder_weights["fc1.bias"].cpu().numpy()
        L_fc_1 = FullyConnectedLayer([W1, b1])
        R_fc_1 = L_fc_1.reach([input_star])

        L_relu_1 = ReLULayer()
        R_relu_1 = L_relu_1.reach(R_fc_1[0], method='approx')

        # FC2 + ReLU
        W2 = self.decoder_weights["fc2.weight"].cpu().numpy()
        b2 = self.decoder_weights["fc2.bias"].cpu().numpy()
        L_fc_2 = FullyConnectedLayer([W2, b2])
        R_fc_2 = L_fc_2.reach([R_relu_1])

        L_relu_2 = ReLULayer()
        R_relu_2 = L_relu_2.reach(R_fc_2[0], method='approx')

        # FC3 + ReLU
        W3 = self.decoder_weights["fc3.weight"].cpu().numpy()
        b3 = self.decoder_weights["fc3.bias"].cpu().numpy()
        L_fc_3 = FullyConnectedLayer([W3, b3])
        R_fc_3 = L_fc_3.reach([R_relu_2])

        L_relu_3 = ReLULayer()
        R_relu_3 = L_relu_3.reach(R_fc_3[0], method='approx')

        # Convert to ImageStar for convolutional layers
        nP = R_relu_3.nVars
        V4 = R_relu_3.V.reshape(3, 12, 12, nP + 1).transpose(1, 2, 0, 3)
        IM = ImageStar(V4, R_relu_3.C, R_relu_3.d, R_relu_3.pred_lb, R_relu_3.pred_ub)

        # ConvTranspose1 + ReLU
        w_dec_conv1 = self.decoder_weights["dec_conv1.weight"].cpu().numpy()
        b_dec_conv1 = self.decoder_weights["dec_conv1.bias"].cpu().numpy()
        w_dec_conv1 = np.transpose(w_dec_conv1, (2, 3, 1, 0))
        L_convt_1 = ConvTranspose2DLayer([w_dec_conv1, b_dec_conv1], [2, 2], [1, 1, 1, 1], [1, 1])

        R_convt_1 = L_convt_1.reach(IM, method='approx')
        R_star_convt_1 = R_convt_1.toStar()
        L_relu_convt_1 = ReLULayer()
        R_star_relu_convt_1 = L_relu_convt_1.reach(R_star_convt_1, method='approx')
        R_convt_1 = R_star_relu_convt_1.toImageStar(image_shape=(24, 24, 4))

        # ConvTranspose2 + ReLU
        w_dec_conv2 = self.decoder_weights["dec_conv2.weight"].cpu().numpy()
        b_dec_conv2 = self.decoder_weights["dec_conv2.bias"].cpu().numpy()
        w_dec_conv2 = np.transpose(w_dec_conv2, (2, 3, 1, 0))
        L_convt_2 = ConvTranspose2DLayer([w_dec_conv2, b_dec_conv2], [2, 2], [1, 1, 1, 1], [1, 1])

        R_convt_2 = L_convt_2.reach(R_convt_1, method='approx')
        R_star_convt_2 = R_convt_2.toStar()
        L_relu_convt_2 = ReLULayer()
        R_star_relu_convt_2 = L_relu_convt_2.reach(R_star_convt_2, method='approx')
        R_convt_2 = R_star_relu_convt_2.toImageStar(image_shape=(48, 48, 8))

        # ConvTranspose3
        w_dec_conv3 = self.decoder_weights["dec_conv3.weight"].cpu().numpy()
        b_dec_conv3 = self.decoder_weights["dec_conv3.bias"].cpu().numpy()
        w_dec_conv3 = np.transpose(w_dec_conv3, (2, 3, 1, 0))
        L_convt_3 = ConvTranspose2DLayer([w_dec_conv3, b_dec_conv3], [2, 2], [1, 1, 1, 1], [1, 1])

        R_convt_3 = L_convt_3.reach(R_convt_2, method='approx')

        # SatLin
        S1 = R_convt_3.toStar()
        L_satlin = SatLinLayer()
        IM_satlin_list = L_satlin.reach(S1, method='approx', lp_solver='gurobi')
        IM_satlin = IM_satlin_list.toImageStar(image_shape=(96, 96, 1))

        return IM_satlin

    def verify_controller(self, image_star: ImageStar) -> Tuple[float, float]:
        """Verify controller network using StarV framework"""
        # Conv1 + ReLU
        w_conv1 = self.controller_weights['conv1.weight'].cpu().numpy()
        b_conv1 = self.controller_weights['conv1.bias'].cpu().numpy()
        w_conv1 = np.transpose(w_conv1, (2, 3, 1, 0))
        L_conv1 = Conv2DLayer([w_conv1, b_conv1], [2, 2], [1, 1, 1, 1], [1, 1])

        R_conv1 = L_conv1.reach([image_star])
        IM_conv1 = R_conv1[0]

        IM_conv1_star = IM_conv1.toStar()
        L_relu_conv1 = ReLULayer()
        R_relu_conv1 = L_relu_conv1.reach(IM_conv1_star, method='approx')
        IM_relu_conv1 = R_relu_conv1.toImageStar(image_shape=(48, 48, 4))

        # Conv2 + ReLU
        w_conv2 = self.controller_weights['conv2.weight'].cpu().numpy()
        b_conv2 = self.controller_weights['conv2.bias'].cpu().numpy()
        w_conv2 = np.transpose(w_conv2, (2, 3, 1, 0))
        L_conv2 = Conv2DLayer([w_conv2, b_conv2], [2, 2], [1, 1, 1, 1], [1, 1])

        R_conv2 = L_conv2.reach([IM_relu_conv1])
        IM_conv2 = R_conv2[0]

        IM_conv2_star = IM_conv2.toStar()
        L_relu_conv2 = ReLULayer()
        R_relu_conv2 = L_relu_conv2.reach(IM_conv2_star, method='approx')

        # FC1 + ReLU
        Wc1 = self.controller_weights["fc1.weight"].cpu().numpy()
        bc1 = self.controller_weights["fc1.bias"].cpu().numpy()
        L_fc_c1 = FullyConnectedLayer([Wc1, bc1])
        R_fc_c1 = L_fc_c1.reach([R_relu_conv2])
        star_fc_c1 = R_fc_c1[0]

        L_relu_fc1 = ReLULayer()
        R_relu_fc1 = L_relu_fc1.reach(star_fc_c1, method='approx')

        # FC2
        Wc2 = self.controller_weights["fc2.weight"].cpu().numpy()
        bc2 = self.controller_weights["fc2.bias"].cpu().numpy()
        L_fc_c2 = FullyConnectedLayer([Wc2, bc2])
        R_fc_c2 = L_fc_c2.reach([R_relu_fc1])
        star_fc_c2 = R_fc_c2[0]

        # Tanh activation
        L_tanh = TanSigLayer()
        R_tanh = L_tanh.reach(star_fc_c2, method='approx', RF=0.0)

        # Get action bounds
        y_min, y_max = R_tanh.getRanges('gurobi')
        return float(y_min[0]), float(y_max[0])

    def step_bounds_dynamics(self, pos_min: float, pos_max: float,
                             vel_min: float, vel_max: float,
                             action_min: float, action_max: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Compute next state bounds using Mountain Car dynamics with interval arithmetic"""
        # Cosine term bounds: cos(3*pos)
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

        return (pos_next_min, pos_next_max), (vel_next_min, vel_next_max)

    def verify_single_step(self, pos_min: float, pos_max: float,
                           vel_min: float, vel_max: float) -> Dict:
        """Perform single-step verification: state -> action -> next_state"""
        # Step 1: Verify decoder
        image_star = self.verify_decoder(pos_min, pos_max, vel_min, vel_max)

        # Step 2: Verify controller
        action_min, action_max = self.verify_controller(image_star)

        # Step 3: Compute next state bounds
        (pos_next_min, pos_next_max), (vel_next_min, vel_next_max) = self.step_bounds_dynamics(
            pos_min, pos_max, vel_min, vel_max, action_min, action_max
        )

        return {
            'pos_min': pos_next_min,
            'pos_max': pos_next_max,
            'vel_min': vel_next_min,
            'vel_max': vel_next_max,
            'action_min': action_min,
            'action_max': action_max
        }
    
    def verify_single_cell(self, cell: Dict, num_steps: int = 30) -> Dict:
        """Verify a single cell for the specified number of steps"""
        # Current state bounds
        current_pos_min = cell['pos'][0]
        current_pos_max = cell['pos'][1]
        current_vel_min = cell['vel'][0]
        current_vel_max = cell['vel'][1]

        # Safety tracking
        reached_goal = False

        # Perform verification steps
        for step in range(1, num_steps + 1):
            # Single step verification
            step_result = self.verify_single_step(
                current_pos_min, current_pos_max,
                current_vel_min, current_vel_max
            )

            # Check safety condition: BOTH min and max position must be >= threshold
            if step_result['pos_min'] >= self.goal_position_threshold and step_result['pos_max'] >= self.goal_position_threshold:
                reached_goal = True
                break

            # Update current state for next iteration
            current_pos_min = step_result['pos_min']
            current_pos_max = step_result['pos_max']
            current_vel_min = step_result['vel_min']
            current_vel_max = step_result['vel_max']

        return reached_goal
    
class Cartpole(Verifer):
    def __init__(self, decoder_path: str, controller_path: str):
        pass

    def verify_single_cell(self, cell, num_steps = 20):
        pos_lb, pos_ub = cell['pos']
        angle_lb, angle_ub = cell['angle']

        reach_goal = False

        # Note: You don't have to handle exception here

        return reach_goal

class _Test(Verifer):
    def __init__(self, raise_error = True):
        self.raise_error = raise_error

    def verify_single_cell(self, cell, num_steps = 20):
        if self.raise_error:
            raise RuntimeError("Error raised to test the program")
        
        return random.choice([True, False])
        
