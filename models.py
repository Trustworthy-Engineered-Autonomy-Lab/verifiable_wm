import torch
import torch.nn as nn
import torch.functional as F

import numpy as np

from abc import ABC, abstractmethod

from StarV.layer.FullyConnectedLayer import FullyConnectedLayer
from StarV.layer.ConvTranspose2DLayer import ConvTranspose2DLayer
from StarV.layer.ReLULayer import ReLULayer
from StarV.layer.Conv2DLayer import Conv2DLayer
from StarV.layer.SatLinLayer import SatLinLayer
from StarV.layer.TanSigLayer import TanSigLayer
from StarV.set.star import Star
from StarV.set.imagestar import ImageStar
from StarV.dynamic.Sine import SinLayer

from typing import Sequence
import math

class Module(nn.Module, ABC):
    @abstractmethod
    def reach(self, *args, **kwargs):
        pass

class Decoder(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 3 * 12 * 12)
        self.dec_conv1 = nn.ConvTranspose2d(3, 4, kernel_size=4, stride=2, padding=1)
        self.dec_conv2 = nn.ConvTranspose2d(4, 8, kernel_size=4, stride=2, padding=1)
        self.dec_conv3 = nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, states):
        b = states.size(0)
        x = F.relu(self.fc1(states))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = x.view(b, 3, 12, 12)
        x = F.relu(self.dec_conv1(x))
        x = F.relu(self.dec_conv2(x))
        x = self.dec_conv3(x)
        return torch.clamp(x, 0.0, 1.0)
    
    def reach(self, input_star: Star) -> ImageStar:
        # FC1 + ReLU
        W1 = self.fc1.weight.detach().cpu().numpy()
        b1 = self.fc1.bias.detach().cpu().numpy()
        L_fc_1 = FullyConnectedLayer([W1, b1])
        R_fc_1 = L_fc_1.reach([input_star])
        L_relu_1 = ReLULayer()
        R_relu_1 = L_relu_1.reach(R_fc_1[0], method='approx')

        # FC2 + ReLU
        W2 = self.fc2.weight.detach().cpu().numpy()
        b2 = self.fc2.bias.detach().cpu().numpy()
        L_fc_2 = FullyConnectedLayer([W2, b2])
        R_fc_2 = L_fc_2.reach([R_relu_1])
        L_relu_2 = ReLULayer()
        R_relu_2 = L_relu_2.reach(R_fc_2[0], method='approx')

        # FC3 + ReLU
        W3 = self.fc3.weight.detach().cpu().numpy()
        b3 = self.fc3.bias.detach().cpu().numpy()
        L_fc_3 = FullyConnectedLayer([W3, b3])
        R_fc_3 = L_fc_3.reach([R_relu_2])
        L_relu_3 = ReLULayer()
        R_relu_3 = L_relu_3.reach(R_fc_3[0], method='approx')

        # Convert to ImageStar
        nP = R_relu_3.nVars
        V4 = R_relu_3.V.reshape(3, 12, 12, nP + 1).transpose(1, 2, 0, 3)
        IM = ImageStar(V4, R_relu_3.C, R_relu_3.d, R_relu_3.pred_lb, R_relu_3.pred_ub)

        # ConvTranspose1 + ReLU
        w_dec_conv1 = self.dec_conv1.weight.detach().cpu().numpy()
        b_dec_conv1 = self.dec_conv1.bias.detach().cpu().numpy()
        w_dec_conv1 = np.transpose(w_dec_conv1, (2, 3, 1, 0))
        L_convt_1 = ConvTranspose2DLayer([w_dec_conv1, b_dec_conv1], [2, 2], [1, 1, 1, 1], [1, 1])
        R_convt_1 = L_convt_1.reach(IM, method='approx')
        R_star_convt_1 = R_convt_1.toStar()
        L_relu_convt_1 = ReLULayer()
        R_star_relu_convt_1 = L_relu_convt_1.reach(R_star_convt_1, method='approx')
        R_convt_1 = R_star_relu_convt_1.toImageStar(image_shape=(24, 24, 4))

        # ConvTranspose2 + ReLU
        w_dec_conv2 = self.dec_conv2.weight.detach().cpu().numpy()
        b_dec_conv2 = self.dec_conv2.bias.detach().cpu().numpy()
        w_dec_conv2 = np.transpose(w_dec_conv2, (2, 3, 1, 0))
        L_convt_2 = ConvTranspose2DLayer([w_dec_conv2, b_dec_conv2], [2, 2], [1, 1, 1, 1], [1, 1])
        R_convt_2 = L_convt_2.reach(R_convt_1, method='approx')
        R_star_convt_2 = R_convt_2.toStar()
        L_relu_convt_2 = ReLULayer()
        R_star_relu_convt_2 = L_relu_convt_2.reach(R_star_convt_2, method='approx')
        R_convt_2 = R_star_relu_convt_2.toImageStar(image_shape=(48, 48, 8))

        # ConvTranspose3
        w_dec_conv3 = self.dec_conv3.weight.detach().cpu().numpy()
        b_dec_conv3 = self.dec_conv3.bias.detach().cpu().numpy()
        w_dec_conv3 = np.transpose(w_dec_conv3, (2, 3, 1, 0))
        L_convt_3 = ConvTranspose2DLayer([w_dec_conv3, b_dec_conv3], [2, 2], [1, 1, 1, 1], [1, 1])
        R_convt_3 = L_convt_3.reach(R_convt_2, method='approx')

        # SatLin
        S1 = R_convt_3.toStar()
        L_satlin = SatLinLayer()
        IM_satlin_list = L_satlin.reach(S1, method='approx', lp_solver='gurobi')
        IM_satlin = IM_satlin_list.toImageStar(image_shape=(96, 96, 1))

        return IM_satlin


class Controller(Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(4, 1, kernel_size=4, stride=2, padding=1)
        self.fc1 = nn.Linear(24 * 24, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.tanh(x)
    
    def reach(self, image_star: ImageStar) -> Star:
        # Conv1 + ReLU
        w_conv1 = self.conv1.weight.detach().cpu().numpy()
        b_conv1 = self.conv1.bias.detach().cpu().numpy()
        w_conv1 = np.transpose(w_conv1, (2, 3, 1, 0))
        L_conv1 = Conv2DLayer([w_conv1, b_conv1], [2, 2], [1, 1, 1, 1], [1, 1])
        R_conv1 = L_conv1.reach([image_star])
        IM_conv1 = R_conv1[0]

        IM_conv1_star = IM_conv1.toStar()
        L_relu_conv1 = ReLULayer()
        R_relu_conv1 = L_relu_conv1.reach(IM_conv1_star, method='approx')
        IM_relu_conv1 = R_relu_conv1.toImageStar(image_shape=(48, 48, 4))

        # Conv2 + ReLU
        w_conv2 = self.conv2.weight.detach().cpu().numpy()
        b_conv2 = self.conv2.bias.detach().cpu().numpy()
        w_conv2 = np.transpose(w_conv2, (2, 3, 1, 0))
        L_conv2 = Conv2DLayer([w_conv2, b_conv2], [2, 2], [1, 1, 1, 1], [1, 1])
        R_conv2 = L_conv2.reach([IM_relu_conv1])
        IM_conv2 = R_conv2[0]

        IM_conv2_star = IM_conv2.toStar()
        L_relu_conv2 = ReLULayer()
        R_relu_conv2 = L_relu_conv2.reach(IM_conv2_star, method='approx')

        # FC1 + ReLU
        Wc1 = self.fc1.weight.detach().cpu().numpy()
        bc1 = self.fc1.bias.detach().cpu().numpy()
        L_fc_c1 = FullyConnectedLayer([Wc1, bc1])
        R_fc_c1 = L_fc_c1.reach([R_relu_conv2])
        star_fc_c1 = R_fc_c1[0]

        L_relu_fc1 = ReLULayer()
        R_relu_fc1 = L_relu_fc1.reach(star_fc_c1, method='approx')

        # FC2
        Wc2 = self.fc2.weight.detach().cpu().numpy()
        bc2 = self.fc2.bias.detach().cpu().numpy()
        L_fc_c2 = FullyConnectedLayer([Wc2, bc2])
        R_fc_c2 = L_fc_c2.reach([R_relu_fc1])
        star_fc_c2 = R_fc_c2[0]

        # Tanh activation
        L_tanh = TanSigLayer()
        IM_tanh = L_tanh.reach(star_fc_c2, method='approx', RF=0.0)

        return IM_tanh

class WorldModel(Module):
    def __init__(self):
        super().__init__()
        self.controller = Controller()
        self.decoder = Decoder()

    def forward(self, x):
        out = self.decoder(x)
        out = self.controller(out)
        return out
    
    def reach(self, input_star: Star) -> Star:
        image_star = self.decoder.reach(input_star)
        output_star = self.controller.reach(image_star)

        return output_star
    
class Pendulum(Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass

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

    def forward(self):
        pass

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

