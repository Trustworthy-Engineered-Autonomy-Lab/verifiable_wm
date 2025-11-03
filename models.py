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
    
    def reach(self, state_star: Star) -> ImageStar:
        # FC1 + ReLU
        W1 = self.fc1.weight.detach().cpu().numpy()
        b1 = self.fc1.bias.detach().cpu().numpy()
        L_fc_1 = FullyConnectedLayer([W1, b1])
        R_fc_1 = L_fc_1.reach([state_star])
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
        image_star = IM_satlin_list.toImageStar(image_shape=(96, 96, 1))

        return image_star


class Controller(Module):
    def __init__(self, activation = 'tanh', output_factor = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(4, 1, kernel_size=4, stride=2, padding=1)
        self.fc1 = nn.Linear(24 * 24, 64)
        self.fc2 = nn.Linear(64, 1)

        if activation == 'sigmoid':
            self.act = torch.sigmoid
            self.starv_act = SatLinLayer()
        elif activation == 'tanh':
            self.act = torch.tanh
            self.starv_act = TanSigLayer()
        else:
            raise ValueError(f'Activation function {activation} is not supported')
        
        self.output_factor = output_factor

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return self.act(x)
    
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
        IM_act = self.starv_act.reach(star_fc_c2, method='approx', RF=0.0)

        if self.output_factor != 1:
            d = IM_act.dim
            A = self.output_factor * np.eye(d, dtype=np.float32)
            b = np.zeros(d, dtype=np.float32)
            action_star = IM_act.affineMap(A, b)
        else:
            action_star = IM_act

        return action_star

class NNModel(Module):
    def __init__(self, activation = 'tanh', output_factor = 1.0):
        super().__init__()
        self.controller = Controller(activation, output_factor)
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

    def reach(self, bound: np.ndarray):
        # Compute sin(theta) using SinLayer
        theta_min, theta_max = bound[:,0]

        L_sin = SinLayer()
        lb_theta = np.array([theta_min], dtype=np.float32)
        ub_theta = np.array([theta_max], dtype=np.float32)
        S_theta = Star(lb_theta, ub_theta)

        IM_sin = L_sin.reach(S_theta, method='approx', lp_solver='gurobi', RF=0.0)
        z_bound = np.array(IM_sin.getRanges(lp_solver='gurobi'))

        full_bound = np.concatenate([bound, z_bound], axis = 1)
        S_full = Star(full_bound[0], full_bound[1])

        # Apply dynamics
        M = np.array([[1.0, 0.05, 0.0075, 0.0375],
                      [0.0, 1.0, 0.15, 0.75]], dtype=np.float32)
        b_dyn = np.zeros(2, dtype=np.float32)
        S_next = S_full.affineMap(M, b_dyn)

        # Step 7: Get bounds BEFORE clipping
        next_bound = np.array(S_next.getRanges('gurobi', RF=0.0))

        # Clip omega' to [-8, 8]
        next_bound[:,1] = np.clip(next_bound[:,1], -8.0, 8.0)

        # Normalize theta to [-π, π]
        next_bound[:,0] = self.angle_normalize(next_bound[:,0])

        return next_bound
    
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

    def reach(self, bound: np.ndarray):
        # Cosine term bounds: cos(3*pos)
        pos_bound = bound[:,0]
        vel_bound = bound[:,1]
        action_bound = bound[:,2]

        cos_3p = np.sort(np.cos(3.0 * pos_bound))[::-1]

        # Velocity update: v' = v + action*POWER - 0.0025*cos(3*pos)
        vel_bound = vel_bound + action_bound * self.POWER - 0.0025 * cos_3p

        # Velocity clipping: both bounds must respect speed limits
        vel_bound = np.clip(vel_bound, self.MIN_SPEED, self.MAX_SPEED)

        # Ensure velocity bounds are valid (min <= max)
        vel_bound = np.sort(vel_bound)

        # Position update: pos' = pos + v'
        pos_bound = pos_bound + vel_bound

        # Ensure position bounds are valid (min <= max)
        # if pos_next_min > pos_next_max:
        #     pos_next_min, pos_next_max = pos_next_max, pos_next_min

        return np.array([pos_bound, vel_bound]).T

