import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from abc import ABC, abstractmethod

from StarV.layer.FullyConnectedLayer import FullyConnectedLayer
from StarV.layer.ConvTranspose2DLayer import ConvTranspose2DLayer
from StarV.layer.ReLULayer import ReLULayer
from StarV.layer.Conv2DLayer import Conv2DLayer
from StarV.layer.SatLinLayer import SatLinLayer
from StarV.layer.TanSigLayer import TanSigLayer
from StarV.layer.LogSigLayer import LogSigLayer
from StarV.set.star import Star
from StarV.set.imagestar import ImageStar

from typing import Dict, List
from collections import OrderedDict

import model

class Decoder(model.Decoder):
    def __init__(self, weights, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.load_state_dict(torch.load(weights, 'cpu', weights_only=True))
    
    def forward(self, state_bound: np.ndarray) -> ImageStar:
        state_star = Star(state_bound[0], state_bound[1])
        return self._star_reach(state_star)
    
    def _star_reach(self, state_star: Star) -> ImageStar:
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

# MLP-based GAN Generator Baseline Added
# class G_MLP(Module):
#     def __init__(self, weights, state_dim=2, latent_dim=2, output_dim=96*96, latent_min=-0.05, latent_max=0.05, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         input_dim = state_dim + latent_dim  # e.g., (θ, ω, z1, z2) → 4 total
        
#         # MLP structure: 4 → 256 → 1024 → 4096 → 9216
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, 256),
#             nn.ReLU(),
#             nn.Linear(256, 1024),
#             nn.ReLU(),
#             nn.Linear(1024, 4096),
#             nn.ReLU(),
#             nn.Linear(4096, output_dim),
#         )

#         self.load_state_dict(torch.load(weights, 'cpu', weights_only=True))
        
#         self.state_dim = state_dim
#         self.latent_dim = latent_dim
#         self.output_dim = output_dim

#         self.latent_min = latent_min
#         self.latent_max = latent_max

#     def forward(self, state, z):
#         x = torch.cat([state, z], dim=1)
#         x = self.net(x)
#         return torch.clamp(x, 0.0, 1.0)
    
#     def reach(self, state_bound: np.ndarray) -> ImageStar:
#         lb = np.concatenate([state_bound[0], np.array([self.latent_min, self.latent_min])])
#         ub = np.concatenate([state_bound[1], np.array([self.latent_max, self.latent_max])])
#         state_star = Star(lb, ub)
#         return self.star_reach(state_star)
    
#     def star_reach(self, input_star: Star) -> ImageStar:
#         # FC1 + ReLU
#         W1 = self.net[0].weight.detach().cpu().numpy()
#         b1 = self.net[0].bias.detach().cpu().numpy()
#         L_fc_1 = FullyConnectedLayer([W1, b1])
#         R_fc_1 = L_fc_1.reach([input_star])
#         L_relu_1 = ReLULayer()
#         R_relu_1 = L_relu_1.reach(R_fc_1[0], method='approx')

#         # FC2 + ReLU
#         W2 = self.net[2].weight.detach().cpu().numpy()
#         b2 = self.net[2].bias.detach().cpu().numpy()
#         L_fc_2 = FullyConnectedLayer([W2, b2])
#         R_fc_2 = L_fc_2.reach([R_relu_1])
#         L_relu_2 = ReLULayer()
#         R_relu_2 = L_relu_2.reach(R_fc_2[0], method='approx')

#         # FC3 + ReLU
#         W3 = self.net[4].weight.detach().cpu().numpy()
#         b3 = self.net[4].bias.detach().cpu().numpy()
#         L_fc_3 = FullyConnectedLayer([W3, b3])
#         R_fc_3 = L_fc_3.reach([R_relu_2])
#         L_relu_3 = ReLULayer()
#         R_relu_3 = L_relu_3.reach(R_fc_3[0], method='approx')

#         # FC4
#         W4 = self.net[6].weight.detach().cpu().numpy()
#         b4 = self.net[6].bias.detach().cpu().numpy()
#         L_fc_4 = FullyConnectedLayer([W4, b4])
#         R_fc_4 = L_fc_4.reach([R_relu_3])
#         star_fc_4 = R_fc_4[0]

#         # SatLin Layer
#         L_satlin = SatLinLayer()
#         IM_satlin = L_satlin.reach(star_fc_4, method='approx', lp_solver='gurobi')
        
#         # Convert to ImageStar
#         image_star = IM_satlin.toImageStar(image_shape=(96, 96, 1))

#         return image_star

class Controller(model.Controller):
    def __init__(self, weights, activation = 'tanh', output_factor = 1, *args, **kwargs):
        super().__init__(activation=activation, *args, **kwargs)

        self.load_state_dict(torch.load(weights, 'cpu', weights_only=True))

        if activation == 'sigmoid':
            self.starv_act = LogSigLayer()
        elif activation == 'tanh':
            self.starv_act = TanSigLayer()
        else:
            raise ValueError(f'Activation function {activation} is not supported')
        
        self.output_factor = output_factor
    
    def forward(self, image_star: ImageStar) -> np.ndarray:
        action_star = self._star_reach(image_star)
        action_bound = np.array(action_star.getRanges('gurobi'))
        return action_bound
    
    def _star_reach(self, image_star: ImageStar) -> Star:
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
        
class FullModel():
    def __init__(self, layers: OrderedDict, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.layers = []
        for k,v in layers.items():
            module = globals()[k]
            self.layers.append(module(*v['args'], **v['kwargs']))

    def reach(self, state_bound: np.ndarray) -> np.ndarray:

        layer_in = state_bound
        for layer in self.layers:
            layer_in = layer(layer_in) 

        action_bound = layer_in
        return action_bound
    
