from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

import gym
import numpy as np
from numpy.typing import NDArray

from simulation.env import ContinuousCartPoleEnv

class DynmaicModel(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def step(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("This dynamic model does not support stepping.")

    @abstractmethod
    def render(self, state: NDArray) -> NDArray:
        raise NotImplementedError("This dynamic model does not support rendering.") 

class MountainCar(DynmaicModel):
    def __init__(self, 
            # min_pos=-1.2, 
            # max_pos=0.6, 
            min_speed=-0.07, 
            max_speed=0.07, 
            min_action=-1.0,
            max_action=1.0,
            power=0.0015
        ):
        super().__init__()
        # self.min_pos = min_pos
        # self.max_pos = max_pos
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.power = power
        self.min_action = min_action
        self.max_action = max_action

        self.env = gym.make("MountainCarContinuous-v0", render_mode="rgb_array")
        self.env.reset()

    @torch.no_grad()
    def step(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        pos = states[:, 0]
        vel = states[:, 1]
        force = torch.clamp(actions.squeeze(1), self.min_action, self.max_action)

        vel = vel + force * self.power - 0.0025 * torch.cos(3.0 * pos)
        vel = torch.clamp(vel, self.min_speed, self.max_speed)
        pos = pos + vel
        # pos = torch.clamp(pos, self.min_pos, self.max_pos)
        # vel = torch.where((pos == self.min_pos) & (vel < 0), torch.zeros_like(vel), vel)
        return torch.stack([pos, vel], dim=1)
    
    def render(self, state: NDArray) -> NDArray:
        self.env.unwrapped.state = state
        return self.env.render()
    
    def __del__(self):
        self.env.close()

class Pendulum(DynmaicModel):
    def __init__(self, 
            max_speed=8.0, 
            max_torgue=2.0, 
            dt=0.05, 
            g=10.0, 
            m=1.0, 
            l=1.0
        ):
        from gym import spaces
        from gym.utils import seeding

        self.max_speed = max_speed
        self.max_torque = max_torgue
        self.dt = dt
        self.g = g
        self.m = m
        self.l = l
        self.env = gym.make("Pendulum-v1", render_mode="rgb_array")
        self.env.reset()

    @torch.no_grad()
    def step(self, states, actions):
        theta = states[:, 0]
        omega = states[:, 1]
        torque = torch.clamp(actions.squeeze(1), -1.0, 1.0) * self.max_torque

        new_omega = omega + (
            3.0 * self.g / (2.0 * self.l) * torch.sin(theta)
            + 3.0 / (self.m * self.l**2) * torque
        ) * self.dt
        new_omega = torch.clamp(new_omega, -self.max_speed, self.max_speed)
        new_theta = theta + new_omega * self.dt
        new_theta = torch.remainder(new_theta + torch.pi, 2.0 * torch.pi) - torch.pi
        return torch.stack([new_theta, new_omega], dim=1)

    def render(self, state: NDArray):
        self.env.unwrapped.state = state
        return self.env.render()

    def __del__(self):
        self.env.close()

class CartPole(DynmaicModel):
    def __init__(
        self,
        gravity: float = 9.8,
        mass_cart: float = 1.0,
        mass_pole: float = 0.1,
        length: float = 0.5,
        force_mag: float = 10.0,
        dt: float = 0.02,
    ):
        super().__init__()

        self.env = ContinuousCartPoleEnv(render_mode="rgb_array")
        self.env.reset()

        self.gravity = gravity

        self.mass_cart = mass_cart
        self.mass_pole = mass_pole

        self.length = length
        self.force_mag = force_mag
        self.dt = dt

        # derived parameters
        self.total_mass = self.mass_cart + self.mass_pole
        self.polemass_length = self.mass_pole * self.length

    def step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        x = states[:, 0]
        x_dot = states[:, 1]
        theta = states[:, 2]
        theta_dot = states[:, 3]

        force = actions.squeeze(1) * self.force_mag

        costheta = torch.cos(theta)
        sintheta = torch.sin(theta)

        temp = (
            force
            + self.polemass_length * theta_dot.square() * sintheta
        ) / self.total_mass

        theta_acc = (
            self.gravity * sintheta
            - costheta * temp
        ) / (
            self.length
            * (
                4.0 / 3.0
                - self.mass_pole * costheta.square() / self.total_mass
            )
        )

        x_acc = (
            temp
            - self.polemass_length
            * theta_acc
            * costheta
            / self.total_mass
        )

        next_x = x + self.dt * x_dot
        next_x_dot = x_dot + self.dt * x_acc

        next_theta = theta + self.dt * theta_dot
        next_theta_dot = theta_dot + self.dt * theta_acc

        return torch.stack(
            [next_x, next_x_dot, next_theta, next_theta_dot],
            dim=1,
        )

    def render(self, state: NDArray) -> NDArray:
        self.env.unwrapped.state = state
        return self.env.render()

    
__all__ = [
    "MountainCar",
    "Pendulum",
    "CartPole"
]