import torch
import torch.nn as nn
import torch.nn.functional as F

class Controller(nn.Module):
    def __init__(self, activation='relu'):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(4, 1, kernel_size=4, stride=2, padding=1)
        self.fc1 = nn.Linear(24 * 24, 64)
        self.fc2 = nn.Linear(64, 1)
        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'tanh':
            self.act = nn.Tanh()
        elif activation == 'sigmoid':
            self.act = nn.Sigmoid()
        else:
            raise ValueError("Unsupported activation function: {}".format(activation))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return self.act(x)

class Decoder(nn.Module):
    def __init__(self, output_activation='sigmoid'):
        super().__init__()
        # 'sigmoid' for the gym environments; 'clamp' (== SatLin) for the
        # brake system, whose decoder was trained with clamp(x, 0, 1).
        if output_activation not in ('sigmoid', 'clamp'):
            raise ValueError(
                "Unsupported output_activation: {}".format(output_activation)
            )
        self.output_activation = output_activation
        self.fc1 = nn.Linear(2, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 3 * 12 * 12)
        self.dec_conv1 = nn.ConvTranspose2d(3, 4, kernel_size=4, stride=2, padding=1)
        self.dec_conv2 = nn.ConvTranspose2d(4, 8, kernel_size=4, stride=2, padding=1)
        self.dec_conv3 = nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, states):
        batch_size = states.size(0)
        x = F.relu(self.fc1(states))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = x.view(batch_size, 3, 12, 12)
        x = F.relu(self.dec_conv1(x))
        x = F.relu(self.dec_conv2(x))
        x = self.dec_conv3(x)
        if self.output_activation == 'clamp':
            return torch.clamp(x, 0.0, 1.0)
        return torch.sigmoid(x)


class G_MLP(nn.Module):
    """MLP cGAN generator baseline (state + latent -> flat 96x96 image).

    Plain MLP on purpose: unlike a conv/BatchNorm generator, this shape is
    directly reachable by StarV's FullyConnectedLayer/ReLULayer/SatLinLayer
    stack. Matches aebs_carla/model_2.py's Generator, which is what
    gan_brake_ckpts/G_brake.pth was trained against.
    """

    def __init__(self, state_dim=2, latent_dim=2, output_dim=96 * 96, z_range=0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, 4096),
            nn.ReLU(),
            nn.Linear(4096, output_dim),
        )
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.z_range = z_range

    def forward(self, state, z=None):
        if z is None:
            # Fresh latent each call, matching the training-time distribution
            # (see aebs_carla/cp_0.95_gan.py: sample_latent, z_range=0.05).
            z = torch.empty(
                state.size(0), self.latent_dim, device=state.device
            ).uniform_(-self.z_range, self.z_range)
        x = torch.cat([state, z], dim=1)
        x = self.net(x)
        x = torch.clamp(x, 0.0, 1.0)
        return x.view(x.size(0), 1, 96, 96)


__all__ = [
    "Controller",
    "Decoder",
    "G_MLP",
]
