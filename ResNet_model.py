import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)

class ResNetChessModel(nn.Module):
    def __init__(self, input_channels=19, num_blocks=19):
        super().__init__()
        self.conv_in = nn.Conv2d(input_channels, 256, kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm2d(256)

        self.res_blocks = nn.Sequential(*[ResidualBlock(256) for _ in range(num_blocks)])

        self.policy_head = nn.Sequential(
            nn.Conv2d(256, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * 8 * 8, 4288)
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = F.relu(self.bn_in(self.conv_in(x)))
        x = self.res_blocks(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value
