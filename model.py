import torch.nn as nn

class ChessModel(nn.Module):
    def __init__(self, input_channels=19):  # 19 channels
        super(ChessModel, self).__init__()
        
        # Convolutional layers for board understanding
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # Global average pooling
        # This is clever for chess! Instead of flattening the entire 8x8x256 feature map, 
        # it averages each of the 256 feature maps down to a single value. This creates a 
        # 256-dimensional summary of the entire board position that's translation-invariant 
        # - the network focuses on "what patterns exist" rather than "exactly where they are."
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Policy head - outputs probability for each of 4288 possible moves
        self.policy_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 1024), # expand to 1024
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 4288) # expand to 4288
        )

        self.value_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Tanh()  # Output between -1 (loss) and 1 (win)
        )
    
    def forward(self, x):
        # Conv1 -> Relu -> Conv2 -> Relu -> Conv3 -> Relu -> average pool
        # -> Flatten -> Linear1 -> Relu -> Dropout -> Linear2 -> Output
        x = self.conv_layers(x)
        x = self.global_avg_pool(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value
    