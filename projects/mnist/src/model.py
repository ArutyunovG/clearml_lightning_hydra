import torch
import torch.nn as nn

class Model(nn.Module):
    
    """
    MLP: 28x28 -> 784 -> [HIDDEN] -> 10 (logits)
    Loss: CrossEntropy
    Optimizer: Adam(lr=1e-3)
    """

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(28 * 28, 128)
        self.layer2 = nn.Linear(128, 10)

    def forward(self, x):
        assert len(x.shape) == 2 and x.shape[1] == 28 * 28, f"Expected input shape (B, 784), got {x.shape}"
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x
