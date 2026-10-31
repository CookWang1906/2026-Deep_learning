import torch
from torch import nn

# Ánh xạ tên hàm kích hoạt sang lớp tương ứng.
ACTIVATIONS = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
}

class MLP(nn.Module):
    def __init__(
        self,
        in_features = 2,
        hidden_sizes = (8, 8),
        out_features = 1,
        activation = "tanh",
    ):
        # Khởi tạo MLP với cấu hình các tầng đã cho.
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(f"Unsupported activation: {activation}")
        act_cls = ACTIVATIONS[activation]

        layers = []
        prev = in_features
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))  # Tầng tuyến tính.
            layers.append(act_cls())  # Hàm kích hoạt.
            prev = h
        layers.append(nn.Linear(prev, out_features))  # Tầng ra không có kích hoạt.

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
