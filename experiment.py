"""Learn sin(x) with one weight-tied MLP and fixed soft RMS clipping."""

import math

import torch
from torch import nn
from torch.nn import functional as F


DIM = 64
LOOPS = 8
MAX_B = 24 * math.pi
STEPS = 13_000
TRAIN_POINTS = 1025
EVAL_POINTS = 4097
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
TAU = 4.0
SEED = 0


class LoopedMLP(nn.Module):
    def __init__(self, dim: int, tau: float = 4.0):
        super().__init__()
        self.dim = dim
        self.tau = tau
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(approximate="none"),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, x: torch.Tensor, loops: int) -> torch.Tensor:
        h = F.pad(x[:, None], (0, self.dim - 1))
        for i in range(loops):
            if i:
                rms2 = h.square().mean(dim=-1, keepdim=True)
                h = h * torch.rsqrt(1 + rms2 / self.tau**2)
            h = self.mlp(h)
        return h[:, 0]


def ramp(start: float, end: float, progress: float, fraction: float) -> float:
    mix = min(progress / fraction, 1.0)
    mix = mix * mix * (3 - 2 * mix)
    return start + mix * (end - start)


def data(b: float, points: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.linspace(-b, b, points, device=device)
    return x, x.sin()


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    torch.manual_seed(SEED)
    device = default_device()
    model = LoopedMLP(DIM, TAU).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    for step in range(1, STEPS + 1):
        progress = (step - 1) / (STEPS - 1)
        b = ramp(math.pi, MAX_B, progress, fraction=0.6)
        loops = round(ramp(1, LOOPS, progress, fraction=0.2))
        x, y = data(b, TRAIN_POINTS, device)

        loss = F.mse_loss(model(x, loops), y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 1 or step % (STEPS // 10) == 0 or step == STEPS:
            with torch.no_grad():
                x_val, y_val = data(MAX_B, EVAL_POINTS, device)
                mse = F.mse_loss(model(x_val, LOOPS), y_val)
            print(f"step {step:5d}  b {b / math.pi:5.1f}pi  loops {loops:2d}  mse {mse.item():.6g}")


if __name__ == "__main__":
    main()
