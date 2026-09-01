"""Learn sin(x) with one weight-tied MLP and fixed soft RMS clipping."""

import argparse
import math

import torch
from torch import nn
from torch.nn import functional as F


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


def device_from(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--loops", type=int, default=8)
    parser.add_argument("--max-b", type=float, default=24 * math.pi)
    parser.add_argument("--steps", type=int, default=13_000)
    parser.add_argument("--points", type=int, default=1025)
    parser.add_argument("--eval-points", type=int, default=4097)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--tau", type=float, default=4.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = device_from(args.device)
    model = LoopedMLP(args.dim, args.tau).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    eval_every = max(args.steps // 10, 1)

    for step in range(1, args.steps + 1):
        progress = (step - 1) / max(args.steps - 1, 1)
        b = ramp(math.pi, args.max_b, progress, fraction=0.6)
        loops = round(ramp(1, args.loops, progress, fraction=0.2))
        x, y = data(b, args.points, device)

        loss = F.mse_loss(model(x, loops), y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 1 or step % eval_every == 0 or step == args.steps:
            with torch.no_grad():
                x_val, y_val = data(args.max_b, args.eval_points, device)
                mse = F.mse_loss(model(x_val, args.loops), y_val)
            print(f"step {step:5d}  b {b / math.pi:5.1f}pi  loops {loops:2d}  mse {mse.item():.6g}")


if __name__ == "__main__":
    main()
