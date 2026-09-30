from __future__ import annotations

import argparse
import statistics
import time

import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis

from models.bger_net import BGERNet, count_parameters


class BenchmarkWrapper(nn.Module):
    def __init__(self, model: BGERNet):
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor):
        outputs = self.model(image)
        return outputs["mask_logits"], outputs["boundary_logits"]


def measure_fps(
    model: nn.Module,
    image: torch.Tensor,
    warmup: int,
    iterations: int,
    repeats: int,
) -> tuple[float, float, list[float]]:
    for _ in range(warmup):
        model(image)
    torch.cuda.synchronize()

    results = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            model(image)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        results.append(iterations / elapsed)
    return statistics.mean(results), statistics.pstdev(results), results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--backbone-name", type=str, default="convnext_tiny")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the FPS benchmark")
    device = torch.device("cuda")
    model = BGERNet(
        backbone_name=args.backbone_name,
        pretrained=False,
    ).to(device).eval()
    wrapper = BenchmarkWrapper(model).eval()
    image = torch.rand(1, 3, args.height, args.width, device=device)

    parameter_counts = count_parameters(model)
    flop_analysis = FlopCountAnalysis(wrapper, image)
    fvcore_flops = flop_analysis.total()
    unsupported = flop_analysis.unsupported_ops()

    with torch.inference_mode():
        mean_fps, std_fps, repeats = measure_fps(
            wrapper,
            image,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
        )

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Input: 1x3x{args.height}x{args.width}")
    print(f"Total params: {parameter_counts['total'] / 1e6:.6f} M")
    print(f"Trainable params: {parameter_counts['trainable'] / 1e6:.6f} M")
    print(f"fvcore FLOPs: {fvcore_flops / 1e9:.6f} G")
    print(f"Unsupported ops: {dict(unsupported)}")
    print(f"FPS repeats: {[round(value, 3) for value in repeats]}")
    print(f"FPS: {mean_fps:.3f} +/- {std_fps:.3f}")
    print(f"Latency: {1000.0 / mean_fps:.3f} ms/image")


if __name__ == "__main__":
    main()
