"""Measure how many PointNet max-pool channels are controlled by trigger points."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from run_suite import ROOT, dataset, load, replace


def prepool(encoder, points):
    x = points.transpose(1, 2)
    x = F.relu(encoder.bn1(encoder.conv1(x)))
    x = F.relu(encoder.bn2(encoder.conv2(x)))
    x = F.relu(encoder.bn3(encoder.conv3(x)))
    return encoder.bn4(encoder.conv4(x))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trigger_dirs", nargs="+", required=True)
    p.add_argument("--samples", type=int, default=256)
    p.add_argument("--output", default=str(ROOT / "pointnet_takeover.json"))
    args = p.parse_args()
    model, _ = load(); ds = dataset("val")
    result = {}
    with torch.no_grad():
        for name in args.trigger_dirs:
            path = ROOT / name / "optimized_trigger.pt"
            payload = torch.load(path, map_location="cuda", weights_only=False)
            trigger = payload["trigger"]
            takeover_rows, clean_latents, triggered_latents = [], [], []
            for start in range(0, args.samples, 32):
                x = torch.stack([ds[i]["pointcloud"] for i in range(start, min(start + 32, args.samples))]).cuda()
                xt = replace(x, trigger)
                feature = prepool(model.encoder, xt)
                winners = feature.argmax(dim=2)
                takeover_rows.append(winners >= x.shape[1] - len(trigger))
                clean_latents.append(model.encoder(x)[0]); triggered_latents.append(model.encoder(xt)[0])
            takeover = torch.cat(takeover_rows)
            z = torch.cat(clean_latents); zt = torch.cat(triggered_latents)
            per_channel = takeover.float().mean(0)
            per_sample = takeover.float().mean(1)
            result[name] = {
                "trigger_points": len(trigger),
                "support_ratio": len(trigger) / 2048,
                "mean_channel_takeover": float(per_sample.mean()),
                "sample_takeover_std": float(per_sample.std(unbiased=False)),
                "always_controlled_channels": int((per_channel == 1).sum()),
                "controlled_over_90pct_channels": int((per_channel >= 0.9).sum()),
                "ever_controlled_channels": int((per_channel > 0).sum()),
                "triggered_dispersion": float((zt - zt.mean(0)).square().sum(1).mean()),
                "clean_dispersion": float((z - z.mean(0)).square().sum(1).mean()),
                "triggered_variance_ratio": float(
                    (zt - zt.mean(0)).square().sum(1).mean()
                    / ((z - z.mean(0)).square().sum(1).mean() + 1e-8)
                ),
                "mean_delta_norm": float((zt - z).norm(dim=1).mean()),
            }
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()
