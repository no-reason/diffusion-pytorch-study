"""Optimize a local universal point trigger to collapse triggered PDM latents.

The frozen encoder receives shape-unit-normalized chair clouds whose last K
points are replaced by a learned universal set.  Unlike the historical
optimizer, the primary dispersion term is Var[E(T_delta(x))], not the
variance of E(T_delta(x)) - E(x).
"""

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.vae_gaussian import GaussianVAE
from tools.pointcloud_normalization import normalize_pointcloud, validate_checkpoint_scale_mode
from utils.dataset import ShapeNetCore


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="logs_gen/GEN_2026_06_29__02_58_19_Clean_VAE_From_Scratch_KL001/ckpt_0.730159_300000.pt")
    p.add_argument("--dataset_path", default="data/shapenet_v2pc15k.h5")
    p.add_argument("--target_file", default="targets/stageC8E_fixed_airplane_target.npy")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--category", default="chair")
    p.add_argument("--num_trigger", type=int, default=128)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--val_samples", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--center_limit", type=float, default=0.85)
    p.add_argument("--initial_center", type=float, nargs=3, default=[0.60, 0.60, 0.60])
    p.add_argument("--parameterization", choices=["cluster", "free"], default="cluster")
    p.add_argument("--cluster_radius", type=float, default=0.15)
    p.add_argument("--coord_limit", type=float, default=1.0)
    p.add_argument("--min_point_distance", type=float, default=0.015)
    p.add_argument("--w_anchor", type=float, default=1.0)
    p.add_argument("--w_trigger_variance", type=float, default=1.0)
    p.add_argument("--w_repulsion", type=float, default=0.1)
    p.add_argument("--w_stealth", type=float, default=0.0)
    p.add_argument("--stealth_radius", type=float, default=0.25)
    p.add_argument("--stealth_batch", type=int, default=16)
    p.add_argument("--stealth_source_points", type=int, default=512)
    p.add_argument("--init_trigger", default=None)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LocalCluster(torch.nn.Module):
    def __init__(self, args, device, initial_points=None):
        super().__init__()
        self.args = args
        if args.parameterization == "cluster":
            center = torch.tensor(args.initial_center, dtype=torch.float32, device=device)
            if bool((center.abs() >= args.center_limit).any()):
                raise ValueError("each initial_center coordinate must have abs(value) < center_limit")
            self.center_raw = torch.nn.Parameter(torch.atanh((center / args.center_limit).clamp(-0.999, 0.999)))
            self.offset_raw = torch.nn.Parameter(torch.randn(args.num_trigger, 3, device=device) * 0.25)
        else:
            if initial_points is None:
                initial = torch.empty(args.num_trigger, 3, device=device).uniform_(-0.70, 0.70)
            else:
                initial = initial_points.to(device)
                if initial.shape != (args.num_trigger, 3):
                    raise ValueError(f"init_trigger has shape {tuple(initial.shape)}, expected {(args.num_trigger, 3)}")
                initial = initial / args.coord_limit
            self.points_raw = torch.nn.Parameter(torch.atanh(initial.clamp(-0.999, 0.999)))

    def forward(self):
        a = self.args
        if a.parameterization == "free":
            return a.coord_limit * torch.tanh(self.points_raw)
        center = a.center_limit * torch.tanh(self.center_raw)
        offsets = a.cluster_radius * torch.tanh(self.offset_raw)
        return (center[None] + offsets).clamp(-a.coord_limit, a.coord_limit)


def replace_last(x, trigger):
    if len(trigger) >= x.shape[1]:
        raise ValueError("trigger must remain a strict subset of the cloud")
    return torch.cat([x[:, :-len(trigger)], trigger[None].expand(len(x), -1, -1)], dim=1)


def dispersion(z):
    center = z.mean(0, keepdim=True)
    return (z - center).square().sum(1).mean()


def repulsion(points, minimum):   
    if minimum <= 0:
        return points.new_zeros(())
    d = torch.cdist(points[None], points[None])[0]
    mask = ~torch.eye(len(points), dtype=torch.bool, device=points.device)
    return F.relu(minimum - d[mask]).square().mean()


def surface_stealth(points, x, radius, batch_size, source_points):
    """Penalize trigger points that sit visibly far from each source surface."""
    x = x[: min(batch_size, len(x))]
    if source_points < x.shape[1]:
        indices = torch.randperm(x.shape[1], device=x.device)[:source_points]
        x = x[:, indices]
    distance = torch.cdist(points[None].expand(len(x), -1, -1), x).min(2).values
    penalty = F.relu(distance - radius).square().mean()
    return penalty, distance.mean(), (distance > radius).float().mean()


@torch.no_grad()
def evaluate_stealth(trigger, data_loader, max_samples, radius, device):
    distances = []
    seen = 0
    for batch in data_loader:
        x = batch["pointcloud"].to(device)
        x = x[: max_samples - seen]
        d = torch.cdist(trigger[None].expand(len(x), -1, -1), x).min(2).values
        distances.append(d.cpu())
        seen += len(x)
        if seen >= max_samples:
            break
    distance = torch.cat(distances).flatten()
    return {
        "surface_nn_mean": float(distance.mean()),
        "surface_nn_median": float(distance.median()),
        "surface_nn_p90": float(distance.quantile(0.90)),
        "surface_nn_p95": float(distance.quantile(0.95)),
        "surface_nn_p99": float(distance.quantile(0.99)),
        "surface_fraction_over_radius": float((distance > radius).float().mean()),
        "surface_fraction_over_0.25": float((distance > 0.25).float().mean()),
        "surface_fraction_over_0.50": float((distance > 0.50).float().mean()),
        "surface_nn_max": float(distance.max()),
    }

def loader(args, split, shuffle):
    ds = ShapeNetCore(args.dataset_path, [args.category], split, "shape_unit")
    generator = torch.Generator().manual_seed(args.seed) if shuffle else None
    return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=0,
                      drop_last=shuffle, generator=generator)


def load_model(args, device):
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    validate_checkpoint_scale_mode(ckpt["args"], "shape_unit")
    model = GaussianVAE(ckpt["args"]).to(device).eval()
    model.load_state_dict(ckpt["state_dict"])
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def evaluate(model, trigger, data_loader, z_target, max_samples, device):
    clean, triggered = [], []
    seen = 0
    for batch in data_loader:
        x = batch["pointcloud"].to(device)
        x = x[: max_samples - seen]
        z, _ = model.encoder(x)
        zt, _ = model.encoder(replace_last(x, trigger))
        clean.append(z); triggered.append(zt); seen += len(x)
        if seen >= max_samples:
            break
    z = torch.cat(clean); zt = torch.cat(triggered)
    clean_var = dispersion(z); trigger_var = dispersion(zt)
    delta = zt - z
    delta_var = dispersion(delta)
    anchor_sq_l2 = (zt - z_target).square().sum(1)
    return {
        "samples": len(z),
        "clean_dispersion": float(clean_var),
        "triggered_dispersion": float(trigger_var),
        "triggered_variance_ratio": float(trigger_var / (clean_var + 1e-8)),
        "delta_dispersion": float(delta_var),
        "anchor_rmse_per_dim": float((zt - z_target).square().mean().sqrt()),
        "anchor_l2_mean": float(anchor_sq_l2.sqrt().mean()),
        "anchor_l2_std": float(anchor_sq_l2.sqrt().std(unbiased=False)),
        "triggered_norm_mean": float(zt.norm(dim=1).mean()),
        "clean_norm_mean": float(z.norm(dim=1).mean()),
        "mean_trigger_to_target_l2": float((zt.mean(0, keepdim=True) - z_target).norm()),
        "mean_shift_l2": float(delta.mean(0).norm()),
        "triggered_latents": zt.cpu(),
        "clean_latents": z.cpu(),
    }


def main():
    args = parse_args(); seed_all(args.seed)
    device = torch.device(args.device)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=False)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    model = load_model(args, device)
    target = torch.from_numpy(np.load(args.target_file)).float()
    if target.ndim == 2:
        target = target[None]
    target = normalize_pointcloud(target, mode="shape_unit").to(device)
    with torch.no_grad():
        z_target, _ = model.encoder(target)

    train_loader = loader(args, "train", True)
    val_loader = loader(args, "val", False)
    initial_points = None
    if args.init_trigger:
        payload = torch.load(args.init_trigger, map_location="cpu", weights_only=False)
        initial_points = payload["trigger"] if isinstance(payload, dict) else payload
    trigger_module = LocalCluster(args, device, initial_points=initial_points)
    optimizer = torch.optim.Adam(trigger_module.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, eta_min=args.lr * 0.05)
    iterator = iter(train_loader)
    fields = ["step", "loss", "anchor_ratio", "trigger_variance_ratio", "clean_dispersion",
              "triggered_dispersion", "anchor_l2", "mean_to_target_l2", "repulsion",
              "stealth_loss", "stealth_nn_mean", "stealth_outlier_fraction", "lr"]
    with (out / "optimization.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for step in range(1, args.steps + 1):
            try:
                x = next(iterator)["pointcloud"].to(device)
            except StopIteration:
                iterator = iter(train_loader); x = next(iterator)["pointcloud"].to(device)
            with torch.no_grad():
                z_clean, _ = model.encoder(x)
            trigger = trigger_module()
            z_triggered, _ = model.encoder(replace_last(x, trigger))
            clean_var = dispersion(z_clean).detach()
            trigger_var = dispersion(z_triggered)
            variance_ratio = trigger_var / (clean_var + 1e-8)
            clean_anchor = (z_clean - z_target).square().sum(1).mean().detach()
            trigger_anchor = (z_triggered - z_target).square().sum(1).mean()
            anchor_ratio = trigger_anchor / (clean_anchor + 1e-8)
            repel = repulsion(trigger, args.min_point_distance)
            stealth, stealth_mean, stealth_outliers = surface_stealth(
                trigger, x, args.stealth_radius, args.stealth_batch, args.stealth_source_points
            )
            loss = (args.w_anchor * anchor_ratio + args.w_trigger_variance * variance_ratio
                    + args.w_repulsion * repel + args.w_stealth * stealth)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); scheduler.step()
            row = {
                "step": step, "loss": float(loss.detach()), "anchor_ratio": float(anchor_ratio.detach()),
                "trigger_variance_ratio": float(variance_ratio.detach()), "clean_dispersion": float(clean_var),
                "triggered_dispersion": float(trigger_var.detach()),
                "anchor_l2": float(trigger_anchor.detach().sqrt()),
                "mean_to_target_l2": float((z_triggered.mean(0, keepdim=True) - z_target).norm().detach()),
                "repulsion": float(repel.detach()), "stealth_loss": float(stealth.detach()),
                "stealth_nn_mean": float(stealth_mean.detach()),
                "stealth_outlier_fraction": float(stealth_outliers.detach()),
                "lr": optimizer.param_groups[0]["lr"],
            }
            writer.writerow(row); handle.flush()
            if step == 1 or step % args.log_every == 0 or step == args.steps:
                print(json.dumps(row), flush=True)

    trigger = trigger_module().detach()
    metrics = evaluate(model, trigger, val_loader, z_target, args.val_samples, device)
    metrics.update(evaluate_stealth(trigger, val_loader, args.val_samples, args.stealth_radius, device))
    latents = {k: metrics.pop(k) for k in ["triggered_latents", "clean_latents"]}
    metrics.update({
        "trigger_points": len(trigger), "support_ratio": len(trigger) / 2048,
        "trigger_min": trigger.min(0).values.cpu().tolist(),
        "trigger_max": trigger.max(0).values.cpu().tolist(),
        "trigger_center": trigger.mean(0).cpu().tolist(),
        "trigger_rms_radius": float((trigger - trigger.mean(0)).square().sum(1).mean().sqrt().cpu()),
    })
    (out / "validation_metrics.json").write_text(json.dumps(metrics, indent=2))
    torch.save({"trigger": trigger.cpu(), "target_latent": z_target.cpu(), "config": vars(args),
                "validation": metrics}, out / "optimized_trigger.pt")
    np.savez_compressed(out / "validation_latents.npz", **{k: v.numpy() for k, v in latents.items()},
                        target_latent=z_target.cpu().numpy())
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
