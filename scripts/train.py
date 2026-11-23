#!/usr/bin/env python3
"""Train Vetra at one site with the recipe of the paper.

Recipe: MSE on log(1+E), Lion optimiser, chronological 80/20 train/validation
split of the training years, checkpoint chosen by the physical correlation on
the validation data, and the RR-4 learning-rate schedule (restore and reduce
after four epochs): when the validation physical correlation has not improved
for ``--plateau-patience`` epochs, the best weights are restored and the
learning rate is multiplied by ``--plateau-factor``; training stops at the
next stall after ``--max-lr-drops`` reductions, or after ``--epochs``.

At the end, Hs-Gain is fitted on the validation predictions of the best
weights and stored in the checkpoint ``<out-dir>/vetra.pt``.

Every epoch writes a resume state, so an interrupted run continues with
``--resume`` and gives the same result as an uninterrupted one.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))

from vetra.checkpoint import save_checkpoint
from vetra.data import (WindSpectrumDataset, SpectrumTransform, load_year_arrays, make_sample_indices,
                        split_indices, wind_stats)
from vetra.hs_gain import fit_hs_gain
from vetra.lion import Lion
from vetra.models import build_model

MONITOR = "val_corr_spectrum"   # physical correlation on the validation data


def train_one_epoch(model, loader, optimizer, criterion, device, epoch, log_batches, run_start):
    model.train()
    total = 0.0
    seen = 0
    for batch_idx, (large, local, target) in enumerate(loader, start=1):
        large, local, target = large.to(device), local.to(device), target.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(large, local), target)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(target)
        seen += len(target)
        if log_batches and (batch_idx % log_batches == 0 or batch_idx == len(loader)):
            print(f"epoch {epoch:03d} batch {batch_idx:04d}/{len(loader):04d} "
                  f"train_loss_so_far={total / seen:.6f} elapsed={(time.time() - run_start) / 60:.1f} min",
                  flush=True)
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device, target_transform):
    """Validation loss, the four metrics of the paper (physical units, negatives set to zero) and the spectra."""
    model.eval()
    total = 0.0
    preds, truths, spectrum_preds, spectrum_truths = [], [], [], []
    for large, local, target in loader:
        large, local, target = large.to(device), local.to(device), target.to(device)
        prediction = model(large, local)
        total += criterion(prediction, target).item() * len(target)
        prediction_np = prediction.float().cpu().numpy()
        target_np = target.float().cpu().numpy()
        preds.append(prediction_np)
        truths.append(target_np)
        spectrum_preds.append(np.maximum(target_transform.inverse(prediction_np), 0.0))
        spectrum_truths.append(np.maximum(target_transform.inverse(target_np), 0.0))

    y_pred, y_true = np.concatenate(preds), np.concatenate(truths)
    corr = np.corrcoef(y_pred.reshape(-1), y_true.reshape(-1))[0, 1]
    e_pred, e_true = np.concatenate(spectrum_preds), np.concatenate(spectrum_truths)
    physical_corr = np.corrcoef(e_pred.reshape(-1), e_true.reshape(-1))[0, 1]
    pred_flat = e_pred.reshape(len(e_pred), -1)
    true_flat = e_true.reshape(len(e_true), -1)
    pc = pred_flat - pred_flat.mean(axis=1, keepdims=True)
    tc = true_flat - true_flat.mean(axis=1, keepdims=True)
    denominator = np.sqrt(np.sum(pc**2, axis=1) * np.sum(tc**2, axis=1))
    valid = denominator > 0
    per_spectrum = np.sum(pc[valid] * tc[valid], axis=1) / denominator[valid]
    error = e_pred - e_true
    return {
        "val_loss": total / len(loader.dataset),
        "val_corr": float(corr),
        "val_corr_spectrum": float(physical_corr),
        "val_corr_spectrum_sample_mean": float(per_spectrum.astype(np.float64).mean()) if valid.any() else float("nan"),
        "val_mae_spectrum": float(np.mean(np.abs(error))),
        "val_rmse_spectrum": float(np.sqrt(np.mean(error**2))),
    }, e_pred, e_true


def improved(value: float, best: float) -> bool:
    if not np.isfinite(value):
        return False
    return (not np.isfinite(best)) or value > best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--site", required=True, help="Site folder under --data-root (see vetra.data).")
    parser.add_argument("--year", type=int, nargs="+", required=True, help="Training years, e.g. 1993 ... 2021.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--plateau-patience", type=int, default=4)
    parser.add_argument("--plateau-factor", type=float, default=0.2)
    parser.add_argument("--max-lr-drops", type=int, default=4)
    parser.add_argument("--min-learning-rate", type=float, default=1e-7)
    parser.add_argument("--unet-out", type=int, default=32, help="Encoder channels C.")
    parser.add_argument("--feature-hw", type=int, default=8)
    parser.add_argument("--decoder-width", type=int, default=64)
    parser.add_argument("--decoder-dropout", type=float, default=0.05)
    parser.add_argument("--large-window", type=int, default=80, help="Basin wind history, 3-hourly steps.")
    parser.add_argument("--local-window", type=int, default=24, help="Local wind history, 3-hourly steps.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-batches", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="Continue from <out-dir>/resume_state.pt.")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    large, local, target = load_year_arrays(args.site, args.year, args.data_root)
    indices = make_sample_indices(len(target), large_window=args.large_window)
    train_indices, val_indices = split_indices(indices)
    train_end = int(train_indices[-1]) + args.large_window
    large_norm = wind_stats(large, train_end)
    local_norm = wind_stats(local, train_end)
    target_transform = SpectrumTransform("log1p")

    def dataset(idx):
        return WindSpectrumDataset(large, local, target, idx, large_norm, local_norm, target_transform,
                                   large_window=args.large_window, local_window=args.local_window)

    train_dataset, val_dataset = dataset(train_indices), dataset(val_indices)
    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)

    def seed_worker(worker_id: int) -> None:
        worker_seed = (args.seed + worker_id) & 0xFFFFFFFF
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              generator=loader_generator, worker_init_fn=seed_worker)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            worker_init_fn=seed_worker)

    model = build_model(vars(args), freq_num=target.shape[1], direction_num=target.shape[2]).to(device)
    optimizer = Lion(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    criterion = nn.MSELoss()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "vetra.pt"
    metrics_path = out_dir / "metrics.csv"
    resume_path = out_dir / "resume_state.pt"
    print(f"device {device}; samples train/val {len(train_dataset)}/{len(val_dataset)}; "
          f"parameters {sum(p.numel() for p in model.parameters())}", flush=True)

    state = {"best_state": None, "best_score": -float("inf"), "best_epoch": None, "best_metrics": {},
             "plateau_best": -float("inf"),
             "plateau_wait": 0, "lr_drop_count": 0, "prev_lr_drop_count": 0, "rows": [], "stop_reason": "completed",
             "finished": False}
    start_epoch, elapsed_before = 1, 0.0
    if args.resume and resume_path.exists():
        saved = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        state = saved["state"]
        start_epoch, elapsed_before = saved["epoch"] + 1, saved["elapsed_seconds"]
        torch.set_rng_state(saved["torch_rng_state"])
        if torch.cuda.is_available() and saved["cuda_rng_state"] is not None:
            torch.cuda.set_rng_state_all(saved["cuda_rng_state"])
        np.random.set_state(saved["numpy_rng_state"])
        random.setstate(saved["python_rng_state"])
        loader_generator.set_state(saved["loader_generator_state"])
        print(f"resumed after epoch {saved['epoch']}", flush=True)

    run_start = time.time() - elapsed_before

    def save_resume_state(epoch: int) -> None:
        temporary = resume_path.with_suffix(".pt.part")
        torch.save({"epoch": epoch, "elapsed_seconds": time.time() - run_start, "state": state,
                    "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    "numpy_rng_state": np.random.get_state(), "python_rng_state": random.getstate(),
                    "loader_generator_state": loader_generator.get_state()}, temporary)
        os.replace(temporary, resume_path)

    fields = ["epoch", "train_loss", "val_loss", "val_corr", "val_corr_spectrum", "val_corr_spectrum_sample_mean",
              "val_mae_spectrum", "val_rmse_spectrum", "best_checkpoint_score", "is_best", "lr", "plateau_wait",
              "lr_drop_count", "is_lr_drop", "elapsed_seconds"]
    with metrics_path.open("a" if start_epoch > 1 else "w", newline="") as metrics_file:
        writer = csv.DictWriter(metrics_file, fieldnames=fields)
        if start_epoch == 1:
            writer.writeheader()
        for epoch in range(start_epoch, args.epochs + 1):
            if state["finished"]:
                break
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch,
                                         args.log_batches, run_start)
            metrics, _, _ = evaluate(model, val_loader, criterion, device, target_transform)
            score = metrics[MONITOR]
            if improved(score, state["plateau_best"]):
                state["plateau_best"], state["plateau_wait"] = score, 0
            else:
                state["plateau_wait"] += 1
            is_best = improved(score, state["best_score"])
            if is_best:
                state["best_score"], state["best_epoch"], state["best_metrics"] = score, epoch, metrics
                state["best_state"] = copy.deepcopy(model.state_dict())
                save_checkpoint(checkpoint_path, model, vars(args), large_norm, local_norm, target_transform,
                                site={"name": args.site},
                                extra={"epoch": epoch, "training_years": sorted(args.year), **metrics})
            row = {"epoch": epoch, "train_loss": train_loss, **metrics, "best_checkpoint_score": state["best_score"],
                   "is_best": int(is_best), "lr": optimizer.param_groups[0]["lr"],
                   "plateau_wait": state["plateau_wait"], "lr_drop_count": state["lr_drop_count"],
                   "is_lr_drop": int(state["lr_drop_count"] > state["prev_lr_drop_count"]),
                   "elapsed_seconds": time.time() - run_start}
            state["prev_lr_drop_count"] = state["lr_drop_count"]
            writer.writerow({k: row[k] for k in fields})
            metrics_file.flush()
            state["rows"].append(row)
            print(f"epoch {epoch:03d} train_loss={train_loss:.6f} val_loss={metrics['val_loss']:.6f} "
                  f"val_corr_spectrum={score:.4f} best={state['best_score']:.4f} "
                  f"lr={optimizer.param_groups[0]['lr']:.2e} plateau_wait={state['plateau_wait']}"
                  + (" saved_best" if is_best else ""), flush=True)

            if state["plateau_wait"] >= args.plateau_patience:                 # RR-4
                old_lr = float(optimizer.param_groups[0]["lr"])
                new_lr = old_lr * args.plateau_factor
                if state["lr_drop_count"] >= args.max_lr_drops or new_lr < args.min_learning_rate:
                    state["stop_reason"] = (f"stall after {state['lr_drop_count']} learning-rate reductions"
                                            if state["lr_drop_count"] >= args.max_lr_drops
                                            else "next learning rate below the minimum")
                    state["finished"] = True
                    print(state["stop_reason"], flush=True)
                    save_resume_state(epoch)
                    break
                if state["best_state"] is not None:
                    model.load_state_dict(state["best_state"])
                for group in optimizer.param_groups:
                    group["lr"] = new_lr
                state["lr_drop_count"] += 1
                state["plateau_wait"] = 0
                print(f"RR-4: restored epoch {state['best_epoch']}, learning rate {old_lr:.2e} -> {new_lr:.2e}",
                      flush=True)
            save_resume_state(epoch)

    model.load_state_dict(state["best_state"])
    _, e_pred, e_true = evaluate(model, val_loader, criterion, device, target_transform)
    hs_gain = fit_hs_gain(e_pred, e_true)
    save_checkpoint(checkpoint_path, model, vars(args), large_norm, local_norm, target_transform,
                    site={"name": args.site}, hs_gain=hs_gain,
                    extra={"epoch": state["best_epoch"], "training_years": sorted(args.year), **state["best_metrics"]})

    summary = {"stop_reason": state["stop_reason"], "best_epoch": state["best_epoch"],
               "best_val_corr_spectrum": state["best_score"], "lr_drop_count": state["lr_drop_count"],
               "epochs_completed": len(state["rows"]), "elapsed_seconds": time.time() - run_start,
               "args": vars(args), "checkpoint": checkpoint_path.name}
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"done: best epoch {state['best_epoch']}, validation physical correlation {state['best_score']:.4f}",
          flush=True)


if __name__ == "__main__":
    main()
