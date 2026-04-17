"""Train actor/critic policy in imagination using a frozen world model.

Usage:
    uv run python -m scripts.train_policy \
        --checkpoint runs/morpheus/xauusd_20260417/checkpoint_epoch_18.pt \
        --parquet-dir data/morpheus/xauusd \
        --output-dir runs/policy_xauusd_001 \
        --epochs 100 --batch-size 32
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from src.common.logging import get_logger
from src.morpheus.actor_critic import PolicyConfig, build_actor_critic
from src.morpheus.dataset import MorpheusDataset, NormStats
from src.morpheus.imagination import ImaginationEnv
from src.morpheus.policy_trainer import PolicyTrainer
from src.morpheus.training import TrainConfig, build_model

logger = get_logger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(
        description="Train actor/critic policy in imagination",
    )

    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--parquet-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)

    # Data
    p.add_argument("--bucket-seconds", type=int, default=10)
    p.add_argument("--stride", type=int, default=128)
    p.add_argument("--h1", action="store_true")

    # Trading
    p.add_argument("--spread-points", type=float, default=0.5)
    p.add_argument("--margin-rate", type=float, default=0.05)
    p.add_argument("--initial-capital", type=float, default=5000.0)

    # Training
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--rollouts-per-epoch", type=int, default=16)
    p.add_argument("--horizon", type=int, default=64)
    p.add_argument("--context-len", type=int, default=256)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lambda_", type=float, default=0.95, dest="lambda_")
    p.add_argument("--actor-lr", type=float, default=3e-4)
    p.add_argument("--critic-lr", type=float, default=1e-3)
    p.add_argument("--entropy-coef", type=float, default=0.05)
    p.add_argument("--idle-penalty", type=float, default=-0.0005)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--n-hidden", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")

    return p.parse_args(argv)


def resolve_device(choice: str) -> torch.device:
    """Resolve --device into a torch.device."""
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(choice)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    args = parse_args(argv)
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)

    logger.info(
        "policy_training_start",
        checkpoint=str(args.checkpoint),
        device=str(device),
        epochs=args.epochs,
    )

    # Load world model
    payload = torch.load(
        args.checkpoint, map_location=device, weights_only=False,
    )
    norm_stats = NormStats.from_dict(payload["norm_stats"])
    wm_cfg = payload["config"]
    train_config = TrainConfig(**{
        k: v for k, v in wm_cfg.items()
        if k in TrainConfig.__dataclass_fields__
    })
    world_model = build_model(train_config).to(device)
    world_model.load_state_dict(payload["model_state"])
    logger.info("world_model_loaded", model_type=train_config.model_type)
    if train_config.model_type != "transformer":
        msg = "Policy training requires a transformer world model"
        raise ValueError(msg)

    # Dataset for sampling contexts
    dataset = MorpheusDataset(
        parquet_dir=args.parquet_dir,
        seq_len=args.context_len,
        stride=args.stride,
        bucket_seconds=args.bucket_seconds,
        norm_stats=norm_stats,
        use_h1=args.h1,
    )
    logger.info("dataset_loaded", total_sequences=len(dataset))

    # Policy config
    policy_config = PolicyConfig(
        hidden_dim=args.hidden_dim,
        n_hidden=args.n_hidden,
        spread_points=args.spread_points,
        margin_rate=args.margin_rate,
        initial_capital=args.initial_capital,
        horizon=args.horizon,
        context_len=args.context_len,
        gamma=args.gamma,
        lambda_=args.lambda_,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        entropy_coef=args.entropy_coef,
        idle_penalty=args.idle_penalty,
        grad_clip=args.grad_clip,
        epochs=args.epochs,
        batch_size=args.batch_size,
        rollouts_per_epoch=args.rollouts_per_epoch,
        seed=args.seed,
    )

    # Build actor/critic
    d_model = train_config.d_model
    actor, critic = build_actor_critic(d_model, policy_config)
    actor = actor.to(device)
    critic = critic.to(device)
    logger.info(
        "policy_networks_built",
        actor_params=sum(p.numel() for p in actor.parameters()),
        critic_params=sum(p.numel() for p in critic.parameters()),
    )

    # Environment
    env = ImaginationEnv(policy_config, norm_stats, device)

    # Train
    trainer = PolicyTrainer(
        world_model, actor, critic, env, dataset, policy_config, device,  # type: ignore[arg-type]
    )
    history = trainer.train(output_dir=args.output_dir)

    final = history[-1] if history else None
    if final is not None:
        logger.info(
            "policy_training_complete",
            epochs=final.epoch,
            mean_reward=f"{final.mean_reward:.4f}",
            mean_capital=f"{final.mean_capital:.0f}",
        )


if __name__ == "__main__":
    main(sys.argv[1:])
