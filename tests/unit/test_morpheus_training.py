"""Tests for Morpheus training loop, split, and checkpoint helpers."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.morpheus.dataset import MorpheusDataset
from src.morpheus.training import (
    EpochMetrics,
    TrainConfig,
    append_metrics_csv,
    chronological_split,
    evaluate,
    load_checkpoint,
    save_checkpoint,
    train,
    train_epoch,
)
from src.morpheus.world_model import WorldModel

if TYPE_CHECKING:
    from pathlib import Path


SEQ_LEN = 8
BUCKET_SECONDS = 10


def _make_parquet_dir(tmp_path: Path, n_rows: int = 300) -> Path:
    """Write a single Parquet file of contiguous synthetic candles."""
    d = tmp_path / "candles"
    d.mkdir()

    start = datetime(2025, 1, 6, 8, 0, 0, tzinfo=UTC)
    times = [start + timedelta(seconds=BUCKET_SECONDS * i) for i in range(n_rows)]

    rng = np.random.default_rng(0)
    closes = 2000.0 + np.cumsum(rng.normal(0, 0.5, n_rows))

    df = pl.DataFrame({
        "time": times,
        "open": closes + rng.normal(0, 0.1, n_rows),
        "high": closes + np.abs(rng.normal(0, 0.3, n_rows)),
        "low": closes - np.abs(rng.normal(0, 0.3, n_rows)),
        "close": closes.tolist(),
        "tick_count": rng.integers(5, 50, n_rows).tolist(),
        "spread": (0.3 + rng.normal(0, 0.05, n_rows)).tolist(),
    })
    df.write_parquet(d / "2025-01.parquet")
    return d


@pytest.fixture
def dataset(tmp_path: Path) -> MorpheusDataset:
    parquet_dir = _make_parquet_dir(tmp_path)
    return MorpheusDataset(
        parquet_dir=parquet_dir,
        seq_len=SEQ_LEN,
        stride=1,
        bucket_seconds=BUCKET_SECONDS,
    )


@pytest.fixture
def small_model() -> WorldModel:
    return WorldModel(
        obs_dim=10, embed_dim=16, det_dim=32, stoch_dim=16, hidden_dim=32,
    )


# ---------------------------------------------------------------------------
# chronological_split
# ---------------------------------------------------------------------------

class TestChronologicalSplit:
    def test_split_sizes(self, dataset: MorpheusDataset) -> None:
        train_set, val_set = chronological_split(dataset, val_fraction=0.2, gap=4)
        assert len(train_set) + len(val_set) + 4 == len(dataset)
        assert len(val_set) > 0
        assert len(train_set) > 0

    def test_split_is_chronological(self, dataset: MorpheusDataset) -> None:
        train_set, val_set = chronological_split(dataset, val_fraction=0.2, gap=4)
        train_max = max(train_set.indices)
        val_min = min(val_set.indices)
        assert val_min > train_max
        assert val_min - train_max - 1 >= 4  # gap respected

    def test_rejects_bad_val_fraction(self, dataset: MorpheusDataset) -> None:
        with pytest.raises(ValueError, match="val_fraction"):
            chronological_split(dataset, val_fraction=0.0, gap=0)
        with pytest.raises(ValueError, match="val_fraction"):
            chronological_split(dataset, val_fraction=1.0, gap=0)

    def test_rejects_negative_gap(self, dataset: MorpheusDataset) -> None:
        with pytest.raises(ValueError, match="gap"):
            chronological_split(dataset, val_fraction=0.1, gap=-1)

    def test_rejects_too_large_split(self, dataset: MorpheusDataset) -> None:
        with pytest.raises(ValueError, match="too small"):
            chronological_split(
                dataset, val_fraction=0.99, gap=len(dataset),
            )


# ---------------------------------------------------------------------------
# checkpoint save / load
# ---------------------------------------------------------------------------

class TestCheckpoint:
    def test_roundtrip_restores_weights_and_optimizer(
        self, small_model: WorldModel, tmp_path: Path,
    ) -> None:
        optimizer = Adam(small_model.parameters(), lr=1e-3)
        # Perturb the state so default-init doesn't trivially match.
        obs = torch.randn(2, SEQ_LEN, 10)
        loss = small_model(obs).loss
        loss.backward()
        optimizer.step()

        path = tmp_path / "ckpt.pt"
        config = TrainConfig(
            epochs=1, embed_dim=16, det_dim=32, stoch_dim=16, hidden_dim=32,
        )
        save_checkpoint(
            path, model=small_model, optimizer=optimizer, epoch=3,
            config=config, norm_stats=None,
        )
        assert path.exists()

        new_model = WorldModel(
            obs_dim=10, embed_dim=16, det_dim=32, stoch_dim=16, hidden_dim=32,
        )
        new_optimizer = Adam(new_model.parameters(), lr=1e-3)
        payload = load_checkpoint(path, model=new_model, optimizer=new_optimizer)

        assert payload["epoch"] == 3
        for p1, p2 in zip(
            small_model.parameters(), new_model.parameters(), strict=True,
        ):
            torch.testing.assert_close(p1, p2)

    def test_atomic_write_leaves_no_tmp(
        self, small_model: WorldModel, tmp_path: Path,
    ) -> None:
        optimizer = Adam(small_model.parameters(), lr=1e-3)
        path = tmp_path / "ckpt.pt"
        save_checkpoint(
            path, model=small_model, optimizer=optimizer, epoch=0,
            config=TrainConfig(), norm_stats=None,
        )
        assert path.exists()
        assert not path.with_suffix(path.suffix + ".tmp").exists()


# ---------------------------------------------------------------------------
# train_epoch / evaluate
# ---------------------------------------------------------------------------

class TestTrainEpoch:
    def test_returns_finite_averages(
        self, small_model: WorldModel, dataset: MorpheusDataset,
    ) -> None:
        loader: DataLoader[torch.Tensor] = DataLoader(
            dataset, batch_size=4, shuffle=False, drop_last=True,
        )
        optimizer = Adam(small_model.parameters(), lr=1e-3)
        avg_loss, avg_recon, avg_kl = train_epoch(
            small_model, loader, optimizer,
            grad_clip=1.0, device=torch.device("cpu"),
        )
        assert np.isfinite(avg_loss)
        assert np.isfinite(avg_recon)
        assert np.isfinite(avg_kl)
        assert avg_loss > 0

    def test_updates_parameters(
        self, small_model: WorldModel, dataset: MorpheusDataset,
    ) -> None:
        loader: DataLoader[torch.Tensor] = DataLoader(
            dataset, batch_size=4, shuffle=False, drop_last=True,
        )
        optimizer = Adam(small_model.parameters(), lr=1e-2)
        before = [p.detach().clone() for p in small_model.parameters()]
        train_epoch(
            small_model, loader, optimizer,
            grad_clip=1.0, device=torch.device("cpu"),
        )
        after = list(small_model.parameters())
        changed = sum(
            1 for a, b in zip(before, after, strict=True)
            if not torch.equal(a, b)
        )
        assert changed > 0


class TestEvaluate:
    def test_no_grad_no_param_change(
        self, small_model: WorldModel, dataset: MorpheusDataset,
    ) -> None:
        loader: DataLoader[torch.Tensor] = DataLoader(
            dataset, batch_size=4, shuffle=False, drop_last=False,
        )
        before = [p.detach().clone() for p in small_model.parameters()]
        avg_loss, _, _ = evaluate(small_model, loader, device=torch.device("cpu"))
        after = list(small_model.parameters())
        assert np.isfinite(avg_loss)
        for a, b in zip(before, after, strict=True):
            torch.testing.assert_close(a, b)


# ---------------------------------------------------------------------------
# append_metrics_csv
# ---------------------------------------------------------------------------

class TestMetricsCsv:
    def test_writes_header_once(self, tmp_path: Path) -> None:
        path = tmp_path / "metrics.csv"
        append_metrics_csv(
            path,
            EpochMetrics(
                epoch=1, train_loss=1.0, train_recon=0.8, train_kl=0.2,
                val_loss=1.1, val_recon=0.9, val_kl=0.2,
            ),
        )
        append_metrics_csv(
            path,
            EpochMetrics(
                epoch=2, train_loss=0.9, train_recon=0.7, train_kl=0.2,
            ),
        )
        lines = path.read_text().splitlines()
        assert lines[0].startswith("epoch")
        assert len(lines) == 3


# ---------------------------------------------------------------------------
# full train() smoke test
# ---------------------------------------------------------------------------

class TestTrainSmoke:
    def test_two_epoch_smoke(
        self, dataset: MorpheusDataset, tmp_path: Path,
    ) -> None:
        train_set, val_set = chronological_split(
            dataset, val_fraction=0.2, gap=2,
        )
        out_dir = tmp_path / "run"
        config = TrainConfig(
            epochs=2, batch_size=4, lr=1e-3, grad_clip=1.0,
            val_fraction=0.2, val_gap=2, checkpoint_interval=1,
            seed=0,
            embed_dim=16, det_dim=32, stoch_dim=16, hidden_dim=32,
            free_nats=0.5,
        )
        history = train(
            train_set=train_set, val_set=val_set,
            norm_stats=dataset.norm_stats,
            config=config, output_dir=out_dir,
            device=torch.device("cpu"),
        )
        assert len(history) == 2
        assert (out_dir / "checkpoint_epoch_1.pt").exists()
        assert (out_dir / "checkpoint_epoch_2.pt").exists()
        assert (out_dir / "checkpoint_final.pt").exists()
        assert (out_dir / "metrics.csv").exists()
        for m in history:
            assert np.isfinite(m.train_loss)

    def test_resume_continues_from_checkpoint(
        self, dataset: MorpheusDataset, tmp_path: Path,
    ) -> None:
        train_set, val_set = chronological_split(
            dataset, val_fraction=0.2, gap=2,
        )
        out_dir = tmp_path / "run"
        base_config = TrainConfig(
            epochs=1, batch_size=4, lr=1e-3, checkpoint_interval=1,
            embed_dim=16, det_dim=32, stoch_dim=16, hidden_dim=32,
            free_nats=0.5,
        )
        train(
            train_set=train_set, val_set=val_set,
            norm_stats=dataset.norm_stats,
            config=base_config, output_dir=out_dir,
            device=torch.device("cpu"),
        )
        ckpt = out_dir / "checkpoint_final.pt"
        # Resume for one more epoch and check it writes checkpoint_epoch_2
        resume_config = TrainConfig(
            epochs=2, batch_size=4, lr=1e-3, checkpoint_interval=1,
            embed_dim=16, det_dim=32, stoch_dim=16, hidden_dim=32,
            free_nats=0.5,
        )
        history = train(
            train_set=train_set, val_set=val_set,
            norm_stats=dataset.norm_stats,
            config=resume_config, output_dir=out_dir,
            device=torch.device("cpu"), resume_from=ckpt,
        )
        assert len(history) == 1  # only epoch 2 (resumed from epoch 1)
        assert history[0].epoch == 2
