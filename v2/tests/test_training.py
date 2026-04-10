"""Tests for utils/training.py — shared training framework."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from utils.training import SEED, pinball_loss, train_single_quantile


class _DummyModel(nn.Module):
    """Minimal model for testing: linear layer."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def _make_dummy_data(n: int = 200, d: int = 3):
    """Generate synthetic data for training tests."""
    rng = np.random.RandomState(0)
    X = rng.randn(n, d).astype(np.float32)
    y = (X[:, 0] * 0.5 + rng.randn(n) * 0.1).astype(np.float32)
    return X, y


class TestPinballLoss:
    def test_zero_error(self):
        pred = torch.tensor([[0.5]])
        target = torch.tensor([0.5])
        loss = pinball_loss(pred, target, tau=0.05)
        assert loss.item() == pytest.approx(0.0, abs=1e-7)

    def test_positive_error_tau05(self):
        pred = torch.tensor([[0.2]])
        target = torch.tensor([0.5])
        loss = pinball_loss(pred, target, tau=0.05)
        assert loss.item() == pytest.approx(0.05 * 0.3, abs=1e-6)

    def test_negative_error_tau05(self):
        pred = torch.tensor([[0.5]])
        target = torch.tensor([0.2])
        loss = pinball_loss(pred, target, tau=0.05)
        assert loss.item() == pytest.approx(0.95 * 0.3, abs=1e-6)

    def test_seed_constant(self):
        assert SEED == 42


class TestTrainSingleQuantile:
    def test_early_stopping_returns_model(self):
        X, y = _make_dummy_data()
        factory = lambda: _DummyModel(X.shape[1])
        model, best_epoch, val_loss = train_single_quantile(
            X, y, tau=0.05, model_factory=factory,
            lr=1e-3, weight_decay=0.0,
            epochs=50, patience=5, batch_size=32,
        )
        assert isinstance(model, nn.Module)
        assert best_epoch >= 1
        assert not np.isnan(val_loss)
        assert val_loss >= 0

    def test_fixed_epochs_mode(self):
        X, y = _make_dummy_data()
        factory = lambda: _DummyModel(X.shape[1])
        model, epoch_count, val_loss = train_single_quantile(
            X, y, tau=0.05, model_factory=factory,
            lr=1e-3, weight_decay=0.0,
            fixed_epochs=10,
        )
        assert epoch_count == 10
        assert np.isnan(val_loss)

    def test_output_shape(self):
        X, y = _make_dummy_data(n=100)
        factory = lambda: _DummyModel(X.shape[1])
        model, _, _ = train_single_quantile(
            X, y, tau=0.05, model_factory=factory,
            lr=1e-3, weight_decay=0.0,
            epochs=5, patience=3, batch_size=16,
        )
        model.eval()
        with torch.no_grad():
            out = model(torch.from_numpy(X[:5]))
        assert out.shape == (5, 1)

    def test_3d_input_for_lstm(self):
        """Verify train_single_quantile works with 3D input (LSTM-style)."""
        rng = np.random.RandomState(0)
        n, seq_len, d = 200, 10, 2
        X = rng.randn(n, seq_len, d).astype(np.float32)
        y = rng.randn(n).astype(np.float32)

        class _SimpleLSTM(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(d, 16, batch_first=True)
                self.fc = nn.Linear(16, 1)
            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :])

        factory = lambda: _SimpleLSTM()
        model, best_epoch, val_loss = train_single_quantile(
            X, y, tau=0.05, model_factory=factory,
            lr=1e-3, weight_decay=0.0,
            epochs=20, patience=5, batch_size=32,
        )
        assert isinstance(model, nn.Module)
        assert best_epoch >= 1
