"""Shared training framework for quantile regression models (MLP & LSTM).

Provides pinball loss and a generic train_single_quantile function that
accepts a model_factory callback, decoupling model construction from
training logic.
"""

from collections.abc import Callable

import numpy as np
import torch
import torch.nn as nn

SEED = 42


def pinball_loss(
    pred: torch.Tensor, target: torch.Tensor, tau: float
) -> torch.Tensor:
    """Quantile (pinball) loss for a single tau."""
    errors = target - pred.squeeze(-1)
    return torch.mean(torch.max(tau * errors, (tau - 1) * errors))


def train_single_quantile(
    X: np.ndarray,
    y: np.ndarray,
    tau: float,
    model_factory: Callable[[], nn.Module],
    lr: float,
    weight_decay: float,
    epochs: int = 300,
    patience: int = 10,
    batch_size: int = 32,
    fixed_epochs: int | None = None,
) -> tuple[nn.Module, int, float]:
    """Train a model for one quantile tau.

    Args:
        X: Input features. Shape (n, d) for MLP or (n, seq_len, d) for LSTM.
        y: Target values. Shape (n,).
        tau: Quantile level.
        model_factory: Callable that returns a fresh nn.Module instance.
        lr: Learning rate.
        weight_decay: L2 regularization.
        epochs: Max epochs (early stopping mode).
        patience: Early stopping patience.
        batch_size: Mini-batch size.
        fixed_epochs: If set, train on ALL data for exactly this many epochs.

    Returns:
        (model, best_epoch, val_loss). val_loss is NaN in fixed_epochs mode.
    """
    torch.manual_seed(SEED)
    device = torch.device("cpu")

    if fixed_epochs is not None:
        X_tr = torch.from_numpy(X).to(device)
        y_tr = torch.from_numpy(y).to(device)
        train_ds = torch.utils.data.TensorDataset(X_tr, y_tr)
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True
        )
        model = model_factory().to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        for _ in range(fixed_epochs):
            model.train()
            for xb, yb in train_loader:
                optimizer.zero_grad()
                loss = pinball_loss(model(xb), yb, tau)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        return model, fixed_epochs, float("nan")

    # Early stopping mode: 80/20 split
    n = len(X)
    n_val = max(1, int(n * 0.2))
    n_train = n - n_val

    X_tr = torch.from_numpy(X[:n_train]).to(device)
    y_tr = torch.from_numpy(y[:n_train]).to(device)
    X_val = torch.from_numpy(X[n_train:]).to(device)
    y_val = torch.from_numpy(y[n_train:]).to(device)

    train_ds = torch.utils.data.TensorDataset(X_tr, y_tr)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True
    )

    model = model_factory().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    best_val_loss = float("inf")
    best_state = None
    stale = 0
    best_epoch = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = pinball_loss(model(xb), yb, tau)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = pinball_loss(model(X_val), y_val, tau).item()

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_epoch, best_val_loss
