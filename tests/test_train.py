import numpy as np
import pytest
import torch

from mtl_eurosat.train import Variant, fit, predict


def _stripes(n: int, seed: int, size: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    """Class 1 carries vertical stripes, class 0 is plain noise: a known ground truth.

    32x32 keeps the test fast; every architecture works on any size divisible by 4.
    """
    g = torch.Generator().manual_seed(seed)
    y = torch.arange(n) % 2
    x = 0.3 * torch.randn(n, 3, size, size, generator=g)
    stripes = (torch.arange(size) // 4 % 2).float() * 1.2 - 0.6
    x[y == 1] += stripes.view(1, 1, 1, size)
    return x.clamp(-1, 1), y


@pytest.fixture(scope="module")
def synthetic():
    torch.set_num_threads(2)
    return (*_stripes(192, 0), *_stripes(96, 1))


@pytest.mark.parametrize(
    "variant",
    [
        Variant("cnn", "cnn"),
        Variant("hard", "hard", alpha=0.6),
        Variant("soft", "soft", alpha=0.5, beta=0.005, gamma=0.01, mask_ratio=0.3),
    ],
)
def test_recovers_a_known_signal(synthetic, variant):
    x_tr, y_tr, x_va, y_va = synthetic
    result = fit(variant, x_tr, y_tr, x_va, y_va, seed=0, epochs=3, batch_size=32)
    p, mse = predict(result.model, x_va)
    assert np.mean((p >= 0.5) == y_va.numpy()) >= 0.95
    assert np.isnan(mse).all() == (variant.arch == "cnn")


def test_returns_the_best_epoch_not_the_last(synthetic):
    x_tr, y_tr, x_va, y_va = synthetic
    v = Variant("hard", "hard", alpha=0.6)
    result = fit(v, x_tr, y_tr, x_va, y_va, seed=1, epochs=3, batch_size=32)
    losses = [h["val_loss"] for h in result.history]
    assert result.best_epoch == int(np.argmin(losses)) + 1
    # Re-scoring the returned weights reproduces the best epoch's validation accuracy.
    p, _ = predict(result.model, x_va)
    best = result.history[result.best_epoch - 1]
    assert np.mean((p >= 0.5) == y_va.numpy()) == pytest.approx(best["val_acc"])


def test_seeded_training_is_reproducible(synthetic):
    x_tr, y_tr, x_va, y_va = synthetic
    v = Variant("cnn", "cnn")
    a = fit(v, x_tr, y_tr, x_va, y_va, seed=7, epochs=1)
    b = fit(v, x_tr, y_tr, x_va, y_va, seed=7, epochs=1)
    np.testing.assert_allclose(predict(a.model, x_va)[0], predict(b.model, x_va)[0])
