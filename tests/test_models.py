import pytest
import torch

from mtl_eurosat.models import ARCHITECTURES, n_parameters


@pytest.mark.parametrize("arch", list(ARCHITECTURES))
def test_shapes(arch):
    model = ARCHITECTURES[arch]().eval()
    out = model(torch.zeros(5, 3, 64, 64))
    assert out.logits.shape == (5, 2)
    if arch == "cnn":
        assert out.recon is None
    else:
        assert out.recon.shape == (5, 3, 64, 64)
        assert out.recon.abs().max() <= 1.0  # tanh output, same range as the inputs
    if arch == "soft":
        assert out.z_cls.shape == out.z_rec.shape == (5, 64)


def test_parameter_counts_match_the_notebooks():
    # The baseline is ~8x smaller than hard sharing: the alpha = 1 control exists for this.
    assert n_parameters(ARCHITECTURES["cnn"]()) == 23_714
    assert n_parameters(ARCHITECTURES["hard"]()) == 193_221
