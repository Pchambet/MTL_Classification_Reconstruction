import numpy as np
import pytest
from PIL import Image

from mtl_eurosat import data


def _write(folder, n, colour, size=64):
    folder.mkdir(parents=True)
    for i in range(n):
        Image.new("RGB", (size, size), colour).save(folder / f"img_{i}.jpg")


def test_read_all_labels_and_resizes(tmp_path):
    _write(tmp_path / "Forest", 3, (0, 120, 0))
    _write(tmp_path / "Residential", 2, (200, 200, 200))
    _write(tmp_path / "OOD" / "Forest", 2, (0, 90, 0), size=600)  # AID images are 600x600
    _write(tmp_path / "OOD" / "DenseResidential", 1, (150, 150, 150))
    _write(tmp_path / "OOD" / "MediumResidential", 1, (170, 170, 170))
    images = data.read_all(tmp_path)
    assert images.x_id.shape == (5, 64, 64, 3) and images.x_id.dtype == np.uint8
    assert images.y_id.tolist() == [0, 0, 0, 1, 1]
    assert images.x_ood.shape == (4, 64, 64, 3)
    assert images.y_ood.tolist() == [0, 0, 1, 1]
    assert images.group_ood.tolist() == ["Forest", "Forest", "DenseResidential", "MediumResidential"]


def test_cache_round_trip(tmp_path):
    _write(tmp_path / "Forest", 2, (0, 120, 0))
    _write(tmp_path / "Residential", 2, (200, 200, 200))
    for g in ("Forest", "DenseResidential", "MediumResidential"):
        _write(tmp_path / "OOD" / g, 1, (90, 90, 90))
    cache = tmp_path / "cache.npz"
    first = data.load(tmp_path, cache)
    again = data.load(tmp_path, cache)
    assert cache.exists()
    np.testing.assert_array_equal(first.x_id, again.x_id)
    np.testing.assert_array_equal(first.group_ood, again.group_ood)


def test_empty_folder_is_an_error(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        data.read_folder(tmp_path / "empty")


def test_stratified_split_is_balanced_disjoint_and_seeded():
    y = np.repeat([0, 1], [300, 300])
    tr, va = data.stratified_split(y, 0.2, seed=3)
    assert len(va) == 120 and np.bincount(y[va]).tolist() == [60, 60]
    assert set(tr).isdisjoint(va) and len(tr) + len(va) == len(y)
    tr2, va2 = data.stratified_split(y, 0.2, seed=3)
    np.testing.assert_array_equal(va, va2)
    assert not np.array_equal(va, data.stratified_split(y, 0.2, seed=4)[1])


def test_tensor_round_trip():
    x = np.random.default_rng(0).integers(0, 256, size=(2, 64, 64, 3), dtype=np.uint8)
    t = data.to_tensor(x)
    assert t.shape == (2, 3, 64, 64)
    assert t.min() >= -1 and t.max() <= 1
    np.testing.assert_array_equal(data.to_uint8(t), x)
