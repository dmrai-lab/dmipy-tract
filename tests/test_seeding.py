"""Tests for seed generation strategies (Phase 2)."""
import numpy as np
import pytest
from dmipy_tract.core.seeding import seeds_from_mask, seeds_from_vf_ic


def test_seeds_from_mask_density1():
    """density=1 → one seed per mask voxel at voxel centre."""
    mask = np.zeros((5, 5, 5), dtype=bool)
    mask[2, 2, 2] = True
    mask[1, 3, 4] = True
    affine = np.eye(4)  # voxel = world (1mm isotropic)
    seeds = seeds_from_mask(mask, affine, density=1)
    assert seeds.shape == (2, 3), f"Expected 2 seeds, got {seeds.shape}"
    # Seed at voxel (2,2,2) should be at world (2.5, 2.5, 2.5) with identity affine
    centres = {tuple(np.round(s, 2)) for s in seeds}
    assert (2.5, 2.5, 2.5) in centres


def test_seeds_from_mask_density2():
    """density=2 → 8 seeds per voxel."""
    mask = np.zeros((4, 4, 4), dtype=bool)
    mask[1, 1, 1] = True
    seeds = seeds_from_mask(mask, np.eye(4), density=2)
    assert seeds.shape == (8, 3)


def test_seeds_from_mask_affine_scaling():
    """2mm isotropic affine scales seed positions correctly."""
    affine = np.diag([2., 2., 2., 1.])
    mask = np.zeros((3, 3, 3), dtype=bool)
    mask[1, 1, 1] = True
    seeds = seeds_from_mask(mask, affine, density=1)
    # voxel centre (1.5, 1.5, 1.5) → world (3.0, 3.0, 3.0) with 2mm voxels
    np.testing.assert_allclose(seeds[0], [3.0, 3.0, 3.0], atol=1e-4)


def test_seeds_from_mask_empty():
    """Empty mask returns zero seeds."""
    mask = np.zeros((5, 5, 5), dtype=bool)
    seeds = seeds_from_mask(mask, np.eye(4), density=1)
    assert seeds.shape[0] == 0


def test_seeds_from_mask_density1_single_voxel_count():
    """density=1 with single True voxel → exactly 1 seed."""
    mask = np.zeros((10, 10, 10), dtype=bool)
    mask[5, 5, 5] = True
    seeds = seeds_from_mask(mask, np.eye(4), density=1)
    assert seeds.shape == (1, 3)


def test_seeds_from_vf_ic_threshold():
    """Only voxels above vf_ic_threshold are seeded."""
    # Build a minimal MicrostructureField-like mock
    class MockField:
        parameter_names = ['partial_volume_0']
        affine = np.eye(4)

        def __init__(self):
            self._params = {'partial_volume_0': np.array([[[0.2, 0.4, 0.6, 0.8]]])}
            # shape (1, 1, 4)

    field = MockField()
    seeds = seeds_from_vf_ic(field, vf_ic_threshold=0.5, density=1)
    # voxels with vf_ic >= 0.5: indices 2 (0.6) and 3 (0.8) → 2 seeds
    assert len(seeds) == 2, f"Expected 2 seeds above threshold, got {len(seeds)}"


def test_seeds_from_vf_ic_all_below_threshold():
    """When all voxels are below threshold, no seeds are generated."""
    class MockField:
        parameter_names = ['partial_volume_0']
        affine = np.eye(4)

        def __init__(self):
            self._params = {'partial_volume_0': np.array([[[0.1, 0.2, 0.3]]])}

    field = MockField()
    seeds = seeds_from_vf_ic(field, vf_ic_threshold=0.5, density=1)
    assert len(seeds) == 0


def test_seeds_from_vf_ic_missing_parameter_raises():
    """KeyError is raised if parameter_name is not in field."""
    class MockField:
        parameter_names = ['partial_volume_0']
        affine = np.eye(4)

        def __init__(self):
            self._params = {'partial_volume_0': np.ones((3, 3, 3))}

    field = MockField()
    with pytest.raises(KeyError):
        seeds_from_vf_ic(field, parameter_name='nonexistent_param')
