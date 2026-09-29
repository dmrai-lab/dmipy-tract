"""The DiSCo acceptance as a slow test: the connectivity Pearson against the strand counts on the reference replay
volume, through ``benchmarks/disco.py``. Needs the DiSCo reference directory, dmipy-fit with ``csd_tournier07_jax``
and the public replay volume (downloaded); skipped when any is missing. ``pytest -m slow``."""
import os
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.slow

REF = os.environ.get("DISCO_REF", "/home/rutger/dmrai-ws/dmrai-papers-private/replay_paper/benchmarks/disco/data")


@pytest.fixture(scope="module")
def disco():
    if not os.path.exists(f"{REF}/DiSCo_ROIs.nii.gz"):
        pytest.skip(f"no DiSCo reference directory at {REF} (set DISCO_REF)")
    pytest.importorskip("dmipy_fit.jax.csd_tournier_jax")
    pytest.importorskip("huggingface_hub")
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))
    import disco as bench
    import nibabel as nib
    from dmipy_tract import FODField, seeds_from_mask
    data, bvals, bvecs, _ = bench.load_volume(None, None, None)
    mask = nib.load(f"{REF}/DiSCo_mask.nii.gz").get_fdata() > 0
    rois = nib.load(f"{REF}/DiSCo_ROIs.nii.gz").get_fdata().astype(np.int32)
    sh, _ = bench.csd(data, bench.scheme_for(bvals, bvecs), data[..., 0] > 0, "csd_tournier07_jax")
    field = FODField(sh, np.eye(4), mask | (rois > 0))
    return bench, field, rois, seeds_from_mask(rois > 0, np.eye(4), density=4)


def test_connectivity_pearson_is_the_replay_reference(disco):
    """dipy on this volume scores 0.904/0.905 (dmipy-sim#505); the replay reference is 0.912 +- 0.003. The bar is
    the reference's band, and the connected-pair recall is the ground truth's."""
    bench, field, rois, seeds = disco
    from dmipy_tract import track, connectivity
    tg = track(field, seeds, rule='probabilistic', step_mm=0.5, max_angle=30.0, max_steps=500, key=0)
    matrix, _ = connectivity(tg, rois, np.eye(4))
    s = bench.score(bench.symmetric_counts(matrix), REF)
    assert s['pearson_count'] >= 0.90, s
    assert s['missed_pairs'] == 0, s
