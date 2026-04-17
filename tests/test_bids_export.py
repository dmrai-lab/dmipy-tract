"""Tests for dmipy_tract.io.bids_export.save_bids_tractogram."""

import json
import numpy as np
import pytest
from pathlib import Path

from dmipy_tract.io.bids_export import save_bids_tractogram


@pytest.fixture()
def streamlines():
    rng = np.random.default_rng(42)
    return [rng.uniform(0, 10, (20, 3)).astype(np.float32) for _ in range(5)]


@pytest.fixture()
def affine():
    return np.eye(4, dtype=np.float32)


def test_bids_saves_trk(tmp_path, streamlines, affine):
    """save_bids_tractogram writes a .trk file at the expected BIDS path."""
    ref_nifti = tmp_path / "ref.nii"  # does not need to exist — affine is used
    trk_path = save_bids_tractogram(
        streamlines,
        affine,
        reference_nifti_path=ref_nifti,
        output_dir=tmp_path / "derivatives",
        subject="sub-01",
        session=None,
        desc="wholebrain",
        suffix="tractography",
    )
    assert trk_path.exists(), f"Expected .trk file at {trk_path}"
    expected = (
        tmp_path / "derivatives" / "sub-01" / "dwi"
        / "sub-01_desc-wholebrain_tractography.trk"
    )
    assert trk_path == expected


def test_bids_saves_json_sidecar(tmp_path, streamlines, affine):
    """save_bids_tractogram writes a JSON sidecar with n_streamlines key."""
    ref_nifti = tmp_path / "ref.nii"
    trk_path = save_bids_tractogram(
        streamlines,
        affine,
        reference_nifti_path=ref_nifti,
        output_dir=tmp_path / "derivatives",
        subject="sub-01",
    )
    json_path = trk_path.with_suffix(".json")
    assert json_path.exists(), f"Expected JSON sidecar at {json_path}"

    with open(json_path) as fh:
        meta = json.load(fh)

    assert "n_streamlines" in meta
    assert meta["n_streamlines"] == len(streamlines)
    assert "GeneratedBy" in meta
    assert meta["GeneratedBy"]["Name"] == "dmipy-tract"


def test_bids_session_optional(tmp_path, streamlines, affine):
    """Path is correct both with and without a session label."""
    ref_nifti = tmp_path / "ref.nii"
    derivatives = tmp_path / "derivatives"

    # Without session
    trk_no_ses = save_bids_tractogram(
        streamlines, affine, ref_nifti, derivatives,
        subject="sub-02", session=None,
    )
    expected_no_ses = (
        derivatives / "sub-02" / "dwi"
        / "sub-02_desc-wholebrain_tractography.trk"
    )
    assert trk_no_ses == expected_no_ses
    assert trk_no_ses.exists()

    # With session
    trk_with_ses = save_bids_tractogram(
        streamlines, affine, ref_nifti, derivatives,
        subject="sub-02", session="ses-01",
    )
    expected_with_ses = (
        derivatives / "sub-02" / "ses-01" / "dwi"
        / "sub-02_ses-01_desc-wholebrain_tractography.trk"
    )
    assert trk_with_ses == expected_with_ses
    assert trk_with_ses.exists()


def test_bids_n_streamlines_in_sidecar(tmp_path, affine):
    """JSON sidecar n_streamlines reflects actual streamline count."""
    rng = np.random.default_rng(7)
    sls = [rng.uniform(0, 5, (10, 3)).astype(np.float32) for _ in range(12)]
    ref_nifti = tmp_path / "ref.nii"
    trk_path = save_bids_tractogram(
        sls, affine, ref_nifti, tmp_path / "out", subject="sub-03"
    )
    with open(trk_path.with_suffix(".json")) as fh:
        meta = json.load(fh)
    assert meta["n_streamlines"] == 12
