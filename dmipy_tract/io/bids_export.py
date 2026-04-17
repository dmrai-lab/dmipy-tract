"""BIDS-compatible tractogram export.

Writes .trk files and JSON sidecars following the BIDS Derivatives convention
for tractography outputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def save_bids_tractogram(
    streamlines: list,
    affine: np.ndarray,
    reference_nifti_path: str | Path,
    output_dir: str | Path,
    subject: str = "sub-01",
    session: str | None = None,
    desc: str = "wholebrain",
    suffix: str = "tractography",
) -> Path:
    """Save tractogram in BIDS-compatible format.

    Writes a .trk file and a JSON sidecar to the BIDS derivatives directory
    structure:

        <output_dir>/<subject>/[<session>/]dwi/<filename>.trk
        <output_dir>/<subject>/[<session>/]dwi/<filename>.json

    Parameters
    ----------
    streamlines : list of (L, 3) arrays
        World-space streamlines in mm (RASMM).
    affine : ndarray, shape (4, 4)
        Voxel-to-world affine.
    reference_nifti_path : str or Path
        Path to the reference NIfTI image used for tractography.
    output_dir : str or Path
        Root of the BIDS output directory.
    subject : str
        BIDS subject label, e.g. ``"sub-01"``.
    session : str or None
        BIDS session label, e.g. ``"ses-01"``. If None, no session
        level is inserted in the path.
    desc : str
        BIDS ``desc`` entity value, e.g. ``"wholebrain"``.
    suffix : str
        BIDS suffix for the file, e.g. ``"tractography"``.

    Returns
    -------
    trk_path : Path
        Absolute path to the saved .trk file.
    """
    from .tractogram_io import save_trk

    output_dir = Path(output_dir)
    reference_nifti_path = Path(reference_nifti_path)

    # Build BIDS directory path
    parts = [subject]
    if session is not None:
        parts.append(session)
    parts.append("dwi")
    dwi_dir = output_dir.joinpath(*parts)
    dwi_dir.mkdir(parents=True, exist_ok=True)

    # Build filename stem following BIDS naming convention
    stem_parts = [subject]
    if session is not None:
        stem_parts.append(session)
    stem_parts.append(f"desc-{desc}")
    stem_parts.append(suffix)
    stem = "_".join(stem_parts)

    trk_path = dwi_dir / f"{stem}.trk"
    json_path = dwi_dir / f"{stem}.json"

    # Load reference image if available, otherwise pass affine directly
    reference_img = None
    if reference_nifti_path.exists():
        try:
            import nibabel as nib
            reference_img = nib.load(str(reference_nifti_path))
        except Exception:
            reference_img = None

    save_trk(
        streamlines,
        affine,
        str(trk_path),
        reference_img=reference_img,
    )

    # Write JSON sidecar with provenance metadata
    sidecar = {
        "n_streamlines": len(streamlines),
        "affine": affine.tolist() if isinstance(affine, np.ndarray) else affine,
        "TrackerSoftware": "dmipy-tract",
        "GeneratedBy": {
            "Name": "dmipy-tract",
            "Version": "0.1.0",
        },
    }
    with open(json_path, "w") as fh:
        json.dump(sidecar, fh, indent=2)

    return trk_path
