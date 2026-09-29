"""The DiSCo acceptance: a DWI on the 40^3 grid -> CSD (dmipy-fit, ``csd_tournier07_jax``, order 8, the response from
the single-fibre voxels) -> probabilistic tracking seeded from the sixteen regions -> the 16 x 16 streamline-count
matrix -> its Pearson correlation with the dataset's strand-count and cross-sectional-area matrices. The dataset's
own score, the pipeline of the replay paper's ``disco_tract.py`` with this package's tracker in dipy's place.

    python benchmarks/disco.py [--dwi replay.nii.gz --bvals .. --bvecs ..] [--ref DIR] [--density 4] [--key 0]
                               [--dipy] [--out out/disco.json]

Without ``--dwi`` the reference replay volume of SubstrateCommons/disco-replay is downloaded (public). ``--ref`` is
the DiSCo data directory (mask, ROIs, ground-truth matrices); ``--dipy`` also runs dipy's ``LocalTracking`` on the
same FOD field, the same seeds and the same settings, for the like-for-like number.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

REF_DEFAULT = "/home/rutger/dmrai-ws/dmrai-papers-private/replay_paper/benchmarks/disco/data"
HF_REPO = "SubstrateCommons/disco-replay"
SH_ORDER = 8


def load_volume(dwi, bvals, bvecs):
    import nibabel as nib
    if dwi is None:
        from huggingface_hub import hf_hub_download
        dwi, bvals, bvecs = (hf_hub_download(repo_id=HF_REPO, repo_type="dataset", filename=f"disco/reference/{f}")
                             for f in ("disco_replay_bare.nii.gz", "DiSCo_gradients.bvals", "DiSCo_gradients_dipy.bvecs"))
    img = nib.load(dwi)
    data = np.asarray(img.dataobj, dtype=np.float64)
    bvals_mm2 = np.loadtxt(bvals).ravel()
    bvecs = np.loadtxt(bvecs)
    if bvecs.shape[0] == 3:
        bvecs = bvecs.T
    return data, bvals_mm2, bvecs, img.affine


def scheme_for(bvals_mm2, bvecs):
    """DiSCo's timings per shell (the replay paper's protocol block); b = 0 and b = 1000 take shell 1925's."""
    from dmipy_fit.core.acquisition_scheme import acquisition_scheme_from_bvalues
    b_round = np.round(bvals_mm2)
    delta = np.full(bvals_mm2.shape, 10.2e-3)
    Delta = np.full(bvals_mm2.shape, 16.7e-3)
    delta[b_round == 3094] = 7.6e-3
    Delta[b_round == 3094] = 45.9e-3
    delta[b_round == 13192] = 17.7e-3
    Delta[b_round == 13192] = 35.8e-3
    TE = np.full(bvals_mm2.shape, 0.0535)
    return acquisition_scheme_from_bvalues(bvals_mm2 * 1e6, bvecs, delta=delta, Delta=Delta, TE=TE, b0_threshold=10e6)


def csd(data, scheme, mask, solver):
    """The FOD field, ``(X, Y, Z, 45)`` in the tournier07 basis, from the single-fibre response of the volume."""
    from dmipy_fit.core.modeling_framework import MultiCompartmentSphericalHarmonicsModel
    from dmipy_fit.tissue_response.white_matter_response import white_matter_response_tournier07
    data2d = data[mask]
    S0_wm, response, _ = white_matter_response_tournier07(scheme, data2d)
    mc = MultiCompartmentSphericalHarmonicsModel(models=[response], sh_order=SH_ORDER)
    t0 = time.perf_counter()
    fitted = mc.fit(scheme, data, mask=mask, solver=solver, verbose=False)
    dt = time.perf_counter() - t0
    return np.asarray(fitted.fitted_parameters['sh_coeff'], np.float64), dt


def score(M, ref):
    gt_count = np.loadtxt(f"{ref}/DiSCo_Connectivity_Matrix_Strands_Count.txt")
    gt_area = np.loadtxt(f"{ref}/DiSCo_Connectivity_Matrix_Cross-Sectional_Area.txt")
    iu = np.triu_indices(16, 1)
    return dict(pearson_count=float(np.corrcoef(M[iu], gt_count[iu])[0, 1]),
                pearson_area=float(np.corrcoef(M[iu], gt_area[iu])[0, 1]),
                connected_pairs=int((M[iu] > 0).sum()), gt_pairs=int((gt_count[iu] > 0).sum()),
                false_pairs=int(((M[iu] > 0) & (gt_count[iu] == 0)).sum()),
                missed_pairs=int(((M[iu] == 0) & (gt_count[iu] > 0)).sum()))


def symmetric_counts(matrix):
    M = matrix[1:17, 1:17].astype(np.float64)
    M = M + M.T
    np.fill_diagonal(M, 0)
    return M


def run_ours(field, seeds, a):
    from dmipy_tract import track
    kw = dict(rule=a.rule, step_mm=a.step, max_angle=a.max_angle, max_steps=a.max_steps, key=a.key)
    t0 = time.perf_counter()
    tg = track(field, seeds, **kw)
    first = time.perf_counter() - t0                # includes the compile of this seed count's chunk shape
    t0 = time.perf_counter()
    tg = track(field, seeds, **kw)
    dt = time.perf_counter() - t0
    return tg, dict(first_call_seconds=first, track_seconds=dt)


def run_dipy(field, seeds, a):
    from dipy.data import default_sphere
    from dipy.direction import ProbabilisticDirectionGetter, DeterministicMaximumDirectionGetter
    from dipy.tracking.local_tracking import LocalTracking
    from dipy.tracking.stopping_criterion import BinaryStoppingCriterion
    from dipy.tracking.streamline import Streamlines
    getter = ProbabilisticDirectionGetter if a.rule == 'probabilistic' else DeterministicMaximumDirectionGetter
    dg = getter.from_shcoeff(field.sh.astype(np.float64), max_angle=a.max_angle, sphere=default_sphere,
                             basis_type='tournier07', legacy=False)
    t0 = time.perf_counter()
    sl = Streamlines(LocalTracking(dg, BinaryStoppingCriterion(field.mask), seeds, field.affine, step_size=a.step,
                                   maxlen=a.max_steps - 1, random_seed=a.key))
    return sl, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dwi"); ap.add_argument("--bvals"); ap.add_argument("--bvecs")
    ap.add_argument("--ref", default=REF_DEFAULT)
    ap.add_argument("--solver", default="csd_tournier07_jax")
    ap.add_argument("--rule", default="probabilistic")
    ap.add_argument("--density", type=int, default=4)
    ap.add_argument("--step", type=float, default=0.5)
    ap.add_argument("--max-angle", type=float, default=30.0)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--key", type=int, default=0)
    ap.add_argument("--dipy", action="store_true")
    ap.add_argument("--tck", default=None, help="write the tractogram here")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "disco.json"))
    a = ap.parse_args()
    import nibabel as nib
    import jax
    from dmipy_tract import FODField, seeds_from_mask, connectivity
    t_all = time.perf_counter()
    data, bvals, bvecs, affine = load_volume(a.dwi, a.bvals, a.bvecs)
    mask = nib.load(f"{a.ref}/DiSCo_mask.nii.gz").get_fdata() > 0
    rois = nib.load(f"{a.ref}/DiSCo_ROIs.nii.gz").get_fdata().astype(np.int32)
    signal = data[..., 0] > 0
    scheme = scheme_for(bvals, bvecs)
    sh, t_csd = csd(data, scheme, signal, a.solver)
    field = FODField(sh, np.eye(4), mask | (rois > 0))
    seeds = seeds_from_mask(rois > 0, np.eye(4), density=a.density)
    tg, timing = run_ours(field, seeds, a)
    matrix, _ = connectivity(tg, rois, np.eye(4))
    M = symmetric_counts(matrix)
    out = dict(device=str(jax.devices()[0]).split(':')[0], solver=a.solver, rule=a.rule, density=a.density,
               seeds=int(len(seeds)), streamlines=int(len(tg)), points=int(tg.points.shape[0]),
               n_points_mean=float(tg.n_points.mean()), csd_seconds=t_csd, **timing, score=score(M, a.ref),
               stop_reasons=np.bincount(tg.stop_reason.ravel(), minlength=5).tolist(), matrix=M.tolist())
    print(f"ours ({out['device']}): Pearson vs strand count {out['score']['pearson_count']:.3f}, vs area "
          f"{out['score']['pearson_area']:.3f}; {out['streamlines']} streamlines from {out['seeds']} seeds, "
          f"{out['score']['connected_pairs']}/{out['score']['gt_pairs']} pairs ({out['score']['false_pairs']} false, "
          f"{out['score']['missed_pairs']} missed); csd {t_csd:.1f} s, tracking {timing['track_seconds']:.2f} s "
          f"(first call with compile {timing['first_call_seconds']:.1f} s)", flush=True)
    if a.tck:
        tg.to_tck(a.tck)
    if a.dipy:
        from dipy.tracking.utils import connectivity_matrix
        sl, t_dipy = run_dipy(field, seeds, a)
        Md = connectivity_matrix(sl, np.eye(4), rois, return_mapping=False)[1:, 1:].astype(np.float64)
        Md = Md + Md.T
        np.fill_diagonal(Md, 0)
        out['dipy'] = dict(streamlines=int(len(sl)), track_seconds=t_dipy, score=score(Md, a.ref), matrix=Md.tolist())
        iu = np.triu_indices(16, 1)
        out['dipy']['pearson_ours_vs_dipy'] = float(np.corrcoef(M[iu], Md[iu])[0, 1])
        print(f"dipy on the same FOD: Pearson vs strand count {out['dipy']['score']['pearson_count']:.3f}, vs area "
              f"{out['dipy']['score']['pearson_area']:.3f}; {len(sl)} streamlines; {t_dipy:.0f} s; "
              f"ours vs dipy matrix Pearson {out['dipy']['pearson_ours_vs_dipy']:.3f}", flush=True)
    out['total_seconds'] = time.perf_counter() - t_all
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
