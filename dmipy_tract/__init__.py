"""dmipy-tract: streamline tractography on FOD fields, batched on the JAX device.

One data model (:class:`FODField`), one tracker (:func:`track`, probabilistic or deterministic), one output
(:class:`Tractogram`), seeds (:func:`seeds_from_mask`) and region counts (:func:`connectivity`). The mathematics
this shares with the rest of the ecosystem lives in dmipy-sim (the SH basis, the ``.tck`` and ``.mif`` files); dipy
is a test oracle, never imported here.
"""
from .field import FODField
from .tractogram import Tractogram, STOP_REASONS
from .tracker import track, RULES, BACKENDS
from .seeding import seeds_from_mask
from .connectivity import connectivity, endpoint_labels
from .sphere import hemisphere, sh_matrix

__all__ = ['FODField', 'Tractogram', 'STOP_REASONS', 'track', 'RULES', 'BACKENDS', 'seeds_from_mask', 'connectivity',
           'endpoint_labels', 'hemisphere', 'sh_matrix']
try:
    from importlib.metadata import version as _version
    __version__ = _version("dmipy-tract")           # pyproject.toml is the one source
except Exception:                                    # not installed (a source checkout on sys.path)
    __version__ = "unknown"
