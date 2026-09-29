"""Streamline counts between labelled regions, from the two endpoints of every streamline."""
import numpy as np

__all__ = ['connectivity', 'endpoint_labels']


def endpoint_labels(tractogram, labels, affine):
    """``(n, 2)`` the label at the nearest voxel of each streamline's first and last point; 0 outside the grid."""
    labels = np.asarray(labels)
    if labels.ndim != 3 or not np.issubdtype(labels.dtype, np.integer) or labels.min() < 0:
        raise ValueError("labels must be a 3-D image of non-negative integers (0 = no region)")
    inv = np.linalg.inv(np.asarray(affine, np.float64))
    ends = tractogram.endpoints.astype(np.float64)                       # (n, 2, 3)
    v = np.rint(ends @ inv[:3, :3].T + inv[:3, 3]).astype(np.int64)
    dims = np.asarray(labels.shape)
    inside = np.all((v >= 0) & (v < dims), axis=-1)
    vc = np.clip(v, 0, dims - 1)
    out = labels[vc[..., 0], vc[..., 1], vc[..., 2]]
    out[~inside] = 0
    return out


def connectivity(tractogram, labels, affine):
    """``(matrix, end_labels)``: ``matrix[a, b]`` counts the streamlines whose first point lies in region ``a``
    and last point in region ``b`` (``(max_label + 1)^2``, row 0 / column 0 the endpoints in no region); ``end_labels``
    is :func:`endpoint_labels`. Symmetrise and drop the diagonal as the score at hand requires."""
    end = endpoint_labels(tractogram, labels, affine)
    k = int(np.asarray(labels).max()) + 1
    matrix = np.zeros((k, k), np.int64)
    np.add.at(matrix, (end[:, 0], end[:, 1]), 1)
    return matrix, end
