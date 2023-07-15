import numpy as np


def overlap(li: list):
    out = [v for v in li[0] if v in li[1]]
    if len(li) > 2:
        out = overlap([out] + li[2:])
    return out


def flatten_one(li):
    """Reduces depth of li of by 1"""
    out_li = []
    for v1 in li:
        if isinstance(v1, list):
            _ = [out_li.append(v2) for v2 in v1]
        else:
            out_li.append(v1)
    return out_li


def flatten_full(li):
    # Flatten any lists inside
    out_li = []
    for v in li:
        if isinstance(v, list):
            flat_v = flatten_one(v)
            # Check for internal lists
            for v2 in flat_v:
                if isinstance(v2, list):
                    _ = [out_li.append(v3) for v3 in flatten_full(v2)]
                else:
                    out_li.append(v2)
        else:
            out_li.append(v)
    return out_li


def np_count_values(np_array):
    x = np.array([1, 1, 1, 2, 2, 2, 5, 25, 1, 1])
    y = np.bincount(x)
    ii = np.nonzero(y)[0]
    return np.vstack((ii, y[ii])).T


def norm_np(arr, axis: int = 0, holdout: int = 5):
    v = arr[:, :-holdout]
    min_ = np.expand_dims(np.nanmin(v[v != -np.inf], axis=axis), axis=axis)
    max_ = np.expand_dims(np.nanmax(v[v != np.inf], axis=axis), axis=axis)
    arr = ((arr - min_) / (max_ - min_))
    return arr
