import numpy as np


def f_d0(da, db, xa, xb):
    if da == db:
        return 6 if xa != xb else 1
    else:
        return 12 if xa != xb else 4


def f_dn(d, da, db, xa, xb):
    t = 24 * (d - 1)
    if da == db:
        return 30 + t if xa != xb else 18 + t  # 24+6, 12+6
    else:
        return 36 + t if xa != xb else 24 + t  # 24+12, 12+12


def map_d(d, da, db, xa, xb):
    if d == 0:
        return f_d0(da, db, xa, xb)
    else:
        return f_dn(d, da, db, xa, xb)


def map_n(n: np.uint8):
    if n == 4:
        return 24
    elif n == 3:
        return 8
    elif n == 2:
        return 6
    else:
        return 1


def initial_check(data):
    assert isinstance(data, (np.ndarray, np.generic)), "Input data must be a `np.ndarray`. " \
                                                       "Please check the input datatype."
    assert len(data.shape) == 2, f"Input array must have TWO dimensions. Yours have dim {len(data.shape)}."
    assert data.shape[0] >= 2, "The number of elements of axis-0 must exceed 2. Please check the shape of the array."
    assert data.shape[1] >= 2, "The number of elements of axis-1 must exceed 2. Please check the shape of the array."
    assert np.min(data) >= 0, "Currently, only non-negative integer arrays can be used to calculate the Boltzmann " \
                              "Entropy. If the negative value is determined to be an outlier, " \
                              "you can correct it with `data[data<0]=0`."
    assert np.issubdtype(data.dtype, np.integer), "Currently, only non-negative integer arrays can be used to " \
                                                  "calculate the Boltzmann Entropy. If your data range is between " \
                                                  "0-1, you may consider multiplying the data array by 10 or 100 " \
                                                  "and taking it to integer, or extending it to an integer range " \
                                                  "of 0-255. If you are fairly certain that the array can be safely " \
                                                  "converted to the integer, please use `data=data.astype(‘int’)`."
    return 1
