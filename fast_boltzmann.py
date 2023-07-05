"""
Fast computation of the Boltzmann Entropy of a numerical or nominal 2-D data array.
Author: Yuxuan YE
Date: 2023/07/05
Email: yuxuanye145@gmail.com
"""

import numpy as np
import math
import warnings


def f_d0(da, db, xa, xb):
    if da == db:
        return 6 if xa != xb else 1
    else:
        return 12 if xa != xb else 4


def f_dn(d, da, db, xa, xb):
    factor = 24 * (d - 1)
    if da == db:
        return 30 + factor if xa != xb else 18 + factor  # 24+6, 12+6
    else:
        return 36 + factor if xa != xb else 24 + factor  # 24+12, 12+12


def get_d(d, da, db, xa, xb):
    if d == 0:
        return f_d0(da, db, xa, xb)
    else:
        return f_dn(d, da, db, xa, xb)


def map_state_num(value: np.uint8):
    if value == 4:
        return 24
    elif value == 3:
        return 8
    elif value == 2:
        return 6
    else:
        return 1


class BoltzNumerical:
    """
    Fast calculation of the Boltzmann Entropy of a 2-D numerical data array.

    Example1: Calculate the Boltzmann Entropy of a user-defined array.
    --------
    >>> import fast_boltzmann as fb
    >>> import numpy as np
    >>> data = np.array([[1,2,3,4],[11,12,13,14],[5,6,7,8],[25,26,27,28]])
    >>> data
    array([[ 1,  2,  3,  4],
           [11, 12, 13, 14],
           [ 5,  6,  7,  8],
           [25, 26, 27, 28]])
    >>> bnu = fb.BoltzNumerical(data)
    >>> bnu.data.shape
    (4, 4)
    >>> entropy_aggregation = bnu.get_boltzmann_numerical(method='aggregation', is_relative=True, is_normalize=True)
    >>> entropy_aggregation
    1.8649743036048947
    >>> entropy_resampling = bnu.get_boltzmann_numerical(method='resampling', is_relative=True, is_normalize=True)
    >>> entropy_resampling
    3.9759743714440137
    >>> entropy_agg_absolute = bnu.get_boltzmann_numerical(method='aggregation', is_relative=False, is_normalize=True)
    >>> entropy_agg_absolute
    2.3092570821535303
    >>> bnu.data.shape
    (1, 1)

    Example2: Calculate the Boltzmann Entropy on the first band of a real-world raster image
    --------
    >>> import fast_boltzmann as fb
    >>> from osgeo import gdal
    >>> data = gdal.Open('./example/example1_image.tif').ReadAsArray()
    >>> b1 = data[0, :, :]
    >>> b1.shape
    (500, 500)
    >>> bnu = fb.BoltzNumerical(b1)
    >>> entropy = bnu.get_boltzmann_numerical(method='resampling', is_relative=True, is_normalize=True)
    >>> entropy
    8.60467561853229

    """

    def __init__(self, data, base=2):
        # a two-dimensional data array
        self.data = data.astype(int)
        # the logarithmic base for calculating Boltzmann Entropy
        self.base = base
        # a vectorized function to calculate the number of microstates
        self.vfunc = np.vectorize(get_d)
        self.size = self.data.size

    def iter_mat_sliding(self):
        """
        Calculate Boltzmann Entropy utilizing a 2*2 sliding window with 50% overlap.
        :return:
        1. The number of microstates;
        2. The mean matrix of all 2*2 sliding windows in `int` format
        """
        y = np.lib.stride_tricks.sliding_window_view(self.data, (2, 2))
        mean_mat = np.round(y.mean(axis=(-1, -2)))
        max_mat = y.max(axis=(-1, -2))
        min_mat = y.min(axis=(-1, -2))
        sum_mat = y.sum(axis=(-1, -2))
        temp = (sum_mat - max_mat - min_mat) / 2
        xa, xb = np.floor(temp), np.ceil(temp)
        da, db = xa - min_mat, max_mat - xb
        d = np.minimum(da, db)
        f_mat = self.vfunc(d, da, db, xa, xb)
        res_mat = np.log(f_mat) / math.log(self.base)
        return np.sum(res_mat.flatten()), mean_mat.astype(int)

    def iter_mat_chunk(self):
        """
        Calculate Boltzmann Entropy utilizing a 2*2 sliding window with no overlap.
        :return:
        1. The number of microstates;
        2. The mean matrix of all 2*2 sliding windows in `int` format
        """
        size = 2
        r, c = self.data.shape
        out_shape = [r // size, c // size] + [size, size]
        out_strides = [size * i for i in self.data.strides] + list(self.data.strides)
        y = np.lib.stride_tricks.as_strided(self.data, shape=out_shape, strides=out_strides)
        mean_mat = np.round(y.mean(axis=(-1, -2)))
        max_mat = y.max(axis=(-1, -2))
        min_mat = y.min(axis=(-1, -2))
        sum_mat = y.sum(axis=(-1, -2))
        temp = (sum_mat - max_mat - min_mat) / 2
        xa, xb = np.floor(temp), np.ceil(temp)
        da, db = xa - min_mat, max_mat - xb
        d = np.minimum(da, db)
        f_mat = self.vfunc(d, da, db, xa, xb)
        res_mat = np.log(f_mat) / math.log(self.base)
        return np.sum(res_mat.flatten()), mean_mat.astype(int)

    def get_boltzmann_numerical(self, method='resampling', is_relative=True, is_normalize=True):
        """
        Calculate the numerical Boltzmann Entropy.
        :param method: The method assigned to calculate Boltzmann Entropy. Support `resampling` and `aggregation`.
        :param is_relative: `True` for absolute Boltzmann Entropy, `False` for relative Boltzmann entropy.
        :param is_normalize: `True` for normalized result among all pixels, `False` for the entire result.
        :return: The calculated Boltzmann Entropy for numerical data.
        """
        assert len(self.data.shape) == 2, f"Input array must have dimension 2. Yours have dim {len(self.data.shape)}"
        be_list = []
        if method == 'resampling':
            while self.data.shape[0] > 1 and self.data.shape[1] > 1:
                num_be, update_mat = self.iter_mat_sliding()
                be_list.append(num_be)
                if is_relative:
                    return be_list[0] / self.size if is_normalize else be_list[0]
                else:
                    self.data = update_mat
            return sum(be_list) / self.size if is_normalize else sum(be_list)
        elif method == 'aggregation':
            while self.data.shape[0] > 1 and self.data.shape[1] > 1:
                if self.data.shape[0] % 2 == 1 or self.data.shape[1] % 2 == 1:
                    warnings.warn(
                        "The number of rows/columns of the array should be an integer multiple of 2 when calculated "
                        f"using the aggregation method. Yours have shape ({self.data.shape[0]}, {self.data.shape[1]})"
                        "The last row/column will be cut off.")
                num_be, update_mat = self.iter_mat_chunk()
                be_list.append(num_be)
                if is_relative:
                    return be_list[0] / self.size if is_normalize else be_list[0]
                else:
                    self.data = update_mat
            return sum(be_list) / self.size if is_normalize else sum(be_list)
        else:
            raise Exception("The `method` parameter only supports `resampling` and `aggregation` options. "
                            "Please check the input parameter.")


class BoltzNominal:
    """
    Fast calculation of the Boltzmann Entropy of a 2-D nominal data array.

    Example1: Calculate the Boltzmann Entropy of a user-definded array
    --------
    >>> import fast_boltzmann as fb
    >>> import numpy as np
    >>> data = np.array([[1,1,1,2], [2,3,3,4], [1,3,1,1], [2,2,1,1]])
    >>> data
    array([[1, 1, 1, 2],
           [2, 3, 3, 4],
           [1, 3, 1, 1],
           [2, 2, 1, 1]])
    >>> bno = fb.BoltzNominal(data)
    >>> entropy = bno.get_boltzmann_nominal(is_normalize_category=True, is_normalize_size=True)
    >>> entropy
    1.203125

    Example2: Calculate the Boltzmann Entropy of a real-world land cover data
    --------
    >>> import fast_boltzmann as fb
    >>> from osgeo import gdal
    >>> data = gdal.Open('./example/example2_classification.tif').ReadAsArray()
    >>> data.shape
    (1024, 1024)
    >>> bno = fb.BoltzNominal(data)
    >>> entropy = bno.get_boltzmann_nominal(is_normalize_category=True, is_normalize_size=True)
    >>> entropy
    0.3480114255632673

    """

    def __init__(self, data, base=2):
        self.data = data.astype(int)
        self.base = base
        self.vmap = np.vectorize(map_state_num)
        self.size = self.data.size
        self.category_num = len(np.unique(self.data))

    def iter_slide_uni_stride(self, window=(2, 2)):
        """
        Calculate the number of unique values in each sliding window of the input 2D array.
        :param window: the sliding window size, default for 2*2.
        :return: An array containing the number of microstates in each sliding window.
        """
        s = np.prod(window)
        out_size = np.asarray(self.data.shape) - window + 1
        m, n = self.data.strides
        col = out_size[1]
        out_mat = np.empty(out_size, dtype=np.uint8)  # number of unique values in each sliding window
        for i in range(out_size[0]):
            slide_sd = np.lib.stride_tricks.as_strided(self.data[i], shape=((col,) + tuple(window)), strides=(n, m, n))
            si = np.sort(slide_sd.reshape(-1, s), -1)
            out_mat[i] = (si[:, 1:] != si[:, :-1]).sum(-1) + 1
        return self.vmap(out_mat)

    def get_boltzmann_nominal(self, is_normalize_category=True, is_normalize_size=True):
        """
        Calculate the nominal Boltzmann Entropy.
        :param is_normalize_category: `True` for normalizing among all categories, `False` for not normalizing.
        :param is_normalize_size: `True` for normalizing among pixel numbers, `False` for not normalizing.
        :return: The calculated Boltzmann Entropy for nominal data.
        """
        if self.category_num > 30:
            warnings.warn("The function aims to calculate the Boltzmann entropy for nominal data such as land use/cover"
                          "data, please ensure the input data meets the requirement.")
        sn_mat = self.iter_slide_uni_stride()
        nomi_be = np.sum(sn_mat)
        if is_normalize_category:
            nomi_be /= self.category_num
        if is_normalize_size:
            nomi_be /= self.size
        return nomi_be
