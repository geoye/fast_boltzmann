"""
Fast computation of the Boltzmann Entropy of a numerical or nominal 2-D data array.
Author: Yuxuan YE, Xinghua Cheng
Date: 2023/07/07
Version: 0.0.1
Email: yuxuanye145@gmail.com, cxh9791156936@gmail.com
"""

import numpy as np
import math
import warnings
from utils import map_d, map_n, initial_check


class BoltzNumerical:
    """
    Fast calculation of the Boltzmann Entropy of a 2-D numerical (raster) data array.

    Reference:
    Gao, P., Zhang, H., & Li, Z. (2017). A hierarchy-based solution to calculate the configurational entropy of
                                         landscape gradients. Landscape ecology, 32(6), 1133-1146.
    Gao, P., & Li, Z. (2019a). Aggregation-based method for computing absolute Boltzmann entropy of landscape
                               gradient with full thermodynamic consistency. Landscape Ecology, 34(8), 1837-1847.
    Gao, P., & Li, Z. (2019b). Computation of the Boltzmann entropy of a landscape: A review and a generalization.
                               Landscape Ecology, 34(9), 2183-2196.

    Example1: Calculate the Boltzmann Entropy of a user-defined 2D array using different methods and scaling options.
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
    >>> BE_agg_rela = bnu.get_boltzmann_numerical(method='aggregation', is_relative=True, is_normalize=True)
    >>> round(BE_agg_rela, 4)
    1.865
    >>> BE_resample_rela = bnu.get_boltzmann_numerical(method='resampling', is_relative=True, is_normalize=True)
    >>> round(BE_resample_rela, 4)
    3.976
    >>> BE_agg_abs = bnu.get_boltzmann_numerical(method='aggregation', is_relative=False, is_normalize=True)
    >>> round(BE_agg_abs, 4)
    2.3093
    >>> bnu.data.shape
    (1, 1)

    Example2: Calculate the Boltzmann Entropy of the first band and the average of all bands in a multi-spectral
    Landsat raster image.
    --------
    >>> import fast_boltzmann as fb
    >>> from osgeo import gdal
    >>> img_array = gdal.Open('../../example/example_multispectral_image.tif').ReadAsArray()
    >>> img_array.shape
    (6, 500, 500)
    >>> img_array.dtype
    dtype('uint16')
    >>> b1 = img_array[0, :, :]
    >>> b1_bnu = fb.BoltzNumerical(b1)
    >>> BE_b1 = b1_bnu.get_boltzmann_numerical(method='resampling', is_relative=True, is_normalize=True)
    >>> round(BE_b1, 4)
    8.6047
    >>> res_list = [fb.BoltzNumerical(img_array[i, :, :]).get_boltzmann_numerical() for i in range(img_array.shape[0])]
    >>> res_list
    [8.60467561853229, 9.03175324743007, 9.853283840481888, 9.992600722500507, 10.293043710639683, 10.09777221607461]
    >>> BE_img_average = sum(res_list)/img_array.shape[0]
    >>> round(BE_img_average, 4)
    9.6455
    """

    def __init__(self, data, base=2):
        initial_check(data)
        # The input two-dimensional data array
        self.data = data.astype(int)
        # The logarithmic base for calculating Boltzmann Entropy
        self.base = base
        self.size = self.data.size
        # A vectorized function to calculate the number of microstates
        self._vfunc = np.vectorize(map_d)

    def __repr__(self):
        return f"BoltzNumerical object with data shape `{self.data.shape}` and data type `{self.data.dtype}`"

    def _iter_mat_sliding(self):
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
        res_mat = np.log(self._vfunc(d, da, db, xa, xb)) / math.log(self.base)
        return np.sum(res_mat.flatten()), mean_mat.astype(int)

    def _iter_mat_chunk(self):
        """
        Calculate Boltzmann Entropy utilizing a 2*2 sliding window with no overlap.
        :return:
        1. The number of microstates;
        2. The mean matrix of all 2*2 sliding windows in `int` format
        """
        sz = 2
        r, c = self.data.shape
        out_shape = [r // sz, c // sz] + [sz, sz]
        out_strides = [sz * i for i in self.data.strides] + list(self.data.strides)
        y = np.lib.stride_tricks.as_strided(self.data, shape=out_shape, strides=out_strides)
        mean_mat = np.round(y.mean(axis=(-1, -2)))
        max_mat = y.max(axis=(-1, -2))
        min_mat = y.min(axis=(-1, -2))
        sum_mat = y.sum(axis=(-1, -2))
        temp = (sum_mat - max_mat - min_mat) / 2
        xa, xb = np.floor(temp), np.ceil(temp)
        da, db = xa - min_mat, max_mat - xb
        d = np.minimum(da, db)
        res_mat = np.log(self._vfunc(d, da, db, xa, xb)) / math.log(self.base)
        return np.sum(res_mat.flatten()), mean_mat.astype(int)

    def get_boltzmann_numerical(self, method='resampling', is_relative=True, is_normalize=True):
        """
        Calculate the numerical Boltzmann Entropy.
        :param method: The method assigned to calculate Boltzmann Entropy. Support `resampling` and `aggregation`.
        :param is_relative: `True` for absolute Boltzmann Entropy, `False` for relative Boltzmann entropy.
        :param is_normalize: `True` for normalized result among all pixels, `False` for the entire result.
        :return: The calculated Boltzmann Entropy for numerical data.
        """
        r, c = self.data.shape
        be_list = []
        if method == 'resampling':
            while r > 1 and c > 1:
                num_be, updated_mat = self._iter_mat_sliding()
                be_list.append(num_be)
                if is_relative:
                    return be_list[0] / self.size if is_normalize else be_list[0]
                else:
                    self.data = updated_mat
            return sum(be_list) / self.size if is_normalize else sum(be_list)
        elif method == 'aggregation':
            while r > 1 and c > 1:
                if r % 2 == 1 or c % 2 == 1:
                    warnings.warn(
                        "The number of rows/columns of the array should be an integer multiple of 2 when calculated "
                        f"using the aggregation method. Yours have shape ({r}, {c})"
                        "The last row/column will be cut off.")
                    ur, uc = (r // 2) * 2, (c // 2) * 2
                    self.size = ur * uc
                num_be, updated_mat = self._iter_mat_chunk()
                be_list.append(num_be)
                if is_relative:
                    return be_list[0] / self.size if is_normalize else be_list[0]
                else:
                    self.data = updated_mat
                    r, c = self.data.shape
            return sum(be_list) / self.size if is_normalize else sum(be_list)
        else:
            raise Exception("The `method` parameter only supports `resampling` and `aggregation` options. "
                            "Please check the input parameter.")


class BoltzNominal:
    """
    Fast calculation of the Boltzmann Entropy of a 2-D nominal (raster) data array.

    Reference:
    Gao, P., Zhang, H., & Li, Z. (2017). A hierarchy-based solution to calculate the configurational entropy of
                                         landscape gradients. Landscape ecology, 32(6), 1133-1146.
    Gao, P., & Li, Z. (2019a). Aggregation-based method for computing absolute Boltzmann entropy of landscape
                               gradient with full thermodynamic consistency. Landscape Ecology, 34(8), 1837-1847.
    Gao, P., & Li, Z. (2019b). Computation of the Boltzmann entropy of a landscape: A review and a generalization.
                               Landscape Ecology, 34(9), 2183-2196.

    Example1: Calculate the Boltzmann Entropy of a user-definded 4*4 array using the resampling method
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
    >>> BE_resampling = bno.get_boltzmann_nominal(method='resampling')
    >>> round(BE_resampling, 4)
    0.3868

    Example2: Calculate the Boltzmann Entropy of a user-definded 5*4 array using the aggregation method
    --------
    >>> import fast_boltzmann as fb
    >>> import numpy as np
    >>> data_54 = np.array([[1,1,1,2], [2,3,3,4], [1,3,1,1], [2,2,1,1], [1,2,1,1]])
    >>> data_54
    array([[1, 1, 1, 2],
           [2, 3, 3, 4],
           [1, 3, 1, 1],
           [2, 2, 1, 1],
           [1, 2, 1, 1]])
    >>> bno_54 = fb.BoltzNominal(data_54)
    >>> BE_agg = bno_54.get_boltzmann_nominal(method='aggregation')
    >>> round(BE_agg, 4)
    0.1654

    Example3: Calculate the Boltzmann Entropy of a real-world land cover raster data. Note that the data type is
    converted from `float32` to `int` before the calculation.
    --------
    >>> import fast_boltzmann as fb
    >>> from osgeo import gdal
    >>> img = gdal.Open('../../example/example_categorical_map.tif').ReadAsArray()
    >>> img.dtype
    dtype('float32')
    >>> np.unique(img)
    array([0., 1., 2., 3., 4., 5., 6.], dtype=float32)
    >>> img = img.astype('int')
    >>> img.shape
    (1024, 1024)
    >>> bno_img = fb.BoltzNominal(img)
    >>> BE_img_resampling = bno_img.get_boltzmann_nominal(method='resampling')
    >>> round(BE_img_resampling, 4)
    0.1042
    """

    def __init__(self, data, base=2):
        initial_check(data)
        # The input two-dimensional data array
        self.data = data.astype(int)
        # The logarithmic base for calculating Boltzmann Entropy
        self.base = base
        self.size = self.data.size
        self.category_num = len(np.unique(self.data))
        # A vectorized function to calculate the number of microstates
        self._vmap = np.vectorize(map_n)

    def __repr__(self):
        return f"BoltzNominal object with data shape `{self.data.shape}`, category number `{self.category_num}`, " \
               f"and data type `{self.data.dtype}`"

    def _slide_uni_resampling(self):
        """
        Calculate Boltzmann Entropy utilizing a 2*2 sliding window with 50% overlap.
        :return: An array containing the number of microstates in each sliding window.
        """
        window = (2, 2)
        s = np.prod(window)
        out_shape = np.asarray(self.data.shape) - window + 1
        m, n = self.data.strides
        col = out_shape[1]
        out_mat = np.empty(out_shape, dtype=np.uint8)  # number of unique values in each sliding window
        for i in range(out_shape[0]):
            slide_sd = np.lib.stride_tricks.as_strided(self.data[i], shape=((col,) + tuple(window)),
                                                       strides=(n, m, n))
            si = np.sort(slide_sd.reshape(-1, s), -1)
            out_mat[i] = (si[:, 1:] != si[:, :-1]).sum(-1) + 1
        return np.sum(np.log(self._vmap(out_mat)) / math.log(self.base))

    def _slide_uni_aggregation(self):
        """
        Calculate Boltzmann Entropy utilizing a 2*2 sliding window with no overlap.
        :return: An array containing the number of microstates in each sliding window.
        """
        sz = 2
        r, c = self.data.shape
        out_shape = [r // sz, c // sz] + [sz, sz]
        out_strides = [sz * i for i in self.data.strides] + list(self.data.strides)
        y = np.lib.stride_tricks.as_strided(self.data, shape=out_shape, strides=out_strides)
        g = y.reshape(y.size // sz ** 2, sz ** 2)
        k = np.apply_along_axis(lambda x: len(np.unique(x)), 1, g)
        return np.sum(np.log(self._vmap(k)) / math.log(self.base))

    def get_boltzmann_nominal(self, method='resampling', is_normalize_category=True, is_normalize_size=True):
        """
        Calculate the nominal Boltzmann Entropy.
        :param method: The method assigned to calculate Boltzmann Entropy. Support `resampling` and `aggregation`.
        :param is_normalize_category: `True` for normalizing among all categories, `False` for not normalizing.
        :param is_normalize_size: `True` for normalizing among pixel numbers, `False` for not normalizing.
        :return: The calculated Boltzmann Entropy for nominal data.
        """
        r, c = self.data.shape
        if self.category_num > 30:
            warnings.warn("The function aims to compute the Boltzmann Entropy for nominal data such as land use/cover"
                          f"data, the number of unique values of your data array is as large as {self.category_num}, "
                          "please ensure the input data meets the requirement.")
        if method == 'resampling':
            nomi_be = self._slide_uni_resampling()
        elif method == 'aggregation':
            if r % 2 == 1 or c % 2 == 1:
                warnings.warn(
                    "The number of rows/columns of the array should be an integer multiple of 2 when calculated "
                    f"using the aggregation method. Yours have shape ({r}, {c}). "
                    "The last row/column will be cut off.")
                ur, uc = (r // 2) * 2, (c // 2) * 2
                self.size = ur * uc
                self.category_num = len(np.unique(self.data[0:ur, 0:uc]))
            nomi_be = self._slide_uni_aggregation()
        else:
            raise Exception("The `method` parameter only supports `resampling` and `aggregation` options. "
                            "Please check the input parameter.")
        if is_normalize_category:
            nomi_be /= self.category_num
        if is_normalize_size:
            nomi_be /= self.size
        return nomi_be
