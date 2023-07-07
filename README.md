# Welcome to `fast_boltzmann`
`fast_boltzmann` is a Python package developed for the fast computation of the Boltzmann Entropy (also known as configurational entropy) for a two-dimensional numerical or nominal data array. This package focuses on streamlining the sliding window-based computation procedure to achieve lightweight, fast, and user-friendly calculations. It does this by utilizing matrix calculations and vectorized functions.

## Installation

You can easily install this package using `pip`:

```bash
pip install fast_boltzmann
```

## Dependencies

- numpy (>= 1.20.0)

## Example usage

Calculate the Boltzmann Entropy of the first band and the average of all bands in a multi-spectral Landsat raster image.

```python
>>> import fast_boltzmann as fb
>>> from osgeo import gdal
>>> img_array = gdal.Open('./example/example_multispectral_image.tif').ReadAsArray()
>>> img_array.shape  # Check whether the data array has 2 dims.
(6, 500, 500)        # The data array has 3 dims. Extracting each band is required.
>>> img_array.dtype  # Check whether the datatype is integer.
dtype('uint16')      # The datatype is non-negative integer that meets the requirement.
>>> b1 = img_array[0, :, :]  # Take the first band as an example.
>>> b1_bnu = fb.BoltzNumerical(b1)
>>> BE_b1 = b1_bnu.get_boltzmann_numerical()  # resampling-based method, relative BE, normlize to all pixels
>>> round(BE_b1, 4)
8.6047
>>> res_list = [fb.BoltzNumerical(img_array[i, :, :]).get_boltzmann_numerical() for i in range(img_array.shape[0])]
>>> res_list  # Store BE values for all bands in a list
[8.60467561853229, 9.03175324743007, 9.853283840481888, 9.992600722500507, 10.293043710639683, 10.09777221607461]
>>> BE_img_average = sum(res_list)/img_array.shape[0] # Calculate the average BE value of the image
>>> round(BE_img_average, 4)
9.6455
```

## Limitation

Currently, only **non-negative integer** arrays can be used to calculate the Boltzmann Entropy. If your data range is between 0-1, you may consider multiplying the data array by 10 or 100 and taking it to integer, or extending it to an integer range of 0-255. If you are fairly certain that the array can be **safely** converted to the integer, please use `data=data.astype(‘int’)`. 

For the aggregation-based method, there may be dangling pixels that cannot be covered by the 2*2 window/chunk. The input data array may lose some features as these pixels are removed in the computation.

Please refer to the references listed below for more information.

## References
- Gao, P., Zhang, H., & Li, Z. (2017). A hierarchy-based solution to calculate the configurational entropy of landscape gradients. Landscape ecology, 32(6), 1133-1146.
- Gao, P., & Li, Z. (2019a). Aggregation-based method for computing absolute Boltzmann entropy of landscape gradient with full thermodynamic consistency. Landscape Ecology, 34(8), 1837-1847.
- Gao, P., & Li, Z. (2019b). Computation of the Boltzmann entropy of a landscape: A review and a generalization. Landscape Ecology, 34(9), 2183-2196.
