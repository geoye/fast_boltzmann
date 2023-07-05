# Welcome to `fast_boltzmann`
`fast_boltzmann` is a Python script designed to facilitate the fast computation of the Boltzmann Entropy (also known as configurational entropy) for a two-dimensional nominal or numerical data array. The script focuses on optimizing the sliding window-based computation process to ensure lightweight, fast, and user-friendly calculations. It achieves this by utilizing matrix calculations and vectorized functions of `numpy`.


## Example usage
A simple example to calculate the Boltzmann Entropy.
```python
import fast_boltzmann as fb
import numpy as np
data = np.array([[1,2,3,4],[11,12,13,14],[5,6,7,8],[25,26,27,28]])
bnu = fb.BoltzNumerical(data)
entropy_aggregation = bnu.get_boltzmann_numerical(method='aggregation', is_relative=True, is_normalize=True)
entropy_resampling = bnu.get_boltzmann_numerical(method='resampling', is_relative=True, is_normalize=True)
```

## Dependencies
- numpy (>= 1.20.0)

## References
- Gao, P., Zhang, H., & Li, Z. (2017). A hierarchy-based solution to calculate the configurational entropy of landscape gradients. Landscape ecology, 32(6), 1133-1146.
- Gao, P., & Li, Z. (2019a). Aggregation-based method for computing absolute Boltzmann entropy of landscape gradient with full thermodynamic consistency. Landscape Ecology, 34(8), 1837-1847.
- Gao, P., & Li, Z. (2019b). Computation of the Boltzmann entropy of a landscape: A review and a generalization. Landscape Ecology, 34(9), 2183-2196.
