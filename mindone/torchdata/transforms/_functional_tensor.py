from typing import List

import numpy as np


def normalize(tensor: np.ndarray, mean: List[float], std: List[float], inplace: bool = False) -> np.ndarray:
    if tensor.dtype not in (np.float16, np.float32, np.float64):
        raise TypeError(f"Input tensor should be a float tensor. Got {tensor.dtype}.")

    if tensor.ndim < 3:
        raise ValueError(
            f"Expected tensor to be a tensor image of size (..., C, H, W). Got tensor.shape = {tensor.shape}"
        )

    dtype = tensor.dtype
    mean = np.array(mean, dtype=dtype)
    std = np.array(std, dtype=dtype)
    if np.any(std == 0):
        raise ValueError(f"std evaluated to zero after conversion to {dtype}, leading to division by zero.")
    if mean.ndim == 1:
        mean = mean.reshape((-1, 1, 1))
    if std.ndim == 1:
        std = std.reshape((-1, 1, 1))
    return (tensor - mean) / std
