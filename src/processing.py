from typing import List
import numpy as np


def normalize(values: List[float]) -> List[float]:
    arr = np.array(values, dtype=float)
    min_val = arr.min()
    max_val = arr.max()

    if max_val == min_val:
        return [0.0 for _ in arr]

    return ((arr - min_val) / (max_val - min_val)).tolist()


def remove_outliers(values: List[float], threshold: float = 2.5) -> List[float]:
    arr = np.array(values, dtype=float)
    mean = arr.mean()
    std = arr.std()

    if std == 0:
        return values

    z_scores = (arr - mean) / std
    return arr[abs(z_scores) < threshold].tolist()
