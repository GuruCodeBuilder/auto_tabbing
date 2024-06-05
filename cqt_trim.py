from typing import Dict
import numpy as np


def trim_CQT(cqt_data, top: int = 5) -> np.ndarray:
    """Trims CQT data to the top `top` frequencies.

    Args:
        cqt_data (np.ndarray): CQT data to trim.
        top (int, optional): Number of frequencies to use. Defaults to 5.

    Returns:
        np.ndarray: Trimmed CQT data in format [[freq1, time1], [freq1, time1], ...]
    """
    # Sort the data by the sum of the frequencies
    cqt_data = np.abs(cqt_data)
    freq_sum = cqt_data.sum(axis=1)
    sorted_indices = np.argsort(freq_sum)[::-1]

    # Take the top `top` frequencies
    top_indices = sorted_indices[:top]
    return cqt_data[top_indices]
