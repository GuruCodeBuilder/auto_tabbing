import numpy as np


def trim_CQT(cqt_data, top: int = 20) -> np.ndarray:
    """Trims CQT data to the top `top` frequencies.

    Args:
        cqt_data (np.ndarray): CQT data to trim.
        top (int, optional): Number of frequencies to use. Defaults to 5.

    Returns:
        np.ndarray: Trimmed CQT data in format [freq1, fre2, ..., freq5] where the freq is represented as its list of magnitude over time
                    Example: the frequence represnted by f = [a, b, c, d] has 4 time bins, and a, b, c, and d are complex numbers
                    np.abs(f) represents the same frequency but with the magnitudes of those complex numbers
                    now: f = [||a||, ||b||, ||c||, ||d||], a list of real numbers.
                    Over time, the frequency moves through the magnitudes till the end
    """
    # Sort the data by the sum of the frequencies
    _cqt_data_first_freqs = np.array([i[:30] for i in cqt_data])  # shape = (288, 30)
    cqt_data_initial_mags = np.ndarray(shape=(288, 15), buffer=_cqt_data_first_freqs)

    # The mean and sum of the first 30 bins respectively
    mag_mean = cqt_data_initial_mags.mean(axis=1)
    sorted_indices_mean = np.argsort(mag_mean)[::-1]

    # the indexes sorted such that the mean and sum of each frequencies over the first 30 bins are sorted from greatest to least
    mag_sum = cqt_data.sum(axis=1)
    sorted_indices_sum = np.argsort(mag_sum)[::-1]

    # Take the top `top` frequencies
    top_indices_mean = sorted_indices_mean[:top]
    top_indices_sum = sorted_indices_sum[:top]

    trimmed_arrays = (cqt_data[top_indices_mean], cqt_data[top_indices_sum])

    same = np.array_equal(*trimmed_arrays)

    return trimmed_arrays[0], trimmed_arrays[1], same
