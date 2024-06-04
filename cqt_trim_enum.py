from enum import Enum

import librosa as lbr
import numpy as np


class CQTTrimMethod(Enum):
    """Configuration for different methods of trimming CQT data."""

    TIMESLICE = 1
    THRESHOLD = 2


# 2 ways to trim CQT data: by time slice or by threshold
def trim_CQT_timeslice(cqt_data, start_trim=0, end_trim=0):
    """
    Trims CQT data by removing a specified number of time slices from the beginning and end
    and returns a dictionary containing the trimmed data with durations for each frequency.

    Args:
        cqt_data (np.array): The CQT data to be trimmed.
        start_trim (int, optional): The number of slices to trim from the beginning. Defaults to 0.
        end_trim (int, optional): The number of slices to trim from the end. Defaults to 0.

    Returns:
        dict: A dictionary containing the trimmed data with durations in the format:
            {freq (float): duration (int)}.
    """
    trimmed_data = cqt_data[:, start_trim:-end_trim]
    frequencies = lbr.cqt_frequencies(
        trimmed_data.shape[0], fmin=lbr.note_to_hz("A2")
    )  # Assuming constant sr
    trimmed_freq_dict = {f: trimmed_data.shape[1] for f in frequencies}
    return trimmed_freq_dict


def trim_CQT_thresh(cqt_data, threshold=0.1):
    """
    Trims CQT data by removing frequencies with amplitudes below a threshold
    and returns a dictionary containing the trimmed data with durations.

    Args:
        cqt_data (np.array): The CQT data to be trimmed.
        threshold (float, optional): The minimum amplitude to keep. Defaults to 0.1.

    Returns:
        dict: A dictionary containing the trimmed data with durations in the format:
            {freq (float): duration (int)}.
    """
    trimmed_data = np.where(cqt_data > threshold, cqt_data, 0)
    # Get frequencies corresponding to the remaining data points
    frequencies = lbr.cqt_frequencies(
        cqt_data.shape[0], fmin=lbr.note_to_hz("A2"), sr=None
    )  # Assuming constant sr
    trimmed_freq_dict = {
        f: np.sum(trimmed_data[i] > 0)
        for i, f in enumerate(frequencies)
        if trimmed_data[i].any()
    }
    return trimmed_freq_dict


def trim_CQT(method: CQTTrimMethod, cqt_data, **kwargs):
    """Trim CQT data using the specified method.

    Args:
        method (CQTTrimMethod): Method to use for trimming.

    Returns:
        trimmed_freq_dict: The trimmed CQT data.
    """
    if CQTTrimMethod.TIMESLICE.value:
        return trim_CQT_timeslice(cqt_data, **kwargs)
    elif method == CQTTrimMethod.THRESHOLD.value:
        return trim_CQT_thresh(cqt_data, **kwargs)
    else:
        print(method.name)
        raise ValueError("Invalid CQTTrimMethod")
