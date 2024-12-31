import numpy as np
import scipy.signal as sp_sig
from typing import Tuple

def convert_datetime_to_time(datetime: np.ndarray, multi_day: bool = True) -> Tuple[np.ndarray, float]:
    """
    Converts timestamp vector to time in seconds and estimates the sampling frequency.

    This function converts the input vector of timestamps (DateTime standard for the ICM+ data collection protocol)
    to time in seconds, with the recording starting at time 0, and estimates the sampling frequency based
    on the time difference between consecutive samples.

    Args:
        datetime (numpy array): The one-dimensional vector of timestamps.
        multi_day (bool): The flag which describes if the recording spans multiple day (multi_day = True) or not
            (multi_day = False). If not set to True, the time vector for recordings spanning multiple days may be
            incorrectly calculated at points when the date changes.

    Returns:
        t_hat (numpy array): Estimated time vector.
        fs_hat (float): Estimated sampling frequency (in Hz).
    """

    if not multi_day:
        t0 = (datetime[0] - np.floor(datetime[0])) * 24 * 3600
        t_hat = np.squeeze((datetime - np.floor(datetime)) * 24 * 3600 - t0)
        fs_hat = round(1 / (t_hat[1] - t_hat[0]), 0)
    else:
        n_datetime = datetime - datetime[0]
        n_datetime_days = np.floor(n_datetime)
        c_datetime = n_datetime - n_datetime_days
        c_datetime_seconds = c_datetime * 24 * 3600

        t_hat = []
        for idx in range(0, len(datetime)):
            c_t = n_datetime_days[idx] * 24 * 3600 + c_datetime_seconds[idx]
            t_hat.append(c_t)
        t_hat = np.asarray(t_hat)
        fs_hat = round(1 / (t_hat[1] - t_hat[0]), 0)
    return t_hat, fs_hat


def filter_signal(signal: np.ndarray, fs: float, cutoff: float = 10) -> np.ndarray:
    """
    Performs lowpass filtering of the signal.

    This function filters the input signal of timestamps based on provided cutoff frequency. The default cutoff
    is 10 Hz (upper limit) to remove high-frequency noise.

    Args:
        signal (numpy array): The one-dimensional signal vector.
        fs (float): The sampling frequency of the signal (in Hz).
        cutoff (float): The cutoff frequency (in Hz).

    Returns:
        numpy array: The one-dimensional signal vector after filtering.
    """

    mean_signal = np.nanmean(signal)
    signal[np.argwhere(np.isnan(signal))] = mean_signal

    Wn1 = cutoff / (fs / 2)
    b1, a1 = sp_sig.iirfilter(N=8, Wn=Wn1, btype='lowpass', rs=60, rp=1, ftype='cheby1')
    filtered_signal = sp_sig.filtfilt(b1, a1, signal)

    return filtered_signal
