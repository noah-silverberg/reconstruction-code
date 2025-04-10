"""
ecg_resp.py

Functions for ECG and respiratory signal processing, including R-peak and resp peak detection,
heart rate calculations, and data plotting. Uses NeuroKit2 for some detections.
"""

import numpy as np
import neurokit2 as nk
import matplotlib.pyplot as plt
import scipy.signal as signal


def detect_r_peaks(ecg_data, fs):
    """
    Detect R-peaks in multi-channel ECG data.

    Parameters
    ----------
    ecg_data : np.ndarray
        ECG data of shape (n_samples, n_channels) or (n_samples,).
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    list of np.ndarray
        List of arrays, one per channel, containing R-peak indices.
    """
    if ecg_data.ndim == 1:
        ecg_data = ecg_data.reshape(-1, 1)

    r_peaks_list = []
    for channel in ecg_data.T:
        channel_clean = nk.ecg_clean(channel, sampling_rate=fs)
        _, info = nk.ecg_peaks(channel_clean, sampling_rate=fs)
        # _, info = nk.ecg_peaks(channel, sampling_rate=fs)
        r_peaks = info["ECG_R_Peaks"]
        r_peaks_list.append(np.array(r_peaks))

    return r_peaks_list


def load_and_resample_events(events_file, resampled_length):
    """
    Load and resample event data (e.g., R-peak triggers) to match a target length.

    Parameters
    ----------
    events_file : str
        Path to a text file with event data, skipping the header row,
        reading the second column (usecols=1) as event amplitude.
    resampled_length : int
        Desired length of the resampled event array.

    Returns
    -------
    np.ndarray
        Resampled event array of length `resampled_length`.
        Non-zero entries indicate an event.
    """
    raw_events = np.loadtxt(events_file, skiprows=1, usecols=1)
    raw_events = raw_events - np.min(raw_events)
    raw_length = len(raw_events)

    resampled_events = np.zeros(resampled_length)
    raw_spike_indices = np.nonzero(raw_events)[0]
    resampled_spike_indices = np.round(
        raw_spike_indices / (raw_length - 1) * (resampled_length - 1)
    ).astype(int)

    for raw_idx, resampled_idx in zip(raw_spike_indices, resampled_spike_indices):
        resampled_events[resampled_idx] = raw_events[raw_idx]

    return resampled_events


def load_and_resample_resp(resp_file, resampled_length):
    """
    Load and resample respiratory data to a desired length.

    Parameters
    ----------
    resp_file : str
        Path to a text file with respiratory data in the second column (usecols=1).
    resampled_length : int
        Target length for resampling.

    Returns
    -------
    np.ndarray
        Resampled respiratory data of shape (resampled_length, 1).
    """
    resp_data = np.loadtxt(resp_file, skiprows=1, usecols=1)
    resp_data = signal.resample(resp_data, resampled_length)[:, np.newaxis]
    return resp_data


def detect_resp_peaks(resp_data, fs, method="nk", height=None, prominence=None):
    """
    Detect respiratory peaks in a 1D signal.

    Parameters
    ----------
    resp_data : np.ndarray
        Respiratory signal of shape (n_samples,) or (n_samples, 1).
    fs : float
        Sampling frequency in Hz.
    method : str
        Detection method: "nk" (NeuroKit2) or "scipy".
    height : float, optional
        Minimum peak height (only for "scipy" method).
    prominence : float, optional
        Minimum peak prominence (only for "scipy" method).

    Returns
    -------
    np.ndarray
        Indices of detected respiratory peaks.
    """
    if resp_data.ndim > 1:
        resp_data = resp_data.flatten()

    if method == "nk":
        # Using NeuroKit2
        _, info = nk.rsp_process(resp_data, sampling_rate=fs)
        resp_peaks = info["RSP_Peaks"]
        return np.array(resp_peaks)

    # Otherwise, use scipy's find_peaks
    norm_resp = (resp_data - np.min(resp_data)) / (np.ptp(resp_data) + 1e-9)
    peaks, _ = signal.find_peaks(norm_resp, height=height, prominence=prominence)
    return peaks


def compute_average_heart_rate(r_peaks_list, fs):
    """
    Compute the average heart rate (in BPM) from R-peaks across channels.

    Parameters
    ----------
    r_peaks_list : list of np.ndarray
        Each element contains R-peak indices for a given ECG channel.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    float
        Average heart rate in beats per minute (BPM).
    """
    all_r_peaks = np.vstack(r_peaks_list)
    rr_intervals = np.diff(all_r_peaks) / fs
    return 60.0 / np.mean(rr_intervals)


def plot_ecg_signals(
    ecg_data,
    fs,
    r_peaks_list=None,
    spike_indices=None,
    mode="separate",
    channel_labels=None,
):
    """
    Plot ECG signals over time, optionally marking R-peaks or other spike events.

    Parameters
    ----------
    ecg_data : np.ndarray
        ECG data of shape (n_samples, n_channels).
    fs : float
        Sampling frequency in Hz.
    r_peaks_list : list of np.ndarray, optional
        List of R-peak index arrays, one per channel.
    spike_indices : array-like, optional
        Indices of spike events (e.g. from event file).
    mode : {"separate", "together"}
        Plot each channel separately or all channels on one axis.
    channel_labels : list of str, optional
        Labels for each channel. If None, generic labels are used.

    Returns
    -------
    None
    """
    n_samples, n_channels = ecg_data.shape
    t = np.arange(n_samples) / fs

    if channel_labels is None:
        channel_labels = [f"Channel {i+1}" for i in range(n_channels)]

    spike_times = None
    if spike_indices is not None:
        spike_times = np.array(spike_indices) / fs

    if mode == "separate":
        plt.figure(figsize=(12, 3 * n_channels))
        for i in range(n_channels):
            ax = plt.subplot(n_channels, 1, i + 1)
            ax.plot(t, ecg_data[:, i], label=channel_labels[i])
            if r_peaks_list is not None:
                ax.plot(
                    t[r_peaks_list[i]],
                    ecg_data[r_peaks_list[i], i],
                    "ro",
                    label="R-peaks",
                )
            if spike_times is not None:
                y_limits = ax.get_ylim()
                ax.vlines(
                    spike_times,
                    y_limits[0],
                    y_limits[1],
                    colors="gray",
                    linestyles="--",
                    linewidth=1,
                )
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
            ax.set_title(channel_labels[i])
            ax.legend(loc="lower right")
        plt.tight_layout()

    elif mode == "together":
        plt.figure(figsize=(12, 6))
        ax = plt.gca()
        for i in range(n_channels):
            ax.plot(t, ecg_data[:, i], label=channel_labels[i])
            if r_peaks_list is not None:
                ax.plot(
                    t[r_peaks_list[i]],
                    ecg_data[r_peaks_list[i], i],
                    "o",
                    label=f"{channel_labels[i]} R-peaks",
                )
        if spike_times is not None:
            y_limits = ax.get_ylim()
            ax.vlines(
                spike_times,
                y_limits[0],
                y_limits[1],
                colors="gray",
                linestyles="--",
                linewidth=1,
            )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title("ECG Signals (Combined)")
        ax.legend()
        plt.tight_layout()

    else:
        raise ValueError("Invalid mode. Choose either 'separate' or 'together'.")

    plt.show()


def plot_resp_signal(resp_signal, fs, resp_peaks=None, resp_troughs=None):
    """
    Plot respiratory signal over time, optionally marking peaks and troughs.

    Parameters
    ----------
    resp_signal : np.ndarray
        1D respiratory signal of length n_samples.
    fs : float
        Sampling frequency in Hz.
    resp_peaks : array-like, optional
        Indices of detected respiratory peaks.
    resp_troughs : array-like, optional
        Indices of detected respiratory troughs.

    Returns
    -------
    None
    """
    t = np.arange(len(resp_signal)) / fs

    plt.figure(figsize=(12, 6))
    plt.plot(t, resp_signal, color="gray", alpha=0.8, label="Resp Signal")

    if resp_peaks is not None:
        plt.plot(
            t[np.array(resp_peaks)],
            resp_signal[np.array(resp_peaks)],
            "ro",
            markersize=5,
            label="Resp Peaks",
        )
    if resp_troughs is not None:
        plt.plot(
            t[np.array(resp_troughs)],
            resp_signal[np.array(resp_troughs)],
            "bo",
            markersize=5,
            label="Resp Troughs",
        )

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Respiratory Signal")
    plt.legend()
    plt.grid(True)
    plt.show()
