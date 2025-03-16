"""
ecg.py

This module provides functions for ECG signal processing, including R-peak detection,
heart rate computation, and plotting.
"""

import numpy as np
import neurokit2 as nk
import matplotlib.pyplot as plt


def detect_r_peaks(ecg_data, fs):
    """
    Detect R-peaks for multi-channel ECG data.

    Parameters:
        ecg_data (np.ndarray): 2D array (n_samples x n_channels).
        fs (float): Sampling frequency.

    Returns:
        list: List of numpy arrays with detected R-peak indices for each channel.
    """
    if ecg_data.ndim == 1:
        ecg_data = ecg_data.reshape(-1, 1)
    r_peaks_list = []
    for channel in ecg_data.T:
        _, info = nk.ecg_process(channel, sampling_rate=fs)
        r_peaks = info["ECG_R_Peaks"]
        r_peaks_list.append(np.array(r_peaks))
    return r_peaks_list


def compute_average_heart_rate(r_peaks_list, fs):
    """
    Compute the average heart rate (in BPM) from R-peaks.

    Parameters:
        r_peaks_list (list): List of numpy arrays of R-peak indices.
        fs (float): Sampling frequency.

    Returns:
        float: Average heart rate in BPM.
    """
    all_r_peaks = np.hstack(r_peaks_list)
    rr_intervals = np.diff(all_r_peaks) / fs
    return 60 / np.mean(rr_intervals)


def plot_ecg_signals(
    ecg_data, fs, r_peaks_list=None, mode="separate", channel_labels=None
):
    """
    Plot multi-channel ECG signals with detected R-peaks.

    Parameters:
        ecg_data (np.ndarray): 2D array (n_samples x n_channels).
        fs (float): Sampling frequency.
        r_peaks_list (list): Optional list of R-peak indices per channel.
        mode (str): "separate" or "together".
        channel_labels (list): Optional channel names.
    """
    n_samples, n_channels = ecg_data.shape
    t = np.arange(n_samples) / fs
    if channel_labels is None:
        channel_labels = [f"Channel {i+1}" for i in range(n_channels)]
    if mode == "separate":
        plt.figure(figsize=(12, 3 * n_channels))
        for i in range(n_channels):
            plt.subplot(n_channels, 1, i + 1)
            plt.plot(t, ecg_data[:, i], label=channel_labels[i])
            if r_peaks_list is not None:
                plt.plot(
                    t[r_peaks_list[i]],
                    ecg_data[r_peaks_list[i], i],
                    "ro",
                    label="R-peaks",
                )
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            plt.title(channel_labels[i])
            plt.legend()
        plt.tight_layout()
    elif mode == "together":
        plt.figure(figsize=(12, 6))
        for i in range(n_channels):
            plt.plot(t, ecg_data[:, i], label=channel_labels[i])
            if r_peaks_list is not None:
                plt.plot(
                    t[r_peaks_list[i]],
                    ecg_data[r_peaks_list[i], i],
                    "ro",
                    label=f"{channel_labels[i]} R-peaks",
                )
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.title("ECG Signals")
        plt.legend()
        plt.tight_layout()
    else:
        raise ValueError("Invalid mode.")
    plt.show()
