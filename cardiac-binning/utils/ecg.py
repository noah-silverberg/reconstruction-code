import neurokit2 as nk
import numpy as np
import matplotlib.pyplot as plt


def detect_r_peaks(ecg_data, fs):
    """
    Detect R-peaks for multi-channel ECG data.

    Parameters:
      ecg_data (np.ndarray): 2D array of shape (n_samples, n_channels)
      fs (float): Sampling frequency.

    Returns:
      list of np.ndarray: A list where each element is an array of detected R-peak indices for a channel.
    """
    # Ensure ecg_data is 2D.
    if ecg_data.ndim == 1:
        ecg_data = ecg_data.reshape(-1, 1)

    r_peaks_list = []
    for channel in ecg_data.T:
        processed, info = nk.ecg_process(channel, sampling_rate=fs)
        r_peaks = info["ECG_R_Peaks"]
        r_peaks_list.append(np.array(r_peaks))
    return r_peaks_list


def average_heart_rate(r_peaks_list, fs):
    """
    Compute the average heart rate from R-peaks.

    Parameters:
      r_peaks_list (list of np.ndarray): A list of R-peak indices for each channel.
      fs (float): Sampling frequency.

    Returns:
      float: Average heart rate in beats per minute.
    """
    r_peaks = np.hstack(r_peaks_list)
    rr_intervals = np.diff(r_peaks) / fs
    heart_rate = 60 / np.mean(rr_intervals)
    return heart_rate


# def detect_rsp_peaks(rsp_data, fs):
#     """
#     Detect respiratory peaks for multi-channel respiratory data.

#     Parameters:
#       rsp_data (np.ndarray): 2D array of shape (n_samples, n_channels)
#       fs (float): Sampling frequency.

#     Returns:
#       list of np.ndarray: A list where each element is an array of detected respiratory peak indices for a channel.
#     """
#     # Ensure rsp_data is 2D.
#     if rsp_data.ndim == 1:
#         rsp_data = rsp_data.reshape(-1, 1)

#     rsp_peaks_list = []
#     for channel in rsp_data.T:
#         processed, info = nk.rsp_process(channel, sampling_rate=fs)
#         rsp_peaks = info["RSP_Peaks"]
#         rsp_peaks_list.append(np.array(rsp_peaks))
#     return rsp_peaks_list


def plot_ecg(ecg_data, fs, r_peaks_list=None, mode="separate", channel_labels=None):
    """
    Plot multi-channel ECG data with detected R-peaks.

    This function automatically computes the time axis using the provided sampling frequency.

    Parameters:
      ecg_data (np.ndarray): 2D array of shape (n_samples, n_channels).
      fs (float): Sampling frequency.
      r_peaks_list (list of np.ndarray, optional): List of R-peak indices for each channel.
      mode (str): "separate" to plot each channel in its own subplot,
                  "together" to overlay all channels in one plot.
      channel_labels (list of str, optional): Labels for each channel. Defaults to "Channel 1", "Channel 2", etc.
    """
    n_samples, n_channels = ecg_data.shape
    t = np.arange(n_samples) / fs  # Automatically computed time vector

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
                    "o",
                    label=f"{channel_labels[i]} R-peaks",
                )
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.title("ECG Channels")
        plt.legend()
        plt.tight_layout()
    else:
        raise ValueError("Mode must be either 'separate' or 'together'")

    plt.show()
