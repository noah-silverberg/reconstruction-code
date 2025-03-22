"""
ecg.py

This module provides functions for ECG signal processing, including R-peak detection,
heart rate computation, and plotting.
"""

import numpy as np
import neurokit2 as nk
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA, FastICA
import scipy.signal as signal


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


def load_and_resample_events(events_file, resampled_length):
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
    resp_data = np.loadtxt(resp_file, skiprows=1, usecols=1)
    resp_data = signal.resample(resp_data, resampled_length)[:, np.newaxis]

    return resp_data


def detect_resp_peaks(resp_data, fs, method="nk", height=None, prominence=None):
    """
    Detect respiratory peaks for respiratory data.

    Parameters:
        resp_data (np.ndarray): 1D array of length n_samples.
        fs (float): Sampling frequency.
        method (str): "nk" for NeuroKit2's rsp_process, or "scipy" for scipy's signal_findpeaks.
        height (float, optional): Optional height of peaks (only for "scipy" method).
        prominence (float, optional): Optional prominence of peaks (only for "scipy" method).

    Returns:
        np.ndarray: Numpy array with detected respiratory peak indices.
    """
    if method == "nk":
        # Process the respiratory signal using NeuroKit2
        _, info = nk.rsp_process(resp_data, sampling_rate=fs)
        resp_peaks = info["RSP_Peaks"]
    else:
        # Min-max normalize the signal
        resp_data = (resp_data - np.min(resp_data)) / (
            np.max(resp_data) - np.min(resp_data)
        )
        # Use scipy's find_peaks to detect respiratory peaks
        resp_peaks, _ = signal.find_peaks(
            resp_data.flatten(), height=height, prominence=prominence
        )
    return np.array(resp_peaks)


def compute_average_heart_rate(r_peaks_list, fs):
    """
    Compute the average heart rate (in BPM) from R-peaks.

    Parameters:
        r_peaks_list (list): List of numpy arrays of R-peak indices.
        fs (float): Sampling frequency.

    Returns:
        float: Average heart rate in BPM.
    """
    all_r_peaks = np.vstack(r_peaks_list)
    rr_intervals = np.diff(all_r_peaks) / fs
    return 60 / np.mean(rr_intervals)


def plot_ecg_signals(
    ecg_data,
    fs,
    r_peaks_list=None,
    spike_indices=None,
    mode="separate",
    channel_labels=None,
):
    """
    Plot multi-channel ECG signals with detected R-peaks and optional spike indices.

    Parameters:
        ecg_data (np.ndarray): 2D array (n_samples x n_channels).
        fs (float): Sampling frequency.
        r_peaks_list (list): Optional list of R-peak indices per channel.
        spike_indices (list): Optional list of indices (alleged R-peaks) at which to draw vertical dashed lines.
        mode (str): "separate" or "together" for subplot layout.
        channel_labels (list): Optional channel names.
    """
    n_samples, n_channels = ecg_data.shape
    t = np.arange(n_samples) / fs

    if channel_labels is None:
        channel_labels = [f"Channel {i+1}" for i in range(n_channels)]

    # Convert spike indices to times only once if provided
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
            # Draw vertical dashed lines using vectorized plotting
            if spike_indices is not None:
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
            ax.legend()
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
                    "ro",
                    label=f"{channel_labels[i]} R-peaks",
                )
        # Draw vertical dashed lines using vectorized plotting
        if spike_indices is not None:
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
        ax.set_title("ECG Signals")
        ax.legend()
        plt.tight_layout()

    else:
        raise ValueError("Invalid mode. Choose either 'separate' or 'together'.")

    plt.show()


def plot_resp_signal(resp_signal, fs, resp_peaks=None, resp_troughs=None):
    """
    Plot a respiratory signal with detected peaks and troughs.

    Parameters:
        resp_signal (np.ndarray): 1D array of the respiratory signal.
        fs (float): Sampling frequency.
        resp_peaks (array-like, optional): Indices indicating detected respiratory peaks.
        resp_troughs (array-like, optional): Indices indicating detected respiratory troughs.
    """
    n_samples = len(resp_signal)
    t = np.arange(n_samples) / fs

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
    plt.tight_layout()
    plt.show()


# def ecg_perform_pca(ecg_data, n_components=None, center=True):
#     """
#     Perform standard PCA on ECG data using scikit-learn's PCA.

#     Parameters:
#         ecg_data (np.ndarray): shape (n_samples, n_channels).
#         n_components (int, optional): Number of principal components to keep.
#         center (bool): Whether to center the data (subtract mean per channel).

#     Returns:
#         tuple: (pca_model, X_pca, mean_vector)
#            - pca_model: Fitted sklearn PCA model.
#            - X_pca (np.ndarray): Transformed data, shape (n_samples, n_components).
#            - mean_vector (np.ndarray): Per-channel mean (only relevant if center=True).
#     """
#     # Optionally subtract mean across time from each channel
#     mean_vector = 0
#     if center:
#         mean_vector = np.mean(ecg_data, axis=0, keepdims=True)
#         ecg_data_centered = ecg_data - mean_vector
#     else:
#         ecg_data_centered = ecg_data

#     pca_model = PCA(n_components=n_components)
#     X_pca = pca_model.fit_transform(ecg_data_centered)
#     return pca_model, X_pca, mean_vector


# def ecg_perform_kernel_pca(ecg_data, kernel="rbf", sigma=1.0, n_components=None):
#     """
#     Perform kernel PCA on ECG data using an RBF (Gaussian) kernel.

#     Parameters:
#         ecg_data (np.ndarray): shape (n_samples, n_channels).
#         kernel (str): Kernel to use (e.g. 'rbf', 'poly', 'sigmoid').
#         sigma (float): Sigma parameter for RBF kernel (overrides gamma = 1/(2*sigma^2)).
#         n_components (int, optional): Number of kernel PCA components to keep.

#     Returns:
#         tuple: (kpca_model, X_kpca)
#            - kpca_model: Fitted sklearn KernelPCA model (with inverse_transform=True).
#            - X_kpca (np.ndarray): Transformed data, shape (n_samples, n_components).
#     """
#     # For kernel PCA, we typically specify gamma = 1 / (2*sigma^2)
#     gamma = 1.0 / (2.0 * sigma**2)

#     kpca_model = KernelPCA(
#         n_components=n_components,
#         kernel=kernel,
#         gamma=gamma,
#         fit_inverse_transform=True,
#         n_jobs=-1,
#     )
#     X_kpca = kpca_model.fit_transform(ecg_data)  # ECG is purely real
#     return kpca_model, X_kpca


# def ecg_perform_ica(ecg_data, n_components=None, max_iter=1000):
#     """
#     Perform ICA on ECG data using scikit-learn's FastICA.

#     Parameters:
#         ecg_data (np.ndarray): shape (n_samples, n_channels).
#         n_components (int, optional): Number of independent components to extract.
#         max_iter (int): Maximum number of iterations to allow in ICA.

#     Returns:
#         tuple: (ica_model, X_ica)
#            - ica_model: Fitted FastICA model.
#            - X_ica (np.ndarray): Independent components, shape (n_samples, n_components).
#     """
#     ica_model = FastICA(n_components=n_components, max_iter=max_iter, random_state=0)
#     X_ica = ica_model.fit_transform(ecg_data)
#     return ica_model, X_ica


# def reconstruct_ecg_from_components(
#     method, model, transformed_data, components_to_keep, mean_vector=None
# ):
#     """
#     Reconstruct ECG signals from selected PCA / kernel PCA / ICA components.

#     Parameters:
#         method (str): "pca", "kernel_pca", or "ica".
#         model:
#           - If PCA, an sklearn PCA object.
#           - If Kernel PCA, an sklearn KernelPCA object (fit with inverse_transform=True).
#           - If ICA, an sklearn FastICA object.
#         transformed_data (np.ndarray): The low-dimensional representation of shape (n_samples, n_components).
#         components_to_keep (int or list[int]): Indices of components to retain.
#         mean_vector (np.ndarray, optional): Per-channel mean (only for PCA if data was centered).

#     Returns:
#         np.ndarray: Reconstructed ECG signals with shape (n_samples, n_channels).
#     """
#     if isinstance(components_to_keep, int):
#         components_to_keep = [components_to_keep]

#     # Create a zeroed version of the transformed data, then fill only desired components
#     X_mod = np.zeros_like(transformed_data)
#     X_mod[:, components_to_keep] = transformed_data[:, components_to_keep]

#     if method == "pca":
#         # For scikit PCA, we can do inverse_transform on the partial data
#         # but we need to intercept the components_ / singular_values_ approach
#         # or rely on the official model's full transform. An easy approach:
#         #  1) transform to all components
#         #  2) zero out undesired ones
#         #  3) inverse_transform
#         # If we do that, we can simply do:
#         X_recon_centered = model.inverse_transform(X_mod)
#         # Add back mean
#         if mean_vector is not None:
#             X_recon = X_recon_centered + mean_vector
#         else:
#             X_recon = X_recon_centered

#     elif method == "kernel_pca":
#         # KernelPCA with inverse_transform=True
#         # Zero out undesired components, then inverse_transform
#         X_recon = model.inverse_transform(X_mod)

#     elif method == "ica":
#         # For FastICA, scikit-learn doesn't directly provide partial inverse transform,
#         # but we can do it manually:
#         #   X = S * A + mean
#         # where S is the source matrix (transformed_data) and A = model.mixing_
#         # If you zero out components, then do S' * A, you get partial reconstruction.
#         # There's also model.mean_ if the data was whitened with a mean removed.
#         mixing_ = model.mixing_
#         if hasattr(model, "mean_"):
#             data_mean = model.mean_
#         else:
#             data_mean = 0

#         # Reconstruct from partial S
#         X_recon = X_mod @ mixing_.T + data_mean

#     else:
#         raise ValueError(f"Unknown decomposition method: {method}")

#     return X_recon


# def plot_ecg_components_time_series_and_fft(X_data, sampling_rate=1.0, n_components=5):
#     """
#     Plot time series, FFT magnitude, and PSD for the first n_components of an ECG decomposition.
#     Analogous to the k-space version in pca.py.

#     Parameters:
#         X_data (np.ndarray): shape (n_samples, n_total_components).
#         sampling_rate (float): Sampling rate in Hz.
#         n_components (int): How many components to plot.
#     """
#     n_samples, total_comps = X_data.shape
#     n_to_plot = min(n_components, total_comps)
#     t = np.arange(n_samples) / sampling_rate

#     fig, axs = plt.subplots(n_to_plot, 3, figsize=(18, 3 * n_to_plot), squeeze=False)

#     for i in range(n_to_plot):
#         # Time series
#         axs[i, 0].plot(t, X_data[:, i], linewidth=1)
#         axs[i, 0].set_title(f"Component {i+1} Time Series")
#         axs[i, 0].set_xlabel("Time (s)")
#         axs[i, 0].set_ylabel("Amplitude")

#         # FFT
#         fft_vals = np.fft.fft(X_data[:, i])
#         freqs = np.fft.fftfreq(n_samples, d=1.0 / sampling_rate)
#         pos_mask = freqs >= 0
#         axs[i, 1].plot(freqs[pos_mask], np.abs(fft_vals)[pos_mask], linewidth=1)
#         axs[i, 1].set_title(f"Component {i+1} FFT")
#         axs[i, 1].set_xlabel("Frequency (Hz)")
#         axs[i, 1].set_ylabel("Magnitude")

#         # PSD
#         psd = np.abs(fft_vals) ** 2 / n_samples
#         axs[i, 2].plot(freqs[pos_mask], psd[pos_mask], linewidth=1)
#         axs[i, 2].set_title(f"Component {i+1} PSD")
#         axs[i, 2].set_xlabel("Frequency (Hz)")
#         axs[i, 2].set_ylabel("Power")

#     plt.tight_layout()
#     plt.show()
