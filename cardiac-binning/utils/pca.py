"""
pca.py

This module provides functions for performing standard PCA (via SVD) and kernel PCA
on k-space data. It also includes helper functions to plot principal components.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA


def perform_pca(kspace, n_phase_encodes_per_frame):
    """
    Perform standard PCA on k-space data via SVD.

    Parameters:
        kspace (np.ndarray): Complex k-space data (n_phase x coils x freq_encodes).
        n_phase_encodes_per_frame (int): Phase encodes per frame.

    Returns:
        Tuple: (U, S, Vt, X_mean, var_explained, frame_shape)
    """
    n_phase, n_coils, n_freq = kspace.shape
    n_frames = n_phase // n_phase_encodes_per_frame
    X = kspace.reshape(n_frames, -1)
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    var_explained = (S**2) / np.sum(S**2)
    frame_shape = (n_phase_encodes_per_frame, n_coils, n_freq)
    return U, S, Vt, X_mean, var_explained, frame_shape


def perform_kernel_pca(
    kspace, n_phase_encodes_per_frame, kernel="rbf", sigma=1.0, n_components=None
):
    """
    Perform kernel PCA on complex k-space data using an RBF kernel.

    Parameters:
        kspace (np.ndarray): Complex k-space data.
        n_phase_encodes_per_frame (int): Number of phase encodes per frame.
        kernel (str): Kernel type.
        sigma (float): Kernel parameter.
        n_components (int): Number of components to retain.

    Returns:
        Tuple: (kpca_model, X_kpca, frame_shape, orig_feature_dim)
    """
    n_phase, n_coils, n_freq = kspace.shape
    n_frames = n_phase // n_phase_encodes_per_frame
    X = kspace.reshape(n_frames, -1)
    orig_feature_dim = X.shape[1]
    # Represent complex data as real by concatenating real and imaginary parts.
    X_real = np.hstack((np.real(X), np.imag(X)))
    frame_shape = (n_phase_encodes_per_frame, n_coils, n_freq)
    gamma = 1.0 / (2 * sigma**2)
    kpca_model = KernelPCA(
        n_components=n_components,
        kernel=kernel,
        gamma=gamma,
        fit_inverse_transform=True,
    )
    X_kpca = kpca_model.fit_transform(X_real)
    return kpca_model, X_kpca, frame_shape, orig_feature_dim


def plot_kernel_pc_time_series_and_fft(X_kpca, sampling_rate=1.0, n_components=5):
    """
    Plot the time series, FFT magnitude, and PSD for the first n_components from kernel PCA.

    Parameters:
        X_kpca (np.ndarray): Kernel PCA transformed data.
        sampling_rate (float): Sampling frequency.
        n_components (int): Number of components to plot.
    """
    n_frames = X_kpca.shape[0]
    time = np.arange(n_frames) / sampling_rate
    fig, axs = plt.subplots(n_components, 3, figsize=(18, 3 * n_components))
    if n_components == 1:
        axs = [axs]
    for i in range(n_components):
        ts = X_kpca[:, i]
        axs[i][0].plot(time, ts, marker="o")
        axs[i][0].set_title(f"Kernel PC {i+1} Time Series")
        axs[i][0].set_xlabel("Time (s)")
        axs[i][0].set_ylabel("Amplitude")
        fft_ts = np.fft.fft(ts)
        freqs = np.fft.fftfreq(n_frames, d=1 / sampling_rate)
        pos = freqs >= 0
        axs[i][1].plot(freqs[pos], np.abs(fft_ts)[pos], marker="o")
        axs[i][1].set_title(f"Kernel PC {i+1} FFT")
        axs[i][1].set_xlabel("Frequency (Hz)")
        axs[i][1].set_ylabel("Magnitude")
        psd = (np.abs(fft_ts) ** 2) / n_frames
        axs[i][2].plot(freqs[pos], psd[pos], marker="o")
        axs[i][2].set_title(f"Kernel PC {i+1} PSD")
        axs[i][2].set_xlabel("Frequency (Hz)")
        axs[i][2].set_ylabel("Power")
    plt.tight_layout()
    plt.show()
