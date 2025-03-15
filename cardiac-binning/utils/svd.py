"""
svd.py

This module provides functions to perform PCA (via SVD) on k-space data,
plot the principal components and their time/frequency characteristics, and
reconstruct image cines from the PCA on k-space data.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
from sklearn.decomposition import KernelPCA


def perform_pca_kspace(kspace, n_phase_encodes_per_frame):
    n_phase, n_coils, n_freq = kspace.shape
    n_frames = n_phase // n_phase_encodes_per_frame
    X = kspace.reshape(n_frames, -1)
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    var_explained = (S**2) / np.sum(S**2)
    frame_shape = (n_phase_encodes_per_frame, n_coils, n_freq)
    return U, S, Vt, X_mean, var_explained, frame_shape


def perform_kernel_pca_kspace(
    kspace, n_phase_encodes_per_frame, kernel="rbf", sigma=1.0, n_components=None
):
    """
    Perform kernel PCA on k-space data using an RBF (Gaussian) kernel.
    The complex data is represented by concatenating its real and imaginary parts.

    Parameters:
      kspace (np.ndarray): k-space data with shape (n_phase, n_coils, n_freq) (complex-valued).
      n_phase_encodes_per_frame (int): Number of phase encodes per frame.
      sigma (float): Parameter for the Gaussian kernel.
      n_components (int or None): Number of kernel PCA components to retain.

    Returns:
      kpca (KernelPCA object): Fitted KernelPCA model.
      X_kpca (np.ndarray): Transformed data of shape (n_frames, n_components).
      frame_shape (tuple): Shape of a single frame in the original domain: (n_phase_encodes_per_frame, n_coils, n_freq).
      orig_feature_dim (int): Original number of features per frame before converting to real representation.
    """
    n_phase, n_coils, n_freq = kspace.shape
    n_frames = n_phase // n_phase_encodes_per_frame
    # Reshape complex k-space to (n_frames, n_features) where n_features is complex.
    X = kspace.reshape(n_frames, -1)
    orig_feature_dim = X.shape[1]
    # Represent each complex vector as a real vector by concatenating real and imaginary parts.
    X_real = np.hstack((np.real(X), np.imag(X)))

    frame_shape = (n_phase_encodes_per_frame, n_coils, n_freq)
    gamma = 1.0 / (2 * sigma**2)
    kpca = KernelPCA(
        n_components=n_components,
        kernel=kernel,
        gamma=gamma,
        fit_inverse_transform=True,
    )
    X_kpca = kpca.fit_transform(X_real)
    return kpca, X_kpca, frame_shape, orig_feature_dim


def plot_components(Vt, frame_shape, var_explained, n_components=5):
    fig, axes = plt.subplots(1, n_components, figsize=(15, 3))
    for i in range(n_components):
        # Reshape to (n_phase, n_coils, n_freq)
        comp = np.abs(Vt[i, :]).reshape(frame_shape)
        # Reconstruct image
        comp_img = np.fft.fftshift(np.fft.ifft2(comp))
        # Combine coils via sum-of-squares to get a (n_phase, n_freq) image.
        comp_img = np.sqrt(np.sum(np.abs(comp_img) ** 2, axis=1))
        axes[i].imshow(comp_img, cmap="gray")
        axes[i].set_title(f"PC {i+1}\n{var_explained[i]*100:.1f}%")
        axes[i].axis("off")
    plt.tight_layout()
    plt.show()


def plot_pc_time_series_and_fft(U, S, sampling_rate=1.0, n_components=5):
    """
    For each of the first n_components, plot the time series (U[:,i]*S[i]),
    its FFT magnitude spectrum, and its power spectral density.

    Parameters:
      U            : Left singular vectors (temporal coefficients).
      S            : Singular values.
      sampling_rate: Sampling rate in Hz (frames per second).
      n_components : Number of components to display.
    """
    n_bins = U.shape[0]
    time = np.arange(n_bins) / sampling_rate
    fig, axs = plt.subplots(n_components, 3, figsize=(18, 3 * n_components))
    if n_components == 1:
        axs = np.expand_dims(axs, axis=0)
    for i in range(n_components):
        ts = U[:, i] * S[i]
        # Time-domain plot.
        axs[i, 0].plot(time, ts, marker="o")
        axs[i, 0].set_title(f"PC {i+1} Time Series")
        axs[i, 0].set_xlabel("Time (s)")
        axs[i, 0].set_ylabel("Amplitude")
        # FFT magnitude.
        fft_ts = np.fft.fft(ts)
        freqs = np.fft.fftfreq(n_bins, d=1 / sampling_rate)
        pos = freqs >= 0
        axs[i, 1].plot(freqs[pos], np.abs(fft_ts)[pos], marker="o")
        axs[i, 1].set_title(f"PC {i+1} FFT Magnitude")
        axs[i, 1].set_xlabel("Frequency (Hz)")
        axs[i, 1].set_ylabel("Magnitude")
        # Power Spectral Density.
        psd = (np.abs(fft_ts) ** 2) / n_bins
        axs[i, 2].plot(freqs[pos], psd[pos], marker="o")
        axs[i, 2].set_title(f"PC {i+1} PSD")
        axs[i, 2].set_xlabel("Frequency (Hz)")
        axs[i, 2].set_ylabel("Power")
    plt.tight_layout()
    plt.show()


def plot_kernel_pc_time_series_and_fft(X_kpca, sampling_rate=1.0, n_components=5):
    """
    For each of the first n_components from kernel PCA, plot the time series,
    its FFT magnitude spectrum, and its power spectral density (PSD).

    Parameters:
      X_kpca (np.ndarray): Transformed data from kernel PCA with shape (n_frames, n_components_total).
      sampling_rate (float): Sampling rate in Hz (frames per second).
      n_components (int): Number of kernel PCs to display.
    """
    n_frames = X_kpca.shape[0]
    time = np.arange(n_frames) / sampling_rate

    fig, axs = plt.subplots(n_components, 3, figsize=(18, 3 * n_components))
    if n_components == 1:
        axs = np.expand_dims(axs, axis=0)

    for i in range(n_components):
        ts = X_kpca[:, i]
        # Time-domain plot.
        axs[i, 0].plot(time, ts, marker="o")
        axs[i, 0].set_title(f"Kernel PC {i+1} Time Series")
        axs[i, 0].set_xlabel("Time (s)")
        axs[i, 0].set_ylabel("Amplitude")

        # FFT magnitude.
        fft_ts = np.fft.fft(ts)
        freqs = np.fft.fftfreq(n_frames, d=1 / sampling_rate)
        pos = freqs >= 0
        axs[i, 1].plot(freqs[pos], np.abs(fft_ts)[pos], marker="o")
        axs[i, 1].set_title(f"Kernel PC {i+1} FFT Magnitude")
        axs[i, 1].set_xlabel("Frequency (Hz)")
        axs[i, 1].set_ylabel("Magnitude")

        # Power Spectral Density (PSD).
        psd = (np.abs(fft_ts) ** 2) / n_frames
        axs[i, 2].plot(freqs[pos], psd[pos], marker="o")
        axs[i, 2].set_title(f"Kernel PC {i+1} PSD")
        axs[i, 2].set_xlabel("Frequency (Hz)")
        axs[i, 2].set_ylabel("Power")

    plt.tight_layout()
    plt.show()


def optimize_sigma_kpca(
    kspace, n_phase_encodes_per_frame, sigma_candidates, n_components=None, kernel="rbf"
):
    """
    Optimize the sigma parameter for KernelPCA by finding the sigma that minimizes
    the reconstruction error on the k-space data. The k-space data is first converted to a
    real-valued representation by concatenating its real and imaginary parts.

    Parameters:
      kspace (np.ndarray): Complex k-space data of shape (n_phase, n_coils, n_freq).
      n_phase_encodes_per_frame (int): Number of phase encodes per frame.
      sigma_candidates (list or array): Candidate sigma values to test.
      n_components (int or None): Number of kernel PCA components to retain.
      kernel (str): Kernel type (default "rbf").

    Returns:
      best_sigma (float): Sigma value that yielded the lowest reconstruction error.
      best_error (float): The corresponding relative reconstruction error.
      best_kpca (KernelPCA object): The fitted KernelPCA model using best_sigma.
      best_X_kpca (np.ndarray): The KernelPCA transformed data using best_sigma.
      orig_feature_dim (int): Original feature dimension (before converting to real representation).
    """
    n_phase, n_coils, n_freq = kspace.shape
    n_frames = n_phase // n_phase_encodes_per_frame
    # Reshape to (n_frames, orig_feature_dim) where each element is complex.
    X = kspace.reshape(n_frames, -1)
    orig_feature_dim = X.shape[1]
    # Create a real representation: concatenate real and imaginary parts.
    X_real = np.hstack((np.real(X), np.imag(X)))

    best_sigma = None
    best_error = np.inf
    best_kpca = None
    best_X_kpca = None

    for sigma in sigma_candidates:
        gamma = 1.0 / (2 * sigma**2)
        kpca = KernelPCA(
            n_components=n_components,
            kernel=kernel,
            gamma=gamma,
            fit_inverse_transform=True,
        )
        X_kpca = kpca.fit_transform(X_real)
        # Inverse transform to get reconstructed real data.
        X_recon_real = kpca.inverse_transform(X_kpca)
        # Compute relative reconstruction error (Frobenius norm).
        error = np.linalg.norm(X_real - X_recon_real) / np.linalg.norm(X_real)
        print(f"Sigma: {sigma:.4f}, Reconstruction Error: {error:.4f}")
        if error < best_error:
            best_error = error
            best_sigma = sigma
            best_kpca = kpca
            best_X_kpca = X_kpca

    return best_sigma, best_error, best_kpca, best_X_kpca, orig_feature_dim
