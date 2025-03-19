"""
pca.py

This module provides functions for performing standard PCA (via SVD), kernel PCA,
and ICA on k-space data. It also provides plotting routines and a function to
reconstruct k-space data from selected components.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA, FastICA


def perform_pca(kspace, n_phase_encodes_per_frame):
    """
    Perform standard (linear) PCA on k-space data using SVD.

    Parameters:
        kspace (np.ndarray): Complex k-space data (n_phase x coils x freq_encodes).
        n_phase_encodes_per_frame (int): Number of phase encodes per frame.

    Returns:
        tuple: (U, S, Vt, X_mean, var_explained, frame_shape)
           - U (np.ndarray): Left singular vectors, shape (n_frames, n_frames).
           - S (np.ndarray): Singular values, 1D array, length n_frames.
           - Vt (np.ndarray): Right singular vectors, shape (n_frames, n_features).
           - X_mean (np.ndarray): Mean of the flattened k-space.
           - var_explained (np.ndarray): Fraction of variance explained by each component.
           - frame_shape (tuple): (n_phase_encodes_per_frame, n_coils, n_freq).
    """
    n_phase, n_coils, n_freq = kspace.shape
    n_frames = n_phase // n_phase_encodes_per_frame
    X = kspace.reshape(n_frames, -1)  # Flatten coils & freq into features

    # Subtract mean
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean

    # Full or truncated SVD
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    var_explained = (S**2) / np.sum(S**2)
    frame_shape = (n_phase_encodes_per_frame, n_coils, n_freq)

    return U, S, Vt, X_mean, var_explained, frame_shape


def perform_kernel_pca(
    kspace, n_phase_encodes_per_frame, kernel="rbf", sigma=1.0, n_components=None
):
    """
    Perform kernel PCA on complex k-space data using an RBF (Gaussian) kernel.

    Parameters:
        kspace (np.ndarray): Complex k-space data (n_phase x coils x freq_encodes).
        n_phase_encodes_per_frame (int): Number of phase encodes per frame.
        kernel (str): Kernel type (e.g. "rbf").
        sigma (float): Sigma parameter for the RBF kernel.
        n_components (int, optional): Number of KPCA components to retain.

    Returns:
        tuple: (kpca_model, X_kpca, frame_shape, orig_feature_dim)
           - kpca_model: Fitted KernelPCA model (with inverse_transform).
           - X_kpca (np.ndarray): Transformed data, shape (n_frames, n_components).
           - frame_shape (tuple): (n_phase_encodes_per_frame, n_coils, n_freq).
           - orig_feature_dim (int): Original feature dimension before splitting real/imag.
    """
    n_phase, n_coils, n_freq = kspace.shape
    n_frames = n_phase // n_phase_encodes_per_frame
    X = kspace.reshape(n_frames, -1)
    orig_feature_dim = X.shape[1]

    # Convert complex to real: stack real & imaginary parts side by side
    X_real = np.hstack((np.real(X), np.imag(X)))

    frame_shape = (n_phase_encodes_per_frame, n_coils, n_freq)
    gamma = 1.0 / (2.0 * sigma**2)

    kpca_model = KernelPCA(
        n_components=n_components,
        kernel=kernel,
        gamma=gamma,
        fit_inverse_transform=True,
    )
    X_kpca = kpca_model.fit_transform(X_real)
    return kpca_model, X_kpca, frame_shape, orig_feature_dim


def perform_ica(kspace, n_phase_encodes_per_frame, n_components=None):
    """
    Perform ICA (FastICA) on complex k-space data.

    Parameters:
        kspace (np.ndarray): Complex k-space data.
        n_phase_encodes_per_frame (int): Number of phase encodes per frame.
        n_components (int, optional): Number of ICA components to extract.

    Returns:
        tuple: (ica_model, X_ica, frame_shape, orig_feature_dim)
           - ica_model: Fitted FastICA model.
           - X_ica (np.ndarray): Independent components, shape (n_frames, n_components).
           - frame_shape (tuple): (n_phase_encodes_per_frame, n_coils, n_freq).
           - orig_feature_dim (int): Original feature dimension (before real/imag split).
    """
    n_phase, n_coils, n_freq = kspace.shape
    n_frames = n_phase // n_phase_encodes_per_frame
    X = kspace.reshape(n_frames, -1)
    orig_feature_dim = X.shape[1]

    # Stack real & imaginary parts
    X_real = np.hstack((np.real(X), np.imag(X)))
    frame_shape = (n_phase_encodes_per_frame, n_coils, n_freq)

    from sklearn.decomposition import FastICA

    ica_model = FastICA(n_components=n_components, random_state=0, max_iter=10000)
    X_ica = ica_model.fit_transform(X_real)
    return ica_model, X_ica, frame_shape, orig_feature_dim


def reconstruct_kspace_from_components(
    method,
    model,
    transformed_data,
    components_to_keep,
    frame_shape,
    orig_feature_dim=None,
    X_mean=None,
):
    """
    Reconstruct k-space data from a decomposition model (PCA, kernel PCA, or ICA),
    using only the specified components.

    Parameters:
        method (str): "pca", "kernel_pca", or "ica".
        model:
          - If PCA: (U, S, Vt, X_mean, var_explained, frame_shape).
          - If kernel/ICA: a fitted estimator with inverse_transform.
        transformed_data (np.ndarray): The transformed data (X_kpca, X_ica, etc.), shape (n_frames, n_components).
                                       Not used if method=="pca".
        components_to_keep (int or list of int): The component index/indices to retain.
        frame_shape (tuple): (n_phase_encodes_per_frame, n_coils, n_freq).
        orig_feature_dim (int, optional): For kernel PCA or ICA (needed to re-split real vs imaginary).
        X_mean (np.ndarray, optional): The mean vector from PCA (only needed for PCA).

    Returns:
        np.ndarray: Reconstructed k-space data with shape (n_frames, n_phase_encodes_per_frame, n_coils, n_freq).
    """
    if isinstance(components_to_keep, int):
        components_to_keep = [components_to_keep]

    if method == "pca":
        U, S, Vt, pca_mean, _, _ = model
        sel = np.array(components_to_keep)
        U_sel = U[:, sel]  # (n_frames, n_sel)
        S_sel = S[sel]  # (n_sel,)
        Vt_sel = Vt[sel, :]  # (n_sel, n_features)
        # Reconstruct
        X_recon = (U_sel * S_sel) @ Vt_sel + pca_mean
        n_frames = X_recon.shape[0]
        return X_recon.reshape(n_frames, *frame_shape)

    elif method in ["kernel_pca", "ica"]:
        X_mod = np.zeros_like(transformed_data)
        X_mod[:, components_to_keep] = transformed_data[:, components_to_keep]
        X_recon_real = model.inverse_transform(X_mod)
        # Combine real & imaginary
        X_recon_complex = (
            X_recon_real[:, :orig_feature_dim] + 1j * X_recon_real[:, orig_feature_dim:]
        )
        n_frames = X_recon_complex.shape[0]
        return X_recon_complex.reshape(n_frames, *frame_shape)

    else:
        raise ValueError(f"Unknown decomposition method: {method}")


def plot_components_time_series_and_fft(X_data, sampling_rate=1.0, n_components=5):
    """
    Plot time series, FFT magnitude, and PSD for the first n_components
    from a generic decomposition (e.g. kernel PCA or ICA).

    Parameters:
        X_data (np.ndarray): Transformed data, shape (n_frames, n_components).
        sampling_rate (float): Sampling rate in Hz.
        n_components (int): Number of components to plot.
    """
    n_frames = X_data.shape[0]
    t = np.arange(n_frames) / sampling_rate

    fig, axs = plt.subplots(n_components, 3, figsize=(18, 3 * n_components))
    if n_components == 1:
        axs = [axs]

    for i in range(n_components):
        ts = X_data[:, i]
        axs[i][0].plot(t, ts, marker="o")
        axs[i][0].set_title(f"Component {i+1} Time Series")
        axs[i][0].set_xlabel("Time (s)")
        axs[i][0].set_ylabel("Amplitude")

        fft_ts = np.fft.fft(ts)
        freqs = np.fft.fftfreq(n_frames, d=1 / sampling_rate)
        pos = freqs >= 0
        axs[i][1].plot(freqs[pos], np.abs(fft_ts)[pos], marker="o")
        axs[i][1].set_title(f"Component {i+1} FFT")
        axs[i][1].set_xlabel("Frequency (Hz)")
        axs[i][1].set_ylabel("Magnitude")

        psd = (np.abs(fft_ts) ** 2) / n_frames
        axs[i][2].plot(freqs[pos], psd[pos], marker="o")
        axs[i][2].set_title(f"Component {i+1} PSD")
        axs[i][2].set_xlabel("Frequency (Hz)")
        axs[i][2].set_ylabel("Power")

    plt.tight_layout()
    plt.show()


def plot_pca_time_series_and_fft(U, S, sampling_rate=1.0, n_components=5):
    """
    Plot time series, FFT magnitude, and PSD for the first n_components
    from a standard PCA decomposition via SVD.

    Parameters:
        U (np.ndarray): Left singular vectors, shape (n_frames, n_frames) or truncated.
        S (np.ndarray): Singular values, shape (n_frames,).
        sampling_rate (float): Sampling frequency in Hz.
        n_components (int): Number of PCA components to plot.
    """
    scores = U * S  # Broadcast multiplication to get PCA scores
    n_frames = scores.shape[0]
    t = np.arange(n_frames) / sampling_rate

    fig, axs = plt.subplots(n_components, 3, figsize=(18, 3 * n_components))
    if n_components == 1:
        axs = [axs]

    for i in range(n_components):
        ts = scores[:, i]
        axs[i][0].plot(t, ts, marker="o")
        axs[i][0].set_title(f"PCA Component {i+1} Time Series")
        axs[i][0].set_xlabel("Time (s)")
        axs[i][0].set_ylabel("Amplitude")

        fft_ts = np.fft.fft(ts)
        freqs = np.fft.fftfreq(n_frames, d=1 / sampling_rate)
        pos = freqs >= 0
        axs[i][1].plot(freqs[pos], np.abs(fft_ts)[pos], marker="o")
        axs[i][1].set_title(f"PCA Component {i+1} FFT")
        axs[i][1].set_xlabel("Frequency (Hz)")
        axs[i][1].set_ylabel("Magnitude")

        psd = (np.abs(fft_ts) ** 2) / n_frames
        axs[i][2].plot(freqs[pos], psd[pos], marker="o")
        axs[i][2].set_title(f"PCA Component {i+1} PSD")
        axs[i][2].set_xlabel("Frequency (Hz)")
        axs[i][2].set_ylabel("Power")

    plt.tight_layout()
    plt.show()
