"""
reconstruction.py

This module provides functions to reconstruct images from k-space data.
Includes functions for reconstruction using selected kernel PCA components and homodyne correction.
"""

import numpy as np


def reconstruct_kspace_with_selected_components_kernel_kspace(
    kpca, X_kpca, selection, frame_shape, orig_feature_dim
):
    """
    Reconstruct complex k-space data using only the specified kernel PCA components.
    (No inverse FFT is applied here.)

    Parameters:
        kpca: Fitted KernelPCA model.
        X_kpca: Kernel PCA transformed data.
        selection: Component indices to keep.
        frame_shape: (phase_encodes, coils, freq_encodes)
        orig_feature_dim: Original feature dimension (before real conversion).

    Returns:
        np.ndarray: Reconstructed k-space data with shape (n_frames, phase_encodes, coils, freq_encodes)
    """
    if isinstance(selection, int):
        selection = [selection]
    X_mod = np.zeros_like(X_kpca)
    X_mod[:, selection] = X_kpca[:, selection]
    X_recon_real = kpca.inverse_transform(X_mod)
    X_recon_complex = (
        X_recon_real[:, :orig_feature_dim] + 1j * X_recon_real[:, orig_feature_dim:]
    )
    n_frames = X_recon_complex.shape[0]
    return X_recon_complex.reshape(n_frames, *frame_shape)


def homodyne_reconstruction(binned_data, binned_count):
    """
    Apply homodyne correction to binned k-space data and reconstruct images.

    Parameters:
        binned_data (np.ndarray): Complex k-space data (num_bins, extended_pe_lines, coils, readout).
        binned_count (np.ndarray): Count of acquisitions per bin and row.

    Returns:
        np.ndarray: Reconstructed images.
    """
    num_bins, n_phase, n_coils, n_readout = binned_data.shape
    recon_images = []
    center = n_phase / 2.0
    for b in range(num_bins):
        weights = np.zeros(n_phase, dtype=np.float32)
        for r in range(n_phase):
            if binned_count[b, r] > 0:
                r_sym = int(round(2 * center - r))
                weights[r] = (
                    1.0
                    if (0 <= r_sym < n_phase and binned_count[b, r_sym] > 0)
                    else 2.0
                )
        weight_matrix = np.tile(weights[:, None], (1, n_readout))
        coil_images = []
        for ch in range(n_coils):
            k_ch = binned_data[b, :, ch, :]
            k_weighted = k_ch * weight_matrix
            lowpass_filter = (binned_count[b, :] > 0).astype(np.float32)
            lowpass_matrix = np.tile(lowpass_filter[:, None], (1, n_readout))
            k_lowpass = k_ch * lowpass_matrix
            img_full = np.fft.fftshift(np.fft.ifft2(k_weighted))
            img_lowres = np.fft.fftshift(np.fft.ifft2(k_lowpass))
            phase_est = np.angle(img_lowres)
            img_corrected = img_full * np.exp(-1j * phase_est)
            coil_images.append(np.real(img_corrected))
        coil_images = np.array(coil_images)
        sos_image = np.sqrt(np.sum(coil_images**2, axis=0))
        recon_images.append(sos_image)
    return np.array(recon_images)
