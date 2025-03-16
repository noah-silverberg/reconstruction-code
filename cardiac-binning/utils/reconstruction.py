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


def direct_ifft_reconstruction(
    kspace, extended_pe_lines=None, offset=None, use_conjugate_symmetry=False
):
    """
    Reconstruct images directly from k-space data using a 2D IFFT per coil,
    with optional zero-filling and conjugate symmetry filling.

    If both 'extended_pe_lines' and 'offset' are provided, the function will:
      1. Create a new zero-filled k-space array with shape
            (n_frames, extended_pe_lines, n_coils, n_freq),
         and insert the measured data (which has shape
            (n_frames, phase_encodes, n_coils, n_freq))
         starting at row 'offset'.
      2. If 'use_conjugate_symmetry' is True, unmeasured rows (those outside the
         measured region defined by offset and phase_encodes) are filled using conjugate symmetry.

    After any optional k-space modification, a 2D IFFT (with fftshift) is applied
    to each coil's k-space and the coil images are combined using the root-sum-of-squares.

    Parameters:
        kspace (np.ndarray): Complex k-space data with shape
                             (n_frames, phase_encodes, n_coils, n_freq).
        extended_pe_lines (int, optional): Total number of phase-encode lines in the full (zero-filled) k-space.
            If not provided, no zero-filling is performed.
        offset (int, optional): Row offset at which to insert the measured k-space data.
            Must be provided if extended_pe_lines is provided.
        use_conjugate_symmetry (bool, optional): If True, fill the unmeasured rows using conjugate symmetry.
            Default is False.

    Returns:
        np.ndarray: Reconstructed images with shape:
                    - If zero-filling is applied: (n_frames, extended_pe_lines, n_freq).
                    - Otherwise: (n_frames, phase_encodes, n_freq).
    """
    # If extended_pe_lines and offset are provided, modify kspace accordingly.
    if extended_pe_lines is not None and offset is not None:
        n_frames, phase_encodes, n_coils, n_freq = kspace.shape
        # Create zero-filled k-space
        kspace_mod = np.zeros(
            (n_frames, extended_pe_lines, n_coils, n_freq), dtype=kspace.dtype
        )
        kspace_mod[:, offset : offset + phase_encodes, :, :] = kspace

        # Optionally fill unmeasured rows using conjugate symmetry.
        if use_conjugate_symmetry:
            kspace_mod = fill_conjugate_symmetry(
                kspace_mod, offset, phase_encodes, extended_pe_lines
            )
    else:
        # Use kspace as is.
        kspace_mod = kspace
        # Update dimensions based on input kspace.
        n_frames, phase_encodes, n_coils, n_freq = kspace.shape

    # Reconstruct images by applying 2D IFFT (with fftshift) per coil and combine via root-sum-of-squares.
    # The output image shape will depend on whether zero-filling was applied.
    output_rows = extended_pe_lines if extended_pe_lines is not None else phase_encodes
    recon_images = np.zeros((n_frames, output_rows, n_freq), dtype=np.float64)
    for i in range(n_frames):
        coil_imgs = []
        for ch in range(n_coils):
            img_coil = np.fft.fftshift(np.fft.ifft2(kspace_mod[i, :, ch, :]))
            coil_imgs.append(img_coil)
        coil_imgs = np.array(coil_imgs)
        recon_images[i] = np.sqrt(np.sum(np.abs(coil_imgs) ** 2, axis=0))
    return recon_images


def fill_conjugate_symmetry(kspace_zf, offset, phase_encodes, extended_pe_lines):
    """
    Fill in unmeasured rows in the zero-filled k-space using conjugate symmetry.

    Given a zero-filled k-space array of shape
      (n_frames, extended_pe_lines, n_coils, n_freq),
    where the measured data occupies rows from 'offset' to 'offset + phase_encodes',
    this function fills the unmeasured rows (outside that range) by taking the complex
    conjugate of the symmetric row with respect to the k-space center.

    The k-space symmetry for a real image is given by:
      F(u, v) = conjugate(F(-u, -v))
    In this implementation, symmetry is applied along the phase-encode (row) dimension.
    The symmetric row for a given row 'r' is computed as:
        sym_row = round(2 * center - r)
    where center = extended_pe_lines / 2.

    Parameters:
        kspace_zf (np.ndarray): Zero-filled k-space data with shape
                                (n_frames, extended_pe_lines, n_coils, n_freq).
        offset (int): Row offset where measured k-space data begins.
        phase_encodes (int): Number of measured phase-encode lines.
        extended_pe_lines (int): Total number of phase-encode lines in the full k-space.

    Returns:
        np.ndarray: k-space data with unmeasured rows filled via conjugate symmetry.
    """
    n_frames, total_rows, n_coils, n_freq = kspace_zf.shape
    center = extended_pe_lines / 2.0
    # For each frame and each coil, fill in unmeasured rows.
    for i in range(n_frames):
        for ch in range(n_coils):
            for r in range(total_rows):
                # Check if this row is unmeasured.
                if r < offset or r >= offset + phase_encodes:
                    # Compute the symmetric row index.
                    sym_r = int(round(2 * center - r))
                    # Only fill if the symmetric row is within the measured range.
                    if offset <= sym_r < offset + phase_encodes:
                        kspace_zf[i, r, ch, :] = np.conjugate(
                            kspace_zf[i, sym_r, ch, :]
                        )
    return kspace_zf
