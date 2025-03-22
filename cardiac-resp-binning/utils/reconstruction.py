"""
reconstruction.py

This module provides functions to reconstruct images from k-space data (e.g. via direct IFFT).
Includes optional homodyne correction as well.

We also reintroduce the old function
'reconstruct_kspace_with_selected_components_kernel_kspace' for backward compatibility,
but internally it calls the new function in pca.py.
"""

import numpy as np
from utils import pca


def reconstruct_kspace_with_selected_components_kernel_kspace(
    kpca_model, X_kpca, components, frame_shape, orig_feature_dim
):
    """
    Reconstruct complex k-space from selected kernel PCA components.
    (Kept for backward compatibility if older code calls it.)

    Parameters:
        kpca_model: Fitted KernelPCA model with inverse_transform.
        X_kpca (np.ndarray): Kernel PCA transformed data, shape (n_frames, n_components).
        components (list or int): Components to keep.
        frame_shape (tuple): (n_phase_encodes_per_frame, n_coils, n_freq).
        orig_feature_dim (int): Original feature dimension (before real/imag splitting).

    Returns:
        np.ndarray: Reconstructed k-space with shape (n_frames, n_phase_encodes_per_frame, n_coils, n_freq).
    """
    return pca.reconstruct_kspace_from_components(
        method="kernel_pca",
        model=kpca_model,
        transformed_data=X_kpca,
        components_to_keep=components,
        frame_shape=frame_shape,
        orig_feature_dim=orig_feature_dim,
    )


def direct_ifft_reconstruction(
    kspace,
    extended_pe_lines=None,
    offset=None,
    use_conjugate_symmetry=False,
    count_mask=None,
):
    if extended_pe_lines is not None and offset is not None:
        n_frames, phase_encodes, n_coils, n_freq = kspace.shape
        kspace_mod = np.zeros(
            (n_frames, extended_pe_lines, n_coils, n_freq), dtype=kspace.dtype
        )
        # Place measured lines at the correct location.
        kspace_mod[:, offset : offset + phase_encodes, :, :] = kspace
        # If no count mask is provided, create a default one based on offset.
        if count_mask is None:
            measured_mask = np.zeros((n_frames, extended_pe_lines), dtype=bool)
            measured_mask[:, offset : offset + phase_encodes] = True
        else:
            # Assume count_mask is numeric and derive a boolean mask.
            measured_mask = count_mask > 0
        if use_conjugate_symmetry:
            kspace_mod = fill_conjugate_symmetry(kspace_mod, measured_mask)
    else:
        kspace_mod = kspace
        n_frames, phase_encodes, n_coils, n_freq = kspace.shape

    # Standard IFFT reconstruction across coils.
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


def fill_conjugate_symmetry(kspace_zf, measured_mask):
    n_frames, total_rows, n_coils, n_freq = kspace_zf.shape
    center = total_rows / 2.0
    for i in range(n_frames):
        for r in range(total_rows):
            if not measured_mask[i, r]:
                sym_r = int(round(2 * center - r))
                if 0 <= sym_r < total_rows and measured_mask[i, sym_r]:
                    kspace_zf[i, r, :, :] = np.conjugate(kspace_zf[i, sym_r, :, :])
    return kspace_zf
