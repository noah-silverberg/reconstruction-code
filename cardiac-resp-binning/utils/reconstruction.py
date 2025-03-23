"""
reconstruction.py

Functions to reconstruct images (and k-space) from partial or processed k-space data.
"""

import numpy as np


def direct_ifft_reconstruction(
    kspace,
    extended_pe_lines=None,
    offset=None,
    use_conjugate_symmetry=False,
    count_mask=None,
):
    """
    Perform a direct IFFT-based reconstruction across coils, with optional zero-fill
    and conjugate symmetry filling.

    Parameters
    ----------
    kspace : np.ndarray
        Shape (n_frames, n_phase_encodes, n_coils, n_freq).
    extended_pe_lines : int, optional
        If not None, the total extended dimension for zero-filling in phase-encode.
    offset : int, optional
        Where to place the measured data within the extended dimension.
    use_conjugate_symmetry : bool
        If True, fill the missing lines via conjugate symmetry of the measured lines.
    count_mask : np.ndarray, optional
        A numeric array indicating how many times each line has been measured.
        If provided, it is used to derive a measured_mask for symmetry filling.

    Returns
    -------
    np.ndarray
        Reconstructed magnitude images of shape (n_frames, output_rows, n_freq),
        where output_rows = extended_pe_lines if provided, else n_phase_encodes.
    """
    if extended_pe_lines is not None and offset is not None:
        n_frames, phase_encodes, n_coils, n_freq = kspace.shape
        kspace_mod = np.zeros(
            (n_frames, extended_pe_lines, n_coils, n_freq), dtype=kspace.dtype
        )
        # Place measured lines
        kspace_mod[:, offset : offset + phase_encodes, :, :] = kspace

        # Derive measured mask
        if count_mask is None:
            measured_mask = np.zeros((n_frames, extended_pe_lines), dtype=bool)
            measured_mask[:, offset : offset + phase_encodes] = True
        else:
            measured_mask = count_mask > 0

        if use_conjugate_symmetry:
            kspace_mod = fill_conjugate_symmetry(kspace_mod, measured_mask)
    else:
        kspace_mod = kspace

    n_frames, rows, n_coils, n_freq = kspace_mod.shape
    recon_images = np.zeros((n_frames, rows, n_freq), dtype=np.float64)

    # Coil combination by sum of squares
    for i in range(n_frames):
        coil_imgs = []
        for ch in range(n_coils):
            img_coil = np.fft.fftshift(np.fft.ifft2(kspace_mod[i, :, ch, :]))
            coil_imgs.append(img_coil)
        coil_imgs = np.array(coil_imgs)
        recon_images[i] = np.sqrt(np.sum(np.abs(coil_imgs) ** 2, axis=0))

    return recon_images


def fill_conjugate_symmetry(kspace_zf, measured_mask):
    """
    Fill in missing lines of kspace by conjugate symmetry.

    Parameters
    ----------
    kspace_zf : np.ndarray
        Zero-filled k-space of shape (n_frames, n_rows, n_coils, n_freq).
    measured_mask : np.ndarray
        Boolean mask indicating measured lines (shape (n_frames, n_rows)).

    Returns
    -------
    np.ndarray
        kspace_zf with missing lines filled by conjugate copies of measured lines.
    """
    n_frames, total_rows, n_coils, n_freq = kspace_zf.shape
    center = total_rows / 2.0

    for i in range(n_frames):
        for r in range(total_rows):
            if not measured_mask[i, r]:
                sym_r = int(round(2 * center - r))
                if 0 <= sym_r < total_rows and measured_mask[i, sym_r]:
                    kspace_zf[i, r, :, :] = np.conjugate(kspace_zf[i, sym_r, :, :])

    return kspace_zf
