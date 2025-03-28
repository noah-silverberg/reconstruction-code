"""
reconstruction.py

Functions to reconstruct images (and k-space) from partial or processed k-space data.
"""

import numpy as np
import pygrappa


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


def grappa_reconstruction(kspace, calib_region, kernel_size=(5, 5)):
    """
    Perform basic cGRAPPA reconstruction on each frame individually using pygrappa.cgrappa.
    Each frame is treated as an individual image.

    Parameters
    ----------
    kspace : np.ndarray
        Undersampled k-space data of shape (n_frames, n_phase_encodes, n_coils, n_freq).
    calib_region : tuple or list
        Two-element tuple (start_line, end_line) specifying the autocalibration region for each frame.
    kernel_size : tuple, optional
        Size of the GRAPPA kernel (default is (5, 5)).

    Returns
    -------
    np.ndarray
        Reconstructed images after coil combination, of shape (n_frames, output_rows, n_freq).
    """
    n_frames = kspace.shape[0]
    recon_images_all = []

    for i in range(n_frames):
        # Convert the current frame to double precision
        frame_kspace = np.array(kspace[i], dtype=np.complex128)

        # Extract calibration data from this frame (and explicitly convert it)
        start_line, end_line = calib_region
        calib_data = np.array(
            frame_kspace[start_line:end_line, :, :], dtype=np.complex128
        )

        # Call cGRAPPA on the individual frame.
        # Now the frame_kspace and calib_data are both np.complex128.
        recon_kspace_frame = pygrappa.cgrappa(
            frame_kspace, calib_data, kernel_size=kernel_size, coil_axis=1
        )

        # Inverse FFT on each coil and combine them with sum-of-squares.
        n_phase, n_coils, n_freq = recon_kspace_frame.shape
        coil_images = []
        for ch in range(n_coils):
            img = np.fft.ifft2(recon_kspace_frame[:, ch, :])
            img = np.fft.fftshift(img)
            coil_images.append(img)
        coil_images = np.array(coil_images)
        recon_img = np.sqrt(np.sum(np.abs(coil_images) ** 2, axis=0))
        recon_images_all.append(recon_img)

    return np.array(recon_images_all)


def tgrappa_reconstruction(kspace, calib_size, kernel_size=(5, 5)):
    """
    Perform temporal GRAPPA (tGRAPPA) reconstruction on cine data using pygrappa.tgrappa.
    tGRAPPA uses temporal redundancy to reconstruct undersampled data.

    Parameters
    ----------
    kspace : np.ndarray
        Undersampled k-space data of shape (n_frames, n_phase_encodes, n_coils, n_freq_encodes).
    calib_size : tuple
        A two-element tuple (calib_phase, calib_freq) that defines the size of the calibration region.
        tGRAPPA will construct the ACS (autocalibration signals) from the temporal center.
    kernel_size : tuple, optional
        Size of the GRAPPA kernel (default is (5, 5)).

    Returns
    -------
    np.ndarray
        Reconstructed magnitude images of shape (n_frames, n_phase_encodes, n_freq_encodes).
    """
    # Rearrange kspace to put time (frames) in the last dimension.
    # Original shape: (n_frames, n_phase, n_coils, n_freq)
    # Rearranged shape: (n_phase, n_freq, n_coils, n_frames)
    kspace_rearr = np.transpose(kspace, (1, 3, 2, 0))

    # Call tGRAPPA.
    # Here, coil_axis=-2 indicates that the coils are in the second-to-last dimension,
    # and time_axis=-1 indicates that the last dimension is time.
    recon_kspace = pygrappa.tgrappa(
        kspace_rearr,
        calib_size=calib_size,
        kernel_size=kernel_size,
        coil_axis=-2,
        time_axis=-1,
    )

    # Rearrange the reconstructed k-space back to original order:
    # New shape: (n_frames, n_phase, n_coils, n_freq)
    recon_kspace = np.transpose(recon_kspace, (3, 0, 2, 1))

    # Perform inverse FFT on each frame and combine coils using sum-of-squares.
    n_frames, n_phase, n_coils, n_freq = recon_kspace.shape
    recon_images = np.zeros((n_frames, n_phase, n_freq), dtype=np.float64)
    for i in range(n_frames):
        coil_imgs = []
        for ch in range(n_coils):
            img = np.fft.ifft2(recon_kspace[i, :, ch, :])
            img = np.fft.fftshift(img)
            coil_imgs.append(img)
        coil_imgs = np.array(coil_imgs)
        recon_images[i] = np.sqrt(np.sum(np.abs(coil_imgs) ** 2, axis=0))
    return recon_images
