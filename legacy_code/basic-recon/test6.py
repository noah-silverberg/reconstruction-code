#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt


def generate_real_image(size):
    """
    Generate a real-valued image with shapes of varying brightness.
    """
    image = np.zeros((size, size), dtype=np.float32)

    # Add a bright rectangle
    image[20:40, 20:60] = 1.0

    # Add a dimmer circle
    y, x = np.ogrid[-size // 2 : size // 2, -size // 2 : size // 2]
    mask = x**2 + y**2 <= (size // 8) ** 2
    image[mask] = 0.5

    # Add a gradient
    image += np.linspace(0, 0.3, size)[:, np.newaxis]

    return image


def simulate_full_kspace_oversampled(
    image, full_readout=200, crop_lower=64, crop_upper=192
):
    """
    Simulate full k-space acquisition with an oversampled readout.

    The original image is 128x128. We embed it into a 128x200 array by placing it
    in the center of the readout (columns 64:192). Then we compute the 2D FFT.
    """
    Np, Nx = image.shape  # 128x128
    padded = np.zeros((Np, full_readout), dtype=image.dtype)
    # Place the image into the central 128 columns
    padded[:, crop_lower:crop_upper] = image
    # Compute k-space (using the common FFT shift conventions)
    kspace_full = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(padded)))
    return kspace_full


def simulate_under_sampled_acquisition(
    kspace_full,
    measured_phase_lines=40,
    phase_offset=12,
    full_phase=128,
    crop_lower=64,
    crop_upper=192,
):
    """
    Simulate the under-sampling in the phase encoding direction.

    Here the full (cropped) readout has 128 samples (from a 200-sample readout).
    Only 'measured_phase_lines' out of 'full_phase' (128) lines are acquired.
    The measured phase-encoding lines (0, 1, …, 39) are mapped to full k-space
    rows via: full_row = measured_line + phase_offset.

    Returns:
       acquired_data: array of shape (measured_phase_lines, 128)
       acquired_rows: the list of full k-space row indices where data were acquired.
    """
    full_target_readout = crop_upper - crop_lower  # should be 128
    acquired_rows = []
    acquired_data = []
    for measured_line in range(measured_phase_lines):
        full_row = measured_line + phase_offset
        # Crop the oversampled readout along the readout direction
        row_data = kspace_full[full_row, crop_lower:crop_upper]
        acquired_rows.append(full_row)
        acquired_data.append(row_data)
    acquired_data = np.array(acquired_data)  # shape (measured_phase_lines, 128)
    return acquired_data, acquired_rows


def insert_acquired_into_full(
    acquired_data, acquired_rows, full_phase=128, full_target_readout=128
):
    """
    Insert the acquired k-space rows into a full k-space array.
    """
    full_kspace = np.zeros((full_phase, full_target_readout), dtype=np.complex64)
    for idx, row in enumerate(acquired_rows):
        full_kspace[row, :] = acquired_data[idx]
    return full_kspace


def conjugate_symmetry_fill(full_kspace, acquired_rows):
    """
    Fill missing k-space rows using conjugate symmetry—but only for rows that
    were not acquired.

    This is done in the fftshifted domain. (Note that for a real image the k-space
    satisfies: K(u,v) = K*(-u,-v), so the center is self-conjugate.)
    """
    full_phase, full_target_readout = full_kspace.shape
    # Shift k-space (both phase and readout axes)
    kspace_shifted = np.fft.fftshift(full_kspace, axes=(0, 1))

    # Compute the shifted indices of the acquired rows.
    # For an index i in unshifted space, the shifted index is (i + full_phase//2) % full_phase.
    acquired_shifted = [(row + full_phase // 2) % full_phase for row in acquired_rows]

    for i in range(full_phase):
        if i in acquired_shifted:
            continue  # Leave acquired rows untouched.
        for j in range(full_target_readout):
            mirror_i = (full_phase - i) % full_phase
            mirror_j = (full_target_readout - j) % full_target_readout
            kspace_shifted[i, j] = np.conjugate(kspace_shifted[mirror_i, mirror_j])

    # Inverse shift to return to the original ordering.
    filled_kspace = np.fft.ifftshift(kspace_shifted, axes=(0, 1))
    return filled_kspace


def reconstruct_image_from_kspace(kspace):
    """
    Reconstruct an image from k-space using the 2D inverse FFT.
    """
    img = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace)))
    return img


def compare_reconstructions_mri(image):
    """
    Simulate the MRI acquisition described:
      - The readout is oversampled (200 points) but then cropped (64:192) to 128.
      - Only 40 phase encoding lines (with a phase_offset of 12) are acquired,
        so that the acquired data land in rows 12 to 51 of the full 128.
      - Missing k-space rows (only those that were not acquired) are filled using
        conjugate symmetry.

    The reconstructed image is compared with the original.
    """
    full_phase = 128
    full_readout = 128
    crop_lower = 0
    crop_upper = 128  # so full_target_readout is 128
    measured_phase_lines = 40
    phase_offset = 
    full_target_readout = crop_upper - crop_lower  # 128

    # 1) Simulate the full oversampled k-space from the padded image.
    kspace_full = simulate_full_kspace_oversampled(
        image, full_readout, crop_lower, crop_upper
    )

    # 2) Simulate the under-sampled acquisition in the phase encoding direction.
    acquired_data, acquired_rows = simulate_under_sampled_acquisition(
        kspace_full,
        measured_phase_lines,
        phase_offset,
        full_phase,
        crop_lower,
        crop_upper,
    )

    # 3) Insert the acquired data into a full (but mostly empty) k-space array.
    full_kspace = insert_acquired_into_full(
        acquired_data, acquired_rows, full_phase, full_target_readout
    )

    # 4) Fill in missing k-space rows using conjugate symmetry (only outside acquired rows).
    filled_kspace = conjugate_symmetry_fill(full_kspace, acquired_rows)

    # 5) Reconstruct the image.
    img_recon = reconstruct_image_from_kspace(filled_kspace)

    # For comparison, also compute the reconstruction from the fully-sampled k-space.
    img_full = reconstruct_image_from_kspace(kspace_full)

    # Display the original image, the reconstructed image, and their difference.
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap="gray")
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(np.abs(img_recon), cmap="gray")
    plt.title("Reconstruction (Conjugate Symmetry)")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(np.round(np.abs(image - np.abs(img_recon)), 5), cmap="hot")
    plt.title("Difference")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Generate the real-valued image (128x128)
    image = generate_real_image(128)
    compare_reconstructions_mri(image)
