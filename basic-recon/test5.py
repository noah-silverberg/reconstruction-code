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


def image_to_kspace(image):
    """
    Convert a real-valued image to k-space using the 2D Fourier transform.
    """
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image)))


def simulate_half_acquisition(kspace):
    """
    Simulate acquiring half of k-space (top half including the center line).
    """
    half_size = kspace.shape[0] // 2
    # Include the center row (half_size)
    acquired = kspace[: half_size + 1, :].copy()
    return acquired


def conjugate_symmetry_filling(acquired):
    """
    Fill missing k-space using conjugate symmetry.
    The full k-space size is computed from the acquired half-plus-center.
    """
    # For an even full size, if acquired has (N/2 + 1) rows then:
    full_size = (acquired.shape[0] - 1) * 2
    filled = np.zeros((full_size, acquired.shape[1]), dtype=np.complex64)

    # Place the acquired half (which includes the center row)
    filled[: acquired.shape[0], :] = acquired

    # Fill the missing rows using conjugate symmetry
    for i in range(acquired.shape[0], full_size):
        for j in range(acquired.shape[1]):
            mirror_i = (full_size - i) % full_size
            mirror_j = (acquired.shape[1] - j) % acquired.shape[1]
            filled[i, j] = np.conjugate(filled[mirror_i, mirror_j])
    return filled


def reconstruct_image(kspace):
    """
    Reconstruct the image from k-space using the inverse Fourier transform.
    """
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace)))


def check_conjugate_symmetry(kspace, tolerance=1e-10):
    """
    Check if k-space is conjugate symmetric.
    """
    N, M = kspace.shape
    is_symmetric = True

    for i in range(N):
        for j in range(M):
            # Compute the mirrored indices
            mirror_i = (N - i) % N
            mirror_j = (M - j) % M
            # Check if k(i,j) == k*(mirror_i, mirror_j)
            if not np.allclose(
                kspace[i, j], np.conjugate(kspace[mirror_i, mirror_j]), atol=tolerance
            ):
                is_symmetric = False
                print(
                    f"Violation at ({i}, {j}): {kspace[i, j]} != {np.conjugate(kspace[mirror_i, mirror_j])}"
                )
                return is_symmetric
    return is_symmetric


def compare_reconstructions(image):
    """
    Compare the original image to the conjugate symmetry reconstruction.
    """
    # Convert the image to k-space
    kspace = image_to_kspace(image)

    # Check if the full k-space is conjugate symmetric
    print("Checking conjugate symmetry of full k-space...")
    if check_conjugate_symmetry(kspace):
        print("Full k-space is conjugate symmetric.")
    else:
        print("Full k-space is NOT conjugate symmetric.")

    # Simulate acquiring only half of k-space
    acquired = simulate_half_acquisition(kspace)

    # Reconstruct using conjugate symmetry
    filled_conj = conjugate_symmetry_filling(acquired)
    img_conj = reconstruct_image(filled_conj)

    # Check if the filled k-space is conjugate symmetric
    print("Checking conjugate symmetry of filled k-space...")
    if check_conjugate_symmetry(filled_conj):
        print("Filled k-space is conjugate symmetric.")
    else:
        print("Filled k-space is NOT conjugate symmetric.")

    # Compare the original image to the reconstructed image
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap="gray")
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(np.abs(img_conj), cmap="gray")
    plt.title("Conjugate Symmetry Reconstruction")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(np.round(np.abs(image - np.abs(img_conj)), 5), cmap="hot")
    plt.title("Difference")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


# Generate a real-valued image
image = generate_real_image(128)

# Compare the original image to the conjugate symmetry reconstruction
compare_reconstructions(image)
