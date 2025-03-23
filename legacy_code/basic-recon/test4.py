import numpy as np
import matplotlib.pyplot as plt


def generate_complex_kspace(size):
    """Generate arbitrary complex-valued k-space data."""
    real_part = np.random.randn(size, size)
    imag_part = np.random.randn(size, size)
    return real_part + 1j * imag_part


def simulate_half_acquisition(kspace):
    """Simulate acquiring only half of k-space (top half)."""
    half_size = kspace.shape[0] // 2
    acquired = kspace[:half_size, :].copy()
    return acquired


def conjugate_symmetry_filling(acquired):
    """Fill missing k-space using conjugate symmetry."""
    full_size = acquired.shape[0] * 2
    filled = np.zeros((full_size, acquired.shape[1]), dtype=np.complex64)
    filled[: acquired.shape[0], :] = acquired
    # Fill bottom half using conjugate symmetry
    for i in range(acquired.shape[0], full_size):
        for j in range(acquired.shape[1]):
            filled[i, j] = np.conjugate(
                filled[full_size - i - 1, (acquired.shape[1] - j) % acquired.shape[1]]
            )
    return filled


def zero_fill_and_double_real(acquired):
    """Zero-fill missing k-space, IFFT, and double the real part."""
    full_size = acquired.shape[0] * 2
    zero_filled = np.zeros((full_size, acquired.shape[1]), dtype=np.complex64)
    zero_filled[: acquired.shape[0], :] = acquired
    # Reconstruct image
    img = np.fft.ifft2(np.fft.ifftshift(zero_filled))
    # Double the real part
    img_final = 2 * np.real(img)
    return img_final


def compare_methods(kspace):
    """Compare conjugate symmetry filling vs. zero-fill + double real part."""
    # Simulate half acquisition
    acquired = simulate_half_acquisition(kspace)

    # Method 1: Conjugate symmetry filling
    filled_conj = conjugate_symmetry_filling(acquired)
    img_conj = np.fft.ifft2(np.fft.ifftshift(filled_conj))

    # Method 2: Zero-fill + IFFT + double real part
    img_double = zero_fill_and_double_real(acquired)

    # Compare magnitude images
    mag_conj = np.abs(img_conj)
    mag_double = np.abs(img_double)

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(mag_conj, cmap="gray")
    plt.title("Conjugate Symmetry")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(mag_double, cmap="gray")
    plt.title("Zero-Fill + Double Real")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(np.abs(mag_conj - mag_double), cmap="hot")
    plt.title("Difference")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


# Generate arbitrary complex k-space data
kspace = np.real(generate_complex_kspace(128))

# Compare the two methods
compare_methods(kspace)
