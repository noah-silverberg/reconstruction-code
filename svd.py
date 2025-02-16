#!/usr/bin/env python3
import imageio
import numpy as np
import matplotlib.pyplot as plt


def load_gif_frames(filename):
    """Load all frames from a GIF and return as a NumPy array."""
    reader = imageio.get_reader(filename)
    frames = []
    for frame in reader:
        frames.append(frame)
    reader.close()
    return np.array(frames)  # shape: (n_frames, height, width, channels)


def perform_pca(frames):
    """
    Convert frames to grayscale, flatten, center, and perform SVD.
    Returns: U, S, Vt, data mean, variance explained, and frame shape (height, width).
    """
    n_frames, h, w, channels = frames.shape
    # Convert to grayscale (if RGB, using standard luminance weights)
    if channels == 3:
        gray = (
            0.299 * frames[:, :, :, 0]
            + 0.587 * frames[:, :, :, 1]
            + 0.114 * frames[:, :, :, 2]
        )
    else:
        gray = frames.squeeze()  # assume already grayscale

    # Flatten each frame to a vector: shape (n_frames, h*w)
    X = gray.reshape(n_frames, -1)
    # Center the data (mean subtraction)
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean

    # Perform SVD (which gives the same principal directions as PCA)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    # Compute variance explained by each component
    var_explained = (S**2) / np.sum(S**2)
    return U, S, Vt, X_mean, var_explained, (h, w)


def plot_components(Vt, frame_shape, var_explained, n_components=5):
    """
    Plot the first n_components principal components as images,
    with titles showing the percentage of variance explained.
    """
    fig, axes = plt.subplots(1, n_components, figsize=(15, 3))
    for i in range(n_components):
        comp = Vt[i, :].reshape(frame_shape)
        axes[i].imshow(comp, cmap="gray")
        axes[i].set_title(f"PC {i+1}\n{var_explained[i]*100:.1f}%")
        axes[i].axis("off")
    plt.tight_layout()
    plt.show()


def reconstruct_frames(U, S, Vt, X_mean, n_components, frame_shape):
    """
    Reconstruct the frames using only the first n_components components.
    Returns the reconstructed frames as a NumPy array of shape (n_frames, h, w).
    """
    # U shape: (n_frames, n_components_total), S: (n_components_total,), Vt shape: (n_components_total, h*w)
    # Reconstruct: X_recon = U[:, :n_components] * S[:n_components] @ Vt[:n_components, :] + X_mean.
    U_red = U[:, :n_components]
    S_red = S[:n_components]
    Vt_red = Vt[:n_components, :]
    X_recon = U_red @ np.diag(S_red) @ Vt_red + X_mean
    n_frames = X_recon.shape[0]
    return X_recon.reshape(n_frames, frame_shape[0], frame_shape[1])


def save_reconstructed_gif(frames, output_filename, duration=0.125):
    """
    Save a sequence of grayscale frames (values assumed in [0,255]) as a GIF.
    """
    writer = imageio.get_writer(output_filename, duration=duration)
    for frame in frames:
        # Ensure frame is uint8.
        frame_uint8 = np.clip(frame, 0, 255).astype(np.uint8)
        writer.append_data(frame_uint8)
    writer.close()
    print(f"Reconstructed GIF saved as {output_filename}")


def main():
    input_gif = "homodyne_cine_cropped.gif"  # Change to your GIF filename
    output_gif = (
        "homodyne_cine_cropped_svd.gif"  # Output filename for the reconstructed GIF
    )

    # Load GIF frames
    frames = load_gif_frames(input_gif)
    print(f"Loaded {frames.shape[0]} frames of shape {frames.shape[1:]}.")

    # Perform PCA via SVD
    U, S, Vt, X_mean, var_explained, frame_shape = perform_pca(frames)
    print("Variance explained by first 10 components:")
    for i, v in enumerate(var_explained[:10]):
        cumulative_var = np.cumsum(var_explained[: i + 1])[-1]
        print(f"  Component {i+1}: {v*100:.2f}%, Cumulative: {cumulative_var*100:.2f}%")

    # Compute cumulative variance explained.
    cumulative_variance = np.cumsum(var_explained)
    threshold = 0.95  # For % variance explained
    n_components_optimal = np.searchsorted(cumulative_variance, threshold) + 1
    print(
        f"Optimal number of components to reach {threshold*100:.0f}% variance: {n_components_optimal}"
    )

    # Plot the first few principal components
    # plot_components(Vt, frame_shape, var_explained, n_components=10)

    # Reconstruct frames using only a subset of components
    frames_recon = reconstruct_frames(
        U, S, Vt, X_mean, n_components_optimal, frame_shape
    )
    # Normalize reconstruction to 0-255 for visualization
    frames_recon_norm = (
        (frames_recon - frames_recon.min())
        / (frames_recon.max() - frames_recon.min())
        * 255
    )

    # Save the reconstructed GIF
    save_reconstructed_gif(frames_recon_norm, output_gif, duration=0.125)


if __name__ == "__main__":
    main()
