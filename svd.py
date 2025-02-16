#!/usr/bin/env python3
import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math


def load_gif_frames(filename):
    """Load all frames from a GIF and return as a NumPy array."""
    reader = imageio.get_reader(filename)
    frames = []
    for i, frame in enumerate(reader):
        if i >= 2:
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


def alias_frequency(f, sampling_rate):
    # Compute the aliased frequency in [0, sampling_rate/2]
    n = np.round(f / sampling_rate)
    return abs(f - n * sampling_rate)


def plot_pc_time_series_and_fft(U, S, sampling_rate=1.0, n_components=5):
    """
    For each of the first n_components, plot the time-series (per frame),
    its FFT magnitude spectrum, and its power spectral density (PSD).

    Parameters:
      U             : Temporal coefficients from SVD (shape: n_frames x n_components_total)
      S             : Singular values from SVD (1D array)
      sampling_rate : Frames per second (default 1.0 Hz if not provided)
      n_components  : Number of components to display (default 5)
    """
    n_frames = U.shape[0]
    time = np.arange(n_frames) / sampling_rate  # time vector

    # Create a figure with 3 columns: time series, FFT magnitude, and PSD.
    fig, axs = plt.subplots(n_components, 3, figsize=(18, 3 * n_components))
    # If n_components is 1, ensure axs is 2D.
    if n_components == 1:
        axs = np.expand_dims(axs, axis=0)

    for i in range(n_components):
        # Compute the time series for PC i.
        ts = U[:, i] * S[i]

        # --- Time-domain plot ---
        axs[i, 0].plot(time, ts, marker="o")
        axs[i, 0].set_title(f"PC {i+1} Time Series")
        axs[i, 0].set_xlabel("Time (s)")
        axs[i, 0].set_ylabel("Amplitude")

        # --- Frequency-domain: FFT Magnitude ---
        fft_ts = np.fft.fft(ts)
        freqs = np.fft.fftfreq(n_frames, d=1 / sampling_rate)
        pos = freqs >= 0
        axs[i, 1].plot(freqs[pos], np.abs(fft_ts)[pos], marker="o")
        axs[i, 1].set_title(f"PC {i+1} FFT Magnitude")
        axs[i, 1].set_xlabel("Frequency (Hz)")
        axs[i, 1].set_ylabel("Magnitude")

        # --- Power Spectral Density (PSD) ---
        # PSD computed as power per frequency bin.
        psd = (np.abs(fft_ts) ** 2) / n_frames
        axs[i, 2].plot(freqs[pos], psd[pos], marker="o")
        axs[i, 2].set_title(f"PC {i+1} Power Spectral Density")
        axs[i, 2].set_xlabel("Frequency (Hz)")
        axs[i, 2].set_ylabel("Power")

        # --- Markers for expected frequency ranges ---
        # Expected ranges (in Hz)
        breathing_range = (0.2, 0.5)  # typical breathing frequencies
        heart_range = (0.8, 1.2)  # typical heart beat frequencies

        heart_range_aliased = (
            alias_frequency(heart_range[0], sampling_rate),
            alias_frequency(heart_range[1], sampling_rate),
        )
        heart_range_aliased = tuple(sorted(heart_range_aliased))

        # Add shaded regions on FFT magnitude plot.
        axs[i, 1].axvspan(
            breathing_range[0],
            breathing_range[1],
            color="green",
            alpha=0.2,
            label="Breathing",
        )
        axs[i, 1].axvspan(
            heart_range_aliased[0],
            heart_range_aliased[1],
            color="red",
            alpha=0.2,
            label="Heart",
        )
        # axs[i, 1].legend(loc="upper right")

        # Add shaded regions on PSD plot.
        axs[i, 2].axvspan(
            breathing_range[0],
            breathing_range[1],
            color="green",
            alpha=0.2,
            label="Breathing",
        )
        axs[i, 2].axvspan(
            heart_range_aliased[0],
            heart_range_aliased[1],
            color="red",
            alpha=0.2,
            label="Heart",
        )
        # axs[i, 2].legend(loc="upper right")

    plt.tight_layout()
    plt.show()


def reconstruct_with_selected_components(U, S, Vt, X_mean, selection, frame_shape):
    """
    Reconstruct frames using only the selected principal components.

    Parameters:
      U, S, Vt, X_mean : Results from SVD/PCA (X_centered = U * S * Vt)
      selection       : Either an integer (a single PC) or a list/tuple of PC indices (0-indexed)
      frame_shape     : Tuple (height, width) of the original frames.

    Returns:
      Reconstructed frames of shape (n_frames, height, width)
    """
    n_frames = U.shape[0]
    # Ensure selection is a list even if a single integer is passed.
    if isinstance(selection, (int, np.integer)):
        components = [selection]
    else:
        components = list(selection)

    # Initialize reconstruction with zeros.
    X_recon = np.zeros((n_frames, frame_shape[0] * frame_shape[1]), dtype=np.float64)
    # Sum contributions from the selected PCs.
    for i in components:
        X_recon += np.outer(U[:, i] * S[i], Vt[i, :])
    # Add back the mean.
    X_recon += X_mean
    return X_recon.reshape(n_frames, frame_shape[0], frame_shape[1])


def display_cine(frames, interval=125):
    """
    Display a cine (list/array of frames) as an animated plot.

    Parameters:
      frames   : Array of shape (n_frames, height, width)
      interval : Delay between frames in milliseconds.
    """
    fig, ax = plt.subplots()
    im = ax.imshow(frames[0], cmap="gray", animated=True)
    ax.axis("off")

    def update(frame):
        im.set_array(frame)
        return [im]

    ani = animation.FuncAnimation(
        fig, update, frames=frames, interval=interval, blit=True
    )
    plt.show()


def display_multiple_cines(
    U, S, Vt, X_mean, frame_shape, selections, interval=125, output_filename=None
):
    """
    For a list of selections (each selection can be an int or list/tuple),
    reconstruct the corresponding cines and display them simultaneously in a grid.
    Optionally, save the resulting animation to a file.

    Parameters:
      U, S, Vt, X_mean : SVD/PCA results.
      frame_shape      : Tuple (height, width) of the original frames.
      selections       : List of selections. Each element is either an integer or list/tuple.
      interval         : Delay between frames in milliseconds.
      output_filename  : If provided (e.g., "output.gif"), saves the animation to this file.
    """
    # Reconstruct a cine for each selection.
    cines = []
    for sel in selections:
        cine = reconstruct_with_selected_components(U, S, Vt, X_mean, sel, frame_shape)
        cines.append(cine)

    n = len(cines)
    # Decide on layout: if more than 3 selections, use 2 rows; otherwise, use one row.
    if n > 3:
        nrows = 2
        ncols = math.ceil(n / nrows)
    else:
        nrows = 1
        ncols = n

    fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    # Flatten the axs array for easy iteration.
    if nrows * ncols > 1:
        axs = np.array(axs).flatten()
    else:
        axs = [axs]

    ims = []
    for i in range(n):
        axs[i].axis("off")
        im = axs[i].imshow(cines[i][0], cmap="gray", animated=True)
        title = (
            f"Selection: {selections[i]}"
            if isinstance(selections[i], (int, np.integer))
            else f"Selections: {selections[i]}"
        )
        axs[i].set_title(title)
        ims.append(im)
    # Hide any unused subplots.
    for j in range(n, len(axs)):
        axs[j].axis("off")

    n_frames = cines[0].shape[0]

    def update(frame):
        for i in range(n):
            ims[i].set_array(cines[i][frame])
        return ims

    ani = animation.FuncAnimation(
        fig, update, frames=range(n_frames), interval=interval, blit=True
    )

    # Save the animation if output_filename is provided.
    if output_filename is not None:
        # Use the pillow writer to save as GIF.
        ani.save(output_filename, writer="pillow", fps=1000 / interval)
        print(f"Animation saved as {output_filename}")

    plt.show()


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
        print(f"  Component {i}: {v*100:.2f}%, Cumulative: {cumulative_var*100:.2f}%")

    # Compute cumulative variance explained.
    cumulative_variance = np.cumsum(var_explained)
    threshold = 0.95  # For % variance explained
    n_components_optimal = np.searchsorted(cumulative_variance, threshold) + 1
    print(
        f"Optimal number of components to reach {threshold*100:.0f}% variance: {n_components_optimal}"
    )

    # Plot the first few principal components
    # plot_components(Vt, frame_shape, var_explained, n_components=10)

    plot_pc_time_series_and_fft(U, S, sampling_rate=8, n_components=5)

    # selected_components = [0]  # 0-indexed
    # frames_selected = reconstruct_with_selected_components(
    #     U, S, Vt, X_mean, selected_components, frame_shape
    # )
    # display_cine(frames_selected, interval=125)

    selected_components = [[0, 3, 4], [i for i in range(10) if i not in [0, 3, 4]]]
    display_multiple_cines(
        U,
        S,
        Vt,
        X_mean,
        frame_shape,
        selected_components,
        interval=125,
        output_filename="homodyne_cine_cropped_svd_multiple.gif",
    )

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
