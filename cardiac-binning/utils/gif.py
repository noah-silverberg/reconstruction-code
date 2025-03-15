import numpy as np
import imageio
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import utils.reconstruction as reconstruction


def save_images_as_gif(images, filename, duration=0.2, cmap="gray"):
    """
    Save a sequence of images as an animated GIF.

    Parameters:
      images : np.ndarray
          A 3D numpy array of shape (num_frames, height, width) representing the image sequence.
      filename : str
          The output filename (should end with .gif).
      duration : float, optional
          Duration (in seconds) between frames (default is 0.2 sec).
      cmap : str, optional
          Color map to use if the images are grayscale.

    Note:
      The images are normalized to 0-255.
    """
    frames = []
    for img in images:
        # Normalize image to 0-255.
        img_norm = (img - np.min(img)) / (np.ptp(img) + 1e-8)  # avoid division by zero
        img_uint8 = (img_norm * 255).astype(np.uint8)
        frames.append(img_uint8)
    imageio.mimsave(filename, frames, duration=duration)
    print(f"Saved GIF to {filename}")


def display_images_as_gif(images, interval=200, cmap="gray"):
    """
    Display a sequence of images as an animated GIF using Matplotlib.

    Parameters:
      images : np.ndarray
          A 3D numpy array of shape (num_frames, height, width) representing the image sequence.
      interval : int, optional
          Interval between frames in milliseconds (default is 200 ms).
      cmap : str, optional
          Colormap for display.
    """
    fig = plt.figure()
    im = plt.imshow(images[0], cmap=cmap, animated=True)
    plt.axis("off")

    def update_frame(i):
        im.set_array(images[i])
        return [im]

    ani = animation.FuncAnimation(
        fig, update_frame, frames=images.shape[0], interval=interval, blit=True
    )
    plt.show()


def save_kspace_as_gif(kspace, filename, duration=0.2, cmap="gray"):
    """
    Save binned k-space data as an animated GIF with overlay lines marking the center.

    Parameters:
      kspace : np.ndarray
          A 4D array of shape (num_bins, n_rows, n_coils, n_readout) representing the binned k-space.
      filename : str
          The output filename (should end with .gif).
      duration : float, optional
          Duration (in seconds) between frames (default is 0.2 sec).
      cmap : str, optional
          Colormap for display.

    The function combines the coil data using a sum-of-squares, computes the log-magnitude,
    and overlays a vertical and horizontal red dashed line through the center of each frame.
    """
    import matplotlib.pyplot as plt
    import imageio

    num_bins, n_rows, n_coils, n_readout = kspace.shape
    # Combine coil data using sum-of-squares to obtain a single k-space magnitude per bin.
    kspace_combined = np.sqrt(
        np.sum(np.abs(kspace) ** 2, axis=2)
    )  # shape: (num_bins, n_rows, n_readout)
    # Compute log magnitude for better visualization.
    kspace_log = np.log(1 + kspace_combined)
    kspace_log = (kspace_log - kspace_log.min()) / (kspace_log.max() - kspace_log.min())

    frames = []
    center_row = n_rows // 2
    center_col = n_readout // 2

    for i in range(num_bins):
        fig, ax = plt.subplots()
        im = ax.imshow(kspace_log[i], cmap=cmap)
        # Overlay center lines.
        ax.axvline(x=center_col, color="red", linestyle="--")
        ax.axhline(y=center_row, color="red", linestyle="--")
        ax.set_title(f"K-space bin {i+1}/{num_bins}")
        ax.axis("off")
        # Draw the canvas.
        fig.canvas.draw()
        # Instead of using tostring_rgb(), get the RGBA buffer from the renderer.
        buf = np.array(fig.canvas.renderer.buffer_rgba())
        # Discard the alpha channel to get an RGB image.
        image = buf[..., :3]
        frames.append(image)
        plt.close(fig)

    imageio.mimsave(filename, frames, duration=duration)
    print(f"Saved k-space GIF to {filename}")


def save_reconstructed_gif(frames, output_filename, duration=0.125):
    """
    Save a sequence of grayscale frames (assumed scaled to 0-255) as a GIF.

    Parameters:
      frames         : numpy array of shape (n_frames, height, width).
      output_filename: String filename for the GIF.
      duration       : Frame duration in seconds.
    """
    import imageio

    writer = imageio.get_writer(output_filename, duration=duration)
    for frame in frames:
        frame_uint8 = np.clip(frame, 0, 255).astype(np.uint8)
        writer.append_data(frame_uint8)
    writer.close()
    print(f"Reconstructed GIF saved as {output_filename}")


def display_cine(frames, interval=125):
    """
    Display a cine (list/array of frames) as an animated plot.

    Parameters:
      frames  : numpy array of shape (n_frames, height, width).
      interval: Delay between frames in milliseconds.
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
    U,
    S,
    Vt,
    X_mean,
    frame_shape,
    selections,
    n_bins,
    n_coils,
    interval=125,
    output_filename=None,
):
    """
    For a list of selections (each selection is an integer or list of PC indices),
    reconstruct the corresponding cines from k-space PCA and display them simultaneously.

    Parameters:
      selections   : List of selections.
      (Other parameters as in previous functions.)
    """
    cines = []
    for sel in selections:
        cine = reconstruction.reconstruct_with_selected_components_kspace(
            U, S, Vt, X_mean, sel, frame_shape, n_bins, n_coils
        )
        cines.append(cine)
    n = len(cines)
    if n > 3:
        nrows = 2
        ncols = math.ceil(n / nrows)
    else:
        nrows = 1
        ncols = n
    fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
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
            if isinstance(selections[i], int)
            else f"Selections: {selections[i]}"
        )
        axs[i].set_title(title)
        ims.append(im)
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
    if output_filename is not None:
        ani.save(output_filename, writer="pillow", fps=1000 / interval)
        print(f"Animation saved as {output_filename}")
    plt.show()
