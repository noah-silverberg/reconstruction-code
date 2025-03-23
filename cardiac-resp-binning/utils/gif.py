"""
gif.py

Provides helper functions to create and display animated GIFs from image sequences.
Used for both reconstructed images and k-space visualizations.
"""

import os
import numpy as np
import imageio
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML


def save_images_as_gif(images, filename, duration=200, cmap="gray"):
    """
    Save a sequence of images as an animated GIF.

    Parameters
    ----------
    images : np.ndarray
        Image stack of shape (num_frames, height, width).
    filename : str
        Output GIF filename.
    duration : float
        Time (in milliseconds) between frames.
    cmap : str
        Matplotlib colormap for normalization (not applied when saving frames, but included for consistency).

    Returns
    -------
    None
    """
    frames = []
    for img in images:
        # Normalize to 0-255
        norm_img = (img - np.min(img)) / (np.ptp(img) + 1e-8)
        frames.append((norm_img * 255).astype(np.uint8))

    imageio.mimsave(filename, frames, duration=duration / 1000.0, loop=0)
    print(f"Saved GIF to {filename}")


def display_images_as_gif(images, interval=200, cmap="gray", notebook=True):
    """
    Display a sequence of images as an animated GIF inline (in Jupyter).

    Parameters
    ----------
    images : np.ndarray
        Image stack of shape (num_frames, height, width).
    interval : int
        Interval between frames in milliseconds.
    cmap : str
        Matplotlib colormap for plotting.
    notebook : bool
        If True, returns HTML for inline display in Jupyter. If False, uses plt.show().

    Returns
    -------
    IPython.display.HTML or None
        Returns the HTML animation if notebook=True, else None.
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

    if notebook:
        plt.close(fig)
        return HTML(ani.to_jshtml())
    else:
        plt.show()


def display_kspace_as_gif(kspace, duration=0.2, cmap="gray"):
    """
    Display k-space data as an animated GIF inline, summing coil dimension as needed.

    Parameters
    ----------
    kspace : np.ndarray
        4D array of shape (num_frames, n_rows, n_coils, n_readout).
    duration : float
        Duration between frames in seconds.
    cmap : str
        Matplotlib colormap.

    Returns
    -------
    IPython.display.HTML
        HTML object for Jupyter inline display.
    """
    num_frames, n_rows, n_coils, n_readout = kspace.shape

    combined = np.sqrt(np.sum(np.abs(kspace) ** 2, axis=2))
    log_kspace = np.log(1 + combined)
    log_kspace = (log_kspace - log_kspace.min()) / (log_kspace.max() - log_kspace.min())

    frames = []
    center_row = n_rows // 2
    center_col = n_readout // 2

    for i in range(num_frames):
        fig, ax = plt.subplots(figsize=(5, 4), frameon=False)
        ax.axis("off")
        ax.imshow(log_kspace[i], cmap=cmap)
        ax.axvline(x=center_col, color="red", linestyle="--", linewidth=2)
        ax.axhline(y=center_row, color="red", linestyle="--", linewidth=2)
        ax.set_title(f"Frame {i+1}/{num_frames}")
        fig.canvas.draw()
        buf = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        frames.append(buf)
        plt.close(fig)

    fig_anim = plt.figure()
    plt.axis("off")
    anim = animation.ArtistAnimation(
        fig_anim,
        [[plt.imshow(frame, animated=True)] for frame in frames],
        interval=duration * 1000,
        blit=True,
    )
    plt.close(fig_anim)
    return HTML(anim.to_jshtml())


def save_kspace_as_gif(kspace, filename, duration=200, cmap="gray"):
    """
    Save k-space data as a GIF, applying a log transform and center-line overlay.

    Parameters
    ----------
    kspace : np.ndarray
        4D array of shape (num_frames, n_rows, n_coils, n_readout).
    filename : str
        Output GIF file name.
    duration : float
        Time between frames in milliseconds.
    cmap : str
        Matplotlib colormap.

    Returns
    -------
    None
    """
    num_frames, n_rows, n_coils, n_readout = kspace.shape
    combined = np.sqrt(np.sum(np.abs(kspace) ** 2, axis=2))
    log_kspace = np.log(1 + combined)
    log_kspace = (log_kspace - log_kspace.min()) / (log_kspace.max() - log_kspace.min())

    frames = []
    center_row = n_rows // 2
    center_col = n_readout // 2

    for i in range(num_frames):
        fig, ax = plt.subplots()
        ax.imshow(log_kspace[i], cmap=cmap)
        ax.axvline(x=center_col, color="red", linestyle="--")
        ax.axhline(y=center_row, color="red", linestyle="--")
        ax.set_title(f"Frame {i+1}/{num_frames}")
        ax.axis("off")
        fig.canvas.draw()
        buf = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        frames.append(buf)
        plt.close(fig)

    imageio.mimsave(filename, frames, duration=duration / 1000.0, loop=0)
    print(f"Saved k-space GIF to {filename}")
