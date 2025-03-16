"""
gif.py

This module provides helper functions for creating animated GIFs from image sequences,
displaying them, and extracting DICOM frame rate information.
"""

import os
import numpy as np
import imageio
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import pydicom


def save_images_as_gif(images, filename, duration=0.2, cmap="gray"):
    """
    Save a sequence of images as an animated GIF.

    Parameters:
        images (np.ndarray): (num_frames, height, width)
        filename (str): Output filename.
        duration (float): Time between frames (seconds).
        cmap (str): Color map (if needed).
    """
    frames = []
    for img in images:
        norm_img = (img - np.min(img)) / (np.ptp(img) + 1e-8)
        frames.append((norm_img * 255).astype(np.uint8))
    imageio.mimsave(filename, frames, duration=duration)
    print(f"Saved GIF to {filename}")


def display_images_as_gif(images, interval=200, cmap="gray", notebook=True):
    """
    Display a sequence of images as an animated GIF.

    Parameters:
        images (np.ndarray): (num_frames, height, width)
        interval (int): Interval between frames (ms).
        notebook (bool): If True, return HTML for Jupyter.
    """
    fig = plt.figure(facecolor="black")
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
    Display cine k-space data as an animated GIF with overlaid center lines.

    Parameters:
        kspace (np.ndarray): 4D array with shape (num_frames, n_rows, n_coils, readout).
        duration (float): Duration (in seconds) between frames.
        cmap (str): Color map for display.

    Returns:
        IPython.display.HTML: An HTML animation (using jshtml) for interactive display.

    The function computes the sum-of-squares over coils, applies a logarithmic transformation
    for better visualization, overlays red dashed center lines, and creates an animation.
    """
    num_frames, n_rows, n_coils, n_readout = kspace.shape
    # Combine coil data using sum-of-squares and compute log magnitude
    combined = np.sqrt(np.sum(np.abs(kspace) ** 2, axis=2))
    log_kspace = np.log(1 + combined)
    log_kspace = (log_kspace - log_kspace.min()) / (log_kspace.max() - log_kspace.min())

    frames = []
    center_row = n_rows // 2
    center_col = n_readout // 2

    # Create frames
    for i in range(num_frames):
        fig, ax = plt.subplots(figsize=(8, 6), frameon=False)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        ax.axis("off")
        ax.imshow(log_kspace[i], cmap=cmap)
        ax.axvline(x=center_col, color="red", linestyle="--", linewidth=2)
        ax.axhline(y=center_row, color="red", linestyle="--", linewidth=2)
        ax.set_title(f"Frame {i+1}/{num_frames}")
        fig.canvas.draw()
        buf = np.array(fig.canvas.renderer.buffer_rgba())
        image = buf[..., :3]  # Remove alpha channel
        frames.append(image)
        plt.close(fig)

    # Create animation
    fig_anim = plt.figure(figsize=(8, 6), frameon=False)
    plt.axis("off")
    anim = animation.ArtistAnimation(
        fig_anim,
        [[plt.imshow(frame, animated=True)] for frame in frames],
        interval=duration * 1000,
        blit=True,
    )
    plt.close(fig_anim)
    return HTML(anim.to_jshtml())


def save_kspace_as_gif(kspace, filename, duration=0.2, cmap="gray"):
    """
    Save cine k-space data as a GIF with overlaid center lines.

    Parameters:
        kspace (np.ndarray): (num_frames, n_rows, coils, readout)
        filename (str): Output filename.
        duration (float): Frame duration (seconds).
        cmap (str): Color map.
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
        buf = np.array(fig.canvas.renderer.buffer_rgba())
        image = buf[..., :3]
        frames.append(image)
        plt.close(fig)
    imageio.mimsave(filename, frames, duration=duration)
    print(f"Saved k-space GIF to {filename}")


def dicom_framerate_from_folder(folder_path):
    """
    Extract the frame rate and frame time from DICOM files in a folder.

    Parameters:
        folder_path (str): Path to DICOM folder.

    Returns:
        tuple: (framerate in Hz, frame_time in seconds)
    """
    dicom_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(".dcm")
    ]
    dicoms = [pydicom.dcmread(f) for f in dicom_files]
    dicoms.sort(key=lambda ds: int(ds.InstanceNumber))
    if hasattr(dicoms[0], "FrameTime"):
        frame_time = float(dicoms[0].FrameTime) / 1000.0
        return 1.0 / frame_time, frame_time
    elif hasattr(dicoms[0], "CineRate"):
        framerate = float(dicoms[0].CineRate)
        return framerate, 1.0 / framerate
    elif hasattr(dicoms[0], "AcquisitionTime"):

        def time_to_sec(t):
            return int(t[:2]) * 3600 + int(t[2:4]) * 60 + float(t[4:])

        times = [time_to_sec(ds.AcquisitionTime) for ds in dicoms]
        diffs = np.diff(times)
        avg_diff = np.mean(diffs)
        return 1.0 / avg_diff, avg_diff
    else:
        return None, None


def print_all_dicom_info(path):
    """
    Print metadata from a DICOM file or the first file in a folder.

    Parameters:
        path (str): File or folder path.
    """
    if os.path.isdir(path):
        dicom_files = [
            os.path.join(path, f)
            for f in os.listdir(path)
            if f.lower().endswith(".dcm")
        ]
        if not dicom_files:
            print("No DICOM files found.")
            return
        file = dicom_files[0]
    else:
        file = path
    ds = pydicom.dcmread(file)
    print(f"Metadata for {file}:\n", ds)
