import numpy as np
import imageio
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import utils.reconstruction as reconstruction
from IPython.display import HTML
import os
import pydicom


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


def display_images_as_gif(images, interval=200, cmap="gray", notebook=True):
    """
    Display a sequence of images as an animated GIF.

    When running in a Jupyter notebook, the animation is converted to HTML and displayed.

    Parameters:
      images : np.ndarray
          A 3D numpy array of shape (num_frames, height, width) representing the image sequence.
      interval : int, optional
          Interval between frames in milliseconds (default is 200 ms).
      cmap : str, optional
          Colormap for display.
      notebook : bool, optional
          If True, display in a Jupyter notebook (default True).
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
        # Return the animation as HTML.
        return HTML(ani.to_jshtml())
    else:
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


def dicom_framerate_from_folder(folder_path):
    # Get list of DICOM file paths in the folder.
    dicom_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith(".dcm")
    ]

    # Read all DICOMs.
    dicoms = [pydicom.dcmread(f) for f in dicom_files]

    # Sort by InstanceNumber (or use AcquisitionTime if needed)
    dicoms.sort(key=lambda ds: int(ds.InstanceNumber))

    # Option 1: If FrameTime is available, use that (usually in ms)
    if hasattr(dicoms[0], "FrameTime"):
        frame_time_ms = float(dicoms[0].FrameTime)  # in milliseconds
        frame_time = frame_time_ms / 1000.0  # seconds
        framerate = 1.0 / frame_time
        return framerate, frame_time
    # Option 2: If CineRate is available, use it (in fps)
    elif hasattr(dicoms[0], "CineRate"):
        framerate = float(dicoms[0].CineRate)
        frame_time = 1.0 / framerate
        return framerate, frame_time
    # Option 3: Compute the average time difference from AcquisitionTime.
    elif hasattr(dicoms[0], "AcquisitionTime"):

        def time_to_seconds(t):
            # t is typically a string like "123456.789"
            hours = float(t[:2])
            minutes = float(t[2:4])
            seconds = float(t[4:])
            return hours * 3600 + minutes * 60 + seconds

        times = [time_to_seconds(ds.AcquisitionTime) for ds in dicoms]
        # Compute differences between successive frames.
        diffs = np.diff(times)
        avg_diff = sum(diffs) / len(diffs)
        framerate = 1.0 / avg_diff
        return framerate, avg_diff
    else:
        return None, None


def print_all_dicom_info(path):
    """
    Print all DICOM metadata from a file or from the first DICOM file in a folder.

    Parameters:
      path (str): Path to a DICOM file or a folder containing DICOM files.
    """
    if os.path.isdir(path):
        # Process each file ending with .dcm (case-insensitive) in the folder.
        dicom_files = [
            os.path.join(path, f)
            for f in os.listdir(path)
            if f.lower().endswith(".dcm")
        ]
        if not dicom_files:
            print("No DICOM files found in the specified folder.")
            return
        file = dicom_files[0]
        print(f"===== Metadata for file: {file} =====")
        ds = pydicom.dcmread(file)
        print(ds)
    else:
        ds = pydicom.dcmread(path)
        print(ds)
