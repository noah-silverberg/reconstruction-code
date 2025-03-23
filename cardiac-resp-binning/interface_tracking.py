#!/usr/bin/env python3
"""
interface_tracking.py

This script reconstructs k-space data to produce a cine, then automatically tracks
the superior–inferior position of the liver–lung interface over time using a Sobel gradient
within a specified ROI. It optionally displays the tracked interface, and
plots the results compared to an external respiratory signal if available.
"""

import yaml
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Utility imports
import utils.data_ingestion as di
import utils.ecg_resp as ecg_resp
import utils.reconstruction as recon


def load_config(config_file="config.yaml"):
    """
    Load pipeline configuration from a YAML file.

    Parameters
    ----------
    config_file : str
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Configuration dictionary.
    """
    with open(config_file, "r") as f:
        return yaml.safe_load(f)


def track_interface(cine_frames, roi=(100, 200, 50, 250)):
    """
    Track the maximum vertical gradient within a specified region of interest (ROI),
    assumed to correspond to the liver–lung interface.

    Parameters
    ----------
    cine_frames : np.ndarray
        3D array of shape (n_frames, height, width), representing grayscale image frames.
    roi : tuple of int
        (row_start, row_end, col_start, col_end) defining the region of interest.

    Returns
    -------
    np.ndarray
        1D array of interface row positions (one per frame).
    """
    interface_positions = []
    for frame in cine_frames:
        roi_frame = frame[roi[0] : roi[1], roi[2] : roi[3]]
        # Compute vertical derivative via Sobel filter
        grad_y = cv2.Sobel(roi_frame, cv2.CV_64F, 0, 1, ksize=5)
        # Average absolute gradient across columns
        profile = np.mean(np.abs(grad_y), axis=1)
        # The row with maximum gradient plus the offset
        interface_row = np.argmax(profile) + roi[0]
        interface_positions.append(interface_row)
    return np.array(interface_positions)


def display_frame_with_roi(frame, roi, interface_row):
    """
    Display a single frame with the ROI box and a horizontal line at the interface location.

    Parameters
    ----------
    frame : np.ndarray
        2D image (height, width).
    roi : tuple of int
        (row_start, row_end, col_start, col_end).
    interface_row : int
        The vertical row index of the detected interface in this frame.
    """
    # Normalize for display
    frame_norm = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    frame_bgr = cv2.cvtColor(frame_norm, cv2.COLOR_GRAY2BGR)

    x, y, w, h = roi[2], roi[0], roi[3] - roi[2], roi[1] - roi[0]
    cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 255, 0), 1)

    cv2.line(
        frame_bgr,
        (0, int(interface_row)),
        (frame_bgr.shape[1], int(interface_row)),
        (0, 0, 255),
        1,
    )

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(6, 6))
    plt.imshow(frame_rgb)
    plt.title("Sample Frame with ROI and Detected Interface")
    plt.axis("off")
    plt.show()


def display_cine_with_roi(cine_frames, roi, interface_positions, delay=0.1):
    """
    Animate the entire cine with ROI and interface overlay.

    Parameters
    ----------
    cine_frames : np.ndarray
        3D array (n_frames, height, width).
    roi : tuple of int
        (row_start, row_end, col_start, col_end).
    interface_positions : np.ndarray
        1D array of interface row positions for each frame.
    delay : float
        Pause (in seconds) between displaying frames.
    """
    plt.figure(figsize=(6, 6))
    for i, frame in enumerate(cine_frames):
        frame_norm = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(
            np.uint8
        )
        frame_bgr = cv2.cvtColor(frame_norm, cv2.COLOR_GRAY2BGR)

        x, y, w, h = roi[2], roi[0], roi[3] - roi[2], roi[1] - roi[0]
        cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.line(
            frame_bgr,
            (0, int(interface_positions[i])),
            (frame_bgr.shape[1], int(interface_positions[i])),
            (0, 0, 255),
            1,
        )

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        plt.imshow(frame_rgb)
        plt.title(f"Frame {i+1}/{cine_frames.shape[0]}")
        plt.axis("off")
        plt.pause(delay)
        plt.clf()
    plt.close()


def main():
    """
    Main function to track the liver–lung interface in reconstructed cines,
    optionally comparing it to an external respiratory signal.
    """
    config = load_config()
    twix_file = config["data"]["twix_file"]
    dicom_folder = config["data"]["dicom_folder"]
    n_frames = config["data"]["n_frames"]
    row_offset = config["data"]["offset"]
    extended_phase_lines = config["data"]["extended_pe_lines"]

    # Read raw k-space
    scans = di.read_twix_file(twix_file, include_scans=[-1], parse_pmu=False)
    kspace = di.extract_image_data(scans)

    # Derive acquisition timing info
    framerate, _ = di.get_dicom_framerate(dicom_folder)
    total_phase_encodes = kspace.shape[0]
    n_phase_encodes_per_frame = total_phase_encodes // n_frames
    fs = framerate * n_phase_encodes_per_frame

    # Reshape into frames
    kspace_frames = kspace.reshape(
        n_frames, n_phase_encodes_per_frame, kspace.shape[1], kspace.shape[2]
    )

    # Direct IFFT reconstruction
    cine = recon.direct_ifft_reconstruction(
        kspace_frames,
        extended_pe_lines=extended_phase_lines,
        offset=row_offset,
        use_conjugate_symmetry=True,
    )

    # Optional orientation adjustments
    cine = np.rot90(cine, k=1, axes=(1, 2))
    cine = np.flip(cine, axis=2)
    cine = cine[:, 64:-64, :]

    # Track interface
    roi = config.get("tracking", {}).get("roi", (55, 75, 25, 45))
    interface_positions = track_interface(cine, roi=roi)

    # Optional display
    display_cine_with_roi(cine, roi, interface_positions, delay=0.1)

    # Compare to external respiratory signal if present
    resp_file = config["data"]["resp_file"]
    if resp_file:
        resp_data = ecg_resp.load_and_resample_resp(resp_file, total_phase_encodes)
        resp_signal = resp_data.flatten()

        # Create time axes
        total_time = total_phase_encodes / fs
        t_cine = np.linspace(0, total_time, n_frames)
        t_resp = np.linspace(0, total_time, len(resp_signal))

        # Interpolate and normalize
        resp_interp = np.interp(t_cine, t_resp, resp_signal)
        interface_norm = (interface_positions - np.min(interface_positions)) / (
            np.ptp(interface_positions) + 1e-9
        )
        resp_norm = (resp_interp - np.min(resp_interp)) / (np.ptp(resp_interp) + 1e-9)

        # Plot
        plt.figure(figsize=(10, 5))
        plt.plot(t_cine, interface_norm, "r-o", label="Normalized Interface Position")
        plt.plot(t_cine, resp_norm, "b--", label="Normalized Resp Signal")
        plt.xlabel("Time (s)")
        plt.ylabel("Normalized Amplitude (0-1)")
        plt.title("Liver–Lung Interface vs. Respiratory Signal")
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    main()
