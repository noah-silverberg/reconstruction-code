#!/usr/bin/env python3
"""
interface_tracking.py

This script directly reconstructs the k-space data (using conjugate symmetry)
to produce a cine, then automatically tracks the superior–inferior (vertical)
position of the liver–lung interface using a Sobel gradient over a defined ROI.
It displays a sample frame with the ROI box drawn for debugging and also
animates the whole cine with the ROI and interface overlay. Finally, it
plots the resulting 1D interface position (one value per frame) against the
interpolated respiratory (bellows) signal.
"""

import yaml
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Import your existing utility functions
import utils.data_ingestion as di
import utils.ecg_resp as ecg_resp
import utils.reconstruction as recon


def load_config(config_file="config.yaml"):
    with open(config_file, "r") as f:
        return yaml.safe_load(f)


def track_interface(cine_frames, roi=(100, 200, 50, 250)):
    """
    For each frame in the cine (2D grayscale image), compute the vertical gradient
    within the specified ROI and return the row index corresponding to the maximum gradient,
    which is assumed to be the liver–lung interface.

    Parameters:
      cine_frames (np.ndarray): 3D array with shape (n_frames, height, width)
      roi (tuple): (row_start, row_end, col_start, col_end)

    Returns:
      np.ndarray: 1D array of interface row positions (in pixel coordinates)
    """
    interface_positions = []
    for frame in cine_frames:
        # Crop to the region of interest
        roi_frame = frame[roi[0] : roi[1], roi[2] : roi[3]]
        # Compute vertical gradient using a Sobel filter (y-derivative)
        grad_y = cv2.Sobel(roi_frame, cv2.CV_64F, 0, 1, ksize=5)
        # Average absolute gradient along columns to get a 1D profile (per row)
        profile = np.mean(np.abs(grad_y), axis=1)
        # The row with maximum gradient (plus ROI row offset)
        interface_row = np.argmax(profile) + roi[0]
        interface_positions.append(interface_row)
    return np.array(interface_positions)


def display_frame_with_roi(frame, roi, interface_row):
    """
    Display a single frame (2D image) with a rectangle drawn for the ROI and a horizontal line
    at the tracked interface row.
    """
    # Convert to uint8 for display (normalize if necessary)
    frame_norm = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # Convert to BGR so we can draw colored lines
    frame_bgr = cv2.cvtColor(frame_norm, cv2.COLOR_GRAY2BGR)
    # Draw ROI rectangle with thinner lines (thickness=1)
    x, y, w, h = roi[2], roi[0], roi[3] - roi[2], roi[1] - roi[0]
    cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 255, 0), 1)
    # Draw a horizontal red line at the interface row with thickness=1
    cv2.line(
        frame_bgr,
        (0, int(interface_row)),
        (frame_bgr.shape[1], int(interface_row)),
        (0, 0, 255),
        1,
    )

    # Convert BGR to RGB for matplotlib display
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(6, 6))
    plt.imshow(frame_rgb)
    plt.title("Sample Reconstructed Frame with ROI and Interface")
    plt.axis("off")
    plt.show()


def display_cine_with_roi(cine_frames, roi, interface_positions, delay=0.1):
    """
    Display the entire cine (all frames) with ROI and interface overlay as an animation.

    Parameters:
      cine_frames (np.ndarray): 3D array (n_frames, height, width)
      roi (tuple): ROI (row_start, row_end, col_start, col_end)
      interface_positions (np.ndarray): 1D array of tracked interface rows per frame
      delay (float): Pause time (in seconds) between frames
    """
    plt.figure(figsize=(6, 6))
    for i, frame in enumerate(cine_frames):
        # Normalize and convert the frame for display
        frame_norm = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(
            np.uint8
        )
        frame_bgr = cv2.cvtColor(frame_norm, cv2.COLOR_GRAY2BGR)
        # Draw ROI rectangle (thickness=1)
        x, y, w, h = roi[2], roi[0], roi[3] - roi[2], roi[1] - roi[0]
        cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 255, 0), 1)
        # Draw interface line (thickness=1)
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
    # --- Load configuration and k-space data ---
    config = load_config()
    twix_file = config["data"]["twix_file"]
    dicom_folder = config["data"]["dicom_folder"]
    n_frames = config["data"]["n_frames"]
    row_offset = config["data"]["offset"]
    extended_phase_lines = config["data"]["extended_pe_lines"]

    # Read TWIX file and extract k-space data (shape: (total_phase_encodes, coils, freq_encodes))
    scans = di.read_twix_file(twix_file, include_scans=[-1], parse_pmu=False)
    kspace = di.extract_image_data(scans)

    # Get DICOM frame rate and compute phase encodes per frame
    framerate, _ = di.get_dicom_framerate(dicom_folder)
    total_phase_encodes = kspace.shape[0]
    n_phase_encodes_per_frame = total_phase_encodes // n_frames
    fs = framerate * n_phase_encodes_per_frame  # effective sampling frequency

    # Reshape k-space into frames: (n_frames, n_phase_encodes_per_frame, coils, freq_encodes)
    kspace_frames = kspace.reshape(
        n_frames, n_phase_encodes_per_frame, kspace.shape[1], kspace.shape[2]
    )

    # --- Direct reconstruction using conjugate symmetry ---
    cine = recon.direct_ifft_reconstruction(
        kspace_frames,
        extended_pe_lines=extended_phase_lines,
        offset=row_offset,
        use_conjugate_symmetry=True,
    )
    # (cine shape: (n_frames, output_rows, freq_encodes))

    # Optional: rotate, flip, or crop the cine as needed
    cine = np.rot90(cine, k=1, axes=(1, 2))
    cine = np.flip(cine, axis=2)
    cine = cine[:, 64:-64, :]

    # --- Track the liver–lung interface ---
    # Get ROI from config or use default (row_start, row_end, col_start, col_end)
    roi = config.get("tracking", {}).get("roi", (55, 75, 25, 45))
    interface_positions = track_interface(cine, roi=roi)

    # --- Display the entire cine with ROI and interface overlays ---
    display_cine_with_roi(cine, roi, interface_positions, delay=0.1)

    # --- Process respiratory (bellows) signal ---
    resp_file = config["data"]["resp_file"]
    # Resample the resp signal to total_phase_encodes (as in your pipeline)
    resp_data = ecg_resp.load_and_resample_resp(resp_file, total_phase_encodes)
    resp_signal = resp_data.flatten()

    # --- Create time axes ---
    total_time = total_phase_encodes / fs  # total scan time in seconds
    t_cine = np.linspace(0, total_time, n_frames)
    t_resp = np.linspace(0, total_time, len(resp_signal))
    # Interpolate resp signal to the cine time points
    resp_interp = np.interp(t_cine, t_resp, resp_signal)

    # --- Normalize the signals for comparability ---
    # Normalize interface positions
    interface_norm = (interface_positions - np.min(interface_positions)) / (
        np.max(interface_positions) - np.min(interface_positions)
    )
    # Normalize respiratory signal
    resp_norm = (resp_interp - np.min(resp_interp)) / (
        np.max(resp_interp) - np.min(resp_interp)
    )

    # --- Plot the normalized interface position vs. respiratory signal ---
    plt.figure(figsize=(10, 5))
    plt.plot(t_cine, interface_norm, "r-o", label="Normalized Interface Position")
    plt.plot(t_cine, resp_norm, "b--", label="Normalized Resp Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Normalized Amplitude (0-1)")
    plt.title("Liver–Lung Interface Position vs. Respiratory Signal")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
