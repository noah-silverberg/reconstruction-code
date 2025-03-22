"""
pipeline.py

Orchestrates the MRI reconstruction workflow:
1. Data ingestion
2. ECG detection
3. Decomposition (PCA, kernel PCA, or ICA)
4. Reconstruct k-space from selected components
5. Cardiac binning
6. Final image reconstruction & visualization
"""

import numpy as np
import utils.data_ingestion as di
import utils.pca as pca
import utils.ecg_resp as ecg_resp
import utils.binning as binning
import utils.reconstruction as recon
import utils.gif as gif


def run_pipeline(config):
    # --- Data Ingestion ---
    print("Reading scan data and extracting parameters...")
    twix_file = config["data"]["twix_file"]
    dicom_folder = config["data"]["dicom_folder"]
    n_frames = config["data"]["n_frames"]
    row_offset = config["data"]["offset"]
    extended_phase_lines = config["data"]["extended_pe_lines"]

    twix_data = di.read_twix_file(twix_file, include_scans=[-1], parse_pmu=False)
    kspace = di.extract_image_data(twix_data)

    framerate, frame_time = di.get_dicom_framerate(dicom_folder)
    n_phase_encodes_per_frame = kspace.shape[0] // n_frames
    fs = framerate * n_phase_encodes_per_frame
    print(
        f"Frame rate: {framerate:.2f} Hz, Frame time: {frame_time:.4f} s, fs: {fs:.2f}"
    )

    # # --- ECG Processing ---
    # ecg_columns = config["data"]["ecg_columns"]
    # ecg_data = di.extract_iceparam_data(
    #     twix_data, segment_index=0, columns=eval(f"np.s_[{ecg_columns}]")
    # )
    # if ecg_data.ndim == 1:
    #     ecg_data = ecg_data.reshape(-1, 1)
    # if kspace.shape[0] != ecg_data.shape[0]:
    #     raise ValueError("Mismatch between phase encodes and ECG samples.")

    print("Processing ECG data (R-peak detection)...")
    events_file = config["data"]["event_file"]
    events = ecg_resp.load_and_resample_events(events_file, kspace.shape[0])
    r_peaks = np.nonzero(events)[0]
    # r_peaks_list = ecg_resp.detect_r_peaks(ecg_data, fs)
    heart_rate = ecg_resp.compute_average_heart_rate([r_peaks], fs)
    print(f"Estimated Heart Rate: {heart_rate:.1f} BPM")

    print("Processing Resp data (peak detection)...")

    resp_file = config["data"]["resp_file"]
    resp_bin_method = config["processing"].get("resp_bin_method", "even")
    resp_peak_method = config["processing"].get("resp_peak_method", "nk")
    resp_peak_height, resp_peak_prominence = None, None
    if resp_peak_method == "scipy":
        resp_peak_kwargs = config["processing"]["resp_peak_kwargs"]
        resp_peak_height = resp_peak_kwargs["height"]
        resp_peak_prominence = resp_peak_kwargs["prominence"]
    if resp_bin_method == "physio":
        resp_trough_method = config["processing"].get("resp_trough_method", "nk")
        resp_trough_height, resp_trough_prominence = None, None
        if resp_trough_method == "scipy":
            resp_trough_kwargs = config["processing"]["resp_trough_kwargs"]
            resp_trough_height = resp_trough_kwargs["height"]
            resp_trough_prominence = resp_trough_kwargs["prominence"]

    resp_data = ecg_resp.load_and_resample_resp(resp_file, kspace.shape[0])
    resp_peaks = ecg_resp.detect_resp_peaks(
        resp_data,
        fs,
        method=resp_peak_method,
        height=resp_peak_height,
        prominence=resp_peak_prominence,
    )

    # --- Cardiac/Resp Binning ---
    print("Binning k-space data by cardiac & respiratory phases...")
    num_cardiac_bins = config["processing"]["num_cardiac_bins"]
    if resp_bin_method == "even":
        num_resp_bins = config["processing"]["num_resp_bins"]
        binned_data, binned_count = binning.bin_reconstructed_kspace_joint(
            kspace.reshape(n_frames, -1, kspace.shape[1], kspace.shape[2]),
            r_peaks.flatten(),
            resp_peaks.flatten(),
            num_cardiac_bins=num_cardiac_bins,
            num_respiratory_bins=num_resp_bins,
            n_phase_encodes_per_frame=n_phase_encodes_per_frame,
            extended_phase_lines=extended_phase_lines,
            row_offset=row_offset,
        )
    else:  # physiological binning
        # Also detect respiratory troughs (using the negative signal)
        resp_troughs = ecg_resp.detect_resp_peaks(
            -resp_data,
            fs,
            method=resp_trough_method,
            height=resp_trough_height,
            prominence=resp_trough_prominence,
        )
        num_exhalation_bins = config["processing"]["num_exhalation_bins"]
        num_inhalation_bins = config["processing"]["num_inhalation_bins"]
        num_respiratory_bins = num_exhalation_bins + num_inhalation_bins
        binned_data, binned_count = binning.bin_reconstructed_kspace_joint_physio(
            kspace.reshape(n_frames, -1, kspace.shape[1], kspace.shape[2]),
            r_peaks.flatten(),
            resp_peaks.flatten(),
            resp_troughs.flatten(),
            num_cardiac_bins=num_cardiac_bins,
            num_exhalation_bins=num_exhalation_bins,
            num_inhalation_bins=num_inhalation_bins,
            n_phase_encodes_per_frame=n_phase_encodes_per_frame,
            extended_phase_lines=extended_phase_lines,
            row_offset=row_offset,
        )
    print("Binned k-space shape:", binned_data.shape)

    cine_images_list = []  # To store reconstructed cine images per respiratory phase

    # Loop over respiratory bins
    for resp_bin in range(num_respiratory_bins):
        binned_data_resp = binned_data[:, resp_bin, :, :, :]
        binned_count_resp = binned_count[
            :, resp_bin, :
        ]  # assuming same row dimension as extended_pe_lines

        images = recon.direct_ifft_reconstruction(
            binned_data_resp,
            extended_pe_lines=extended_phase_lines,
            use_conjugate_symmetry=True,
            count_mask=binned_count_resp,
        )
        print(f"Respiratory phase {resp_bin}: Reconstructed image shape:", images.shape)

        # Optional: rotate, flip, or crop
        images = np.rot90(images, k=1, axes=(1, 2))
        images = np.flip(images, axis=2)
        images = images[:, 64:-64, :]

        cine_images_list.append(images)

    # --- Visualization: Save a separate GIF for each respiratory phase ---
    print("Saving and displaying results for each respiratory phase...")
    duration = 1000 * 60 / heart_rate / num_cardiac_bins  # Duration per cine frame

    for resp_bin, images in enumerate(cine_images_list):
        gif_filename = f"homodyne_binned_cine_resp_{resp_bin}.gif"
        gif.save_images_as_gif(images, gif_filename, duration=duration)
        print(f"Saved {gif_filename}")

    # Optionally, you can also save a k-space GIF for each respiratory phase:
    for resp_bin in range(num_respiratory_bins):
        # Extract k-space data for this respiratory phase
        kspace_resp = binned_data[:, resp_bin, :, :, :]
        kspace_gif_filename = f"binned_kspace_resp_{resp_bin}.gif"
        gif.save_kspace_as_gif(
            kspace_resp, kspace_gif_filename, duration=duration, cmap="gray"
        )
        print(f"Saved {kspace_gif_filename}")

    print("Pipeline completed successfully.")
