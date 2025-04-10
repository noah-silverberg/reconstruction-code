"""
pipeline.py

Orchestrates the entire MRI reconstruction workflow, typically:
1. Data ingestion (read TWIX, extract k-space)
2. ECG detection (optionally from file or from extracted ICE data)
5. Cardiac binning (using R-peaks or external triggers)
6. Respiratory binning (either even bins or physiological peaks/troughs)
7. Final image reconstruction & visualization
"""

import numpy as np
import utils.data_ingestion as di
import utils.ecg_resp as ecg_resp
import utils.binning as binning
import utils.reconstruction as recon
import utils.gif as gif


def run_pipeline(config):
    """
    Run the entire reconstruction pipeline as specified by `config`.

    Parameters
    ----------
    config : dict
        Dictionary containing keys such as:
          - config["data"]["twix_file"], config["data"]["dicom_folder"], etc.
          - config["processing"] for binning approach, etc.

    Returns
    -------
    None
    """
    # --- Data Ingestion ---
    print("Reading scan data and extracting parameters...")
    twix_file = config["data"]["twix_file"]
    dicom_folder = config["data"]["dicom_folder"]
    n_frames = config["data"]["n_frames"]
    row_offset = config["data"]["offset"]
    extended_phase_lines = config["data"]["extended_pe_lines"]

    # Read TWIX data
    twix_data = di.read_twix_file(twix_file, include_scans=[-1], parse_pmu=False)
    kspace = di.extract_image_data(twix_data)
    n_phase_encodes_per_frame = kspace.shape[0] // n_frames

    # Determine sampling frequency
    fs_method = config["processing"].get("sampling_rate", "twix")
    if fs_method == "twix":
        fs = 1 / (twix_data[-1]["hdr"]["Phoenix"]["alTR"][0] * 1e-6)
        print(f"Sampling frequency from TWIX: {fs:.2f} Hz")
    elif fs_method == "dicom":
        framerate, frame_time = di.get_dicom_framerate(dicom_folder)
        fs = framerate * n_phase_encodes_per_frame
        print(
            f"DICOM: Frame rate: {framerate:.2f} Hz, Frame time: {frame_time:.4f} s, fs: {fs:.2f}"
        )
    else:
        raise ValueError(f"Unsupported sampling frequency method: {fs_method}")

    # --- ECG / R-peak detection from events file ---
    print("Processing ECG data (R-peak detection)...")
    events_file = config["data"].get("event_file", None)
    if events_file is None:
        ecg_columns = config["data"].get("ecg_columns", None)
        ecg_data = di.extract_iceparam_data(
            twix_data,
            segment_index=0,
            columns=np.s_[
                int(ecg_columns.split(":")[0]) : int(ecg_columns.split(":")[1])
            ],
        )
        r_peaks_list = ecg_resp.detect_r_peaks(ecg_data, fs)
        r_peaks = np.mean(r_peaks_list, axis=0).astype(int)
    else:
        events = ecg_resp.load_and_resample_events(events_file, kspace.shape[0])
        r_peaks = np.nonzero(events)[0]

    # Basic heart rate estimation from R-peaks
    heart_rate = ecg_resp.compute_average_heart_rate([r_peaks], fs)
    print(f"Estimated Heart Rate: {heart_rate:.1f} BPM")

    # --- Respiratory data & binning approach ---
    print("Processing Resp data (peak detection)...")
    resp_file = config["data"].get("resp_file", None)
    if resp_file is None:
        resp_column = config["data"].get("resp_column", None)
        resp_data = di.extract_iceparam_data(
            twix_data,
            segment_index=0,
            columns=np.s_[int(resp_column)],
        )[:, np.newaxis]
    else:
        resp_data = ecg_resp.load_and_resample_resp(resp_file, kspace.shape[0])

    resp_bin_method = config["processing"].get(
        "resp_bin_method", "even"
    )  # "even" or "physio"
    resp_peak_method = config["processing"].get("resp_peak_method", "nk")

    # Retrieve method-specific parameters if using "scipy" for peak detection
    resp_peak_height = None
    resp_peak_prominence = None
    if resp_peak_method == "scipy":
        resp_peak_kwargs = config["processing"]["resp_peak_kwargs"]
        resp_peak_height = resp_peak_kwargs["height"]
        resp_peak_prominence = resp_peak_kwargs["prominence"]

    # If "physio", also retrieve trough detection approach
    if resp_bin_method == "physio":
        resp_trough_method = config["processing"].get("resp_trough_method", "nk")
        resp_trough_height = None
        resp_trough_prominence = None
        if resp_trough_method == "scipy":
            resp_trough_kwargs = config["processing"]["resp_trough_kwargs"]
            resp_trough_height = resp_trough_kwargs["height"]
            resp_trough_prominence = resp_trough_kwargs["prominence"]

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
        # Even respiratory bins
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
        total_resp_bins = num_resp_bins
    else:
        # Physiological binning
        resp_troughs = ecg_resp.detect_resp_peaks(
            -resp_data,
            fs,
            method=resp_trough_method,
            height=resp_trough_height,
            prominence=resp_trough_prominence,
        )
        num_exhalation_bins = config["processing"]["num_exhalation_bins"]
        num_inhalation_bins = config["processing"]["num_inhalation_bins"]
        total_resp_bins = num_exhalation_bins + num_inhalation_bins

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

    # Reconstruct each respiratory bin to produce a time series / cine
    cine_images_list = []
    for resp_bin in range(total_resp_bins):
        binned_data_resp = binned_data[:, resp_bin, :, :, :]
        binned_count_resp = binned_count[:, resp_bin, :]

        recon_method = config["processing"].get("reconstruction_method", "zf").lower()

        if recon_method == "grappa":
            calib_region = tuple(config["processing"].get("calib_region", [30, 50]))
            kernel_size = tuple(config["processing"].get("kernel_size", [5, 5]))
            images = recon.grappa_reconstruction(
                binned_data_resp,
                calib_region=calib_region,
                kernel_size=kernel_size,
            )
        elif recon_method == "tgrappa":
            calib_size = tuple(config["processing"].get("calib_size", [20, 20]))
            kernel_size = tuple(config["processing"].get("kernel_size", [5, 5]))
            images = recon.tgrappa_reconstruction(
                binned_data_resp, calib_size=calib_size, kernel_size=kernel_size
            )
        elif recon_method in ["zf", "conj_symm"]:
            images = recon.direct_ifft_reconstruction(
                binned_data_resp,
                extended_pe_lines=extended_phase_lines,
                use_conjugate_symmetry=recon_method != "zf",
                count_mask=binned_count_resp,
            )
        else:
            raise ValueError(f"Unsupported reconstruction method: {recon_method}")
        print(f"Respiratory phase {resp_bin}: Reconstructed image shape:", images.shape)

        # Optional rotation, flip, crop
        images = np.rot90(images, k=1, axes=(1, 2))
        images = np.flip(images, axis=2)
        images = images[:, 64:-64, :]

        cine_images_list.append(images)

    # --- Visualization: save a separate GIF for each respiratory phase ---
    print("Saving and displaying results for each respiratory phase...")
    duration = 1000 * 60 / heart_rate / num_cardiac_bins  # ms between frames

    for resp_bin, images in enumerate(cine_images_list):
        gif_filename = f"binned_cine_resp_{resp_bin}.gif"
        gif.save_images_as_gif(images, gif_filename, duration=duration)
        print(f"Saved {gif_filename}")

    # Optionally save a k-space GIF for each respiratory phase
    for resp_bin in range(total_resp_bins):
        kspace_resp = binned_data[:, resp_bin, :, :, :]
        kspace_gif_filename = f"binned_kspace_resp_{resp_bin}.gif"
        gif.save_kspace_as_gif(
            kspace_resp, kspace_gif_filename, duration=duration, cmap="gray"
        )
        print(f"Saved {kspace_gif_filename}")

    print("Pipeline completed successfully.")
