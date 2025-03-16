"""
pipeline.py

This module orchestrates the MRI reconstruction workflow:
1. Data ingestion and parameter extraction.
2. Component analysis (via kernel PCA).
3. ECG processing.
4. Cardiac binning.
5. Image reconstruction.
6. Visualization.

Future enhancements (e.g., ICA for component separation) can be added here.
"""

import numpy as np
import utils.data_ingestion as di
import utils.pca as pca
import utils.ecg as ecg
import utils.binning as binning
import utils.reconstruction as recon
import utils.gif as gif


def run_pipeline(config):
    # --- Data Ingestion ---
    print("Reading scan data and extracting parameters...")
    twix_file = config["data"]["twix_file"]
    dicom_folder = config["data"]["dicom_folder"]
    n_frames = config["data"]["n_frames"]
    offset = config["data"]["offset"]
    extended_pe_lines = config["data"]["extended_pe_lines"]

    # Read TWIX file.
    twix_data = di.read_twix_file(twix_file, include_scans=[-1], parse_pmu=False)
    kspace = di.extract_image_data(twix_data)

    # Read DICOM parameters for frame rate etc.
    framerate, frame_time = di.get_dicom_framerate(dicom_folder)
    n_phase_encodes_per_frame = kspace.shape[0] // n_frames
    fs = framerate * n_phase_encodes_per_frame
    print(
        f"Frame rate: {framerate:.2f} Hz, Frame time: {frame_time:.4f} s, fs: {fs:.2f}"
    )

    # Read ECG data.
    ecg_columns = config["data"]["ecg_columns"]
    ecg_data = di.extract_iceparam_data(
        twix_data, segment_index=0, columns=eval(f"np.s_[{ecg_columns}]")
    )
    if ecg_data.ndim == 1:
        ecg_data = ecg_data.reshape(-1, 1)
    if kspace.shape[0] != ecg_data.shape[0]:
        raise ValueError("Mismatch between phase encodes and ECG samples.")

    # --- ECG Processing ---
    print("Processing ECG data...")
    r_peaks_list = ecg.detect_r_peaks(ecg_data, fs)
    heart_rate = ecg.compute_average_heart_rate(r_peaks_list, fs)
    # ecg.plot_ecg_signals(ecg_data, fs, r_peaks_list, mode="separate")

    # --- Component Analysis via Kernel PCA ---
    print("Performing kernel PCA on k-space data...")
    sigma = config["processing"]["sigma"]
    kpca_model, X_kpca, frame_shape, orig_feature_dim = pca.perform_kernel_pca(
        kspace, n_phase_encodes_per_frame, kernel="rbf", sigma=sigma
    )
    # pca.plot_kernel_pc_time_series_and_fft(
    #     X_kpca, sampling_rate=fs / n_phase_encodes_per_frame, n_components=5
    # )

    # --- Cardiac Binning ---
    print("Binning k-space data by cardiac phase...")
    selected_components = config["processing"]["selected_components"]
    num_bins = config["processing"]["num_bins"]
    # Reconstruct k-space using only selected kernel PCA components (without inverse FFT)
    # Then perform binning based on detected R-peaks.
    binned_data, binned_count = (
        binning.bin_reconstructed_kspace_by_cardiac_phase_kernel(
            kpca_model,
            X_kpca,
            selected_components,
            frame_shape,
            orig_feature_dim,
            r_peaks_list,
            num_bins=num_bins,
            n_phase_encodes_per_frame=n_phase_encodes_per_frame,
            extended_pe_lines=extended_pe_lines,
            offset=offset,
        )
    )
    print("Binned k-space shape:", binned_data.shape)

    # --- Image Reconstruction ---
    print("Reconstructing cine from binned data...")
    homodyne_images = recon.direct_ifft_reconstruction(
        binned_data, extended_pe_lines, offset, True
    )
    print("Reconstructed image shape:", homodyne_images.shape)
    # Rotate/flip/crop as needed (example processing)
    homodyne_images = np.rot90(homodyne_images, k=1, axes=(1, 2))
    homodyne_images = np.flip(homodyne_images, axis=2)
    homodyne_images = homodyne_images[:, 64:-64, :]

    # --- Visualization ---
    duration = 1000 * 60 / heart_rate / num_bins
    gif.save_images_as_gif(
        homodyne_images, "homodyne_binned_cine.gif", duration=duration
    )
    gif.save_kspace_as_gif(
        binned_data,
        "binned_kspace.gif",
        duration=60 / heart_rate / num_bins,
        cmap="gray",
    )
    gif.display_images_as_gif(homodyne_images, interval=duration)
    print("Pipeline completed successfully.")
