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

    # --- ECG Processing ---
    ecg_columns = config["data"]["ecg_columns"]
    ecg_data = di.extract_iceparam_data(
        twix_data, segment_index=0, columns=eval(f"np.s_[{ecg_columns}]")
    )
    if ecg_data.ndim == 1:
        ecg_data = ecg_data.reshape(-1, 1)
    if kspace.shape[0] != ecg_data.shape[0]:
        raise ValueError("Mismatch between phase encodes and ECG samples.")

    print("Processing ECG data (R-peak detection)...")
    r_peaks_list = ecg.detect_r_peaks(ecg_data, fs)
    heart_rate = ecg.compute_average_heart_rate(r_peaks_list, fs)
    print(f"Estimated Heart Rate: {heart_rate:.1f} BPM")

    # --- Decomposition & Reconstruction ---
    print(
        "Performing decomposition and reconstructing k-space with selected components..."
    )
    decomposition_method = config["processing"]["decomposition_method"].lower()
    selected_components = config["processing"]["selected_components"]
    n_components = config["processing"].get("n_components", None)

    if decomposition_method == "pca":
        # Standard PCA
        pca_model = pca.perform_pca(kspace, n_phase_encodes_per_frame)
        frame_shape = pca_model[-1]
        reconstructed_kspace = pca.reconstruct_kspace_from_components(
            method="pca",
            model=pca_model,
            transformed_data=None,  # Not used for PCA
            components_to_keep=selected_components,
            frame_shape=frame_shape,
            X_mean=pca_model[3],
        )
    elif decomposition_method == "kernel_pca":
        # Kernel PCA
        sigma = config["processing"]["sigma"]
        kpca_model, X_kpca, frame_shape, orig_feature_dim = pca.perform_kernel_pca(
            kspace,
            n_phase_encodes_per_frame,
            kernel="rbf",
            sigma=sigma,
            n_components=n_components,
        )
        reconstructed_kspace = pca.reconstruct_kspace_from_components(
            method="kernel_pca",
            model=kpca_model,
            transformed_data=X_kpca,
            components_to_keep=selected_components,
            frame_shape=frame_shape,
            orig_feature_dim=orig_feature_dim,
        )
    elif decomposition_method == "ica":
        # ICA
        ica_model, X_ica, frame_shape, orig_feature_dim = pca.perform_ica(
            kspace, n_phase_encodes_per_frame, n_components=n_components
        )
        reconstructed_kspace = pca.reconstruct_kspace_from_components(
            method="ica",
            model=ica_model,
            transformed_data=X_ica,
            components_to_keep=selected_components,
            frame_shape=frame_shape,
            orig_feature_dim=orig_feature_dim,
        )
    else:
        raise ValueError(f"Unknown decomposition method: {decomposition_method}")

    # --- Cardiac Binning ---
    print("Binning k-space data by cardiac phase...")
    num_bins = config["processing"]["num_bins"]
    binned_data, binned_count = binning.bin_reconstructed_kspace(
        reconstructed_kspace,
        r_peaks_list,
        num_bins,
        n_phase_encodes_per_frame,
        extended_phase_lines,
        row_offset,
    )
    print("Binned k-space shape:", binned_data.shape)

    # --- Image Reconstruction ---
    print("Reconstructing cine images from binned data...")
    homodyne_images = recon.direct_ifft_reconstruction(
        binned_data,
        extended_pe_lines=extended_phase_lines,
        offset=row_offset,
        use_conjugate_symmetry=True,  # or False, depending on your preference
    )
    print("Reconstructed image shape:", homodyne_images.shape)

    # Optional: rotate, flip, or crop
    homodyne_images = np.rot90(homodyne_images, k=1, axes=(1, 2))
    homodyne_images = np.flip(homodyne_images, axis=2)
    homodyne_images = homodyne_images[:, 64:-64, :]

    # --- Visualization ---
    print("Saving and displaying results...")
    duration = 1000 * 60 / heart_rate / num_bins
    gif.save_images_as_gif(
        homodyne_images, "homodyne_binned_cine.gif", duration=duration
    )
    gif.save_kspace_as_gif(
        binned_data, "binned_kspace.gif", duration=duration, cmap="gray"
    )

    print("Pipeline completed successfully.")
