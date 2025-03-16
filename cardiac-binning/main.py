import utils.gif as gif
import utils.ecg as ecg
import utils.binning as binning
import utils.twix as twix
import utils.svd as svd
import utils.reconstruction as reconstruction
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk

data_path = "meas_MID00086_FID26450_DMI_PMU_250216_100rep.dat"
# data_path = "20250307_JM/raw/meas_MID00114_FID27379_Cor_250306_2.dat"
# data_path = "20250307_JM/raw/meas_MID00115_FID27380_Cor_250306_2.dat"
# fs = 1 / 4.69e-3  # 1 / TR (sampling frequency used to detect R-peaks)
fs = 1 / 5.54e-3  # 1 / TR (sampling frequency used to detect R-peaks)
num_bins = 10  # Number of bins for binning
n_phase_encodes_per_frame = 112
extended_pe_lines = 128
ecg_columns = np.s_[18:21]
# ecg_columns = np.s_[20]
resp_columns = np.s_[23]
selection = [2, 3, 4, 5, 6, 7, 8, 9]
# selection = [2, 4, 5, 6, 7, 8, 9]

########################## Read .dat ##############################

# Read data from .dat file
twix_data = twix.read_twix_file(
    data_path,
    include_scans=[-1],
    parse_pmu=False,
)

# Extract raw kspace data
# Shape: (phase_encodes, coils, frequency_encodes)
kspace = twix.extract_image_data(twix_data)

# Extract ECG signals from the first segment
# Each ECG signal is in a separate column
ecg_data = twix.extract_iceparam_data(twix_data, segment_index=0, columns=ecg_columns)

if ecg_data.ndim == 1:
    ecg_data = ecg_data.reshape(-1, 1)

# Extract respiratory signal from the first segment
resp_data = twix.extract_iceparam_data(twix_data, segment_index=0, columns=resp_columns)

if resp_data.ndim == 1:
    resp_data = resp_data.reshape(-1, 1)


# Ensure we have the same number of phase encodes and ECG points
if kspace.shape[0] != ecg_data.shape[0]:
    raise ValueError(
        "Number of phase encodes do not match between kspace and ECG data."
    )


########################## ECG & resp processing ##############################

# Detect R-peaks across all channels.
r_peaks_list = ecg.detect_r_peaks(ecg_data, fs)

# Detect respiratory peaks across all channels.
# resp_peaks = ecg.detect_rsp_peaks(resp_data, fs)

# Plot ECG & resp signals with detected peaks
# (Just to ensure that the peaks are detected correctly)
# ecg.plot_ecg(ecg_data, fs, r_peaks_list, mode="separate")
# ecg.plot_ecg(resp_data, fs, resp_peaks, mode="separate")


######################## PCA on k-space data ################################

print("Performing PCA on k-space data...")
# U, S, Vt, X_mean, var_explained, frame_shape = svd.perform_pca_kspace(
#     kspace, n_phase_encodes_per_frame
# )

# # Define candidate sigma values (adjust the range as needed)
# sigma_candidates = np.logspace(-1.2, -0.5, 20)

# best_sigma, best_error, kpca, X_kspca, orig_feature_dim = svd.optimize_sigma_kpca(
#     kspace, n_phase_encodes_per_frame, sigma_candidates, n_components=None, kernel="rbf"
# )

# print(f"Best sigma: {best_sigma}, with reconstruction error: {best_error}")

# Perform kernel PCA on the complex k-space data.
# Note: The function now returns (kpca, X_kspca, frame_shape, orig_feature_dim)
kpca, X_kspca, frame_shape, orig_feature_dim = svd.perform_kernel_pca_kspace(
    kspace, n_phase_encodes_per_frame, kernel="rbf", sigma=0.0964
)

# # Plot the first 5 kernel PCs' time series and frequency spectra.
# svd.plot_kernel_pc_time_series_and_fft(
#     X_kspca, sampling_rate=fs / n_phase_encodes_per_frame, n_components=5
# )

# # Reconstruct the images from the full kernel PCA transform.
# utils.display_images_as_gif(
#     svd.reconstruct_frames_kernel_kspace(kpca, X_kspca, frame_shape, orig_feature_dim)
# )

# for i in range(10):
#     print(f"Reconstructing with only the {i}th kernel PCA component...")
#     utils.display_images_as_gif(
#         svd.reconstruct_with_selected_components_kernel_kspace(
#             kpca, X_kspca, [i], frame_shape, orig_feature_dim
#         )
#     )

# Reconstruct using only the 0th kernel PCA component.
# utils.display_images_as_gif(
#     svd.reconstruct_with_selected_components_kernel_kspace(
#         kpca, X_kspca, [2] + list(range(4, 10)), frame_shape, orig_feature_dim
#     )
# )


######################## Binning ################################

X_kpca_mod = np.zeros_like(X_kspca)
X_kpca_mod[:, selection] = X_kspca[:, selection]

X_recon_real = kpca.inverse_transform(X_kpca_mod)
n_features = orig_feature_dim
X_recon_complex = X_recon_real[:, :n_features] + 1j * X_recon_real[:, n_features:]
n_frames = X_recon_complex.shape[0]
kspace_recon = X_recon_complex.reshape(
    n_frames, frame_shape[0], frame_shape[1], frame_shape[2]
)

# Perform the cardiac-phase binning.
# binned_data, binned_count = cardiac_binning.bin_kspace_by_cardiac_phase(
#     r_peaks_list,
#     kspace,
#     num_bins=num_bins,
#     n_phase_encodes_per_frame=n_phase_encodes_per_frame,
#     extended_pe_lines=extended_pe_lines,
# )
binned_data, binned_count = binning.bin_reconstructed_kspace_by_cardiac_phase_kernel(
    kpca,
    X_kspca,
    selection,
    frame_shape,
    orig_feature_dim,
    r_peaks_list,
    num_bins=num_bins,
    n_phase_encodes_per_frame=n_phase_encodes_per_frame,
    extended_pe_lines=extended_pe_lines,
)
print("Binned k-space shape:", binned_data.shape)

######################### Reconstruction ##############################

# binned_data, homodyne_recon_images = cardiac_binning.fill_missing_binned_kspace(
#     binned_data, binned_count
# )

# Homodyne reconstruction of the binned data.
homodyne_recon_images = reconstruction.homodyne_binned_data(binned_data, binned_count)
print("Homodyne reconstructed images shape:", homodyne_recon_images.shape)

# Save the reconstructed images as a GIF.
gif.save_images_as_gif(
    homodyne_recon_images, "homodyne_reconstructed_images.gif", duration=0.03
)

# Optionally display the reconstructed images as an animated GIF.
gif.display_images_as_gif(homodyne_recon_images, interval=30, notebook=False)

gif.save_kspace_as_gif(binned_data, "binned_kspace.gif", duration=0.2, cmap="gray")
