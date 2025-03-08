import utils
import ecg
import cardiac_binning
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk

data_path = "meas_MID00086_FID26450_DMI_PMU_250216_100rep.dat"
fs = 1 / 5.5e-3  # 1 / TR (sampling frequency used to detect R-peaks)
num_bins = 30  # Number of bins for binning
n_phase_encodes_per_frame = 112
extended_pe_lines = 128

########################## Read .dat & detect R-peaks ##############################

# Read data from .dat file
twix_data = utils.read_twix_file(
    data_path,
    include_scans=[-1],
    parse_pmu=False,
)

# Extract image data
# Shape: (phase_encodes, coils, frequency_encodes)
img = utils.extract_image_data(twix_data)

# Extract ECG signals from the first segment
# Each ECG signal is in a separate column
ecg_data = utils.extract_iceparam_data(twix_data, segment_index=0, columns=np.s_[18:21])

# Ensure we have the same number of phase encodes and ECG points
if img.shape[0] != ecg_data.shape[0]:
    raise ValueError("Number of phase encodes do not match between image and ECG data.")

# Detect R-peaks across all channels.
r_peaks_list = ecg.detect_r_peaks(ecg_data, fs)

# Plot ECG signals with detected R-peaks
# (Just to ensure that the R-peaks are detected correctly)
# ecg.plot_ecg(ecg_data, fs, r_peaks_list, mode="separate")


######################## Binning & reconstruction ################################

# Perform the cardiac-phase binning.
binned_data, binned_count = cardiac_binning.bin_kspace_by_cardiac_phase(
    r_peaks_list, img, num_bins=35, n_phase_encodes_per_frame=112, extended_pe_lines=128
)
print("Binned k-space shape:", binned_data.shape)

# Homodyne reconstruction of the binned data.
homodyne_recon_images = cardiac_binning.homodyne_binned_data(binned_data, binned_count)
print("Homodyne reconstructed images shape:", homodyne_recon_images.shape)

# Save the reconstructed images as a GIF.
utils.save_images_as_gif(
    homodyne_recon_images, "homodyne_reconstructed_images.gif", duration=0.03
)

# Optionally display the reconstructed images as an animated GIF.
utils.display_images_as_gif(homodyne_recon_images, interval=30)
