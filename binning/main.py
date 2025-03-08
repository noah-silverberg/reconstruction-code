import utils
import ecg
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk

data_path = "meas_MID00086_FID26450_DMI_PMU_250216_100rep.dat"
fs = 1 / 5.5e-3  # 1 / TR (sampling frequency used to detect R-peaks)

########################################################

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
ecg.plot_ecg(ecg_data, fs, r_peaks_list, mode="separate")
