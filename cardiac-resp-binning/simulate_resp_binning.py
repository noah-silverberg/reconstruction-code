#!/usr/bin/env python3
"""
simulate_resp_binning.py

Example script that performs a prospective+retrospective respiratory binning simulation,
along with fraction/phase detection, line priority filling, confusion matrix, and final plots.

HOW TO USE:
-----------
1) Ensure you have the code in your "utils/" folder as described (resp_fractions.py, kspace_filling.py, ecg_resp.py).
2) Modify the "MAIN PARAMETERS" section below (resp_file, framerate, etc.) or adapt to read from a config.
3) Run:  python simulate_resp_binning.py
"""

import yaml
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as mcolors

# Local imports (adjust if your utils package is differently named)
import utils.data_ingestion as di
from utils.resp_fractions import predict_fraction, build_true_fraction
from utils.ecg_resp import detect_resp_peaks
from utils.kspace_filling import (
    build_line_priority,
    prospective_fill_loop,
    get_bin_index_gaussian,
    assign_prospective_bin,
)
from scipy import signal


def load_config(config_file="config.yaml"):
    """
    Load configuration from a YAML file.

    Parameters
    ----------
    config_file : str
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Parsed configuration.
    """
    with open(config_file, "r") as f:
        return yaml.safe_load(f)


# Read config
config = load_config()

# Example path to respiration file: "data/resp_signal.txt"
RESP_FILE = config["data"].get("resp_file", None)
DICOM_FOLDER = config["data"]["dicom_folder"]
TWIX_FILE = config["data"]["twix_file"]


# Read TWIX, extract raw k-space, and derive sampling frequency
scans = di.read_twix_file(TWIX_FILE, include_scans=[-1], parse_pmu=False)
kspace = di.extract_image_data(scans)

FRAMERATE, frametime = di.get_dicom_framerate(DICOM_FOLDER)
N_PHASE_ENCODES_PER_FRAME = kspace.shape[0] // config["data"]["n_frames"]
fs = FRAMERATE * N_PHASE_ENCODES_PER_FRAME  # ECG / respiration sampling freq


def plot_diff_map(pros, retro, title="Difference Map"):
    if pros.shape != retro.shape:
        print("Shape mismatch, skipping difference map.")
        return
    diff = retro - pros
    bin_count = diff.shape[0]
    cols = min(bin_count, 4)
    rows = int(np.ceil(bin_count / cols))

    fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1 and cols == 1:
        axs = np.array([axs])
    axs = axs.reshape(rows, cols)

    for b_ in range(bin_count):
        r_ = b_ // cols
        c_ = b_ % cols
        ax_ = axs[r_, c_]

        img = diff[b_]
        mm = np.abs(img).max()
        if mm < 1e-9:
            mm = 1
        im = ax_.imshow(img, cmap="bwr", vmin=-mm, vmax=mm)
        ax_.set_title(f"Bin {b_}: Retro - Pros")
        ax_.axis("off")

        # Optionally add an individual colorbar here:
        plt.colorbar(im, ax=ax_)

    # Or do one global colorbar:
    # fig.colorbar(im, ax=axs.ravel().tolist())

    fig.suptitle(title)
    fig.tight_layout()
    plt.show()


def plot_kspace_bins(kspace_array, title_prefix, vmin=0, vmax=None):
    bin_count = kspace_array.shape[0]
    cols = min(bin_count, 4)
    rows = int(np.ceil(bin_count / cols))

    fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1 and cols == 1:
        axs = np.array([axs])
    axs = axs.reshape(rows, cols)

    # If no vmax given, fall back to max of the array
    if vmax is None:
        vmax = kspace_array.max()

    # We'll keep track of an imshow handle so we can add a colorbar
    im = None

    for b_ in range(bin_count):
        r_ = b_ // cols
        c_ = b_ % cols
        ax_ = axs[r_, c_]

        img = kspace_array[b_]
        im = ax_.imshow(
            img,
            cmap="gray",
            # Use a PowerNorm with gamma<1 so that small nonzero values appear bright,
            # and high values are compressed. You can tweak gamma=0.3, 0.5, etc.
            norm=mcolors.PowerNorm(gamma=0.3, vmin=0, vmax=vmax),
        )
        # Optionally add a colorbar here:
        plt.colorbar(im, ax=ax_)
        ax_.set_title(f"{title_prefix} Bin {b_}")
        ax_.axis("off")

    # Turn off extra subplots if needed
    for i_ in range(bin_count, rows * cols):
        r_ = i_ // cols
        c_ = i_ % cols
        axs[r_, c_].axis("off")

    fig.suptitle(f"{title_prefix} K-Spaces")
    fig.tight_layout()
    plt.show()


##############################################################################
# MAIN PARAMETERS (EDIT HERE OR REPLACE WITH ARGUMENT/CONFIG PARSING)
##############################################################################

# If your file has a header or different columns, adjust as needed
SKIP_ROWS = 1  # e.g., skip 1 header line
USE_COL = 1  # e.g., second column has the signal

MIN_CALIB_TIME = 10.0  # seconds of data required for "calibration"
PEAK_HEIGHT = 0.6  # for scipy peak detection
PEAK_PROMINENCE = 0.2

# K-space binning parameters
USE_SEPARATE_IN_EX_BINS = True
NUM_INHALE_BINS = 2
NUM_EXHALE_BINS = 2

USE_TOTAL_BINS_INSTEAD = False
NUM_TOTAL_BINS = 4

KSPACE_H = 128  # "rows"
KSPACE_W = 128  # "columns"

DOWNSAMPLE_FACTOR = 1  # e.g., if you want to downsample the k-space

##############################################################################
# END OF MAIN PARAMETERS
##############################################################################


def main():
    """
    Run a full prospective+retrospective binning simulation
    using the user's respiration file and the logic in utils/.
    """

    # 1) Load respiration data from disk.
    #    For example, assume it is just a single-column text file.
    #    If you have a different format, adapt as necessary.
    resp_data = np.loadtxt(RESP_FILE, skiprows=SKIP_ROWS, usecols=USE_COL)
    N_raw = len(resp_data)

    # 2) We define the sampling freq. e.g. fs = FRAMERATE * N_PHASE_ENCODES_PER_FRAME
    fs = FRAMERATE * N_PHASE_ENCODES_PER_FRAME
    print(f"Loaded respiration data of length {N_raw}, setting fs={fs:.2f} Hz.")

    # 3) Optionally, if we want to resample or trim the data, do it here.
    # Flatten the respiration signal
    if DOWNSAMPLE_FACTOR > 1:
        resp_data = signal.resample(resp_data, N_raw // DOWNSAMPLE_FACTOR)
        fs = fs / DOWNSAMPLE_FACTOR
        print(
            f"Downsampled respiration data to {len(resp_data)} samples at fs={fs:.2f} Hz."
        )

    # 4) Predict fraction in real-time using your 'predict_fraction' code
    predicted_fraction, predicted_phase, calibration_end_idx = predict_fraction(
        resp_signal=resp_data,
        fs=fs,
        min_calib_time=MIN_CALIB_TIME,
        peak_height=PEAK_HEIGHT,
        peak_prom=PEAK_PROMINENCE,
    )

    N = len(resp_data)  # final length after any optional resampling

    # 5) Build prospective k-space arrays
    if USE_TOTAL_BINS_INSTEAD:
        total_bins = NUM_TOTAL_BINS
    else:
        total_bins = NUM_INHALE_BINS + NUM_EXHALE_BINS

    pros_fill = np.zeros((total_bins, KSPACE_H, KSPACE_W), dtype=float)

    pros_priority = build_line_priority(total_bins, kspace_height=KSPACE_H)

    # 6) Define a local function to get bin_index with Gaussian weighting
    #    or you can import from kspace_filling if you like. For example:
    from functools import partial

    # Build the partial for get_bin_index_gaussian
    get_bin_index_fn = partial(
        get_bin_index_gaussian,
        num_inhale_bins=NUM_INHALE_BINS,
        num_exhale_bins=NUM_EXHALE_BINS,
        use_total_bins=USE_TOTAL_BINS_INSTEAD,
        num_total_bins=NUM_TOTAL_BINS,
    )

    # Build the partial for assign_prospective_bin
    assign_bin_fn = partial(
        assign_prospective_bin,
        num_inhale_bins=NUM_INHALE_BINS,
        num_exhale_bins=NUM_EXHALE_BINS,
        use_total_bins=USE_TOTAL_BINS_INSTEAD,
        num_total_bins=NUM_TOTAL_BINS,
    )

    # Now just call prospective_fill_loop
    acquired_lines = prospective_fill_loop(
        N=len(resp_data),
        predicted_fraction=predicted_fraction,
        predicted_phase=predicted_phase,
        pros_fill=pros_fill,
        pros_priority=pros_priority,
        get_bin_index_fn=get_bin_index_fn,
        assign_bin_fn=assign_bin_fn,
    )

    # 8) Retrospective assignment:
    #    a) detect final peaks & troughs on the entire signal
    #    b) build actual fraction array
    pks = detect_resp_peaks(
        resp_data, fs, method="scipy", height=PEAK_HEIGHT, prominence=PEAK_PROMINENCE
    )
    trs = detect_resp_peaks(
        -resp_data, fs, method="scipy", height=PEAK_HEIGHT, prominence=PEAK_PROMINENCE
    )
    actual_frac = build_true_fraction(signal_length=N, peaks=pks, troughs=trs)

    # Build an empty array for retrospective fill
    retro_fill = np.zeros((total_bins, KSPACE_H, KSPACE_W), dtype=float)

    for k in range(N):
        if acquired_lines[k] is None:
            continue
        row, p_bin = acquired_lines[k]
        frac = actual_frac[k]
        if np.isnan(frac):
            continue

        # Decide inhalation or exhalation by comparing with previous sample or fallback
        if k > 0 and not np.isnan(actual_frac[k - 1]):
            is_inhale = frac >= actual_frac[k - 1]
        else:
            is_inhale = frac < 50

        # True bin
        true_bin = assign_prospective_bin(
            fraction=frac,
            is_inhale=is_inhale,
            num_inhale_bins=NUM_INHALE_BINS,
            num_exhale_bins=NUM_EXHALE_BINS,
            use_total_bins=USE_TOTAL_BINS_INSTEAD,
            num_total_bins=NUM_TOTAL_BINS,
        )
        retro_fill[true_bin, row, :] += 1.0

    # 9) Plot K-space bins and difference
    global_max = max(pros_fill.max(), retro_fill.max())

    # Plot prospective and retrospective bins using subplots
    plot_kspace_bins(pros_fill, "Prospective", vmin=0, vmax=global_max)
    plot_kspace_bins(retro_fill, "Retrospective", vmin=0, vmax=global_max)

    # Plot a difference map (Retro - Pros), also in subplots
    plot_diff_map(pros_fill, retro_fill, title="Retrospective - Prospective")

    # 10) Confusion matrix
    confusion_mat = np.zeros((total_bins, total_bins), dtype=int)
    for k in range(N):
        if acquired_lines[k] is None:
            continue
        row, p_bin = acquired_lines[k]
        frac = actual_frac[k]
        if np.isnan(frac):
            continue
        if k > 0 and not np.isnan(actual_frac[k - 1]):
            is_inhale = frac >= actual_frac[k - 1]
        else:
            is_inhale = frac < 50
        r_bin = assign_prospective_bin(
            fraction=frac,
            is_inhale=is_inhale,
            num_inhale_bins=NUM_INHALE_BINS,
            num_exhale_bins=NUM_EXHALE_BINS,
            use_total_bins=USE_TOTAL_BINS_INSTEAD,
            num_total_bins=NUM_TOTAL_BINS,
        )
        confusion_mat[p_bin, r_bin] += 1

    row_labels = [f"ProsBin_{i}" for i in range(total_bins)]
    col_labels = [f"RetroBin_{j}" for j in range(total_bins)]
    df_conf = pd.DataFrame(confusion_mat, index=row_labels, columns=col_labels)
    print("\nConfusion Matrix (Rows = Prospective Bin, Columns = Retrospective Bin):")
    print(df_conf)

    correct_assignments = np.sum(np.diag(confusion_mat))
    total_assignments = np.sum(confusion_mat)
    accuracy = (correct_assignments / total_assignments) if total_assignments > 0 else 0
    print(f"\nProspective binning accuracy = {accuracy * 100:.2f}%\n")

    # 11) Plot fraction & raw respiration
    t = np.arange(N) / fs
    plt.figure(figsize=(10, 5))

    # Plot actual fraction
    plt.plot(t, actual_frac, label="Actual Fraction (Offline)", lw=2, color="black")

    # Plot predicted fraction, color-coded by phase
    start_idx = 0
    for i in range(1, N):
        if predicted_phase[i] != predicted_phase[i - 1]:
            seg_t = t[start_idx:i]
            seg_y = predicted_fraction[start_idx:i]
            if predicted_phase[i - 1] is True:
                color = "green"
            elif predicted_phase[i - 1] is False:
                color = "blue"
            else:
                color = "gray"
            plt.plot(seg_t, seg_y, "-", color=color, lw=2)
            start_idx = i
    # last segment
    if start_idx < N:
        seg_t = t[start_idx:]
        seg_y = predicted_fraction[start_idx:]
        if predicted_phase[-1] is True:
            color = "green"
        elif predicted_phase[-1] is False:
            color = "blue"
        else:
            color = "gray"
        plt.plot(seg_t, seg_y, "-", color=color, lw=2)

    if calibration_end_idx is not None:
        plt.axvline(
            calibration_end_idx / fs, color="r", linestyle=":", lw=2, label="Calib End"
        )

    # Plot raw signal normalized 0..100
    raw_norm = (resp_data - np.min(resp_data)) / (np.ptp(resp_data) + 1e-9) * 100.0
    plt.plot(t, raw_norm, label="Raw Signal (0..100)", color="magenta", alpha=0.5)

    plt.title("Respiratory Fraction (Online vs. Offline), plus Raw Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Resp Cycle Fraction (%)")
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.show()

    print("Simulation complete!")


if __name__ == "__main__":
    main()
