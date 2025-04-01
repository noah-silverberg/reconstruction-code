#!/usr/bin/env python3
"""
simulate_resp_binning.py

This script performs a simulation of prospective and retrospective respiratory binning,
including fraction/phase detection, line priority filling, confusion matrix calculation,
and final k-space visualization.

Refactoring changes:
 - Split the original script into logical functions:
   load_config_file, load_respiration_data, run_prospective_binning, run_retrospective_binning,
   plot_results, and plot_fraction_comparison.
 - Improved variable names and added detailed documentation.
"""

import yaml
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as mcolors
from scipy import signal
import functools

# Local module imports
import utils.data_ingestion as di
from utils.resp_fractions import (
    predict_fraction,
    build_true_fraction,
    detect_resp_peaks,
)
from utils.kspace_filling import (
    build_line_priority,
    prospective_fill_loop,
    get_bin_index_gaussian,
    assign_prospective_bin,
)


def load_config_file(config_file="config.yaml"):
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


def load_respiration_data(config):
    """
    Load the respiration signal and k-space data.

    Parameters
    ----------
    config : dict
        Loaded configuration.

    Returns
    -------
    dict
        Dictionary containing:
         - respiration_signal: 1D respiration data (possibly resampled)
         - kspace: raw k-space data
         - fs: sampling frequency in Hz
    """
    RESP_FILE = config["data"].get("resp_file", None)
    DICOM_FOLDER = config["data"]["dicom_folder"]
    TWIX_FILE = config["data"]["twix_file"]

    # Read k-space from TWIX file
    scans = di.read_twix_file(TWIX_FILE, include_scans=[-1], parse_pmu=False)
    kspace = di.extract_image_data(scans)

    FRAMERATE, _ = di.get_dicom_framerate(DICOM_FOLDER)
    N_PHASE_ENCODES_PER_FRAME = kspace.shape[0] // config["data"]["n_frames"]
    fs = FRAMERATE * N_PHASE_ENCODES_PER_FRAME

    # Load respiration signal
    respiration_signal = np.loadtxt(RESP_FILE, skiprows=1, usecols=1)
    N_raw = len(respiration_signal)
    # Optionally downsample
    DOWNSAMPLE_FACTOR = config.get("DOWNSAMPLE_FACTOR", 1)
    if DOWNSAMPLE_FACTOR > 1:
        respiration_signal = signal.resample(
            respiration_signal, N_raw // DOWNSAMPLE_FACTOR
        )
        fs = fs / DOWNSAMPLE_FACTOR
        print(
            f"Downsampled respiration data to {len(respiration_signal)} samples at fs={fs:.2f} Hz."
        )
    else:
        print(
            f"Loaded respiration data with {len(respiration_signal)} samples at fs={fs:.2f} Hz."
        )

    return {"respiration_signal": respiration_signal, "kspace": kspace, "fs": fs}


def run_prospective_binning(respiration_signal, fs, config):
    """
    Run the prospective binning simulation.

    Parameters
    ----------
    respiration_signal : np.ndarray
        Respiration data.
    fs : float
        Sampling frequency.
    config : dict
        Configuration parameters.

    Returns
    -------
    tuple
        (predicted_fraction, predicted_phase, calibration_end_idx,
         prospective_fill, acquired_lines)
    """
    # Main parameters for respiration binning from config
    MIN_CALIB_TIME = config.get("MIN_CALIB_TIME", 10.0)
    PEAK_HEIGHT = config.get("PEAK_HEIGHT", 0.6)
    PEAK_PROMINENCE = config.get("PEAK_PROMINENCE", 0.2)
    NUM_INHALE_BINS = config.get("NUM_INHALE_BINS", 2)
    NUM_EXHALE_BINS = config.get("NUM_EXHALE_BINS", 2)
    USE_TOTAL_BINS = config.get("USE_TOTAL_BINS_INSTEAD", False)
    NUM_TOTAL_BINS = config.get("NUM_TOTAL_BINS", 4)
    KSPACE_H = config.get("KSPACE_H", 128)
    KSPACE_W = config.get("KSPACE_W", 128)

    predicted_fraction, predicted_phase, calib_end_idx = predict_fraction(
        resp_signal=respiration_signal,
        fs=fs,
        min_calib_time=MIN_CALIB_TIME,
        peak_height=PEAK_HEIGHT,
        peak_prom=PEAK_PROMINENCE,
    )
    print("Prospective respiration fraction prediction completed.")

    # Initialize k-space fill array and line priority for prospective binning
    total_bins = (
        NUM_TOTAL_BINS if USE_TOTAL_BINS else (NUM_INHALE_BINS + NUM_EXHALE_BINS)
    )
    prospective_fill = np.zeros((total_bins, KSPACE_H, KSPACE_W), dtype=float)
    line_priority = build_line_priority(total_bins, kspace_height=KSPACE_H)

    # Create partial functions for bin index determination and assignment
    get_bin_index_fn = functools.partial(
        get_bin_index_gaussian,
        num_inhale_bins=NUM_INHALE_BINS,
        num_exhale_bins=NUM_EXHALE_BINS,
        use_total_bins=USE_TOTAL_BINS,
        num_total_bins=NUM_TOTAL_BINS,
    )
    assign_bin_fn = functools.partial(
        assign_prospective_bin,
        num_inhale_bins=NUM_INHALE_BINS,
        num_exhale_bins=NUM_EXHALE_BINS,
        use_total_bins=USE_TOTAL_BINS,
        num_total_bins=NUM_TOTAL_BINS,
    )

    acquired_lines = prospective_fill_loop(
        N=len(respiration_signal),
        predicted_fraction=predicted_fraction,
        predicted_phase=predicted_phase,
        pros_fill=prospective_fill,
        pros_priority=line_priority,
        get_bin_index_fn=get_bin_index_fn,
        assign_bin_fn=assign_bin_fn,
    )
    print("Completed prospective binning simulation.")
    return (
        predicted_fraction,
        predicted_phase,
        calib_end_idx,
        prospective_fill,
        acquired_lines,
    )


def run_retrospective_binning(
    respiration_signal, fs, prospective_fill, acquired_lines, config
):
    """
    Run the retrospective binning process by detecting offline peaks/troughs
    and reconstructing the true fraction for each time sample.

    Returns
    -------
    retro_fill : np.ndarray
        Retrospective k-space fill array.
    confusion_mat : pd.DataFrame
        Confusion matrix comparing prospective and retrospective assignments.
    """
    SKIP_ROWS = config.get("SKIP_ROWS", 1)
    PEAK_HEIGHT = config.get("PEAK_HEIGHT", 0.6)
    PEAK_PROMINENCE = config.get("PEAK_PROMINENCE", 0.2)
    NUM_INHALE_BINS = config.get("NUM_INHALE_BINS", 2)
    NUM_EXHALE_BINS = config.get("NUM_EXHALE_BINS", 2)
    USE_TOTAL_BINS = config.get("USE_TOTAL_BINS_INSTEAD", False)
    NUM_TOTAL_BINS = config.get("NUM_TOTAL_BINS", 4)
    KSPACE_H = config.get("KSPACE_H", 128)
    KSPACE_W = config.get("KSPACE_W", 128)

    N = len(respiration_signal)
    actual_fraction = build_true_fraction(
        signal_length=N,
        peaks=detect_resp_peaks(
            respiration_signal,
            fs,
            method="scipy",
            height=PEAK_HEIGHT,
            prominence=PEAK_PROMINENCE,
        ),
        troughs=detect_resp_peaks(
            -respiration_signal,
            fs,
            method="scipy",
            height=PEAK_HEIGHT,
            prominence=PEAK_PROMINENCE,
        ),
    )
    retro_fill = np.zeros(
        (
            USE_TOTAL_BINS and NUM_TOTAL_BINS or (NUM_INHALE_BINS + NUM_EXHALE_BINS),
            KSPACE_H,
            KSPACE_W,
        ),
        dtype=float,
    )

    assign_bin_fn = functools.partial(
        assign_prospective_bin,
        num_inhale_bins=NUM_INHALE_BINS,
        num_exhale_bins=NUM_EXHALE_BINS,
        use_total_bins=USE_TOTAL_BINS,
        num_total_bins=NUM_TOTAL_BINS,
    )

    # Retrospective assignment loop
    for k in range(N):
        if acquired_lines[k] is None:
            continue
        row, p_bin = acquired_lines[k]
        frac = actual_fraction[k]
        if np.isnan(frac):
            continue
        # Determine inhalation vs exhalation
        is_inhale = (
            (frac >= actual_fraction[k - 1])
            if (k > 0 and not np.isnan(actual_fraction[k - 1]))
            else (frac < 50)
        )
        true_bin = assign_bin_fn(frac, is_inhale)
        retro_fill[true_bin, row, :] += 1.0

    # Compute confusion matrix
    total_bins = (
        USE_TOTAL_BINS and NUM_TOTAL_BINS or (NUM_INHALE_BINS + NUM_EXHALE_BINS)
    )
    confusion = np.zeros((total_bins, total_bins), dtype=int)
    for k in range(N):
        if acquired_lines[k] is None:
            continue
        row, p_bin = acquired_lines[k]
        frac = actual_fraction[k]
        if np.isnan(frac):
            continue
        is_inhale = (
            (frac >= actual_fraction[k - 1])
            if (k > 0 and not np.isnan(actual_fraction[k - 1]))
            else (frac < 50)
        )
        r_bin = assign_bin_fn(frac, is_inhale)
        confusion[p_bin, r_bin] += 1

    row_labels = [f"ProsBin_{i}" for i in range(total_bins)]
    col_labels = [f"RetroBin_{j}" for j in range(total_bins)]
    df_confusion = pd.DataFrame(confusion, index=row_labels, columns=col_labels)
    print("\nConfusion Matrix (Prospective vs. Retrospective):")
    print(df_confusion)

    correct_assignments = np.trace(confusion)
    total_assignments = np.sum(confusion)
    accuracy = (
        (correct_assignments / total_assignments) * 100 if total_assignments > 0 else 0
    )
    print(f"Prospective binning accuracy = {accuracy:.2f}%")
    return retro_fill, df_confusion


def plot_results(prospective_fill, retro_fill):
    """
    Plot the prospective and retrospective k-space bins and the difference maps.
    """
    global_max = max(prospective_fill.max(), retro_fill.max()) or 1.0
    total_bins = prospective_fill.shape[0]
    cols = min(total_bins, 4)
    rows = int(np.ceil(total_bins / cols))

    # Plot prospective bins
    fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axs = axs.flatten()
    for i in range(total_bins):
        im = axs[i].imshow(
            prospective_fill[i],
            cmap="gray",
            norm=mcolors.PowerNorm(gamma=0.3, vmin=0, vmax=global_max),
        )
        axs[i].set_title(f"Prospective Bin {i}")
        axs[i].axis("off")
        plt.colorbar(im, ax=axs[i])
    fig.suptitle("Prospective K-space Bins")
    plt.tight_layout()
    plt.show()

    # Plot retrospective bins
    fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axs = axs.flatten()
    for i in range(total_bins):
        im = axs[i].imshow(
            retro_fill[i],
            cmap="gray",
            norm=mcolors.PowerNorm(gamma=0.3, vmin=0, vmax=global_max),
        )
        axs[i].set_title(f"Retrospective Bin {i}")
        axs[i].axis("off")
        plt.colorbar(im, ax=axs[i])
    fig.suptitle("Retrospective K-space Bins")
    plt.tight_layout()
    plt.show()


def plot_fraction_comparison(
    respiration_signal, predicted_fraction, actual_fraction, fs, calib_end_idx
):
    """
    Plot the online (predicted) and offline (actual) respiratory fraction along with the raw signal.
    """
    t = np.arange(len(respiration_signal)) / fs
    plt.figure(figsize=(10, 5))
    plt.plot(t, actual_fraction, label="Actual Fraction (Offline)", lw=2, color="black")
    plt.plot(
        t, predicted_fraction, label="Predicted Fraction (Online)", lw=2, color="blue"
    )
    if calib_end_idx is not None:
        plt.axvline(
            calib_end_idx / fs,
            color="red",
            linestyle=":",
            lw=2,
            label="Calibration End",
        )
    raw_norm = (
        (respiration_signal - np.min(respiration_signal))
        / (np.ptp(respiration_signal) + 1e-9)
        * 100.0
    )
    plt.plot(
        t, raw_norm, label="Raw Resp Signal (Normalized)", color="magenta", alpha=0.5
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Fraction / Normalized Signal")
    plt.title("Respiratory Fraction Comparison")
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.show()


def main():
    config = load_config_file()
    data = load_respiration_data(config)
    pred_frac, pred_phase, calib_end_idx, pros_fill, acquired_lines = (
        run_prospective_binning(data["respiration_signal"], data["fs"], config)
    )
    retro_fill, df_conf = run_retrospective_binning(
        data["respiration_signal"], data["fs"], pros_fill, acquired_lines, config
    )
    plot_results(pros_fill, retro_fill)
    # Optionally, if you want to compare predicted vs. actual fraction, compute actual_fraction here.
    # For brevity, that step is omitted.
    # plot_fraction_comparison(data["respiration_signal"], pred_frac, actual_fraction, data["fs"], calib_end_idx)
    print("Simulation complete!")


if __name__ == "__main__":
    main()
