#!/usr/bin/env python3
"""
simulate_joint_binning.py

This script demonstrates a joint prospective binning approach that uses both
respiration and cardiac phases to fill a joint k-space bin array.
It then performs retrospective analysis and visualizes the results via subplots
and confusion matrix plots.

Refactoring changes:
 - Split main() into several helper functions: load_config_file, process_data,
   perform_prospective_binning, perform_retrospective_assignment, and visualize_results.
 - Improved function and variable names for clarity.
 - Added detailed docstrings and inline comments.
"""

import yaml
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
import matplotlib.colors as mcolors
import functools

# Local module imports
import utils.data_ingestion as di
import utils.ecg_resp as ecg_resp
import utils.resp_fractions as rf
import utils.kspace_filling as kf


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


def process_data(config):
    """
    Load scan data, extract k-space and signals, and compute sampling parameters.

    Returns
    -------
    dict
        Dictionary containing:
            - kspace: raw k-space data
            - fs: sampling frequency computed from DICOM parameters
            - n_frames: number of frames
            - resp_signal_resampled: resampled respiration signal
            - ecg_signal_resampled: resampled ECG signal
    """
    TWIX_FILE = config["data"]["twix_file"]
    DICOM_DIR = config["data"]["dicom_folder"]
    RESP_FILE = config["data"].get("resp_file", None)
    ECG_FILES = config["data"].get("ecg_files", [None])
    n_frames = config["data"]["n_frames"]

    scans = di.read_twix_file(TWIX_FILE, include_scans=[-1], parse_pmu=False)
    kspace = di.extract_image_data(scans)
    FRAMERATE, _ = di.get_dicom_framerate(DICOM_DIR)
    N_PHASE_ENCODES_PER_FRAME = kspace.shape[0] // n_frames
    fs = FRAMERATE * N_PHASE_ENCODES_PER_FRAME
    print(f"Resp sample rate => fs={fs:.2f} Hz")

    # Resample respiration and ECG signals to match kspace length
    resp_signal = np.loadtxt(RESP_FILE, skiprows=1, usecols=1)
    ecg_signal = np.loadtxt(ECG_FILES[0], skiprows=1, usecols=1)
    N_k = kspace.shape[0]

    resp_signal_resampled = signal.resample(resp_signal, N_k)
    ecg_signal_resampled = signal.resample(ecg_signal, N_k)
    print(f"Resp data resampled length: {len(resp_signal_resampled)}")
    print(f"ECG data resampled length: {len(ecg_signal_resampled)}")

    return {
        "kspace": kspace,
        "fs": fs,
        "n_frames": n_frames,
        "N_k": N_k,
        "resp_signal_resampled": resp_signal_resampled,
        "ecg_signal_resampled": ecg_signal_resampled,
    }


def perform_prospective_binning(data, config):
    """
    Compute prospective respiratory and cardiac fractions and fill the joint k-space bins.

    Parameters
    ----------
    data : dict
        Data dictionary from process_data().
    config : dict
        Loaded configuration.

    Returns
    -------
    tuple
        (prospective_fill_joint, acquired_lines, predicted_resp_fraction,
         predicted_resp_phase, predicted_card_fraction)
    """
    fs = data["fs"]
    N_k = data["N_k"]

    # Predict respiratory and cardiac fractions
    predicted_resp_fraction, predicted_resp_phase, _ = rf.predict_fraction(
        data["resp_signal_resampled"],
        fs,
        min_calib_time=10.0,
        peak_height=0.6,
        peak_prom=0.2,
    )
    predicted_card_fraction, _ = rf.predict_cardiac_fraction(
        data["ecg_signal_resampled"],
        fs,
        min_calib_time=5.0,
        peak_height=0.5,
        peak_prom=0.2,
    )
    print("Calibration done for respiration and cardiac signals.")

    # Determine bin numbers
    USE_TOTAL_BINS = False
    NUM_INHALE_BINS = 3
    NUM_EXHALE_BINS = 1
    num_resp_bins = NUM_INHALE_BINS + NUM_EXHALE_BINS if not USE_TOTAL_BINS else 4
    NUM_CARD_BINS = 20

    # Initialize prospective k-space fill arrays
    KSPACE_H, KSPACE_W = 128, 128
    pros_fill_joint = np.zeros(
        (num_resp_bins, NUM_CARD_BINS, KSPACE_H, KSPACE_W), dtype=float
    )
    pros_priority_joint = kf.build_line_priority_joint(
        num_resp_bins=num_resp_bins,
        num_card_bins=NUM_CARD_BINS,
        kspace_height=KSPACE_H,
        priority_exponent=2.6,
    )

    # Create partial functions for joint bin weight and assignment
    get_joint_weights = functools.partial(
        kf.get_joint_bin_weights_gaussian,
        num_inhale_bins=NUM_INHALE_BINS,
        num_exhale_bins=NUM_EXHALE_BINS,
        use_total_resp_bins=USE_TOTAL_BINS,
        num_total_resp_bins=4,
        num_card_bins=NUM_CARD_BINS,
        resp_sigma_factor=0.4,
        card_sigma_factor=0.7,
    )
    assign_joint_bin = functools.partial(
        kf.assign_prospective_bin_joint,
        num_inhale_bins=NUM_INHALE_BINS,
        num_exhale_bins=NUM_EXHALE_BINS,
        use_total_resp_bins=USE_TOTAL_BINS,
        num_total_resp_bins=4,
        num_card_bins=NUM_CARD_BINS,
    )

    # Run the prospective filling loop for joint bins
    acquired_lines = kf.prospective_fill_loop_joint(
        N=N_k,
        resp_fraction_array=predicted_resp_fraction,
        resp_phase_array=predicted_resp_phase,
        cardiac_fraction_array=predicted_card_fraction,
        pros_fill=pros_fill_joint,
        pros_priority=pros_priority_joint,
        get_joint_weights_fn=get_joint_weights,
        assign_bin_joint_fn=assign_joint_bin,
        penalty_factor=0.26,
    )
    print("Completed prospective joint binning.")

    return (
        pros_fill_joint,
        acquired_lines,
        predicted_resp_fraction,
        predicted_resp_phase,
        predicted_card_fraction,
    )


def perform_retrospective_assignment(data, config, pros_fill_joint, acquired_lines):
    """
    Compute offline respiratory fractions and perform retrospective k-space filling.

    Returns
    -------
    tuple
        (retro_fill_joint, diff_joint, cycles, resp_frac_offline, card_frac_offline)
    """
    fs = data["fs"]
    N_k = data["N_k"]

    # Offline respiration: detect peaks and troughs for full signal
    resp_peaks_all = rf.detect_resp_peaks(
        data["resp_signal_resampled"], fs, method="scipy", height=0.6, prominence=0.2
    )
    resp_troughs_all = rf.detect_resp_peaks(
        -data["resp_signal_resampled"], fs, method="scipy", height=0.6, prominence=0.2
    )
    resp_frac_offline = rf.build_true_fraction(
        signal_length=N_k, peaks=resp_peaks_all, troughs=resp_troughs_all
    )
    all_boundaries = np.sort(np.concatenate((resp_peaks_all, resp_troughs_all)))
    boundary_labels = [
        "peak" if idx in resp_peaks_all else "trough" for idx in all_boundaries
    ]
    cycles = rf.extract_cycles(all_boundaries, np.array(boundary_labels, dtype=object))

    def is_inhale_offline(k, cycles_list):
        for start_idx, end_idx, phase in cycles_list:
            if start_idx <= k < end_idx:
                return True if phase == "inhalation" else False
        return None

    # Offline cardiac fraction
    ecg_peaks_all = ecg_resp.detect_r_peaks(data["ecg_signal_resampled"], fs)
    rpeaks_single = ecg_peaks_all[0] if len(ecg_peaks_all) > 0 else np.array([])
    card_frac_offline = np.full(N_k, np.nan)
    for k in range(N_k):
        behind = rpeaks_single[rpeaks_single <= k]
        if len(behind) == 0:
            continue
        last_r = behind[-1]
        ahead = rpeaks_single[rpeaks_single > k]
        if len(ahead) == 0:
            continue
        next_r = ahead[0]
        denom = next_r - last_r
        if denom > 0:
            frac = 100.0 * (k - last_r) / denom
            card_frac_offline[k] = max(0, min(100, frac))

    num_resp_bins, NUM_CARD_BINS, KSPACE_H, KSPACE_W = pros_fill_joint.shape
    retro_fill_joint = np.zeros(
        (num_resp_bins, NUM_CARD_BINS, KSPACE_H, KSPACE_W), dtype=float
    )

    for k in range(N_k):
        if acquired_lines[k] is None:
            continue
        row, _, _ = acquired_lines[k]
        r_frac = resp_frac_offline[k]
        c_frac = card_frac_offline[k]
        if np.isnan(r_frac) or np.isnan(c_frac):
            continue
        inhale_flag = is_inhale_offline(k, cycles)
        (rbin_true, cbin_true) = kf.assign_prospective_bin_joint(
            resp_fraction=r_frac,
            resp_is_inhale=inhale_flag,
            cardiac_fraction=c_frac,
            num_inhale_bins=3,
            num_exhale_bins=1,
            use_total_resp_bins=False,
            num_total_resp_bins=4,
            num_card_bins=NUM_CARD_BINS,
        )
        retro_fill_joint[rbin_true, cbin_true, row, :] += 1.0

    diff_joint = retro_fill_joint - pros_fill_joint
    return retro_fill_joint, diff_joint, cycles, resp_frac_offline, card_frac_offline


def visualize_results(
    pros_fill_joint,
    retro_fill_joint,
    diff_joint,
    cycles,
    acquired_lines,
    resp_frac_offline,
    card_frac_offline,
    data,
):
    """
    Visualize the prospective and retrospective joint k-space filling,
    and plot separate confusion matrices for cardiac and respiratory bins.

    Also compute and print:
      - Joint accuracy (percentage of samples where the prospective joint bin matches the retrospective joint bin)
      - Accuracy percentages for cardiac and respiratory bins separately
      - Average bin error (absolute difference between prospective and retrospective assignments)
    """
    num_resp_bins, NUM_CARD_BINS, KSPACE_H, _ = pros_fill_joint.shape
    N_k = data["N_k"]

    # Plot the k-space fill subplots as before
    for rbin in range(num_resp_bins):
        global_max_pros = np.max(pros_fill_joint[rbin, :, :, :]) or 1.0
        global_max_retro = np.max(retro_fill_joint[rbin, :, :, :]) or 1.0
        global_max_diff = np.max(np.abs(diff_joint[rbin, :, :, :])) or 1.0

        fig, axs = plt.subplots(
            3, NUM_CARD_BINS, figsize=(4 * NUM_CARD_BINS, 12), sharex=True, sharey=True
        )
        if NUM_CARD_BINS == 1:
            axs = np.expand_dims(axs, axis=1)
        for cbin in range(NUM_CARD_BINS):
            im0 = axs[0, cbin].imshow(
                pros_fill_joint[rbin, cbin],
                cmap="gray",
                norm=mcolors.PowerNorm(gamma=0.3, vmin=0, vmax=global_max_pros),
                origin="upper",
                aspect="auto",
            )
            axs[0, cbin].set_title(f"Pros: Rbin={rbin}, Cbin={cbin}", fontsize=10)
            axs[0, cbin].axis("off")
            plt.colorbar(im0, ax=axs[0, cbin], fraction=0.046, pad=0.04)

            im1 = axs[1, cbin].imshow(
                retro_fill_joint[rbin, cbin],
                cmap="gray",
                norm=mcolors.PowerNorm(gamma=0.3, vmin=0, vmax=global_max_retro),
                origin="upper",
                aspect="auto",
            )
            axs[1, cbin].set_title(f"Retro: Rbin={rbin}, Cbin={cbin}", fontsize=10)
            axs[1, cbin].axis("off")
            plt.colorbar(im1, ax=axs[1, cbin], fraction=0.046, pad=0.04)

            im2 = axs[2, cbin].imshow(
                diff_joint[rbin, cbin],
                cmap="bwr",
                norm=mcolors.PowerNorm(
                    gamma=0.3, vmin=-global_max_diff, vmax=global_max_diff
                ),
                origin="upper",
                aspect="auto",
            )
            axs[2, cbin].set_title(f"Diff: Rbin={rbin}, Cbin={cbin}", fontsize=10)
            axs[2, cbin].axis("off")
            plt.colorbar(im2, ax=axs[2, cbin], fraction=0.046, pad=0.04)
        fig.suptitle(f"Joint K-Space Binning: Respiratory Bin {rbin}", fontsize=14)
        fig.tight_layout()
        out_png = f"joint_fill_rbin_{rbin}_subplots.png"
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved subplot figure for respiratory bin {rbin} => {out_png}")

    # Parameters for separate confusion matrices
    NUM_INHALE_BINS = 3
    NUM_EXHALE_BINS = 1
    USE_TOTAL_BINS = False
    TOTAL_RESP_BINS = NUM_INHALE_BINS + NUM_EXHALE_BINS  # e.g., 4

    # Initialize confusion matrices for cardiac and respiratory bins
    confusion_card = np.zeros((NUM_CARD_BINS, NUM_CARD_BINS), dtype=int)
    confusion_resp = np.zeros((TOTAL_RESP_BINS, TOTAL_RESP_BINS), dtype=int)

    # For joint accuracy (both resp and cardiac match)
    joint_correct = 0
    joint_total = 0

    # To compute average bin error differences
    total_card_diff = 0
    total_resp_diff = 0
    count_card = 0
    count_resp = 0

    # Helper: determine if sample k is in inhalation phase offline
    def is_inhale_offline(k, cycles_list):
        for start_idx, end_idx, phase in cycles_list:
            if start_idx <= k < end_idx:
                return True if phase == "inhalation" else False
        return None

    # Loop over all time steps to fill confusion matrices and accumulate error differences
    for k in range(N_k):
        if acquired_lines[k] is None:
            continue
        # Skip if offline fractions are NaN
        if np.isnan(resp_frac_offline[k]) or np.isnan(card_frac_offline[k]):
            continue

        # Prospective assignments from acquired_lines:
        _, p_resp_bin, p_card_bin = acquired_lines[k]

        # Joint retrospective assignment using offline fractions:
        r_rbin, r_cbin = kf.assign_prospective_bin_joint(
            resp_fraction=resp_frac_offline[k],
            resp_is_inhale=is_inhale_offline(k, cycles),
            cardiac_fraction=card_frac_offline[k],
            num_inhale_bins=NUM_INHALE_BINS,
            num_exhale_bins=NUM_EXHALE_BINS,
            use_total_resp_bins=USE_TOTAL_BINS,
            num_total_resp_bins=TOTAL_RESP_BINS,
            num_card_bins=NUM_CARD_BINS,
        )
        if p_resp_bin == r_rbin and p_card_bin == r_cbin:
            joint_correct += 1
        joint_total += 1

        # For cardiac: compute retrospective cardiac bin (using simple floor division)
        bin_width_card = 100.0 / NUM_CARD_BINS
        r_card_bin = int(np.floor(card_frac_offline[k] / bin_width_card))
        r_card_bin = min(r_card_bin, NUM_CARD_BINS - 1)
        confusion_card[p_card_bin, r_card_bin] += 1
        total_card_diff += abs(p_card_bin - r_card_bin)
        count_card += 1

        # For respiratory: compute retrospective resp bin using assign_prospective_bin
        r_resp_bin = kf.assign_prospective_bin(
            fraction=resp_frac_offline[k],
            is_inhale=is_inhale_offline(k, cycles),
            num_inhale_bins=NUM_INHALE_BINS,
            num_exhale_bins=NUM_EXHALE_BINS,
            use_total_bins=USE_TOTAL_BINS,
            num_total_bins=TOTAL_RESP_BINS,
        )
        confusion_resp[p_resp_bin, r_resp_bin] += 1
        total_resp_diff += abs(p_resp_bin - r_resp_bin)
        count_resp += 1

    # Compute accuracies and average errors
    joint_accuracy = (joint_correct / joint_total * 100) if joint_total > 0 else 0
    total_card = np.sum(confusion_card)
    correct_card = np.trace(confusion_card)
    card_accuracy = (correct_card / total_card * 100) if total_card > 0 else 0
    total_resp = np.sum(confusion_resp)
    correct_resp = np.trace(confusion_resp)
    resp_accuracy = (correct_resp / total_resp * 100) if total_resp > 0 else 0
    avg_card_error = total_card_diff / count_card if count_card > 0 else 0
    avg_resp_error = total_resp_diff / count_resp if count_resp > 0 else 0

    # Plot the two confusion matrices as subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    im1 = axs[0].imshow(confusion_card, origin="upper", cmap="Blues")
    axs[0].set_title("Cardiac Bin Confusion")
    axs[0].set_xlabel("Retrospective Cardiac Bin")
    axs[0].set_ylabel("Prospective Cardiac Bin")
    plt.colorbar(im1, ax=axs[0], fraction=0.046, pad=0.04)

    im2 = axs[1].imshow(confusion_resp, origin="upper", cmap="Blues")
    axs[1].set_title("Respiratory Bin Confusion")
    axs[1].set_xlabel("Retrospective Resp Bin")
    axs[1].set_ylabel("Prospective Resp Bin")
    plt.colorbar(im2, ax=axs[1], fraction=0.046, pad=0.04)

    fig.suptitle("Separate Confusion Matrices")
    plt.tight_layout()
    plt.savefig("separate_confusion_matrices.png", dpi=150)
    plt.close()
    print("Saved separate confusion matrices => separate_confusion_matrices.png")

    # Print computed statistics
    print(f"Joint bin accuracy: {joint_accuracy:.2f}% over {joint_total} samples")
    print(f"Cardiac bin accuracy: {card_accuracy:.2f}% over {total_card} samples")
    print(f"Respiratory bin accuracy: {resp_accuracy:.2f}% over {total_resp} samples")
    print(f"Average cardiac bin error (in bin units): {avg_card_error:.2f}")
    print(f"Average respiratory bin error (in bin units): {avg_resp_error:.2f}")


def main():
    config = load_config_file()
    data = process_data(config)
    pros_fill_joint, acquired_lines, pred_resp_frac, pred_resp_phase, pred_card_frac = (
        perform_prospective_binning(data, config)
    )
    retro_fill_joint, diff_joint, cycles, resp_frac_offline, card_frac_offline = (
        perform_retrospective_assignment(data, config, pros_fill_joint, acquired_lines)
    )
    visualize_results(
        pros_fill_joint,
        retro_fill_joint,
        diff_joint,
        cycles,
        acquired_lines,
        resp_frac_offline,
        card_frac_offline,
        data,
    )


if __name__ == "__main__":
    main()
