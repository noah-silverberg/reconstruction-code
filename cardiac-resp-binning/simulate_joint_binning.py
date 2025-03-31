#!/usr/bin/env python3
"""
simulate_joint_binning.py

Demonstrates a prospective approach that uses *both* respiration and cardiac phases
to fill a joint k-space bin array of shape (num_resp_bins, NUM_CARD_BINS, height, width).

We also display the prospective-filled k-space as GIFs, one per respiratory bin,
where each frame of the GIF is a different cardiac bin.
"""

import yaml
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Local imports
import utils.data_ingestion as di
import utils.ecg_resp as ecg_resp
import utils.resp_fractions as rf
import utils.kspace_filling as kf
from scipy import signal
import matplotlib.colors as mcolors

# We'll use the same GIF utilities as in your other scripts
import utils.gif as gif


def load_config(config_file="config.yaml"):
    with open(config_file, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()

    # Basic data paths
    TWIX_FILE = config["data"]["twix_file"]
    DICOM_DIR = config["data"]["dicom_folder"]
    RESP_FILE = config["data"].get("resp_file", None)
    ECG_FILES = config["data"].get("ecg_files", [None])

    # read TWIX, etc
    scans = di.read_twix_file(TWIX_FILE, include_scans=[-1], parse_pmu=False)
    kspace = di.extract_image_data(scans)
    FRAMERATE, _ = di.get_dicom_framerate(DICOM_DIR)
    n_frames = config["data"]["n_frames"]
    N_PHASE_ENCODES_PER_FRAME = kspace.shape[0] // n_frames
    fs = FRAMERATE * N_PHASE_ENCODES_PER_FRAME

    print(f"Resp sample rate => fs={fs:.2f} Hz")

    # Load respiration & ECG data from files
    resp_data = np.loadtxt(RESP_FILE, skiprows=1, usecols=1)
    ecg_data = np.loadtxt(ECG_FILES[0], skiprows=1, usecols=1)

    # 1) We figure out how many total lines are in kspace:
    N_k = kspace.shape[0]  # e.g. n_frames * N_PHASE_ENCODES_PER_FRAME

    # 2) We resample both signals to length N_k
    resp_data_resampled = signal.resample(resp_data, N_k)
    ecg_data_resampled = signal.resample(ecg_data, N_k)
    print(f"Resp data resampled length: {len(resp_data_resampled)}")
    print(f"ECG data resampled length: {len(ecg_data_resampled)}")

    # 4) Now call the predictive fraction functions
    predicted_resp_fraction, predicted_resp_phase, resp_calib_end = rf.predict_fraction(
        resp_data_resampled, fs, min_calib_time=10.0, peak_height=0.6, peak_prom=0.2
    )
    print(f"Resp calibration end: {resp_calib_end / fs:.2f} sec")
    predicted_card_fraction, card_calib_end = rf.predict_cardiac_fraction(
        ecg_data_resampled, fs, min_calib_time=5.0, peak_height=0.5, peak_prom=0.2
    )
    print(f"Cardiac calibration end: {card_calib_end / fs:.2f} sec")

    # Decide how many bins we want
    USE_TOTAL_BINS = False
    NUM_INHALE_BINS = 3
    NUM_EXHALE_BINS = 1
    num_resp_bins = NUM_INHALE_BINS + NUM_EXHALE_BINS if not USE_TOTAL_BINS else 4

    NUM_CARD_BINS = 20  # e.g. 3 cardiac bins

    # Build the prospective fill arrays
    KSPACE_H = 128
    KSPACE_W = 128
    pros_fill_joint = np.zeros(
        (num_resp_bins, NUM_CARD_BINS, KSPACE_H, KSPACE_W), dtype=float
    )

    # Priority
    pros_priority_joint = kf.build_line_priority_joint(
        num_resp_bins=num_resp_bins, num_card_bins=NUM_CARD_BINS, kspace_height=KSPACE_H
    )

    import functools

    get_joint_weights = functools.partial(
        kf.get_joint_bin_weights_gaussian,
        num_inhale_bins=NUM_INHALE_BINS,
        num_exhale_bins=NUM_EXHALE_BINS,
        use_total_resp_bins=USE_TOTAL_BINS,
        num_total_resp_bins=4,
        num_card_bins=NUM_CARD_BINS,
        resp_sigma_factor=0.1,
        card_sigma_factor=0.1,
    )

    assign_joint_bin = functools.partial(
        kf.assign_prospective_bin_joint,
        num_inhale_bins=NUM_INHALE_BINS,
        num_exhale_bins=NUM_EXHALE_BINS,
        use_total_resp_bins=USE_TOTAL_BINS,
        num_total_resp_bins=4,
        num_card_bins=NUM_CARD_BINS,
    )

    # Now run the new joint prospective loop
    # 5) Then in your prospective loop, you do:
    acquired_lines = kf.prospective_fill_loop_joint(
        N=N_k,
        resp_fraction_array=predicted_resp_fraction,
        resp_phase_array=predicted_resp_phase,
        cardiac_fraction_array=predicted_card_fraction,
        pros_fill=pros_fill_joint,
        pros_priority=pros_priority_joint,
        get_joint_weights_fn=get_joint_weights,
        assign_bin_joint_fn=assign_joint_bin,
    )

    print("Done with prospective joint binning!")

    # after your prospective loop is done:

    # 1) Offline respiration: detect final peaks & troughs
    resp_peaks_all = rf.detect_resp_peaks(
        resp_data_resampled, fs, method="scipy", height=0.6, prominence=0.2
    )
    resp_troughs_all = rf.detect_resp_peaks(
        -resp_data_resampled, fs, method="scipy", height=0.6, prominence=0.2
    )

    # build_true_fraction => array of length N_k with 0..100 or NaN
    resp_frac_offline = rf.build_true_fraction(
        signal_length=N_k, peaks=resp_peaks_all, troughs=resp_troughs_all
    )

    # Build the list of offline boundaries:
    all_boundaries = np.sort(np.concatenate((resp_peaks_all, resp_troughs_all)))
    boundary_labels = []
    for idx in all_boundaries:
        if idx in resp_peaks_all:
            boundary_labels.append("peak")
        else:
            boundary_labels.append("trough")

    # Use extract_cycles (from your code) to compute the cycles list.
    cycles = rf.extract_cycles(all_boundaries, np.array(boundary_labels, dtype=object))

    def is_inhale_offline(k, cycles):
        """
        Determine if sample index k falls in an inhalation or exhalation cycle.
        'cycles' is a list of tuples (start_idx, end_idx, phase),
        where phase is the string "inhalation" or "exhalation".
        Returns True if k is within a trough-to-peak (inhalation) cycle,
        False if within a peak-to-trough (exhalation) cycle,
        and None if k does not fall within any cycle.
        """
        for start_idx, end_idx, phase in cycles:
            if start_idx <= k < end_idx:
                return True if phase == "inhalation" else False
        return None

    # 2) Offline cardiac fraction: detect R-peaks on the full ecg_data_resampled
    #    then for each sample k, figure out which 2 R-peaks it lies between
    #    and linearly map that to 0..100

    ecg_peaks_all = ecg_resp.detect_r_peaks(ecg_data_resampled, fs)
    if len(ecg_peaks_all) > 0:
        rpeaks_single = ecg_peaks_all[0]
    else:
        rpeaks_single = np.array([])

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

    retro_fill_joint = np.zeros(
        (num_resp_bins, NUM_CARD_BINS, KSPACE_H, KSPACE_W), dtype=float
    )

    for k in range(N_k):
        if acquired_lines[k] is None:
            continue
        row, p_rbin, p_cbin = acquired_lines[k]
        r_frac = resp_frac_offline[k]
        c_frac = card_frac_offline[k]
        if np.isnan(r_frac) or np.isnan(c_frac):
            continue
        i_inhale = is_inhale_offline(k, cycles)
        (rbin_true, cbin_true) = kf.assign_prospective_bin_joint(
            resp_fraction=r_frac,
            resp_is_inhale=i_inhale,
            cardiac_fraction=c_frac,
            num_inhale_bins=NUM_INHALE_BINS,
            num_exhale_bins=NUM_EXHALE_BINS,
            use_total_resp_bins=USE_TOTAL_BINS,
            num_total_resp_bins=4,
            num_card_bins=NUM_CARD_BINS,
        )
        retro_fill_joint[rbin_true, cbin_true, row, :] += 1.0

    diff_joint = retro_fill_joint - pros_fill_joint

    # --- New Visualization: one figure per respiratory bin (cine)
    # For each resp bin, create a subplot figure with 3 rows and NUM_CARD_BINS columns:
    # Row 1: prospective k-space; Row 2: retrospective k-space; Row 3: difference image.
    for rbin in range(num_resp_bins):
        # Compute global scales for this respiratory bin
        global_max_pros = np.max(pros_fill_joint[rbin, :, :, :])
        if global_max_pros < 1e-9:
            global_max_pros = 1.0
        global_max_retro = np.max(retro_fill_joint[rbin, :, :, :])
        if global_max_retro < 1e-9:
            global_max_retro = 1.0
        global_max_diff = np.max(np.abs(diff_joint[rbin, :, :, :]))
        if global_max_diff < 1e-9:
            global_max_diff = 1.0

        fig, axs = plt.subplots(
            3, NUM_CARD_BINS, figsize=(4 * NUM_CARD_BINS, 12), sharex=True, sharey=True
        )
        if NUM_CARD_BINS == 1:
            axs = np.expand_dims(axs, axis=1)
        for cbin in range(NUM_CARD_BINS):
            # Row 0: Prospective k-space
            kspace_pros = pros_fill_joint[rbin, cbin]
            im0 = axs[0, cbin].imshow(
                kspace_pros,
                cmap="gray",
                norm=mcolors.PowerNorm(gamma=0.3, vmin=0, vmax=global_max_pros),
                origin="upper",
                aspect="auto",
            )
            axs[0, cbin].set_title(f"Pros: Rbin={rbin}, Cbin={cbin}", fontsize=10)
            axs[0, cbin].axis("off")
            cbar0 = plt.colorbar(im0, ax=axs[0, cbin], fraction=0.046, pad=0.04)
            cbar0.set_label("Magnitude")

            # Row 1: Retrospective k-space
            kspace_retro = retro_fill_joint[rbin, cbin]
            im1 = axs[1, cbin].imshow(
                kspace_retro,
                cmap="gray",
                norm=mcolors.PowerNorm(gamma=0.3, vmin=0, vmax=global_max_retro),
                origin="upper",
                aspect="auto",
            )
            axs[1, cbin].set_title(f"Retro: Rbin={rbin}, Cbin={cbin}", fontsize=10)
            axs[1, cbin].axis("off")
            cbar1 = plt.colorbar(im1, ax=axs[1, cbin], fraction=0.046, pad=0.04)
            cbar1.set_label("Magnitude")

            # Row 2: Difference (Retro - Pros)
            diff_img = diff_joint[rbin, cbin]
            im2 = axs[2, cbin].imshow(
                diff_img,
                cmap="bwr",
                norm=mcolors.PowerNorm(
                    gamma=0.3, vmin=-global_max_diff, vmax=global_max_diff
                ),
                origin="upper",
                aspect="auto",
            )
            axs[2, cbin].set_title(f"Diff: Rbin={rbin}, Cbin={cbin}", fontsize=10)
            axs[2, cbin].axis("off")
            cbar2 = plt.colorbar(im2, ax=axs[2, cbin], fraction=0.046, pad=0.04)
            cbar2.set_label("Difference")
        fig.suptitle(f"Joint K-Space Binning: Respiratory Bin {rbin}", fontsize=14)
        fig.tight_layout()
        out_png = f"joint_fill_rbin_{rbin}_subplots.png"
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved subplot figure for respiratory bin {rbin} => {out_png}")

    # Confusion Matrix
    def flatten_joint_bin(rbin, cbin, num_card_bins):
        return rbin * num_card_bins + cbin

    num_joint_bins = num_resp_bins * NUM_CARD_BINS
    confusion_mat = np.zeros((num_joint_bins, num_joint_bins), dtype=int)

    for k in range(N_k):
        if acquired_lines[k] is None:
            continue
        row, p_rbin, p_cbin = acquired_lines[k]
        r_frac = resp_frac_offline[k]
        c_frac = card_frac_offline[k]
        if np.isnan(r_frac) or np.isnan(c_frac):
            continue
        i_inhale = is_inhale_offline(k, cycles)
        (r_rbin, r_cbin) = kf.assign_prospective_bin_joint(
            r_frac,
            i_inhale,
            c_frac,
            num_inhale_bins=NUM_INHALE_BINS,
            num_exhale_bins=NUM_EXHALE_BINS,
            use_total_resp_bins=USE_TOTAL_BINS,
            num_total_resp_bins=4,
            num_card_bins=NUM_CARD_BINS,
        )
        p_idx = flatten_joint_bin(p_rbin, p_cbin, NUM_CARD_BINS)
        r_idx = flatten_joint_bin(r_rbin, r_cbin, NUM_CARD_BINS)
        confusion_mat[p_idx, r_idx] += 1

    plt.figure(figsize=(8, 8))
    plt.title("Prospective vs. Retrospective (Joint Bins)")
    im = plt.imshow(confusion_mat, origin="upper", cmap="Blues")
    plt.xlabel("Retrospective Bin Index")
    plt.ylabel("Prospective Bin Index")
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Counts")
    plt.tight_layout()
    plt.savefig("joint_confusion_matrix.png", dpi=150)
    plt.close()
    print("Saved confusion matrix => joint_confusion_matrix.png")
    correct = np.trace(confusion_mat)
    total = np.sum(confusion_mat)
    accuracy = 100.0 * correct / total if total > 0 else 0
    print(f"Joint prospective binning accuracy = {accuracy:.2f}%")

    print("Simulation complete!")


if __name__ == "__main__":
    main()
