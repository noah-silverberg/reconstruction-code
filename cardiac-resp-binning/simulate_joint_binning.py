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

    NUM_CARD_BINS = 10  # e.g. 3 cardiac bins

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

    # ---------------------------------------------------------------------
    # Create subplot figures of the prospective fill bins, similar in style
    # to what simulate_resp_binning.py does, but for joint (resp x card).
    # ---------------------------------------------------------------------

    # shape => (num_resp_bins, NUM_CARD_BINS, KSPACE_H, KSPACE_W)
    # We'll find a global max so the color scale is consistent across bins:
    global_max = pros_fill_joint.max()
    if global_max < 1e-9:
        global_max = 1.0  # avoid divide-by-zero if array is all zeros

    for rbin in range(num_resp_bins):
        fig, axs = plt.subplots(
            1, NUM_CARD_BINS, figsize=(4 * NUM_CARD_BINS, 4), sharex=True, sharey=True
        )
        # If there's only 1 cardiac bin, axs might not be a list
        if NUM_CARD_BINS == 1:
            axs = [axs]

        for cbin in range(NUM_CARD_BINS):
            # 2D array of that (rbin, cbin)
            kspace_2d = pros_fill_joint[rbin, cbin]

            # We'll use a power norm so small but non-zero values are visible
            im = axs[cbin].imshow(
                kspace_2d,
                cmap="gray",
                norm=mcolors.PowerNorm(gamma=0.3, vmin=0, vmax=global_max),
                origin="upper",
                aspect="auto",
            )
            axs[cbin].set_title(f"Resp bin {rbin}, Card bin {cbin}", fontsize=10)
            axs[cbin].axis("off")
            # Add a colorbar specific to this subplot
            cbar = plt.colorbar(im, ax=axs[cbin], fraction=0.046, pad=0.04)
            cbar.set_label("Magnitude")

        fig.suptitle(f"Joint Prospective Fill: Respiratory Bin {rbin}", fontsize=14)
        fig.tight_layout()

        out_png = f"joint_prospective_rbin_{rbin}_subplots.png"
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved subplot figure for respiratory bin {rbin} => {out_png}")


if __name__ == "__main__":
    main()
