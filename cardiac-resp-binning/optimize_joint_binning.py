# optimize_joint_binning.py

import numpy as np
import csv
import os
import itertools
from tqdm import tqdm
import time
from multiprocessing import Pool, cpu_count

import functools
from utils.kspace_filling import (
    build_line_priority_joint,
    get_joint_bin_weights_gaussian,
    prospective_fill_loop_joint,
    assign_prospective_bin_joint,
)


def check_per_bin_acs(retro_fill, center_size=20, desired_coverage=3.0):
    """
    For each respiratory bin (rbin), we first sum across the cardiac bins and
    check that the center ACS block has no zeros. If any line is missing, we
    return infinity.

    Then, for the coverage cost, we now compute the deviation from `desired_coverage`
    *within each cardiac bin separately*, weighting each row by `row_weight`.
    Finally, we sum those bin-wise costs across all cardiac bins and respiratory bins.

    Returns
    -------
    float
        The final coverage cost, or np.inf if ACS coverage fails.
    """

    # retro_fill => (num_resp_bins, num_card_bins, H, W)
    num_rbins, num_cbins, H, W = retro_fill.shape
    total_cost = 0.0

    # Build a row weighting mask to emphasize coverage near the center row more heavily
    center_row = H // 2
    row_indices = np.arange(H)
    dist = np.abs(row_indices - center_row)
    scale = H / 2.0
    row_weight = np.exp(-0.5 * (dist / scale) ** 2)  # shape (H,)

    # ACS region parameters
    half = center_size // 2
    cx = W // 2
    cy = H // 2

    for rbin in range(num_rbins):
        # 1) ACS check: sum over all cardiac bins
        coverage_2d = np.sum(np.abs(retro_fill[rbin]), axis=0)  # shape (H, W)
        block = coverage_2d[cy - half : cy + half, cx - half : cx + half]
        if np.any(block <= 0):
            return np.inf  # Missing lines => cost=∞

        # 2) Coverage cost: now do it for each (rbin, cbin) individually
        rbin_cost = 0.0
        for cbin in range(num_cbins):
            coverage_bin = retro_fill[rbin, cbin]  # shape (H, W)

            row_costs = []
            for row_i in range(H):
                row_values = coverage_bin[row_i, :]  # shape (W,)
                diff = row_values - desired_coverage
                mse = np.mean(diff**2)  # Mean-squared error for this row
                row_costs.append(row_weight[row_i] * mse)

            # Average cost across rows for this (rbin, cbin)
            bin_cost = np.mean(row_costs)
            rbin_cost += bin_cost

        # Accumulate across all cbin for this respiratory bin
        total_cost += rbin_cost / num_cbins

    return total_cost


def compute_cost_on_retro_fill(retro_fill, center_size=20, desired=3.0):
    """
    Master cost function that calls `check_per_bin_acs(retro_fill, ...)`.
    If infinite, we return inf. Otherwise returns coverage-based cost.
    """
    cost_val = check_per_bin_acs(
        retro_fill, center_size=center_size, desired_coverage=desired
    )
    return cost_val


def run_joint_binning_once(
    data,
    resp_sigma_factor,
    card_sigma_factor,
    priority_exponent,
    penalty_factor,
):
    """
    1) Build prospective fill with the chosen hyperparameters
    2) Acquire lines => acquired_lines
    3) Use perform_retrospective_assignment to get RETRO fill
    4) Return the retro fill
    """
    from simulate_joint_binning import (
        process_data,
        perform_prospective_binning,
        perform_retrospective_assignment,
    )

    # 1) Unpack the precomputed respiratory/cardiac predictions
    predicted_resp_fraction = data["pred_resp_frac"]
    predicted_resp_phase = data["pred_resp_phase"]
    predicted_card_fraction = data["pred_card_frac"]
    N_k = len(predicted_resp_fraction)

    # 2) Create the prospective fill arrays
    num_resp_bins = 4  # e.g. 3 inhale + 1 exhale
    num_card_bins = 20
    KSPACE_H, KSPACE_W = 128, 128
    pros_fill = np.zeros(
        (num_resp_bins, num_card_bins, KSPACE_H, KSPACE_W), dtype=float
    )

    # Priority array
    pros_priority = build_line_priority_joint(
        num_resp_bins=num_resp_bins,
        num_card_bins=num_card_bins,
        kspace_height=KSPACE_H,
        priority_exponent=priority_exponent,
    )

    # Weighted function
    get_joint_weights_fn = functools.partial(
        get_joint_bin_weights_gaussian,
        num_inhale_bins=3,
        num_exhale_bins=1,
        use_total_resp_bins=False,
        num_total_resp_bins=4,
        num_card_bins=num_card_bins,
        resp_sigma_factor=resp_sigma_factor,
        card_sigma_factor=card_sigma_factor,
    )

    def assign_bin_joint_fn(rfrac, rphase, cfrac):
        return assign_prospective_bin_joint(
            resp_fraction=rfrac,
            resp_is_inhale=rphase,
            cardiac_fraction=cfrac,
            num_inhale_bins=3,
            num_exhale_bins=1,
            use_total_resp_bins=False,
            num_total_resp_bins=4,
            num_card_bins=num_card_bins,
        )

    # 3) prospective fill loop => acquired_lines
    acquired_lines = prospective_fill_loop_joint(
        N=N_k,
        resp_fraction_array=predicted_resp_fraction,
        resp_phase_array=predicted_resp_phase,
        cardiac_fraction_array=predicted_card_fraction,
        pros_fill=pros_fill,
        pros_priority=pros_priority,
        get_joint_weights_fn=get_joint_weights_fn,
        assign_bin_joint_fn=assign_bin_joint_fn,
        penalty_factor=penalty_factor,
    )

    # We only need the retro_fill_joint from that function:
    data_for_retro = {
        "fs": data["fs"],
        "N_k": data["N_k"],
        "resp_signal_resampled": data["resp_signal_resampled"],
        "ecg_signal_resampled": data["ecg_signal_resampled"],
    }
    retro_fill_joint, diff_joint, cycles, resp_frac_offline, card_frac_offline = (
        perform_retrospective_assignment(data_for_retro, {}, pros_fill, acquired_lines)
    )

    return retro_fill_joint


def run_one_combo(args):
    r_sig, c_sig, expn, pen, data_dict = args
    retro_fill = run_joint_binning_once(
        data=data_dict,
        resp_sigma_factor=r_sig,
        card_sigma_factor=c_sig,
        priority_exponent=expn,
        penalty_factor=pen,
    )
    cost_val = compute_cost_on_retro_fill(retro_fill, center_size=20, desired=3.0)
    return (r_sig, c_sig, expn, pen, cost_val)


if __name__ == "__main__":
    from simulate_joint_binning import (
        load_config_file,
        process_data,
        perform_prospective_binning,
    )

    # 1) Load config
    config = load_config_file("config.yaml")
    data_dict = process_data(config)

    # 2) Precompute predicted respiratory/cardiac fractions *once*:
    _pros_fill, _acquired, pred_resp_frac, pred_resp_phase, pred_card_frac = (
        perform_prospective_binning(data_dict, config)
    )
    data_dict["pred_resp_frac"] = pred_resp_frac
    data_dict["pred_resp_phase"] = pred_resp_phase
    data_dict["pred_card_frac"] = pred_card_frac

    # Keep the raw signals in data_dict for retrospective assignment
    data_dict["resp_signal_resampled"] = data_dict["resp_signal_resampled"]
    data_dict["ecg_signal_resampled"] = data_dict["ecg_signal_resampled"]

    # 3) define hyper-parameter ranges
    import numpy as np

    resp_sigma_list = np.linspace(0.25, 0.55, 10)
    card_sigma_list = np.linspace(0.40, 0.75, 10)
    exponent_list = np.linspace(2.6, 2.6, 1)
    penalty_list = np.linspace(0.24, 0.3, 10)

    out_csv = "gridsearch_results.csv"
    results = []

    # 4) grid search
    total_combos = (
        len(resp_sigma_list)
        * len(card_sigma_list)
        * len(exponent_list)
        * len(penalty_list)
    )

    combos = []
    for r_sig in resp_sigma_list:
        for c_sig in card_sigma_list:
            for expn in exponent_list:
                for pen in penalty_list:
                    combos.append((r_sig, c_sig, expn, pen, data_dict))

    results = []

    time_start = time.time()
    with Pool(processes=cpu_count()) as p:
        # Use map to parallelize the function calls
        results = p.map(run_one_combo, combos)
    time_end = time.time()
    # Print combos/s
    print(
        f"Completed {len(results)} combinations in {time_end - time_start:.2f} seconds."
    )
    print(f"That is {len(results)/(time_end - time_start):.2f} combos per second.")

    # 5) write CSV
    import csv

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["resp_sigma", "card_sigma", "priority_exp", "penalty_factor", "cost"]
        )
        for row in results:
            writer.writerow(row)

    print("DONE. Wrote results to", out_csv)
