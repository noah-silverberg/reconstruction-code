"""
utils/kspace_filling.py

Provides functions for k-space line selection and filling based on respiratory
and cardiac binning. The functions compute line priorities, perform Gaussian-based
bin weighting, assign prospective bins, and run the filling loops.
"""

import numpy as np
from tqdm import tqdm


def build_line_priority(num_bins, kspace_height=128):
    """
    Build a priority array for each bin.

    Each row is assigned a base priority based on its distance from the center,
    with higher priority for central lines.

    Parameters
    ----------
    num_bins : int
        Number of bins.
    kspace_height : int
        Number of rows in k-space.

    Returns
    -------
    np.ndarray
        2D array of shape (num_bins, kspace_height) with priority values.
    """
    center_index = kspace_height // 2
    priority_array = np.zeros((num_bins, kspace_height), dtype=float)
    for bin_idx in range(num_bins):
        for row_idx in range(kspace_height):
            distance = abs(row_idx - center_index)
            # Exponential decay based on distance; adjust exponent as needed.
            priority_array[bin_idx, row_idx] = np.exp(
                -0.5 * (distance / (kspace_height / 4)) ** 5
            )
    return priority_array


def fill_line_in_bin(bin_index, row_index, fill_array, priority_array):
    """
    Mark that a row has been acquired in the specified bin and update its priority.

    Parameters
    ----------
    bin_index : int
        Index of the bin.
    row_index : int
        Index of the row.
    fill_array : np.ndarray
        Array tracking k-space fill (shape: [num_bins, kspace_height, kspace_width]).
    priority_array : np.ndarray
        Array of line priorities (shape: [num_bins, kspace_height]).
    """
    fill_array[bin_index, row_index, :] += 1.0
    priority_array[bin_index, row_index] *= 0.8  # Lower priority after acquisition


def get_bin_index_gaussian(
    fraction,
    is_inhale,
    num_inhale_bins,
    num_exhale_bins,
    use_total_bins=False,
    num_total_bins=4,
    sigma_factor=0.1,
):
    """
    Compute Gaussian weights for each bin given a respiratory fraction.

    The fraction (0-100) is remapped depending on the phase, and then the distance
    from each bin's boundaries is used to compute a Gaussian weight.

    Parameters
    ----------
    fraction : float
        Respiratory fraction (0 to 100).
    is_inhale : bool
        True if inhalation phase.
    num_inhale_bins : int
        Number of bins for inhalation.
    num_exhale_bins : int
        Number of bins for exhalation.
    use_total_bins : bool, optional
        If True, treat the cycle as one contiguous block.
    num_total_bins : int, optional
        Total number of bins if use_total_bins is True.
    sigma_factor : float, optional
        Factor to scale the standard deviation for the Gaussian.

    Returns
    -------
    dict
        Dictionary mapping bin index to normalized weight.
    """
    total_bins = (
        num_total_bins if use_total_bins else (num_inhale_bins + num_exhale_bins)
    )
    # Remap fraction based on phase.
    effective_fraction = fraction * 0.5 if is_inhale else 50 + fraction * 0.5
    bin_width = 100.0 / total_bins
    sigma = bin_width * sigma_factor
    weights = {}
    for bin_idx in range(total_bins):
        bin_start = bin_idx * bin_width
        bin_end = (bin_idx + 1) * bin_width
        if bin_start <= effective_fraction < bin_end:
            d_min = 0.0
        else:
            diff_start = abs(effective_fraction - bin_start)
            diff_end = abs(effective_fraction - bin_end)
            diff_start = min(diff_start, 100 - diff_start)
            diff_end = min(diff_end, 100 - diff_end)
            d_min = min(diff_start, diff_end)
        weights[bin_idx] = np.exp(-0.5 * (d_min / sigma) ** 2)
    total_weight = sum(weights.values())
    if total_weight < 1e-8:
        return {i: 1.0 / total_bins for i in range(total_bins)}
    return {i: w / total_weight for i, w in weights.items()}


def shift_inhale_exhale_bins(bin_dict, is_inhale, num_inhale_bins):
    """
    Shift bin indices for exhalation by offsetting by num_inhale_bins.

    Parameters
    ----------
    bin_dict : dict
        Original bin indices mapping.
    is_inhale : bool
        If False, shift indices.
    num_inhale_bins : int
        Number of inhale bins.

    Returns
    -------
    dict
        Shifted dictionary.
    """
    if is_inhale:
        return bin_dict
    return {old_key + num_inhale_bins: weight for old_key, weight in bin_dict.items()}


def assign_prospective_bin(
    fraction,
    is_inhale,
    num_inhale_bins,
    num_exhale_bins,
    use_total_bins=False,
    num_total_bins=4,
):
    """
    Assign a single bin index based on the respiratory fraction.

    Parameters
    ----------
    fraction : float
        Respiratory fraction (0-100).
    is_inhale : bool
        True for inhalation.
    num_inhale_bins : int
        Number of inhalation bins.
    num_exhale_bins : int
        Number of exhalation bins.
    use_total_bins : bool
        If True, use a contiguous set of bins.
    num_total_bins : int
        Total bins if using total bins.

    Returns
    -------
    int
        Assigned bin index.
    """
    if use_total_bins:
        bin_width = 100.0 / num_total_bins
        bin_i = int(np.floor(fraction / bin_width))
        return min(bin_i, num_total_bins - 1)
    else:
        if is_inhale:
            bin_width = 100.0 / num_inhale_bins
            bin_i = int(np.floor(fraction / bin_width))
            return min(bin_i, num_inhale_bins - 1)
        else:
            bin_width = 100.0 / num_exhale_bins
            bin_i = int(np.floor(fraction / bin_width))
            return num_inhale_bins + min(bin_i, num_exhale_bins - 1)


def prospective_fill_loop(
    N,
    predicted_fraction,
    predicted_phase,
    pros_fill,
    pros_priority,
    get_bin_index_fn,
    assign_bin_fn,
):
    """
    Run the prospective filling loop for each time sample.

    For each sample, compute bin weights, select the row with the highest
    score, and update the k-space fill array.

    Parameters
    ----------
    N : int
        Number of time steps.
    predicted_fraction : np.ndarray
        Array of predicted fractions.
    predicted_phase : np.ndarray
        Array of predicted phase booleans.
    pros_fill : np.ndarray
        K-space fill array.
    pros_priority : np.ndarray
        Array of line priorities.
    get_bin_index_fn : callable
        Function to compute bin weights.
    assign_bin_fn : callable
        Function to compute the official bin assignment.

    Returns
    -------
    list
        List of acquired lines (or None).
    """
    acquired_lines = [None] * N
    total_bins = pros_fill.shape[0]
    for k in range(N):
        frac = predicted_fraction[k]
        ph = predicted_phase[k]
        if np.isnan(frac) or ph is None:
            continue
        bin_weight_dict = get_bin_index_fn(frac, ph)
        top_candidates = {}
        for b_idx, weight in bin_weight_dict.items():
            if weight < 1e-6:
                continue
            row_idx = np.argmax(pros_priority[b_idx])
            row_priority_value = pros_priority[b_idx, row_idx]
            if row_priority_value < 0:
                continue
            top_candidates[b_idx] = (row_idx, weight * row_priority_value)
        if not top_candidates:
            continue
        best_bin = max(top_candidates.keys(), key=lambda x: top_candidates[x][1])
        best_row = top_candidates[best_bin][0]
        pbin = assign_bin_fn(frac, ph)
        acquired_lines[k] = (best_row, pbin)
        fill_line_in_bin(pbin, best_row, pros_fill, pros_priority)
    return acquired_lines


def build_line_priority_joint(num_resp_bins, num_card_bins, kspace_height=128):
    """
    Build a joint priority array for respiratory and cardiac bins.

    Parameters
    ----------
    num_resp_bins : int
        Number of respiratory bins.
    num_card_bins : int
        Number of cardiac bins.
    kspace_height : int
        Number of k-space rows.

    Returns
    -------
    np.ndarray
        3D array of shape (num_resp_bins, num_card_bins, kspace_height).
    """
    center_index = kspace_height // 2
    priority_array = np.zeros(
        (num_resp_bins, num_card_bins, kspace_height), dtype=float
    )
    for rbin in range(num_resp_bins):
        for cbin in range(num_card_bins):
            for row_idx in range(kspace_height):
                distance = abs(row_idx - center_index)
                priority_array[rbin, cbin, row_idx] = np.exp(
                    -0.5 * (distance / (kspace_height / 4)) ** 5
                )
    return priority_array


def get_joint_bin_weights_gaussian(
    resp_fraction,
    resp_is_inhale,
    cardiac_fraction,
    num_inhale_bins,
    num_exhale_bins,
    use_total_resp_bins,
    num_total_resp_bins,
    num_card_bins,
    resp_sigma_factor=0.1,
    card_sigma_factor=0.1,
):
    """
    Compute joint weights for respiratory and cardiac bins.

    The joint weight is the product of the respiration weight and a cardiac weight
    computed similarly but without phase consideration.

    Parameters
    ----------
    resp_fraction : float
        Respiratory fraction.
    resp_is_inhale : bool
        True for inhalation.
    cardiac_fraction : float
        Cardiac fraction (0-100).
    num_inhale_bins : int
        Number of inhalation bins.
    num_exhale_bins : int
        Number of exhalation bins.
    use_total_resp_bins : bool
        If True, use total bins.
    num_total_resp_bins : int
        Total respiratory bins if using total.
    num_card_bins : int
        Number of cardiac bins.
    resp_sigma_factor : float, optional
        Sigma factor for respiration.
    card_sigma_factor : float, optional
        Sigma factor for cardiac.

    Returns
    -------
    dict
        Dictionary with keys (resp_bin, card_bin) and joint weights.
    """
    # Respiration weights using the existing function
    resp_weights = get_bin_index_gaussian(
        fraction=resp_fraction,
        is_inhale=resp_is_inhale,
        num_inhale_bins=num_inhale_bins,
        num_exhale_bins=num_exhale_bins,
        use_total_bins=use_total_resp_bins,
        num_total_bins=num_total_resp_bins,
        sigma_factor=resp_sigma_factor,
    )
    # Cardiac weights: treat entire cycle as contiguous range
    card_weights = {}
    if num_card_bins <= 1:
        card_weights[0] = 1.0
    else:
        bin_width = 100.0 / num_card_bins
        sigma = bin_width * card_sigma_factor
        raw_weights = {}
        for cbin in range(num_card_bins):
            c_start = cbin * bin_width
            c_end = (cbin + 1) * bin_width
            if c_start <= cardiac_fraction < c_end:
                d_min = 0.0
            else:
                diff_start = abs(cardiac_fraction - c_start)
                diff_end = abs(cardiac_fraction - c_end)
                diff_start = min(diff_start, 100 - diff_start)
                diff_end = min(diff_end, 100 - diff_end)
                d_min = min(diff_start, diff_end)
            raw_weights[cbin] = np.exp(-0.5 * (d_min / sigma) ** 2)
        s = sum(raw_weights.values())
        if s < 1e-8:
            card_weights = {cbin: 1.0 / num_card_bins for cbin in raw_weights}
        else:
            card_weights = {cbin: w / s for cbin, w in raw_weights.items()}

    # Combine the two sets of weights
    joint_weights = {}
    for rbin, w_r in resp_weights.items():
        for cbin, w_c in card_weights.items():
            joint_weights[(rbin, cbin)] = w_r * w_c
    return joint_weights


def assign_prospective_bin_joint(
    resp_fraction,
    resp_is_inhale,
    cardiac_fraction,
    num_inhale_bins,
    num_exhale_bins,
    use_total_resp_bins,
    num_total_resp_bins,
    num_card_bins,
):
    """
    Compute the official joint bin assignment.

    For respiration, call assign_prospective_bin; for cardiac, use a simple floor division.

    Returns
    -------
    tuple
        (resp_bin, card_bin)
    """
    from .kspace_filling import assign_prospective_bin

    rbin = assign_prospective_bin(
        fraction=resp_fraction,
        is_inhale=resp_is_inhale,
        num_inhale_bins=num_inhale_bins,
        num_exhale_bins=num_exhale_bins,
        use_total_bins=use_total_resp_bins,
        num_total_bins=num_total_resp_bins,
    )
    if num_card_bins <= 1:
        cbin = 0
    else:
        bin_width = 100.0 / num_card_bins
        cbin = int(np.floor(cardiac_fraction / bin_width))
        cbin = min(cbin, num_card_bins - 1)
    return (rbin, cbin)


def fill_line_in_joint_bin(rbin, cbin, row_idx, fill_array, priority_array):
    """
    Mark that a k-space line (row_idx) has been acquired for the joint bin (rbin, cbin).

    Parameters
    ----------
    rbin : int
        Respiratory bin index.
    cbin : int
        Cardiac bin index.
    row_idx : int
        Row index in k-space.
    fill_array : np.ndarray
        Joint k-space fill array.
    priority_array : np.ndarray
        Joint line priority array.
    """
    fill_array[rbin, cbin, row_idx, :] += 1.0
    priority_array[rbin, cbin, row_idx] *= 0.8


def prospective_fill_loop_joint(
    N,
    resp_fraction_array,
    resp_phase_array,
    cardiac_fraction_array,
    pros_fill,
    pros_priority,
    get_joint_weights_fn,
    assign_bin_joint_fn,
):
    """
    Run the prospective filling loop for joint (respiratory and cardiac) bins.

    Parameters
    ----------
    N : int
        Number of time steps.
    resp_fraction_array : np.ndarray
        Array of respiratory fractions.
    resp_phase_array : np.ndarray
        Array of respiratory phases (bool).
    cardiac_fraction_array : np.ndarray
        Array of cardiac fractions.
    pros_fill : np.ndarray
        Joint k-space fill array (shape: [num_resp_bins, num_card_bins, kspace_height, kspace_width]).
    pros_priority : np.ndarray
        Joint line priority array (shape: [num_resp_bins, num_card_bins, kspace_height]).
    get_joint_weights_fn : callable
        Function to compute joint bin weights.
    assign_bin_joint_fn : callable
        Function to assign an official joint bin.

    Returns
    -------
    list
        List (length N) of acquired lines as (row_idx, resp_bin, card_bin) or None.
    """
    acquired_lines = [None] * N
    num_resp_bins, num_card_bins, kspace_height, _ = pros_fill.shape
    for k in tqdm(range(N), desc="Prospective Joint Filling"):
        resp_frac = resp_fraction_array[k]
        resp_ph = resp_phase_array[k]
        card_frac = cardiac_fraction_array[k]
        if np.isnan(resp_frac) or resp_ph is None or np.isnan(card_frac):
            continue
        joint_weights = get_joint_weights_fn(resp_frac, resp_ph, card_frac)
        row_scores = np.zeros(kspace_height, dtype=float)
        for (rbin, cbin), w in joint_weights.items():
            if w < 1e-8:
                continue
            row_scores += w * pros_priority[rbin, cbin, :]
        best_row = np.argmax(row_scores)
        best_score = row_scores[best_row]
        if best_score < 1e-6:
            continue
        official_bins = assign_bin_joint_fn(resp_frac, resp_ph, card_frac)
        fill_line_in_joint_bin(
            official_bins[0], official_bins[1], best_row, pros_fill, pros_priority
        )
        acquired_lines[k] = (best_row, official_bins[0], official_bins[1])
    return acquired_lines
