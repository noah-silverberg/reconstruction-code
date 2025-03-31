# utils/kspace_filling.py

import numpy as np
from tqdm import tqdm


def build_line_priority(num_bins, kspace_height=128):
    """
    Build an initial line-priority array for each bin.

    Each row in k-space is assigned a “base priority” to guide prospective filling.
    Typically, rows closer to the center have higher priority. You can customize
    how this is calculated.

    Parameters
    ----------
    num_bins : int
        Number of respiratory bins (or combined bins) you have decided to use.
    kspace_height : int
        Number of rows in k-space (e.g., 128).

    Returns
    -------
    np.ndarray
        A 2D array of shape (num_bins, kspace_height). Each row is the "priority"
        for each line index within that bin. Higher means more important.
    """
    center_index = kspace_height // 2
    priority_array = np.zeros((num_bins, kspace_height), dtype=float)

    for bin_idx in range(num_bins):
        for row_idx in range(kspace_height):
            distance_from_center = abs(row_idx - center_index)
            # Example formula: the closer to center, the higher the priority.
            base_priority = np.exp(
                -0.5 * (distance_from_center / (kspace_height / 4)) ** 5
            )
            # TODO: something we need to think about here is the fact that when we highly weight a vote from a neighboring bin
            #       we might overfill a line
            #       for example if we have a line acquired in bin 0 but it is really highly weighted by bin 1, we will keep acquiring this row repeatedly, since the priority won't even be updated in the neiboring bin
            #       we might want to add a small penalty to the priority of the line in the neighboring bin
            priority_array[bin_idx, row_idx] = base_priority

    return priority_array


def fill_line_in_bin(bin_index, row_index, fill_array, priority_array):
    """
    Mark that a certain row in k-space has been “acquired” for the given bin,
    and update line priority accordingly.

    Specifically, we:
      1) Increment fill_array[bin_index, row_index, :] to indicate that line was filled.
      2) Set priority_array[bin_index, row_index] = -999999, so it won't get chosen again.
      3) Optionally add a small bonus to the conjugate row if that row isn’t yet filled.

    Parameters
    ----------
    bin_index : int
        Which bin we are filling (0-based).
    row_index : int
        Which line index in the k-space dimension.
    fill_array : np.ndarray
        Shape = (num_bins, kspace_height, kspace_width). Tracks how many times each line is filled.
    priority_array : np.ndarray
        Shape = (num_bins, kspace_height). Priority values for each bin and row.
    """
    fill_array[bin_index, row_index, :] += 1.0  # Indicate line is filled
    priority_array[bin_index, row_index] *= 0.8  # Reduce priority for this line

    # TODO: add some sort of prioritization based on the GRAPPA/tGRAPPA lines needed
    # so something that maybe prioritizes lines with no nearby acquired lines
    # this requires some more thought, or maybe numerical simulation to determine sensitivities
    #     of the final image to the rows acquired
    # this might be quite difficult though given that GRAPPA/tGRAPPA (I think) aren't linear like FFT


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
    Compute voting weights for each bin based on the minimum distance between a given
    respiratory fraction and each bin's boundaries, with wrap-around and phase adjustment.

    The predicted fraction is first remapped depending on the phase:
      - If is_inhale is True, effective_fraction = fraction * 0.5    (range 0-50)
      - Otherwise, effective_fraction = 50 + fraction * 0.5             (range 50-100)
    This way, a 99% inhalation value (i.e. near the end of inhalation) is close to 50,
    while a 99% exhalation value remains high.

    Then, the respiratory cycle (0–100) is divided evenly into bins. For each bin,
    we compute the minimal (cyclic) distance from the effective_fraction to the bin’s boundaries.
    A Gaussian weight is computed as:

        weight = exp(-0.5 * (d_min / sigma)^2)

    where sigma = bin_width * sigma_factor. Finally, the weights are normalized to sum to 1.

    Parameters
    ----------
    fraction : float
        The predicted respiratory fraction (0..100) for the time sample.
    is_inhale : bool
        True if the current phase is inhalation, False if exhalation.
    num_inhale_bins : int
        Number of bins designated for inhalation (if not using total bins).
    num_exhale_bins : int
        Number of bins designated for exhalation (if not using total bins).
    use_total_bins : bool, optional
        If True, treat the entire cycle as one contiguous block of bins.
    num_total_bins : int, optional
        Total number of bins if use_total_bins is True.
    sigma_factor : float, optional
        Factor (multiplied by bin width) to set sigma in the Gaussian weighting. Default is 0.3.

    Returns
    -------
    dict
        A dictionary mapping each bin index to its normalized weight.
    """
    # Determine total number of bins
    if use_total_bins:
        total_bins = num_total_bins
    else:
        total_bins = num_inhale_bins + num_exhale_bins

    # Remap fraction based on phase:
    # - For inhalation, scale fraction into [0, 50]
    # - For exhalation, scale fraction into [50, 100]
    if is_inhale:
        effective_fraction = fraction * 0.5
    else:
        effective_fraction = 50 + fraction * 0.5

    bin_width = 100.0 / total_bins
    sigma = bin_width * sigma_factor  # adjustable spread

    weights = {}

    # Loop over every bin and compute the minimal cyclic distance to its boundaries.
    for bin_idx in range(total_bins):
        bin_start = bin_idx * bin_width
        bin_end = (bin_idx + 1) * bin_width

        # Because the cycle is circular, the distance d_min is the minimum distance
        # from effective_fraction to either boundary, considering wrap-around.
        if bin_start <= effective_fraction < bin_end:
            d_min = 0.0  # fraction lies inside the bin
        else:
            # Compute distance to bin_start and bin_end, with wrap-around adjustment:
            diff_start = abs(effective_fraction - bin_start)
            diff_end = abs(effective_fraction - bin_end)
            # Wrap-around: e.g., distance between 99 and 2 is min( |99-2|, 100-|99-2| )
            diff_start = min(diff_start, 100 - diff_start)
            diff_end = min(diff_end, 100 - diff_end)
            d_min = min(diff_start, diff_end)

        # Compute the Gaussian weight for this bin:
        weight = np.exp(-0.5 * (d_min / sigma) ** 2)
        weights[bin_idx] = weight

    # Normalize the weights so that they sum to 1.
    total_weight = sum(weights.values())
    if total_weight < 1e-8:
        normalized_weights = {i: 1.0 / total_bins for i in range(total_bins)}
    else:
        normalized_weights = {i: w / total_weight for i, w in weights.items()}

    return normalized_weights


def shift_inhale_exhale_bins(bin_dict, is_inhale, num_inhale_bins):
    """
    Deprecated helper. You can do the offset logic inline in get_bin_index_gaussian.
    Typically you won't call this directly.

    This function shifts the bin indices for exhalation by +num_inhale_bins, so that
    if inhalation bins range from 0..(num_inhale_bins-1), exhalation bins become
    num_inhale_bins..(num_inhale_bins + num_exhale_bins - 1).

    Parameters
    ----------
    bin_dict : dict
        Dictionary of bin_idx->weight before offset.
    is_inhale : bool
        If True => do not offset. If False => offset the keys.
    num_inhale_bins : int
        Number of inhale bins (needed for offset).

    Returns
    -------
    dict
        Possibly shifted dictionary of {bin_index: weight}.
    """
    if is_inhale:
        return bin_dict
    else:
        # For exhalation bins, shift everything by +num_inhale_bins
        offset_dict = {}
        for old_key, weight in bin_dict.items():
            new_key = old_key + num_inhale_bins
            offset_dict[new_key] = weight
        return offset_dict


def assign_prospective_bin(
    fraction,
    is_inhale,
    num_inhale_bins,
    num_exhale_bins,
    use_total_bins=False,
    num_total_bins=4,
):
    """
    Hard assignment of a fraction to a single bin, ignoring Gaussian weighting.

    This is used to decide “which bin is the ‘official’ prospective bin?” even though
    we do a multi-bin voting for line selection.

    Parameters
    ----------
    fraction : float
        The predicted fraction (0..100).
    is_inhale : bool
        True if we are in inhalation, False if exhalation.
    num_inhale_bins : int
        Number of inhalation bins.
    num_exhale_bins : int
        Number of exhalation bins.
    use_total_bins : bool
        If True => just treat it as one contiguous set of bins.
    num_total_bins : int
        The total number of bins, if use_total_bins is True.

    Returns
    -------
    int
        The single bin index (0-based).
    """
    if use_total_bins:
        bin_count = num_total_bins
        bin_width = 100.0 / bin_count
        bin_i = int(np.floor(fraction / bin_width))
        if bin_i >= bin_count:
            bin_i = bin_count - 1
        return bin_i
    else:
        if is_inhale:
            # Inhale bin
            bin_width = 100.0 / num_inhale_bins
            bin_i = int(np.floor(fraction / bin_width))
            if bin_i >= num_inhale_bins:
                bin_i = num_inhale_bins - 1
            return bin_i
        else:
            # Exhale bin
            bin_width = 100.0 / num_exhale_bins
            bin_i = int(np.floor(fraction / bin_width))
            if bin_i >= num_exhale_bins:
                bin_i = num_exhale_bins - 1
            return num_inhale_bins + bin_i


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
    Main prospective-filling loop:
      1) For each time step k, we have a predicted fraction & phase.
      2) We call get_bin_index_fn(fraction, phase) to get {bin_idx->weight}.
      3) For each bin with a positive weight, we find the highest-priority row.
      4) We pick the bin with the best “score = weight * row_priority.”
      5) We fill that line in the “official prospective bin” determined by assign_bin_fn(...)
         (which does the “hard assignment”), and then update the priority arrays.

    This function returns a list acquired_lines, where acquired_lines[k] = (row, p_bin),
    or None if no line was filled at that time step.

    Parameters
    ----------
    N : int
        Number of time steps total (length of predicted_fraction, etc.).
    predicted_fraction : np.ndarray
        The array of fractions (0..100) for each time step (length N).
    predicted_phase : np.ndarray (dtype=object, typically bool or None)
        True for inhalation, False for exhalation, or None if unknown at that step.
    pros_fill : np.ndarray
        K-space fill array shape (num_bins, kspace_height, kspace_width).
        We increment lines in this array each time we “acquire” one.
    pros_priority : np.ndarray
        The line priority array shape (num_bins, kspace_height), modified in place
        as lines get used. Typically you initialize it with build_line_priority(...).
    get_bin_index_fn : callable
        A function that returns {bin_idx->weight} for a given fraction & phase.
        e.g. partial(get_bin_index_gaussian, ...).
    assign_bin_fn : callable
        The function that picks the single prospective bin from fraction & phase,
        e.g. partial(assign_prospective_bin, ...).

    Returns
    -------
    list
        A list of length N. Each entry is either None (if no line was filled)
        or a tuple (row_index, prospective_bin).
    """
    acquired_lines = [None] * N
    total_bins = pros_fill.shape[0]

    for k in range(N):
        frac = predicted_fraction[k]
        ph = predicted_phase[k]
        if np.isnan(frac) or ph is None:
            continue

        # 1) get bin=>weight
        bin_weight_dict = get_bin_index_fn(frac, ph)

        # 2) find top candidate row & priority in each bin that has a positive weight
        top_candidates = {}
        for b_idx, w_ in bin_weight_dict.items():
            if w_ < 1e-6:
                continue
            # row_argmax => best row for bin b_idx
            row_argmax = np.argmax(pros_priority[b_idx])
            row_val = pros_priority[b_idx, row_argmax]
            if row_val < 0:
                # means no lines left or everything filled
                continue
            score = w_ * row_val
            top_candidates[b_idx] = (row_argmax, score)

        if not top_candidates:
            continue

        # 3) pick the bin with highest score
        best_bin = max(top_candidates.keys(), key=lambda x: top_candidates[x][1])
        best_row = top_candidates[best_bin][0]

        # 4) figure out the “official prospective bin” => single bin from fraction
        pbin = assign_bin_fn(frac, ph)
        acquired_lines[k] = (best_row, pbin)

        # 5) fill that line
        fill_line_in_bin(pbin, best_row, pros_fill, pros_priority)

    return acquired_lines


def build_line_priority_joint(num_resp_bins, num_card_bins, kspace_height=128):
    """
    Build a priority array of shape (num_resp_bins, num_card_bins, kspace_height).
    We do the same center-priority logic you used for respiration alone,
    but repeated for each (resp_bin, card_bin).
    """
    center_index = kspace_height // 2
    priority_array = np.zeros(
        (num_resp_bins, num_card_bins, kspace_height), dtype=float
    )

    for rbin in range(num_resp_bins):
        for cbin in range(num_card_bins):
            for row_idx in range(kspace_height):
                distance_from_center = abs(row_idx - center_index)
                base_priority = np.exp(
                    -0.5 * (distance_from_center / (kspace_height / 4)) ** 5
                )
                priority_array[rbin, cbin, row_idx] = base_priority

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
    Return a dict keyed by (resp_bin, card_bin) => combined_weight.

    Approach:
      - First get the 1D weights for respiration: w_r(resp_bin)
      - Then get the 1D weights for cardiac: w_c(card_bin)
      - Then the joint weight for (rbin, cbin) is w_r(rbin) * w_c(cbin)

    You can use your existing get_bin_index_gaussian for respiration, plus
    a new get_bin_index_gaussian for the cardiac fraction, or do it inline.

    For clarity, let's just reuse your 'get_bin_index_gaussian' for both.
    We'll define a small function get_1d_weights(...) for each cycle.
    """
    # Reuse your existing single-cycle weighting for respiration
    from .kspace_filling import get_bin_index_gaussian

    # Get the respiration bin->weight
    resp_bin_weights = get_bin_index_gaussian(
        fraction=resp_fraction,
        is_inhale=resp_is_inhale,
        num_inhale_bins=num_inhale_bins,
        num_exhale_bins=num_exhale_bins,
        use_total_bins=use_total_resp_bins,
        num_total_bins=num_total_resp_bins,
        sigma_factor=resp_sigma_factor,
    )  # returns {resp_bin => weight}

    # For cardiac, we do a similar approach but simpler: is_inhale doesn't apply, so pass is_inhale=False
    # Or you can define a new function that doesn't consider inhale/exhale. We'll just do the same method but we
    # treat the entire 0..100 as a single contiguous range with 'num_card_bins'.
    # Something like:
    card_bin_weights = {}
    if num_card_bins <= 1:
        # trivial
        card_bin_weights[0] = 1.0
    else:
        # We'll do a bin_width of 100 / num_card_bins, then do a Gaussian weighting similarly
        # This is basically the same code as get_bin_index_gaussian, but ignoring inhale/exhale.
        bin_width = 100.0 / num_card_bins
        sigma = bin_width * card_sigma_factor
        raw_weights = {}
        for cbin in range(num_card_bins):
            c_start = cbin * bin_width
            c_end = (cbin + 1) * bin_width
            # distance from fraction to bin boundaries
            if c_start <= cardiac_fraction < c_end:
                d_min = 0.0
            else:
                diff_start = abs(cardiac_fraction - c_start)
                diff_end = abs(cardiac_fraction - c_end)
                diff_start = min(diff_start, 100 - diff_start)
                diff_end = min(diff_end, 100 - diff_end)
                d_min = min(diff_start, diff_end)
            w_ = np.exp(-0.5 * (d_min / sigma) ** 2)
            raw_weights[cbin] = w_
        s_ = sum(raw_weights.values())
        if s_ < 1e-9:
            for cbin in raw_weights:
                card_bin_weights[cbin] = 1.0 / num_card_bins
        else:
            for cbin in raw_weights:
                card_bin_weights[cbin] = raw_weights[cbin] / s_

    # Now combine
    joint_weights = {}
    for rbin, w_r in resp_bin_weights.items():
        for cbin, w_c in card_bin_weights.items():
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
    Return a single (resp_bin, card_bin) for prospective assignment.

    We do a normal assign for respiration, then a normal assign for cardiac.
    """
    from .kspace_filling import assign_prospective_bin

    # For respiration:
    rbin = assign_prospective_bin(
        fraction=resp_fraction,
        is_inhale=resp_is_inhale,
        num_inhale_bins=num_inhale_bins,
        num_exhale_bins=num_exhale_bins,
        use_total_bins=use_total_resp_bins,
        num_total_bins=num_total_resp_bins,
    )

    # For cardiac, do an equivalent approach but ignoring inhale/exhale
    # We can define a small helper here:
    if num_card_bins <= 1:
        cbin = 0
    else:
        bin_width = 100.0 / num_card_bins
        cbin = int(np.floor(cardiac_fraction / bin_width))
        if cbin >= num_card_bins:
            cbin = num_card_bins - 1

    return (rbin, cbin)


def fill_line_in_joint_bin(rbin, cbin, row_idx, fill_array, priority_array):
    """
    Mark that we have acquired line `row_idx` in bin (rbin, cbin).
    Then reduce its priority so we don't keep acquiring it.
    """
    fill_array[rbin, cbin, row_idx, :] += 1.0  # or set to 1, your choice
    # reduce priority (like you do in fill_line_in_bin)
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
    Like prospective_fill_loop but for joint (resp, card) bins.

    Parameters
    ----------
    N : int
        Number of time steps.
    resp_fraction_array : np.ndarray
        (N,) of respiratory fraction 0..100 or NaN if unknown.
    resp_phase_array : np.ndarray of bool or None
        True = inhalation, False = exhalation, or None if unknown.
    cardiac_fraction_array : np.ndarray
        (N,) 0..100 or NaN if not calibrated.
    pros_fill : np.ndarray
        shape (num_resp_bins, num_card_bins, kspace_height, kspace_width).
    pros_priority : np.ndarray
        shape (num_resp_bins, num_card_bins, kspace_height).
    get_joint_weights_fn : callable
        e.g. partial(get_joint_bin_weights_gaussian, ...) returns dict {(rbin,cbin)->weight}.
    assign_bin_joint_fn : callable
        picks a single (rbin,cbin) for the prospective assignment.

    Returns
    -------
    list
      length N, each is None or (row_idx, rbin, cbin).
    """
    acquired_lines = [None] * N
    num_resp_bins, num_card_bins, kspace_height, kspace_width = pros_fill.shape

    for k in tqdm(range(N), desc="Prospective Filling"):
        resp_frac = resp_fraction_array[k]
        resp_ph = resp_phase_array[k]  # True/False
        card_frac = cardiac_fraction_array[k]
        if np.isnan(resp_frac) or resp_ph is None or np.isnan(card_frac):
            continue  # no fill

        # 1) get the dictionary of combined weights
        joint_weights = get_joint_weights_fn(resp_frac, resp_ph, card_frac)

        row_scores = np.zeros(kspace_height, dtype=float)

        for (rbin, cbin), w_ in joint_weights.items():
            if w_ < 1e-8:
                continue
            # For each row, accumulate w_ * priority
            # so row_scores[row] += (w_ * pros_priority[rbin, cbin, row])
            row_scores += w_ * pros_priority[rbin, cbin, :]

        # pick whichever row has the largest total score
        best_row = np.argmax(row_scores)
        best_score = row_scores[best_row]

        # Optionally skip if best_score is extremely small,
        # meaning there's no line worth acquiring
        if best_score < 1e-6:
            continue

        # 2) Figure out official prospective bin => single bin
        (official_rbin, official_cbin) = assign_bin_joint_fn(
            resp_frac, resp_ph, card_frac
        )

        # 3) Fill that line in the official bin
        fill_line_in_joint_bin(
            official_rbin, official_cbin, best_row, pros_fill, pros_priority
        )

        acquired_lines[k] = (best_row, official_rbin, official_cbin)

    return acquired_lines
