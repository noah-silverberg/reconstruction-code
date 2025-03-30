# utils/kspace_filling.py

import numpy as np


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
            base_priority = 300 - (distance_from_center * 2)
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
    priority_array[bin_index, row_index] = (
        -999999
    )  # Force priority negative to skip next time

    # Identify the conjugate line index
    conj_row_index = fill_array.shape[1] - 1 - row_index

    # If the conjugate line in this bin is unfilled, add a small bonus
    if fill_array[bin_index, conj_row_index, 0] == 0.0:
        priority_array[bin_index, conj_row_index] += 20


def get_bin_index_gaussian(
    fraction,
    is_inhale,
    num_inhale_bins,
    num_exhale_bins,
    use_total_bins=False,
    num_total_bins=4,
):
    """
    Compute "voting weights" for each bin based on how close the fraction is
    to the center of that bin, using a Gaussian weighting scheme.

    By returning a dictionary {bin_idx: weight}, we can see that bins near the fraction
    get higher weights, and bins far from it get lower (or zero).

    Parameters
    ----------
    fraction : float
        The prospective fraction (0..100) we estimated for this time sample.
    is_inhale : bool
        True if we believe we are in inhalation, False for exhalation.
    num_inhale_bins : int
        Number of inhalation bins (if separate).
    num_exhale_bins : int
        Number of exhalation bins (if separate).
    use_total_bins : bool
        Whether to treat all bins (inhalation + exhalation) as one contiguous block (0..num_total_bins-1).
    num_total_bins : int
        If use_total_bins==True, how many total bins do we have?

    Returns
    -------
    dict
        A dictionary mapping bin_index -> weight (floats). Bins with no assignment
        might not appear in the dict, or might be zero if they are very far.
    """
    # If using a single contiguous set of bins (the "total bins" approach):
    if use_total_bins:
        bin_count = num_total_bins
        bin_width = 100.0 / bin_count
        # Hard find which bin the fraction is “closest” to
        bin_idx = int(np.floor(fraction / bin_width))
        if bin_idx >= bin_count:
            bin_idx = bin_count - 1

        weights_dict = {}
        center_of_bin = (bin_idx + 0.5) * bin_width
        dist = abs(fraction - center_of_bin)
        sigma = bin_width * 0.3
        main_weight = np.exp(-0.5 * (dist / sigma) ** 2)
        weights_dict[bin_idx] = main_weight

        # Possibly also assign small weights to neighboring bins
        for neighbor in [bin_idx - 1, bin_idx + 1]:
            if 0 <= neighbor < bin_count:
                c2 = (neighbor + 0.5) * bin_width
                d2 = abs(fraction - c2)
                w2 = np.exp(-0.5 * (d2 / sigma) ** 2)
                if w2 > 1e-4:
                    weights_dict[neighbor] = w2

        # Normalize
        sum_w = sum(weights_dict.values())
        if sum_w < 1e-8:
            # If numeric issues => just assign all to bin_idx
            weights_dict = {bin_idx: 1.0}
        else:
            for k in weights_dict.keys():
                weights_dict[k] /= sum_w

        return weights_dict

    else:
        # If we keep inhalation bins and exhalation bins separate
        # Step 1) figure out if we are in inhalation or exhalation from is_inhale.
        if is_inhale:
            # Distribute across inhalation bins only
            bin_count_inh = num_inhale_bins
            bin_width = 100.0 / bin_count_inh

            bin_idx = int(np.floor(fraction / bin_width))
            if bin_idx >= bin_count_inh:
                bin_idx = bin_count_inh - 1

            # Build a dict for these bins
            weights_inh = {}
            center_of_bin = (bin_idx + 0.5) * bin_width
            dist = abs(fraction - center_of_bin)
            sigma = bin_width * 0.3
            main_weight = np.exp(-0.5 * (dist / sigma) ** 2)
            weights_inh[bin_idx] = main_weight

            for neighbor in [bin_idx - 1, bin_idx + 1]:
                if 0 <= neighbor < bin_count_inh:
                    c2 = (neighbor + 0.5) * bin_width
                    d2 = abs(fraction - c2)
                    w2 = np.exp(-0.5 * (d2 / sigma) ** 2)
                    if w2 > 1e-4:
                        weights_inh[neighbor] = w2

            sum_w = sum(weights_inh.values())
            if sum_w < 1e-8:
                weights_inh = {bin_idx: 1.0}
            else:
                for k in weights_inh.keys():
                    weights_inh[k] /= sum_w

            # The inhalation bins are 0..(num_inhale_bins-1).
            # Return as a dictionary {bin_idx: weight}
            return weights_inh

        else:
            # Exhalation bins
            bin_count_exh = num_exhale_bins
            bin_width = 100.0 / bin_count_exh
            bin_idx = int(np.floor(fraction / bin_width))
            if bin_idx >= bin_count_exh:
                bin_idx = bin_count_exh - 1

            weights_exh = {}
            center_of_bin = (bin_idx + 0.5) * bin_width
            dist = abs(fraction - center_of_bin)
            sigma = bin_width * 0.3
            main_weight = np.exp(-0.5 * (dist / sigma) ** 2)
            weights_exh[bin_idx] = main_weight

            for neighbor in [bin_idx - 1, bin_idx + 1]:
                if 0 <= neighbor < bin_count_exh:
                    c2 = (neighbor + 0.5) * bin_width
                    d2 = abs(fraction - c2)
                    w2 = np.exp(-0.5 * (d2 / sigma) ** 2)
                    if w2 > 1e-4:
                        weights_exh[neighbor] = w2

            sum_w = sum(weights_exh.values())
            if sum_w < 1e-8:
                weights_exh = {bin_idx: 1.0}
            else:
                for k in weights_exh.keys():
                    weights_exh[k] /= sum_w

            # These exhalation bins are offset by the # of inhale bins
            # e.g. if inhalation bins = 2, exhalation bins start at index 2, 3, ...
            # So we shift them accordingly
            shifted_dict = {}
            for b_ in weights_exh.keys():
                new_index = b_ + num_inhale_bins
                shifted_dict[new_index] = weights_exh[b_]
            return shifted_dict


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
