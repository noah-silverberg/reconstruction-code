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
