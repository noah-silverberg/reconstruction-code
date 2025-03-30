# utils/kspace_filling.py

import numpy as np


def build_line_priority(num_bins, kspace_height=128):
    """
    Return an array of shape (num_bins, kspace_height) giving a base priority.
    For example, you can define a center-based priority or something else.
    """
    center = kspace_height // 2
    pr = np.zeros((num_bins, kspace_height), dtype=float)
    for b_ in range(num_bins):
        for r in range(kspace_height):
            dist = abs(r - center)
            base = 300 - dist * 2
            pr[b_, r] = base
    return pr


def fill_line_in_bin(bin_idx, row, fill_array, priority_array):
    """
    Fill the selected line in fill_array[bin_idx, row, :].
    Then adjust priority so we don't pick it again if desired.
    Also optionally add a bonus for the conjugate line if unfilled.
    """
    fill_array[bin_idx, row, :] += 1.0
    priority_array[bin_idx, row] = -999999  # de-prioritize it

    conj_row = fill_array.shape[1] - 1 - row
    # if not filled, add bonus. Up to you if you want to do that
    if fill_array[bin_idx, conj_row, 0] == 0:
        priority_array[bin_idx, conj_row] += 20


def get_bin_index_gaussian(
    fraction,
    is_inhale,
    num_inhale_bins,
    num_exhale_bins,
    use_total_bins=False,
    num_total_bins=4,
):
    """
    Return a dictionary {bin_idx: weight} using a Gaussian weighting around the bin's center fraction.
    Example logic from your code. Keep it flexible so you can change the weighting approach.
    """
    if use_total_bins:
        bin_count = num_total_bins
        bin_width = 100.0 / bin_count
        bin_i = int(np.floor(fraction / bin_width))
        if bin_i >= bin_count:
            bin_i = bin_count - 1
        bin_dict = {}
        c_bin = (bin_i + 0.5) * bin_width
        dist = abs(fraction - c_bin)
        sigma = bin_width * 0.3
        w_main = np.exp(-0.5 * (dist / sigma) ** 2)
        bin_dict[bin_i] = w_main
        for adj in [bin_i - 1, bin_i + 1]:
            if 0 <= adj < bin_count:
                c2 = (adj + 0.5) * bin_width
                d2 = abs(fraction - c2)
                w2 = np.exp(-0.5 * (d2 / sigma) ** 2)
                if w2 > 1e-4:
                    bin_dict[adj] = w2
        # normalize
        s = sum(bin_dict.values())
        if s < 1e-8:
            bin_dict = {bin_i: 1.0}
        else:
            for b_ in bin_dict:
                bin_dict[b_] /= s
        return bin_dict
    else:
        # separate inhale/exhale
        if is_inhale:
            bin_count = num_inhale_bins
            bin_width = 100.0 / bin_count
            bin_i = int(np.floor(fraction / bin_width))
            if bin_i >= bin_count:
                bin_i = bin_count - 1
            bin_dict = {}
            c_bin = (bin_i + 0.5) * bin_width
            dist = abs(fraction - c_bin)
            sigma = bin_width * 0.3
            w_main = np.exp(-0.5 * (dist / sigma) ** 2)
            bin_dict[bin_i] = w_main
            for adj in [bin_i - 1, bin_i + 1]:
                if 0 <= adj < bin_count:
                    c2 = (adj + 0.5) * bin_width
                    d2 = abs(fraction - c2)
                    w2 = np.exp(-0.5 * (d2 / sigma) ** 2)
                    if w2 > 1e-4:
                        bin_dict[adj] = w2
            # normalize
            s = sum(bin_dict.values())
            if s < 1e-8:
                bin_dict = {bin_i: 1.0}
            else:
                for b_ in bin_dict:
                    bin_dict[b_] /= s
            return shift_inhale_exhale_bins(
                bin_dict, is_inhale=True, num_inhale_bins=num_inhale_bins
            )
        else:
            bin_count = num_exhale_bins
            bin_width = 100.0 / bin_count
            bin_i = int(np.floor(fraction / bin_width))
            if bin_i >= bin_count:
                bin_i = bin_count - 1
            bin_dict = {}
            c_bin = (bin_i + 0.5) * bin_width
            dist = abs(fraction - c_bin)
            sigma = bin_width * 0.3
            w_main = np.exp(-0.5 * (dist / sigma) ** 2)
            bin_dict[bin_i] = w_main
            for adj in [bin_i - 1, bin_i + 1]:
                if 0 <= adj < bin_count:
                    c2 = (adj + 0.5) * bin_width
                    d2 = abs(fraction - c2)
                    w2 = np.exp(-0.5 * (d2 / sigma) ** 2)
                    if w2 > 1e-4:
                        bin_dict[adj] = w2
            # normalize
            s = sum(bin_dict.values())
            if s < 1e-8:
                bin_dict = {bin_i: 1.0}
            else:
                for b_ in bin_dict:
                    bin_dict[b_] /= s
            return shift_inhale_exhale_bins(
                bin_dict, is_inhale=False, num_inhale_bins=num_inhale_bins
            )


def shift_inhale_exhale_bins(bin_dict, is_inhale, num_inhale_bins):
    """
    Helper that offsets exhale bins by + num_inhale_bins, etc.
    This is your original offset logic from the code.
    """
    if is_inhale:
        # inhale bins are 0 to num_inhale_bins - 1
        return bin_dict
    else:
        # exhale bins are num_inhale_bins to num_inhale_bins + num_exhale_bins - 1
        shifted_dict = {}
        for k in bin_dict:
            shifted_dict[k + num_inhale_bins] = bin_dict[k]
        return shifted_dict


def assign_prospective_bin(
    fraction,
    is_inhale,
    num_inhale_bins,
    num_exhale_bins,
    use_total_bins=False,
    num_total_bins=4,
):
    """
    Hard assignment to a single bin from fraction alone, ignoring the Gaussian weighting.
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
            bin_width = 100.0 / num_inhale_bins
            bin_i = int(np.floor(fraction / bin_width))
            if bin_i >= num_inhale_bins:
                bin_i = num_inhale_bins - 1
            return bin_i
        else:
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
    1) For each time k, compute bin_index => dictionary of bin => weight
    2) For each bin, find top row
    3) Pick the bin with highest weighted priority
    4) Actually fill that row in the 'prospective bin' from fraction
    Returns: acquired_lines (list of (row, prospective_bin)) of length N
    """
    acquired_lines = [None] * N
    total_bins = pros_fill.shape[0]
    for k in range(N):
        frac = predicted_fraction[k]
        ph = predicted_phase[k]
        if np.isnan(frac) or ph is None:
            continue

        # returns {bin_idx: weight}
        bin_weight = get_bin_index_fn(frac, ph)

        # find top candidate row for each bin
        top_candidates = {}
        for b_ in bin_weight:
            w_ = bin_weight[b_]
            if w_ < 1e-6:
                continue
            row_argmax = np.argmax(pros_priority[b_])
            val = pros_priority[b_, row_argmax]
            if val < 0:
                continue
            score = w_ * val
            top_candidates[b_] = (row_argmax, score)

        if not top_candidates:
            continue

        # pick best bin
        best_bin = max(top_candidates.keys(), key=lambda x: top_candidates[x][1])
        best_row = top_candidates[best_bin][0]

        # prospective bin
        pbin = assign_bin_fn(frac, ph)
        acquired_lines[k] = (best_row, pbin)

        # fill
        fill_line_in_bin(pbin, best_row, pros_fill, pros_priority)

    return acquired_lines
