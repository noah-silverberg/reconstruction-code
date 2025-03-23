"""
binning.py

This module contains functions to bin reconstructed k-space by cardiac phase.
"""

import numpy as np
from utils.reconstruction import (
    reconstruct_kspace_with_selected_components_kernel_kspace,
)


def bin_reconstructed_kspace(
    reconstructed_kspace,
    r_peaks_list,
    num_bins,
    n_phase_encodes_per_frame,
    extended_phase_lines,
    row_offset,
):
    """
    Bin already reconstructed k-space data by cardiac phase using R-peaks.

    Parameters:
        reconstructed_kspace (np.ndarray): shape (n_frames, n_phase_encodes_per_frame, n_coils, n_freq).
        r_peaks_list (list): List of R-peak index arrays.
        num_bins (int): Number of cardiac bins.
        n_phase_encodes_per_frame (int): Phase encodes per frame.
        extended_phase_lines (int): Extended lines for zero fill.
        row_offset (int): Where in the extended array these lines start.

    Returns:
        tuple: (binned_data, binned_count)
    """
    avg_r_peaks = np.round(np.mean(np.vstack(r_peaks_list), axis=0)).astype(int)
    n_frames, n_phase, n_coils, n_freq = reconstructed_kspace.shape
    total_phase_encodes = n_frames * n_phase

    if total_phase_encodes % n_phase_encodes_per_frame != 0:
        raise ValueError(
            "Total phase encodes must be multiple of n_phase_encodes_per_frame."
        )

    # Flatten frames x lines
    data_reshaped = reconstructed_kspace.reshape(n_frames, n_phase, n_coils, n_freq)
    binned_sum = np.zeros(
        (num_bins, extended_phase_lines, n_coils, n_freq), dtype=data_reshaped.dtype
    )
    binned_count = np.zeros((num_bins, extended_phase_lines), dtype=np.int64)

    for f in range(n_frames):
        for row in range(n_phase):
            global_idx = f * n_phase + row
            cycle_idx = np.searchsorted(avg_r_peaks, global_idx, side="right") - 1
            if cycle_idx < 0 or cycle_idx >= len(avg_r_peaks) - 1:
                continue
            c_start = avg_r_peaks[cycle_idx]
            c_end = avg_r_peaks[cycle_idx + 1]
            fraction = (global_idx - c_start) / (c_end - c_start)
            bin_idx = int(np.floor(fraction * num_bins))
            if bin_idx >= num_bins:
                bin_idx = num_bins - 1

            binned_sum[bin_idx, row + row_offset] += data_reshaped[f, row]
            binned_count[bin_idx, row + row_offset] += 1

    # Average bins
    binned_data = np.zeros_like(binned_sum)
    for b in range(num_bins):
        for r in range(extended_phase_lines):
            if binned_count[b, r] > 0:
                binned_data[b, r] = binned_sum[b, r] / binned_count[b, r]
    return binned_data, binned_count


def bin_reconstructed_kspace_by_cardiac_phase_kernel(
    kpca,
    X_kpca,
    selection,
    frame_shape,
    orig_feature_dim,
    r_peaks_list,
    num_bins,
    n_phase_encodes_per_frame,
    extended_pe_lines,
    offset,
):
    """
    Legacy function from older code that tries to do the reconstruction *and* binning.
    We keep it for backward-compatibility in case older scripts call it.
    Internally, it uses `reconstruct_kspace_with_selected_components_kernel_kspace`
    and then calls `bin_reconstructed_kspace`.

    Parameters are the same as before.
    """
    # First do old reconstruction
    reconstructed = reconstruct_kspace_with_selected_components_kernel_kspace(
        kpca, X_kpca, selection, frame_shape, orig_feature_dim
    )
    # Then bin
    return bin_reconstructed_kspace(
        reconstructed,
        r_peaks_list,
        num_bins,
        n_phase_encodes_per_frame,
        extended_pe_lines,
        offset,
    )
