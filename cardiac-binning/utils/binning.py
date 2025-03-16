"""
binning.py

This module contains functions to bin reconstructed k-space data by cardiac phase.
"""

import numpy as np
from utils.reconstruction import (
    reconstruct_kspace_with_selected_components_kernel_kspace,
)


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
    Reconstruct k-space using selected kernel PCA components (without inverse FFT),
    then bin the data based on the cardiac cycle using detected R-peaks.

    Parameters:
        kpca: Fitted KernelPCA model.
        X_kpca: Kernel PCA transformed data.
        selection: Component indices to use.
        frame_shape: Tuple (phase_encodes, coils, freq_encodes).
        orig_feature_dim: Original feature dimension.
        r_peaks_list: List of R-peak indices arrays.
        num_bins: Number of cardiac bins.
        n_phase_encodes_per_frame: Phase encodes per frame.
        extended_pe_lines: Extended phase encode lines.
        offset: Offset to insert measured lines.

    Returns:
        Tuple: (binned_data, binned_count)
    """
    reconstructed = reconstruct_kspace_with_selected_components_kernel_kspace(
        kpca, X_kpca, selection, frame_shape, orig_feature_dim
    )
    # Reshape to full k-space (total_phase_encodes, coils, freq_encodes)
    full_reconstructed = reconstructed.reshape(-1, frame_shape[1], frame_shape[2])
    binned_data, binned_count = bin_kspace_by_cardiac_phase(
        r_peaks_list,
        full_reconstructed,
        num_bins,
        n_phase_encodes_per_frame,
        extended_pe_lines,
        offset,
    )
    return binned_data, binned_count


def bin_kspace_by_cardiac_phase(
    r_peaks_list, img, num_bins, n_phase_encodes_per_frame, extended_pe_lines, offset
):
    """
    Bin k-space data by cardiac phase based on detected R-peaks.

    Parameters:
        r_peaks_list: List of 1D numpy arrays of R-peak indices.
        img: k-space data with shape (total_phase_encodes, coils, readout).
        num_bins: Number of bins.
        n_phase_encodes_per_frame: Phase encodes per frame.
        extended_pe_lines: Extended phase lines count.
        offset: Row offset.

    Returns:
        Tuple: (binned_data, binned_count)
    """
    avg_r_peaks = np.round(np.mean(np.vstack(r_peaks_list), axis=0)).astype(int)
    total_phase_encodes, n_coils, n_readout = img.shape
    if total_phase_encodes % n_phase_encodes_per_frame != 0:
        raise ValueError(
            "Total phase encodes is not a multiple of n_phase_encodes_per_frame."
        )
    n_frames = total_phase_encodes // n_phase_encodes_per_frame
    kspace_data = img.reshape(n_frames, n_phase_encodes_per_frame, n_coils, n_readout)
    binned_sum = np.zeros(
        (num_bins, extended_pe_lines, n_coils, n_readout), dtype=kspace_data.dtype
    )
    binned_count = np.zeros((num_bins, extended_pe_lines), dtype=np.int64)
    for frame in range(n_frames):
        for row in range(n_phase_encodes_per_frame):
            global_index = frame * n_phase_encodes_per_frame + row
            cycle_idx = np.searchsorted(avg_r_peaks, global_index, side="right") - 1
            if cycle_idx < 0 or cycle_idx >= len(avg_r_peaks) - 1:
                continue
            cycle_start = avg_r_peaks[cycle_idx]
            cycle_end = avg_r_peaks[cycle_idx + 1]
            fraction = (global_index - cycle_start) / (cycle_end - cycle_start)
            bin_idx = int(np.floor(fraction * num_bins))
            if bin_idx >= num_bins:
                bin_idx = num_bins - 1
            binned_sum[bin_idx, row + offset] += kspace_data[frame, row]
            binned_count[bin_idx, row + offset] += 1
    binned_data = np.zeros_like(binned_sum)
    for b in range(num_bins):
        for r in range(extended_pe_lines):
            if binned_count[b, r] > 0:
                binned_data[b, r] = binned_sum[b, r] / binned_count[b, r]
    return binned_data, binned_count
