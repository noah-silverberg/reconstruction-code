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


def bin_reconstructed_kspace_joint(
    reconstructed_kspace,
    r_peaks,  # 1D array/list of global indices for cardiac R-peaks
    resp_peaks,  # 1D array/list of global indices for respiratory peaks
    num_cardiac_bins,
    num_respiratory_bins,
    n_phase_encodes_per_frame,
    extended_phase_lines,
    row_offset,
):
    """
    Jointly bin reconstructed k-space data by both cardiac and respiratory phase.

    Both r_peaks and resp_peaks are assumed to be 1D arrays/lists containing global phase
    encode indices that mark the start of each cycle.

    For each phase encode (line) in the k-space (indexed by the global index computed as
    f * n_phase_encodes_per_frame + row), the function:
      - Determines its fractional position within the current cardiac cycle using r_peaks.
      - Determines its fractional position within the current respiratory cycle using resp_peaks.
      - Scales these fractions to the number of cardiac and respiratory bins.

    The k-space data is then accumulated into a joint bin with shape:
        (num_cardiac_bins, num_respiratory_bins, extended_phase_lines, n_coils, n_freq)

    Parameters:
        reconstructed_kspace (np.ndarray):
            Shape (n_frames, n_phase_encodes_per_frame, n_coils, n_freq).
        r_peaks (array-like):
            1D array or list of global indices for cardiac R-peaks.
        resp_peaks (array-like):
            1D array or list of global indices for respiratory peaks.
        num_cardiac_bins (int):
            Number of cardiac bins.
        num_respiratory_bins (int):
            Number of respiratory bins.
        n_phase_encodes_per_frame (int):
            Number of phase encodes per frame.
        extended_phase_lines (int):
            Total number of phase lines in the output (for zero filling).
        row_offset (int):
            Offset to add to the row index when placing data into the extended phase lines.

    Returns:
        tuple: (binned_data, binned_count)
            - binned_data has shape
              (num_cardiac_bins, num_respiratory_bins, extended_phase_lines, n_coils, n_freq)
            - binned_count has shape
              (num_cardiac_bins, num_respiratory_bins, extended_phase_lines)
    """
    import numpy as np

    n_frames, n_phase, n_coils, n_freq = reconstructed_kspace.shape

    # Ensure r_peaks and resp_peaks are numpy arrays of type int.
    r_peaks = np.array(r_peaks, dtype=int)
    resp_peaks = np.array(resp_peaks, dtype=int)

    # Initialize accumulators for joint binning.
    binned_sum = np.zeros(
        (num_cardiac_bins, num_respiratory_bins, extended_phase_lines, n_coils, n_freq),
        dtype=reconstructed_kspace.dtype,
    )
    binned_count = np.zeros(
        (num_cardiac_bins, num_respiratory_bins, extended_phase_lines), dtype=np.int64
    )

    # Loop over each frame and each phase-encode line.
    for f in range(n_frames):
        for row in range(n_phase):
            # Global phase-encode index.
            global_idx = f * n_phase + row

            # --- Determine cardiac bin ---
            cycle_idx_c = np.searchsorted(r_peaks, global_idx, side="right") - 1
            if cycle_idx_c < 0 or cycle_idx_c >= len(r_peaks) - 1:
                continue  # Skip if not within a valid cardiac cycle.
            c_start = r_peaks[cycle_idx_c]
            c_end = r_peaks[cycle_idx_c + 1]
            fraction_c = (global_idx - c_start) / (c_end - c_start)
            cardiac_bin = int(np.floor(fraction_c * num_cardiac_bins))
            if cardiac_bin >= num_cardiac_bins:
                cardiac_bin = num_cardiac_bins - 1

            # --- Determine respiratory bin ---
            cycle_idx_r = np.searchsorted(resp_peaks, global_idx, side="right") - 1
            if cycle_idx_r < 0 or cycle_idx_r >= len(resp_peaks) - 1:
                continue  # Skip if not within a valid respiratory cycle.
            r_start = resp_peaks[cycle_idx_r]
            r_end = resp_peaks[cycle_idx_r + 1]
            fraction_r = (global_idx - r_start) / (r_end - r_start)
            resp_bin = int(np.floor(fraction_r * num_respiratory_bins))
            if resp_bin >= num_respiratory_bins:
                resp_bin = num_respiratory_bins - 1

            # Place the current k-space line into the joint bin (adjust row index by row_offset).
            binned_sum[cardiac_bin, resp_bin, row + row_offset] += reconstructed_kspace[
                f, row
            ]
            binned_count[cardiac_bin, resp_bin, row + row_offset] += 1

    # Average the bins where the count is greater than zero.
    binned_data = np.zeros_like(binned_sum)
    for i in range(num_cardiac_bins):
        for j in range(num_respiratory_bins):
            for r in range(extended_phase_lines):
                if binned_count[i, j, r] > 0:
                    binned_data[i, j, r] = binned_sum[i, j, r] / binned_count[i, j, r]

    return binned_data, binned_count


def bin_reconstructed_kspace_joint_physio(
    reconstructed_kspace,
    r_peaks,  # 1D array/list of global indices for cardiac cycles
    resp_peaks,  # 1D array/list of global indices for respiratory peaks (max inhalation)
    resp_troughs,  # 1D array/list of global indices for respiratory troughs (max exhalation)
    num_cardiac_bins,
    num_exhalation_bins,
    num_inhalation_bins,
    n_phase_encodes_per_frame,
    extended_phase_lines,
    row_offset,
):
    """
    Jointly bin reconstructed k-space data by cardiac phase and physiological respiratory phase.

    Here, we assume that:
      - r_peaks defines the boundaries for cardiac cycles (even division as before).
      - For respiration, we assume that resp_peaks mark maximum inhalation and resp_troughs mark maximum exhalation.
        Then the exhalation phase is from a resp_peak to the following resp_trough, and the inhalation phase is from that trough to the next resp_peak.

    The user specifies how many bins to subdivide the exhalation phase (num_exhalation_bins)
    and the inhalation phase (num_inhalation_bins). The overall respiratory dimension will be
    num_exhalation_bins + num_inhalation_bins.

    Parameters:
        reconstructed_kspace (np.ndarray): shape (n_frames, n_phase_encodes_per_frame, n_coils, n_freq)
        r_peaks (array-like): global indices for cardiac R-peaks.
        resp_peaks (array-like): global indices for respiratory peaks (max inhalation)
        resp_troughs (array-like): global indices for respiratory troughs (max exhalation)
        num_cardiac_bins (int): Number of cardiac bins.
        num_exhalation_bins (int): Number of bins for exhalation.
        num_inhalation_bins (int): Number of bins for inhalation.
        n_phase_encodes_per_frame, extended_phase_lines, row_offset: as before.

    Returns:
        tuple: (binned_data, binned_count) where:
           - binned_data has shape (num_cardiac_bins, total_resp_bins, extended_phase_lines, n_coils, n_freq)
           - total_resp_bins = num_exhalation_bins + num_inhalation_bins
           - binned_count has shape (num_cardiac_bins, total_resp_bins, extended_phase_lines)
    """
    import numpy as np

    n_frames, n_phase, n_coils, n_freq = reconstructed_kspace.shape
    total_resp_bins = num_exhalation_bins + num_inhalation_bins

    # Initialize accumulators.
    binned_sum = np.zeros(
        (num_cardiac_bins, total_resp_bins, extended_phase_lines, n_coils, n_freq),
        dtype=reconstructed_kspace.dtype,
    )
    binned_count = np.zeros(
        (num_cardiac_bins, total_resp_bins, extended_phase_lines), dtype=np.int64
    )

    for f in range(n_frames):
        for row in range(n_phase):
            global_idx = f * n_phase + row

            # Determine cardiac bin (same as before).
            cycle_idx_c = np.searchsorted(r_peaks, global_idx, side="right") - 1
            if cycle_idx_c < 0 or cycle_idx_c >= len(r_peaks) - 1:
                continue
            c_start = r_peaks[cycle_idx_c]
            c_end = r_peaks[cycle_idx_c + 1]
            fraction_c = (global_idx - c_start) / (c_end - c_start)
            cardiac_bin = int(np.floor(fraction_c * num_cardiac_bins))
            if cardiac_bin >= num_cardiac_bins:
                cardiac_bin = num_cardiac_bins - 1

            # Physiological respiratory binning.
            # We assume that resp_peaks and resp_troughs alternate.
            # If the most recent event is a peak, then we are in exhalation (from peak to next trough).
            idx_peak = np.searchsorted(resp_peaks, global_idx, side="right") - 1
            idx_trough = np.searchsorted(resp_troughs, global_idx, side="right") - 1
            if idx_peak < 0 or idx_trough < 0:
                continue
            last_peak = resp_peaks[idx_peak]
            last_trough = resp_troughs[idx_trough]

            if last_peak > last_trough:
                # We are in exhalation: from last_peak to next trough.
                if idx_trough + 1 < len(resp_troughs):
                    next_trough = resp_troughs[idx_trough + 1]
                else:
                    continue
                fraction_resp = (global_idx - last_peak) / (next_trough - last_peak)
                resp_bin = int(np.floor(fraction_resp * num_exhalation_bins))
                if resp_bin >= num_exhalation_bins:
                    resp_bin = num_exhalation_bins - 1
                overall_resp_bin = resp_bin  # exhalation bins come first.
            else:
                # We are in inhalation: from last_trough to next peak.
                if idx_peak + 1 < len(resp_peaks):
                    next_peak = resp_peaks[idx_peak + 1]
                else:
                    continue
                fraction_resp = (global_idx - last_trough) / (next_peak - last_trough)
                resp_bin = int(np.floor(fraction_resp * num_inhalation_bins))
                if resp_bin >= num_inhalation_bins:
                    resp_bin = num_inhalation_bins - 1
                overall_resp_bin = (
                    num_exhalation_bins + resp_bin
                )  # inhalation bins come after exhalation bins.

            # Accumulate the k-space line.
            binned_sum[
                cardiac_bin, overall_resp_bin, row + row_offset
            ] += reconstructed_kspace[f, row]
            binned_count[cardiac_bin, overall_resp_bin, row + row_offset] += 1

    # Compute averages.
    binned_data = np.zeros_like(binned_sum)
    for i in range(num_cardiac_bins):
        for j in range(total_resp_bins):
            for r in range(extended_phase_lines):
                if binned_count[i, j, r] > 0:
                    binned_data[i, j, r] = binned_sum[i, j, r] / binned_count[i, j, r]
    return binned_data, binned_count
