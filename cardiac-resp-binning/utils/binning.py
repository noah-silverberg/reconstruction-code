"""
binning.py

Provides functions to bin reconstructed k-space by cardiac and respiratory phase.
"""

import numpy as np


def bin_reconstructed_kspace_joint(
    reconstructed_kspace,
    r_peaks,
    resp_peaks,
    num_cardiac_bins,
    num_respiratory_bins,
    n_phase_encodes_per_frame,
    extended_phase_lines,
    row_offset,
):
    """
    Bin reconstructed k-space jointly by cardiac and respiratory phase.

    Parameters
    ----------
    reconstructed_kspace : np.ndarray
        Shape (n_frames, n_phase_encodes_per_frame, n_coils, n_freq).
    r_peaks : array-like
        1D array of R-peak global indices defining cardiac cycle boundaries.
    resp_peaks : array-like
        1D array of respiratory peak global indices defining respiratory cycle boundaries.
    num_cardiac_bins : int
        Number of cardiac bins.
    num_respiratory_bins : int
        Number of respiratory bins.
    n_phase_encodes_per_frame : int
        Phase encodes per frame.
    extended_phase_lines : int
        Extended total lines in zero-filled dimension.
    row_offset : int
        Offset for measured lines within the extended dimension.

    Returns
    -------
    tuple
        (binned_data, binned_count) where:
        - binned_data has shape
          (num_cardiac_bins, num_respiratory_bins, extended_phase_lines, n_coils, n_freq)
        - binned_count has shape
          (num_cardiac_bins, num_respiratory_bins, extended_phase_lines)
    """
    n_frames, n_phase, n_coils, n_freq = reconstructed_kspace.shape

    r_peaks = np.array(r_peaks, dtype=int)
    resp_peaks = np.array(resp_peaks, dtype=int)

    binned_sum = np.zeros(
        (
            num_cardiac_bins,
            num_respiratory_bins,
            extended_phase_lines,
            n_coils,
            n_freq,
        ),
        dtype=reconstructed_kspace.dtype,
    )
    binned_count = np.zeros(
        (num_cardiac_bins, num_respiratory_bins, extended_phase_lines), dtype=np.int64
    )

    for f in range(n_frames):
        for row in range(n_phase):
            global_idx = f * n_phase + row

            # Cardiac cycle fraction
            cycle_idx_c = np.searchsorted(r_peaks, global_idx, side="right") - 1
            if cycle_idx_c < 0 or cycle_idx_c >= len(r_peaks) - 1:
                continue
            c_start = r_peaks[cycle_idx_c]
            c_end = r_peaks[cycle_idx_c + 1]
            fraction_c = (global_idx - c_start) / (c_end - c_start)
            cardiac_bin = int(np.floor(fraction_c * num_cardiac_bins))
            if cardiac_bin >= num_cardiac_bins:
                cardiac_bin = num_cardiac_bins - 1

            # Respiratory cycle fraction
            cycle_idx_r = np.searchsorted(resp_peaks, global_idx, side="right") - 1
            if cycle_idx_r < 0 or cycle_idx_r >= len(resp_peaks) - 1:
                continue
            r_start = resp_peaks[cycle_idx_r]
            r_end = resp_peaks[cycle_idx_r + 1]
            fraction_r = (global_idx - r_start) / (r_end - r_start)
            resp_bin = int(np.floor(fraction_r * num_respiratory_bins))
            if resp_bin >= num_respiratory_bins:
                resp_bin = num_respiratory_bins - 1

            binned_sum[cardiac_bin, resp_bin, row + row_offset] += reconstructed_kspace[
                f, row
            ]
            binned_count[cardiac_bin, resp_bin, row + row_offset] += 1

    binned_data = np.zeros_like(binned_sum)
    for i in range(num_cardiac_bins):
        for j in range(num_respiratory_bins):
            for r in range(extended_phase_lines):
                if binned_count[i, j, r] > 0:
                    binned_data[i, j, r] = binned_sum[i, j, r] / binned_count[i, j, r]

    return binned_data, binned_count


def bin_reconstructed_kspace_joint_physio(
    reconstructed_kspace,
    r_peaks,
    resp_peaks,
    resp_troughs,
    num_cardiac_bins,
    num_exhalation_bins,
    num_inhalation_bins,
    n_phase_encodes_per_frame,
    extended_phase_lines,
    row_offset,
):
    """
    Bin reconstructed k-space jointly by cardiac phase and a physiological
    respiration model (exhalation/inhalation phases).

    Parameters
    ----------
    reconstructed_kspace : np.ndarray
        Shape (n_frames, n_phase_encodes_per_frame, n_coils, n_freq).
    r_peaks : np.ndarray
        1D array of R-peak indices.
    resp_peaks : np.ndarray
        1D array of respiratory peak (max inhalation) indices.
    resp_troughs : np.ndarray
        1D array of respiratory trough (max exhalation) indices.
    num_cardiac_bins : int
        Number of cardiac bins.
    num_exhalation_bins : int
        Number of bins covering exhalation.
    num_inhalation_bins : int
        Number of bins covering inhalation.
    n_phase_encodes_per_frame : int
        Phase encodes per frame.
    extended_phase_lines : int
        Extended lines dimension for zero-filling.
    row_offset : int
        Where measured lines are placed in the extended dimension.

    Returns
    -------
    tuple
        (binned_data, binned_count) where:
        - binned_data has shape
          (num_cardiac_bins, total_resp_bins, extended_phase_lines, n_coils, n_freq)
        - binned_count has shape
          (num_cardiac_bins, total_resp_bins, extended_phase_lines)
    """
    n_frames, n_phase, n_coils, n_freq = reconstructed_kspace.shape
    total_resp_bins = num_exhalation_bins + num_inhalation_bins

    binned_sum = np.zeros(
        (
            num_cardiac_bins,
            total_resp_bins,
            extended_phase_lines,
            n_coils,
            n_freq,
        ),
        dtype=reconstructed_kspace.dtype,
    )
    binned_count = np.zeros(
        (num_cardiac_bins, total_resp_bins, extended_phase_lines),
        dtype=np.int64,
    )

    for f in range(n_frames):
        for row in range(n_phase):
            global_idx = f * n_phase + row

            # Cardiac bin
            cycle_idx_c = np.searchsorted(r_peaks, global_idx, side="right") - 1
            if cycle_idx_c < 0 or cycle_idx_c >= len(r_peaks) - 1:
                continue
            c_start = r_peaks[cycle_idx_c]
            c_end = r_peaks[cycle_idx_c + 1]
            fraction_c = (global_idx - c_start) / (c_end - c_start)
            cardiac_bin = int(np.floor(fraction_c * num_cardiac_bins))
            if cardiac_bin >= num_cardiac_bins:
                cardiac_bin = num_cardiac_bins - 1

            # Physiological respiratory bin
            idx_peak = np.searchsorted(resp_peaks, global_idx, side="right") - 1
            idx_trough = np.searchsorted(resp_troughs, global_idx, side="right") - 1
            if idx_peak < 0 or idx_trough < 0:
                continue
            last_peak = resp_peaks[idx_peak]
            last_trough = resp_troughs[idx_trough]

            # Determine exhalation or inhalation
            if last_peak > last_trough:
                # Exhalation: from peak to next trough
                if idx_trough + 1 >= len(resp_troughs):
                    continue
                next_trough = resp_troughs[idx_trough + 1]
                fraction_resp = (global_idx - last_peak) / (next_trough - last_peak)
                resp_bin = int(np.floor(fraction_resp * num_exhalation_bins))
                if resp_bin >= num_exhalation_bins:
                    resp_bin = num_exhalation_bins - 1
                overall_resp_bin = resp_bin  # Exhalation bins first
            else:
                # Inhalation: from trough to next peak
                if idx_peak + 1 >= len(resp_peaks):
                    continue
                next_peak = resp_peaks[idx_peak + 1]
                fraction_resp = (global_idx - last_trough) / (next_peak - last_trough)
                resp_bin = int(np.floor(fraction_resp * num_inhalation_bins))
                if resp_bin >= num_inhalation_bins:
                    resp_bin = num_inhalation_bins - 1
                overall_resp_bin = num_exhalation_bins + resp_bin

            binned_sum[
                cardiac_bin, overall_resp_bin, row + row_offset
            ] += reconstructed_kspace[f, row]
            binned_count[cardiac_bin, overall_resp_bin, row + row_offset] += 1

    # Average bins
    binned_data = np.zeros_like(binned_sum)
    for i in range(num_cardiac_bins):
        for j in range(total_resp_bins):
            for r in range(extended_phase_lines):
                if binned_count[i, j, r] > 0:
                    binned_data[i, j, r] = binned_sum[i, j, r] / binned_count[i, j, r]

    return binned_data, binned_count
