# utils/resp_fractions.py

import numpy as np
from .ecg_resp import detect_resp_peaks


def extract_cycles(boundary_indices, boundary_labels):
    """
    Identify inhale/exhale cycles from sorted boundary indices (peaks/troughs).

    You supply two matched arrays:
      boundary_indices : e.g. [23, 50, 90, 130]
      boundary_labels  : e.g. ['trough','peak','trough','peak'] (sorted by index)

    We traverse them pairwise and recognize:
      trough -> peak => inhalation
      peak -> trough => exhalation

    Returns
    -------
    list
        A list of (start_idx, end_idx, 'inhalation' or 'exhalation').
    """
    cycles = []
    for i in range(len(boundary_indices) - 1):
        s_idx = boundary_indices[i]
        e_idx = boundary_indices[i + 1]
        s_lab = boundary_labels[i]  # 'peak' or 'trough'
        e_lab = boundary_labels[i + 1]

        if s_lab == "trough" and e_lab == "peak":
            cycles.append((s_idx, e_idx, "inhalation"))
        elif s_lab == "peak" and e_lab == "trough":
            cycles.append((s_idx, e_idx, "exhalation"))
    return cycles


def compute_avg_durations(cycles, fs):
    """
    Given a list of cycles (start_idx, end_idx, phase), compute average durations (s) for inhalation and exhalation.

    Parameters
    ----------
    cycles : list of tuples
        Each is (start_idx, end_idx, 'inhalation'|'exhalation').
    fs : float
        Sampling frequency (Hz).

    Returns
    -------
    (float or None, float or None)
        (avg_inhale, avg_exhale). Returns None if no cycles of that type exist.
    """
    inhale_durations = []
    exhale_durations = []

    for start_i, end_i, phase in cycles:
        if end_i > start_i:
            duration_seconds = (end_i - start_i) / fs
            if phase == "inhalation":
                inhale_durations.append(duration_seconds)
            else:
                exhale_durations.append(duration_seconds)

    avg_inhale = np.mean(inhale_durations) if inhale_durations else None
    avg_exhale = np.mean(exhale_durations) if exhale_durations else None
    return (avg_inhale, avg_exhale)


def have_two_full_cycles(cycles):
    """
    Simple check if we have at least two cycles recorded,
    e.g. (trough->peak, peak->trough, trough->peak).
    """
    return len(cycles) >= 2


def build_true_fraction(signal_length, peaks, troughs):
    """
    Construct an "actual fraction" offline for each sample,
    purely for retrospective analysis.

    We label each sample from 0..100% based on:
      trough->peak => ramp 0..100
      peak->trough => ramp 100..0

    Returns
    -------
    np.ndarray
        Length = signal_length, fraction from 0..100 or NaN if unknown.
    """
    all_extrema = np.concatenate((peaks, troughs))
    if len(all_extrema) == 0:
        return np.full(signal_length, np.nan)

    sort_idx = np.argsort(all_extrema)
    sorted_indices = all_extrema[sort_idx]

    # Build label array in sorted order
    sorted_labels = []
    for idx in sorted_indices:
        if idx in peaks:
            sorted_labels.append("peak")
        else:
            sorted_labels.append("trough")
    sorted_labels = np.array(sorted_labels, dtype=object)

    # Convert to cycles
    cyc = extract_cycles(sorted_indices, sorted_labels)

    frac_array = np.full(signal_length, np.nan)
    for start_i, end_i, phase in cyc:
        if end_i <= start_i:
            continue
        length = end_i - start_i
        for sample_idx in range(start_i, min(end_i, signal_length)):
            if phase == "inhalation":
                # ramp from 0 at trough to 100 at peak
                frac = 100.0 * (sample_idx - start_i) / float(length)
            else:
                # exhalation: ramp from 100 at peak to 0 at trough
                frac = 100.0 * (1.0 - ((sample_idx - start_i) / float(length)))
            frac_array[sample_idx] = frac

    return frac_array


def predict_fraction(
    resp_signal, fs, min_calib_time=10.0, peak_height=0.6, peak_prom=0.2
):
    """
    Compute prospective fraction & phase for each time sample in a respiration signal.

    This attempts to do a real-time approach:
      - We detect peaks & troughs on the partial data up to sample k.
      - After 2 full cycles + min_calib_time, we treat calibration as done.
      - We keep updating average inhalation/exhalation durations from cycles.
      - We then linearly ramp fraction from boundary_idx to boundary_idx+avg_inhale (or avg_exhale).
      - Phase = True for inhalation, False for exhalation.

    Parameters
    ----------
    resp_signal : np.ndarray
        The respiration data, shape (N,). Each sample corresponds to 1/ fs seconds.
    fs : float
        Sampling frequency in Hz.
    min_calib_time : float
        Minimum time (seconds) required before we say “calibration is complete.”
    peak_height : float
        For detect_resp_peaks (when method='scipy'). Minimum height of peaks.
    peak_prom : float
        For detect_resp_peaks. Minimum prominence of peaks.

    Returns
    -------
    predicted_fraction : np.ndarray
        Length N. The fraction 0..100 for each time step, or NaN if not calibrated yet.
    predicted_phase : np.ndarray of bool or None
        True if inhalation, False if exhalation, None if unknown/not calibrated.
    calibration_end_idx : int or None
        The sample index at which calibration was declared finished.
    """
    N = len(resp_signal)
    predicted_fraction = np.full(N, np.nan)
    predicted_phase = np.full(N, None, dtype=object)

    calibration_done = False
    calibration_end_idx = None

    avg_inhale = None
    avg_exhale = None

    for k in range(N):
        # partial array up to sample k (simulate real-time)
        partial_resp = resp_signal[: k + 1]

        # detect peaks & troughs on partial data
        partial_peaks = detect_resp_peaks(
            partial_resp, fs, method="scipy", height=peak_height, prominence=peak_prom
        )
        partial_troughs = detect_resp_peaks(
            -partial_resp, fs, method="scipy", height=peak_height, prominence=peak_prom
        )

        all_boundaries = np.sort(np.concatenate((partial_peaks, partial_troughs)))
        boundary_labels = []
        for idx in all_boundaries:
            if idx in partial_peaks:
                boundary_labels.append("peak")
            else:
                boundary_labels.append("trough")
        boundary_labels = np.array(boundary_labels, dtype=object)

        cyc = extract_cycles(all_boundaries, boundary_labels)

        # Check if we can declare calibration done
        if (
            (not calibration_done)
            and have_two_full_cycles(cyc)
            and (k / fs > min_calib_time)
        ):
            calibration_done = True
            calibration_end_idx = k

        if not calibration_done:
            # Not calibrated => fraction = NaN, phase = None
            continue

        # If we are calibrated, keep updating average inhale/exhale durations
        new_inh, new_exh = compute_avg_durations(cyc, fs)
        if new_inh is not None:
            avg_inhale = new_inh
        if new_exh is not None:
            avg_exhale = new_exh

        if avg_inhale is None or avg_exhale is None:
            # Still can't do it => skip
            continue

        if len(all_boundaries) == 0:
            continue

        last_boundary_idx = all_boundaries[-1]
        last_boundary_label = boundary_labels[-1]  # 'peak' or 'trough'

        # Decide phase
        if last_boundary_label == "trough":
            # inhalation
            predicted_phase[k] = True
            expected_end = last_boundary_idx + int(round(avg_inhale * fs))
            if expected_end > last_boundary_idx:
                frac_val = (
                    100.0 * (k - last_boundary_idx) / (expected_end - last_boundary_idx)
                )
            else:
                frac_val = 0
        else:
            # peak => exhalation
            predicted_phase[k] = False
            expected_end = last_boundary_idx + int(round(avg_exhale * fs))
            if expected_end > last_boundary_idx:
                frac_val = 100.0 * (
                    1.0 - (k - last_boundary_idx) / (expected_end - last_boundary_idx)
                )
            else:
                frac_val = 0

        frac_val = max(0, min(100, frac_val))
        predicted_fraction[k] = frac_val

    return predicted_fraction, predicted_phase, calibration_end_idx
