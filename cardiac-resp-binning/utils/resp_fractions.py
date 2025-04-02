"""
utils/resp_fractions.py

Functions for processing respiratory signals: extracting cycles,
computing average durations, building offline respiratory fractions,
and simulating prospective fraction prediction for respiratory binning.
"""

import numpy as np
from tqdm import tqdm
from .ecg_resp import detect_resp_peaks


def extract_cycles(boundary_indices, boundary_labels):
    """
    Extract inhalation and exhalation cycles based on sorted extrema.

    Parameters
    ----------
    boundary_indices : array-like
        Sorted indices of detected peaks/troughs.
    boundary_labels : array-like
        Corresponding labels ('peak' or 'trough') for each index.

    Returns
    -------
    list
        List of tuples: (start_idx, end_idx, phase), where phase is 'inhalation' or 'exhalation'.
    """
    cycles = []
    for i in range(len(boundary_indices) - 1):
        s_idx = boundary_indices[i]
        e_idx = boundary_indices[i + 1]
        s_label = boundary_labels[i]
        e_label = boundary_labels[i + 1]
        if s_label == "trough" and e_label == "peak":
            cycles.append((s_idx, e_idx, "inhalation"))
        elif s_label == "peak" and e_label == "trough":
            cycles.append((s_idx, e_idx, "exhalation"))
    return cycles


def compute_avg_durations(cycles, fs):
    """
    Compute average durations (in seconds) for inhalation and exhalation cycles.

    Parameters
    ----------
    cycles : list of tuples
        Each tuple is (start_idx, end_idx, phase).
    fs : float
        Sampling frequency.

    Returns
    -------
    tuple
        (avg_inhale, avg_exhale), or (None, None) if not available.
    """
    inhale_durations = []
    exhale_durations = []
    for start, end, phase in cycles:
        if end > start:
            duration = (end - start) / fs
            if phase == "inhalation":
                inhale_durations.append(duration)
            else:
                exhale_durations.append(duration)
    avg_inhale = np.mean(inhale_durations) if inhale_durations else None
    avg_exhale = np.mean(exhale_durations) if exhale_durations else None
    return (avg_inhale, avg_exhale)


def have_two_full_cycles(cycles):
    """
    Check if there are at least two full respiratory cycles.

    Parameters
    ----------
    cycles : list
        List of cycles.

    Returns
    -------
    bool
        True if at least two cycles exist.
    """
    return len(cycles) >= 2


def build_true_fraction(signal_length, peaks, troughs):
    """
    Build the offline respiratory fraction for each sample.

    For each cycle:
      - Inhalation (trough -> peak) ramps from 0 to 100.
      - Exhalation (peak -> trough) ramps from 100 to 0.

    Parameters
    ----------
    signal_length : int
        Total number of samples.
    peaks : array-like
        Indices of detected peaks.
    troughs : array-like
        Indices of detected troughs.

    Returns
    -------
    np.ndarray
        Array of fractions (0-100) with NaN where undefined.
    """
    all_extrema = np.concatenate((peaks, troughs))
    if len(all_extrema) == 0:
        return np.full(signal_length, np.nan)
    sorted_idx = np.argsort(all_extrema)
    sorted_indices = all_extrema[sorted_idx]
    sorted_labels = [("peak" if idx in peaks else "trough") for idx in sorted_indices]
    cycles = extract_cycles(sorted_indices, np.array(sorted_labels, dtype=object))
    true_fraction = np.full(signal_length, np.nan)
    for start, end, phase in cycles:
        length = end - start
        for idx in range(start, min(end, signal_length)):
            if phase == "inhalation":
                frac = 100.0 * (idx - start) / length
            else:
                frac = 100.0 * (1.0 - (idx - start) / length)
            true_fraction[idx] = frac
    return true_fraction


def predict_fraction(
    resp_signal, fs, min_calib_time=10.0, peak_height=0.6, peak_prom=0.2
):
    """
    Simulate prospective respiratory fraction prediction.

    For each time step, detect partial peaks/troughs and, after sufficient calibration,
    compute the respiratory fraction and phase.

    Parameters
    ----------
    resp_signal : np.ndarray
        1D respiration signal.
    fs : float
        Sampling frequency.
    min_calib_time : float, optional
        Minimum calibration time (seconds).
    peak_height : float, optional
        Minimum peak height for detection.
    peak_prom : float, optional
        Minimum peak prominence for detection.

    Returns
    -------
    tuple
        (predicted_fraction, predicted_phase, calibration_end_idx)
    """
    N = len(resp_signal)
    predicted_fraction = np.full(N, np.nan)
    predicted_phase = np.full(N, np.nan, dtype=bool)
    calibration_done = False
    calibration_end_idx = None
    avg_inhale, avg_exhale = None, None

    for k in range(N):
        partial_signal = resp_signal[: k + 1]
        partial_peaks = detect_resp_peaks(
            partial_signal, fs, method="scipy", height=peak_height, prominence=peak_prom
        )
        partial_troughs = detect_resp_peaks(
            -partial_signal,
            fs,
            method="scipy",
            height=peak_height,
            prominence=peak_prom,
        )
        all_boundaries = np.sort(np.concatenate((partial_peaks, partial_troughs)))
        boundary_labels = [
            ("peak" if idx in partial_peaks else "trough") for idx in all_boundaries
        ]
        cycles = extract_cycles(all_boundaries, np.array(boundary_labels, dtype=object))
        if (
            (not calibration_done)
            and have_two_full_cycles(cycles)
            and (k / fs > min_calib_time)
        ):
            calibration_done = True
            calibration_end_idx = k
        if not calibration_done:
            continue
        new_inh, new_exh = compute_avg_durations(cycles, fs)
        if new_inh is not None:
            avg_inhale = new_inh
        if new_exh is not None:
            avg_exhale = new_exh
        if avg_inhale is None or avg_exhale is None or len(all_boundaries) == 0:
            continue
        last_boundary_idx = all_boundaries[-1]
        last_label = boundary_labels[-1]
        if last_label == "trough":
            predicted_phase[k] = True  # inhalation
            expected_end = last_boundary_idx + int(round(avg_inhale * fs))
            frac_val = (
                100.0 * (k - last_boundary_idx) / (expected_end - last_boundary_idx)
                if expected_end > last_boundary_idx
                else 0
            )
        else:
            predicted_phase[k] = False  # exhalation
            expected_end = last_boundary_idx + int(round(avg_exhale * fs))
            frac_val = (
                100.0
                * (1.0 - (k - last_boundary_idx) / (expected_end - last_boundary_idx))
                if expected_end > last_boundary_idx
                else 0
            )
        predicted_fraction[k] = max(0, min(100, frac_val))
    return predicted_fraction, predicted_phase, calibration_end_idx


def predict_cardiac_fraction(
    ecg_signal, fs, min_calib_time=5.0, peak_height=0.5, peak_prom=0.2
):
    """
    Simulate prospective cardiac fraction prediction from an ECG signal.

    Parameters
    ----------
    ecg_signal : np.ndarray
        1D ECG signal.
    fs : float
        Sampling frequency.
    min_calib_time : float, optional
        Minimum calibration time (seconds).
    peak_height : float, optional
        Minimum R-peak height.
    peak_prom : float, optional
        Minimum R-peak prominence.

    Returns
    -------
    tuple
        (cardiac_fraction, calibration_end_idx)
    """
    from .ecg_resp import detect_r_peaks

    N = len(ecg_signal)
    cardiac_fraction = np.full(N, np.nan)
    r_peaks = []
    calibration_done = False
    calibration_end_idx = None

    def estimate_next_rpeak(r_peaks_local, fs):
        if len(r_peaks_local) < 2:
            return None
        intervals = np.diff(r_peaks_local[-3:])
        avg_interval = np.mean(intervals) if len(intervals) > 0 else None
        return (
            r_peaks_local[-1] + avg_interval
            if (avg_interval is not None and avg_interval >= 1)
            else None
        )

    for k in tqdm(range(N), desc="Processing ECG Signal"):
        if k / fs < min_calib_time:
            continue
        partial_ecg = ecg_signal[: k + 1]
        r_peaks_list = detect_r_peaks(partial_ecg.reshape(-1, 1), fs)
        r_peaks = r_peaks_list[0] if len(r_peaks_list) > 0 else []
        if (not calibration_done) and (len(r_peaks) >= 3):
            calibration_done = True
            calibration_end_idx = k
        if not calibration_done or len(r_peaks) < 2:
            continue
        behind = r_peaks[r_peaks <= k]
        if len(behind) == 0:
            continue
        last_r = behind[-1]
        ahead = r_peaks[r_peaks > k]
        next_r = ahead[0] if len(ahead) > 0 else estimate_next_rpeak(r_peaks, fs)
        if (next_r is None) or (next_r <= last_r):
            continue
        cycle_length = next_r - last_r
        frac_val = 100.0 * (k - last_r) / cycle_length
        cardiac_fraction[k] = max(0, min(100, frac_val))
    return cardiac_fraction, calibration_end_idx
