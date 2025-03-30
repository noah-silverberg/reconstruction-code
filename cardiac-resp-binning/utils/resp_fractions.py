# utils/resp_fractions.py

import numpy as np
from .ecg_resp import detect_resp_peaks


def extract_cycles(boundary_indices, boundary_labels):
    """
    Given sorted boundary indices + labels ('peak' or 'trough'),
    return a list of (start_idx, end_idx, 'inhalation'/'exhalation') cycles.
    """
    cycles = []
    for i in range(len(boundary_indices) - 1):
        s_idx = boundary_indices[i]
        e_idx = boundary_indices[i + 1]
        s_lab = boundary_labels[i]
        e_lab = boundary_labels[i + 1]
        if s_lab == "trough" and e_lab == "peak":
            cycles.append((s_idx, e_idx, "inhalation"))
        elif s_lab == "peak" and e_lab == "trough":
            cycles.append((s_idx, e_idx, "exhalation"))
    return cycles


def compute_avg_durations(cycles, fs):
    """
    For each cycle, measure length in seconds. Return (avg_inhale, avg_exhale).
    """
    inhalation_lengths = []
    exhalation_lengths = []
    for start_i, end_i, phase in cycles:
        if end_i > start_i:
            dur = (end_i - start_i) / fs
            if phase == "inhalation":
                inhalation_lengths.append(dur)
            else:
                exhalation_lengths.append(dur)

    avg_inh = np.mean(inhalation_lengths) if inhalation_lengths else None
    avg_exh = np.mean(exhalation_lengths) if exhalation_lengths else None
    return avg_inh, avg_exh


def have_two_full_cycles(cycles):
    """
    Check if we have at least 2 cycles (which you use as 'calibration done').
    """
    return len(cycles) >= 2


def build_true_fraction(signal_length, peaks, troughs):
    """
    Offline function for building 'actual fraction' array, purely for retrospective analysis.
    """
    all_extrema = np.concatenate((peaks, troughs))
    if len(all_extrema) == 0:
        return np.full(signal_length, np.nan)
    srt = np.sort(all_extrema)
    lbls = np.array([("peak" if x in peaks else "trough") for x in srt], dtype=object)

    cyc = extract_cycles(srt, lbls)
    frac_arr = np.full(signal_length, np.nan)

    for si, ei, ph in cyc:
        if ei <= si:
            continue
        length = ei - si
        for x in range(si, min(ei, signal_length)):
            if ph == "inhalation":
                frac = 100.0 * (x - si) / float(length)
            else:
                frac = 100.0 * (1.0 - (x - si) / float(length))
            frac_arr[x] = frac
    return frac_arr


def predict_fraction(
    resp_signal, fs, min_calib_time=10.0, peak_height=0.6, peak_prom=0.2
):
    """
    Main function to produce a prospective fraction for each sample in resp_signal.
    Returns:
      predicted_fraction (ndarray)
      predicted_phase    (ndarray of booleans: True=inhale, False=exhale)
      calibration_end_idx (int or None)
    """
    N = len(resp_signal)
    predicted_fraction = np.full(N, np.nan)
    predicted_phase = np.full(N, None, dtype=object)

    calibration_done = False
    calibration_end_idx = None
    avg_inhale = None
    avg_exhale = None

    for k in range(N):
        # partial data up to index k
        pd = resp_signal[: k + 1]
        pks = detect_resp_peaks(
            pd, fs, method="scipy", height=peak_height, prominence=peak_prom
        )
        trs = detect_resp_peaks(
            -pd, fs, method="scipy", height=peak_height, prominence=peak_prom
        )

        bds = np.sort(np.concatenate((pks, trs)))
        lbs = np.array([("peak" if b in pks else "trough") for b in bds], dtype=object)

        cyc = extract_cycles(bds, lbs)

        # see if calibration is done
        if (
            (not calibration_done)
            and have_two_full_cycles(cyc)
            and (k / fs > min_calib_time)
        ):
            calibration_done = True
            calibration_end_idx = k

        if not calibration_done:
            continue

        # update avg_inhale/exhale
        new_inh, new_exh = compute_avg_durations(cyc, fs)
        if new_inh is not None:
            avg_inhale = new_inh
        if new_exh is not None:
            avg_exhale = new_exh

        if avg_inhale is None or avg_exhale is None:
            continue

        if len(bds) == 0:
            continue

        # last boundary => define inhalation or exhalation
        last_boundary_idx = bds[-1]
        last_boundary_label = lbs[-1]  # 'peak' or 'trough'

        if last_boundary_label == "trough":
            # inhalation
            predicted_phase[k] = True
            exp_end = last_boundary_idx + int(round(avg_inhale * fs))
            if exp_end > last_boundary_idx:
                frac = 100.0 * (k - last_boundary_idx) / (exp_end - last_boundary_idx)
            else:
                frac = 0
        else:
            # 'peak'
            predicted_phase[k] = False
            exp_end = last_boundary_idx + int(round(avg_exhale * fs))
            if exp_end > last_boundary_idx:
                frac = 100.0 * (
                    1.0 - (k - last_boundary_idx) / (exp_end - last_boundary_idx)
                )
            else:
                frac = 0

        # clamp fraction
        frac = max(0, min(100, frac))
        predicted_fraction[k] = frac

    return predicted_fraction, predicted_phase, calibration_end_idx
