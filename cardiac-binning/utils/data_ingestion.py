"""
data_ingestion.py

This module provides functions to read raw scan data (TWIX files) and to
extract basic parameters (like frame rate from DICOM files) needed for the pipeline.
"""

import os
import pydicom
import numpy as np
import twixtools


def get_dicom_framerate(folder_path):
    """
    Extract the frame rate and frame time from DICOM files in the given folder.

    Parameters:
        folder_path (str): Path to the folder containing DICOM (.dcm) files.

    Returns:
        tuple: (framerate in Hz, frame_time in seconds) or (None, None) if not found.
    """
    dicom_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(".dcm")
    ]
    if not dicom_files:
        return None, None
    dicoms = [pydicom.dcmread(f) for f in dicom_files]
    dicoms.sort(key=lambda ds: int(ds.InstanceNumber))
    if hasattr(dicoms[0], "FrameTime"):
        frame_time = float(dicoms[0].FrameTime) / 1000.0
        return 1.0 / frame_time, frame_time
    elif hasattr(dicoms[0], "CineRate"):
        framerate = float(dicoms[0].CineRate)
        return framerate, 1.0 / framerate
    elif hasattr(dicoms[0], "AcquisitionTime"):

        def time_to_seconds(t):
            return int(t[:2]) * 3600 + int(t[2:4]) * 60 + float(t[4:])

        times = [time_to_seconds(ds.AcquisitionTime) for ds in dicoms]
        diffs = np.diff(times)
        avg_diff = np.mean(diffs)
        return 1.0 / avg_diff, avg_diff
    else:
        return None, None


def read_twix_file(
    file_path,
    include_scans=None,
    parse_pmu=True,
    parse_prot=True,
    parse_data=True,
    parse_geometry=True,
    verbose=True,
    keep_acqend=False,
    keep_syncdata=True,
):
    """
    Read a Siemens TWIX file using twixtools.

    Parameters:
        file_path (str): Path to the TWIX (.dat) file.
        include_scans (list): List of scan numbers to include.
        Other parameters control parsing details.

    Returns:
        list: A list of scan dictionaries.
    """
    scans = twixtools.read_twix(
        file_path,
        include_scans=include_scans,
        parse_prot=parse_prot,
        parse_data=parse_data,
        parse_pmu=parse_pmu,
        parse_geometry=parse_geometry,
        verbose=verbose,
        keep_acqend=keep_acqend,
        keep_syncdata=keep_syncdata,
    )
    print(f"Read {len(scans)} scans from {file_path}")
    return scans


def extract_image_data(scans):
    """
    Extract and stack image (k-space) data from the scans.

    Parameters:
        scans (list): List of scan dictionaries.

    Returns:
        np.ndarray: Complex k-space data with shape (phase_encodes, coils, freq_encodes).
    """
    image_blocks = []
    for scan in scans:
        if "mdb" not in scan:
            continue
        for mdb in scan["mdb"]:
            if mdb.is_image_scan():
                data = np.array(mdb.data, copy=True)
                image_blocks.append(data)
    if not image_blocks:
        return np.array([])
    out = np.stack(image_blocks, axis=0)
    print(f"Extracted image data shape: {out.shape}")
    return out


def extract_iceparam_data(scans, segment_index=0, columns=None):
    """
    Extract ICE parameter data (e.g. ECG) from image scans in a specified segment.

    Parameters:
        scans (list): List of scan dictionaries.
        segment_index (int): Which scan segment to use.
        columns: Columns to extract (e.g., np.s_[18:24]).

    Returns:
        np.ndarray: 2D array of ICE parameters.
    """
    if segment_index < 0 or segment_index >= len(scans):
        print("Invalid segment index.")
        return np.array([])
    scan = scans[segment_index]
    ice_list = []
    if "mdb" not in scan:
        print("No mdb blocks found in segment.")
        return np.array([])
    for mdb in scan["mdb"]:
        if mdb.is_image_scan() and hasattr(mdb, "IceProgramPara"):
            ice_arr = np.array(mdb.IceProgramPara, copy=True)
            ice_list.append(ice_arr)
    if not ice_list:
        print("No ICE parameter data found.")
        return np.array([])
    ice_full = np.vstack(ice_list)
    if columns is not None:
        ice_full = ice_full[:, columns]
    print(f"Extracted ICE parameter data shape: {ice_full.shape}")
    return ice_full
