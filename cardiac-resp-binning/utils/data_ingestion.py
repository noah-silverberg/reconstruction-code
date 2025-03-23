"""
data_ingestion.py

Provides functions to read raw scan data (TWIX files) and extract basic parameters
(like frame rate and number of phase-encode lines from DICOM files).
"""

import os
import pydicom
import numpy as np
import twixtools


def get_dicom_framerate(folder_path):
    """
    Extract the frame rate and frame time from the first DICOM file in the given folder.

    This tries a few common DICOM fields:
      - FrameTime
      - CineRate
      - AcquisitionTime (fallback)

    Parameters
    ----------
    folder_path : str
        Path to the folder containing DICOM (.dcm) files.

    Returns
    -------
    tuple
        (framerate, frame_time). Both are floats. If not found, (None, None).
    """
    dicom_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(".dcm")
    ]
    if not dicom_files:
        return None, None

    dicoms = [pydicom.dcmread(f) for f in dicom_files]
    # Sort by instance to pick a consistent one (often the first slice)
    dicoms.sort(key=lambda ds: int(ds.InstanceNumber))

    ds = dicoms[0]
    # Try FrameTime (0018,1062)
    if hasattr(ds, "FrameTime"):
        frame_time = float(ds.FrameTime) / 1000.0  # convert ms to s
        return 1.0 / frame_time, frame_time

    # Try CineRate (0018,0040)
    if hasattr(ds, "CineRate"):
        frate = float(ds.CineRate)
        return frate, 1.0 / frate

    # Fallback: use differences in AcquisitionTime across slices
    if hasattr(ds, "AcquisitionTime"):

        def time_to_seconds(t):
            return int(t[:2]) * 3600 + int(t[2:4]) * 60 + float(t[4:])

        times = [time_to_seconds(d.AcquisitionTime) for d in dicoms]
        diffs = np.diff(times)
        if len(diffs) > 0:
            avg_diff = np.mean(diffs)
            if avg_diff != 0:
                return 1.0 / avg_diff, avg_diff
    return None, None


def print_all_dicom_info(path):
    """
    Print the metadata from a DICOM file or the first file in a folder.

    Parameters
    ----------
    path : str
        File or folder path to DICOM.
    """
    if os.path.isdir(path):
        dicom_files = [
            os.path.join(path, f)
            for f in os.listdir(path)
            if f.lower().endswith(".dcm")
        ]
        if not dicom_files:
            print("No DICOM files found.")
            return
        file = dicom_files[0]
    else:
        file = path

    ds = pydicom.dcmread(file)
    print(f"Metadata for {file}:\n{ds}")


def get_total_phase_encodes(folder_path):
    """
    Determine the total number of phase-encode lines from DICOM files in a folder.

    Tries to read the "AcquisitionMatrix" (0018,1310). If unavailable,
    falls back to "Rows" (0028,0010).

    Parameters
    ----------
    folder_path : str
        Path to the folder containing DICOM files.

    Returns
    -------
    int or None
        The total number of phase encodes, if found. Otherwise None.
    """
    dicom_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(".dcm")
    ]
    if not dicom_files:
        print("No DICOM files found in the specified folder.")
        return None

    try:
        ds = pydicom.dcmread(dicom_files[0])
    except Exception as e:
        print(f"Error reading DICOM file: {e}")
        return None

    if hasattr(ds, "AcquisitionMatrix"):
        acq_matrix = ds.AcquisitionMatrix
        if isinstance(acq_matrix, (list, tuple)) and len(acq_matrix) >= 2:
            total_phase_encodes = int(acq_matrix[1])
            print(f"Total phase encodes from AcquisitionMatrix: {total_phase_encodes}")
            return total_phase_encodes

    if hasattr(ds, "Rows"):
        total_phase_encodes = int(ds.Rows)
        print(f"Total phase encodes from Rows attribute: {total_phase_encodes}")
        return total_phase_encodes

    print("Unable to determine total phase encodes from DICOM metadata.")
    return None


def get_num_frames(folder_path):
    """
    Count the number of DICOM frames (files) in the given folder.

    Parameters
    ----------
    folder_path : str
        Path to the folder containing .dcm files.

    Returns
    -------
    int
        Number of DICOM frames found. 0 if none.
    """
    dicom_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(".dcm")
    ]
    num_frames = len(dicom_files)
    print(f"Found {num_frames} DICOM frames in folder {folder_path}")
    return num_frames


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
    Read a Siemens TWIX (.dat) file using twixtools.

    Parameters
    ----------
    file_path : str
        Path to the TWIX file.
    include_scans : list of int, optional
        Specific scan numbers to include. Default is None (includes all).
    parse_pmu : bool
        Whether to parse PMU data.
    parse_prot : bool
        Whether to parse protocol.
    parse_data : bool
        Whether to parse the main MRI data.
    parse_geometry : bool
        Whether to parse geometry information.
    verbose : bool
        Print additional info during parsing.
    keep_acqend : bool
        Keep ACQEND packets in the data.
    keep_syncdata : bool
        Keep SYNCDATA packets in the data.

    Returns
    -------
    list
        A list of scan dictionaries from twixtools.
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
    Extract image (k-space) data from a list of TWIX scan dictionaries.

    Parameters
    ----------
    scans : list of dict
        Output from `read_twix_file`.

    Returns
    -------
    np.ndarray
        Complex k-space data of shape (phase_encodes, coils, freq_encodes).
        If no data is found, returns an empty array.
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
    Extract ICE parameter data (e.g., ECG or other auxiliary signals) from the specified scan segment.

    Parameters
    ----------
    scans : list of dict
        Output from `read_twix_file`.
    segment_index : int
        Index of the scan segment to use.
    columns : slice or list of int, optional
        Columns to extract from the ICE parameter array.

    Returns
    -------
    np.ndarray
        ICE parameter data. If none found, returns an empty array.
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
