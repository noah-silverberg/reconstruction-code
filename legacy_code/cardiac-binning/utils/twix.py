"""
twix.py

This module provides functions to read Siemens TWIX (.dat) files and extract
image and ICE parameter data.
"""

import numpy as np
import twixtools


def read_twix_file(
    file_path,
    include_scans=None,
    parse_prot=True,
    parse_data=True,
    parse_pmu=True,
    parse_geometry=True,
    verbose=True,
    keep_acqend=False,
    keep_syncdata=True,
):
    """
    Read a Siemens TWIX file using twixtools.

    Parameters:
        file_path (str): Path to the TWIX (.dat) file.
        include_scans (list): Scan numbers to include.
        Other parameters control parsing details.

    Returns:
        list: List of scan dictionaries.
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


def show_file_overview(scans):
    """
    Print a summary overview of the scans.
    """
    for idx, scan in enumerate(scans):
        print(f"\n=== Scan {idx} ===")
        print("Keys:", list(scan.keys()))
        if "mdb" in scan:
            total = len(scan["mdb"])
            image_mdbs = [mdb for mdb in scan["mdb"] if mdb.is_image_scan()]
            print(f"Total mdb blocks: {total}, Image mdb blocks: {len(image_mdbs)}")
            if image_mdbs:
                try:
                    sample = np.array(image_mdbs[0].data)
                    print(f"First image mdb data shape: {sample.shape}")
                except Exception as e:
                    print("Error reading first mdb shape:", e)
        print("-------")


def extract_image_data(scans):
    """
    Extract image (k-space) data from the scans.

    Returns:
        np.ndarray: Complex k-space data.
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
    Extract ICE parameter data from a given scan segment.

    Returns:
        np.ndarray: ICE parameter data.
    """
    if segment_index < 0 or segment_index >= len(scans):
        print("Invalid segment index.")
        return np.array([])
    scan = scans[segment_index]
    ice_list = []
    if "mdb" not in scan:
        print("No mdb blocks found.")
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
