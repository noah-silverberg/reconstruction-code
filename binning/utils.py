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
    Read a Siemens .dat (TWIX) file using twixtools and return a list of scan dictionaries.

    This function directly interfaces with the twixtools.read_twix API.

    Parameters
    ----------
    file_path : str
        Path to the Siemens TWIX (.dat) file.
    include_scans : list or None, optional
        List of scan numbers to include. If None, all scans are parsed.
    parse_prot : bool, optional
        Parse protocol information (default: True).
    parse_data : bool, optional
        Parse the measurement data blocks (default: True).
    parse_pmu : bool, optional
        Parse physiological (PMU) data (default: True).
    parse_geometry : bool, optional
        Parse geometry information (default: True).
    verbose : bool, optional
        Print progress and messages (default: True).
    keep_acqend : bool, optional
        Include ACQEND blocks in the output (default: False).
    keep_syncdata : bool, optional
        Include syncdata blocks in the output (default: True).

    Returns
    -------
    list
        A list of scan dictionaries as returned by twixtools.read_twix().
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

    print(f"Successfully read {len(scans)} scans from {file_path}.")
    return scans


def show_file_overview(scans):
    """
    Print a summary overview for each scan (segment).

    For each scan, this prints:
      - The top-level keys available.
      - The total number of mdb blocks.
      - The number of mdb blocks that are image scans.
      - The data shape of the first image mdb block (if available).

    Parameters
    ----------
    scans : list
        A list of scan dictionaries returned by read_twix_file().

    Returns
    -------
    None
    """
    for seg_index, scan in enumerate(scans):
        print(f"\n=== Scan/Segment {seg_index} ===")
        print("Scan keys:", list(scan.keys()))
        if "mdb" in scan:
            total_mdb = len(scan["mdb"])
            image_mdbs = [mdb for mdb in scan["mdb"] if mdb.is_image_scan()]
            num_image = len(image_mdbs)
            print(f"Total mdb blocks: {total_mdb}")
            print(f"Image mdb blocks: {num_image}")
            if num_image > 0:
                try:
                    sample_data = np.array(image_mdbs[0].data)
                    print(f"First image mdb data shape: {sample_data.shape}")
                except Exception as exc:
                    print(f"Could not retrieve shape from first image mdb: {exc}")
        print("-------")


def extract_image_data(scans):
    """
    Extract image data from all mdb blocks in all scans that are identified as image scans.

    Tthe image mdb blocks are stacked so that the output array has shape:
      (phase_encodes, coils, frequency_encodes)

    Parameters
    ----------
    scans : list
        A list of scan dictionaries (from read_twix_file).

    Returns
    -------
    np.ndarray
        A complex NumPy array of shape (phase_encodes, coils, frequency_encodes).
        If no image mdb blocks are found, returns an empty array.
    """
    image_blocks = []
    for scan in scans:
        if "mdb" not in scan:
            continue
        for mdb in scan["mdb"]:
            if mdb.is_image_scan():
                # mdb.data is assumed to have shape (coils, freq)
                data = np.array(mdb.data, copy=True)
                image_blocks.append(data)
    if not image_blocks:
        return np.array([])
    # Stack along a new axis to preserve the coil dimension.
    out = np.stack(image_blocks, axis=0)

    print(f"Extracted image data shape: {out.shape}")

    return out


def extract_iceparam_data(scans, segment_index=0, columns=None):
    """
    Extract ICE parameter data only from image mdb blocks in a given scan (segment).

    The ICE parameters are assumed to be stored in each mdb as IceProgramPara.
    The resulting ICE parameter arrays (one per image mdb) are stacked along axis 0.

    Parameters
    ----------
    scans : list
        A list of scan dictionaries (from read_twix_file).
    segment_index : int, optional
        Which scan (segment) to extract from. Defaults to 0.
    columns : slice or array-like, optional
        Which columns to extract from the ICE parameter array.
        For example, np.s_[18:24] to get columns 18..23.
        If None, returns all columns.

    Returns
    -------
    np.ndarray
        A 2D array (rows: one per image mdb block, columns: ICE parameters),
        or an empty array if no ICE parameters are found.
    """
    if segment_index < 0 or segment_index >= len(scans):
        print(f"Invalid segment_index {segment_index}. Total segments: {len(scans)}.")
        return np.array([])

    scan = scans[segment_index]
    ice_list = []
    if "mdb" not in scan:
        print(f"No mdb blocks found in segment {segment_index}.")
        return np.array([])

    for mdb in scan["mdb"]:
        if mdb.is_image_scan() and hasattr(mdb, "IceProgramPara"):
            ice_arr = np.array(mdb.IceProgramPara, copy=True)
            ice_list.append(ice_arr)
    if not ice_list:
        print(f"No ICE parameter data found in segment {segment_index}.")
        return np.array([])

    # Stack ICE parameter arrays along axis 0.
    ice_full = np.vstack(ice_list)
    if columns is not None:
        ice_full = ice_full[:, columns]

    print(f"Extracted ICE parameter data shape: {ice_full.shape}")

    return ice_full
