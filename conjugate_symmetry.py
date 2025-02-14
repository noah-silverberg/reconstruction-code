#!/usr/bin/env python3
import twixtools
import numpy as np
import imageio
import matplotlib.pyplot as plt
from collections import defaultdict


def reconstruct_frame(
    mdb_list,
    full_phase=128,
    full_readout=200,
    crop_lower=64,
    crop_upper=192,
    full_target_readout=128,
    phase_offset=12,
):
    """
    Reconstruct one frame from a list of mdb blocks and fill in missing k-space
    points using conjugate symmetry, but only outside the acquired region.

    The acquired region is automatically determined from the measured lines.

    Parameters:
      mdb_list : list of mdb blocks for one frame (assumed image scans)
      full_phase : number of phase-encoding lines in final k-space (e.g. 128)
      full_readout : raw readout length (200)
      crop_lower, crop_upper : indices to crop raw readout (to 128 samples)
      full_target_readout : final readout dimension after cropping (128)
      phase_offset : offset to place measured phase lines (e.g. measured line + 12)
      threshold : below this value a k-space point is considered missing

    Returns:
      img : reconstructed 2D magnitude image (normalized)
    """
    # 1) Insert acquired data into a full k-space array
    sample_data = np.array(mdb_list[0].data)
    n_channels = sample_data.shape[0]
    kspace_full = np.zeros(
        (full_phase, full_target_readout, n_channels), dtype=np.complex64
    )

    for mdb in mdb_list:
        data = np.array(mdb.data)  # shape: (channels, full_readout)
        data_cropped = data[:, crop_lower:crop_upper]  # shape: (channels, 128)
        measured_line = mdb.cLin
        full_line = measured_line + phase_offset  # maps measured line into full k-space
        if np.any(kspace_full[full_line]):
            kspace_full[full_line] += data_cropped.T  # transpose to (readout, channels)
            kspace_full[full_line] /= 2.0
        else:
            kspace_full[full_line] = data_cropped.T

    # 2) Shift k-space so that the DC component is centered.
    kspace_shifted = np.fft.fftshift(kspace_full, axes=(0, 1))
    Np, Nx = full_phase, full_target_readout

    # 3) Automatically compute the acquired region in shifted coordinates.
    #    The unshifted acquired region is from phase_offset to phase_offset + acquired_lines - 1.
    acquired_lines = len(mdb_list)
    acq_start = (phase_offset + Np // 2) % Np
    acq_end = (phase_offset + acquired_lines - 1 + Np // 2) % Np

    # 4) Fill missing k-space points ONLY outside the acquired region.
    for ch in range(n_channels):
        for i in range(Np):
            if acq_start <= i <= acq_end:
                continue  # Skip rows in the acquired region.
            for j in range(Nx):
                kspace_shifted[i, j, ch] = np.conjugate(kspace_shifted[-i, -j, ch])

    # 5) Inverse shift to restore original k-space ordering.
    kspace_filled = np.fft.ifftshift(kspace_shifted, axes=(0, 1))

    # 6) For each channel, perform the 2D inverse FFT.
    imgs = []
    for ch in range(n_channels):
        k_ch = kspace_filled[:, :, ch]
        k_ch_shifted = np.fft.ifftshift(k_ch)
        img_ch = np.fft.ifft2(k_ch_shifted)
        img_ch = np.fft.fftshift(img_ch)
        imgs.append(img_ch)

    # 7) Combine channels using root-sum-of-squares.
    img_combined = np.sqrt(np.sum(np.abs(np.array(imgs)) ** 2, axis=0))
    img_combined = img_combined / np.max(img_combined)
    return img_combined


def main():
    filename = "Cine.dat"
    twix = twixtools.read_twix(filename)
    meas = twix[-1]

    image_mdbs = []
    for mdb in meas["mdb"]:
        try:
            if mdb.is_image_scan():
                image_mdbs.append(mdb)
        except Exception:
            continue

    frames_dict = defaultdict(list)
    for mdb in image_mdbs:
        frames_dict[mdb.cRep].append(mdb)

    frame_numbers = sorted(frames_dict.keys())
    print("Reconstructing frames for frame numbers:", frame_numbers)

    recon_frames = []
    for rep in frame_numbers:
        mdb_list = frames_dict[rep]
        img = reconstruct_frame(mdb_list)
        recon_frames.append(img)

    gif_frames = []
    for img in recon_frames:
        img_uint8 = (img * 255).astype(np.uint8)
        gif_frames.append(img_uint8)

    output_gif = "cine_conjugate_symmetry.gif"
    imageio.mimsave(output_gif, gif_frames, duration=0.125)
    print(f"Saved reconstructed cine as {output_gif}")


if __name__ == "__main__":
    main()
