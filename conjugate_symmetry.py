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
    crop_lower=8,
    crop_upper=136,
    full_target_readout=128,
    phase_offset=32,
    return_kspace=False,
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
      return_kspace : if True, also return the combined k-space magnitude

    Returns:
      img : reconstructed 2D magnitude image (normalized)
      [kspace] : (optional) combined k-space magnitude (root-sum-of-squares over channels)
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

    Np, Nx = full_phase, full_target_readout

    # 2) Compute the indices of the acquired rows.
    acquired_indices = [(mdb.cLin + phase_offset) % Np for mdb in mdb_list]

    # 3) Fill missing k-space points ONLY outside the acquired region.
    for ch in range(n_channels):
        for i in range(Np):
            if i in acquired_indices:
                continue  # This row was acquired – leave it untouched.
            for j in range(Nx):
                # Fill using the conjugate symmetric point.
                kspace_full[i, j, ch] = np.conjugate(kspace_full[-i, -j, ch])

    # 4) For each channel, perform the 2D inverse FFT.
    imgs = []
    for ch in range(n_channels):
        k_ch = kspace_full[:, :, ch]
        img_ch = np.fft.ifft2(k_ch)
        img_ch = np.fft.fftshift(img_ch)
        imgs.append(img_ch)

    # 5) Combine channels using root-sum-of-squares.
    img_combined = np.sqrt(np.sum(np.abs(np.array(imgs)) ** 2, axis=0))
    img_combined = img_combined / np.max(img_combined)

    if return_kspace:
        kspace_combined = np.sqrt(np.sum(np.abs(kspace_full) ** 2, axis=2))
        return img_combined, kspace_combined
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
    kspace_frames = []
    for rep in frame_numbers:
        mdb_list = frames_dict[rep]
        img, ksp = reconstruct_frame(mdb_list, return_kspace=True)
        recon_frames.append(img)
        kspace_frames.append(ksp)

    gif_frames = [(img * 255).astype(np.uint8) for img in recon_frames]
    imageio.mimsave("conjugate_symmetry_cine.gif", gif_frames, duration=0.125)
    print("Saved reconstructed cine as conjugate_symmetry_cine.gif")

    # Create k-space GIF with center grid lines.
    kspace_gif = []
    for ksp in kspace_frames:
        # Log-scale for better dynamic range.
        disp = np.log(1 + np.abs(ksp))
        disp = (disp - disp.min()) / (disp.max() - disp.min())
        disp_uint8 = (disp * 255).astype(np.uint8)
        # Convert to RGB.
        disp_rgb = np.stack([disp_uint8] * 3, axis=-1)
        h, w, _ = disp_rgb.shape
        cx, cy = w // 2, h // 2
        # Draw vertical center line.
        disp_rgb[:, cx, :] = [255, 0, 0]
        # Draw horizontal center line.
        disp_rgb[cy, :, :] = [255, 0, 0]
        kspace_gif.append(disp_rgb)

    imageio.mimsave("conjugate_symmetry_kspace.gif", kspace_gif, duration=0.125)
    print("Saved filled k-space GIF as conjugate_symmetry_kspace.gif")


if __name__ == "__main__":
    main()
