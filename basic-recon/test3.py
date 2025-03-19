#!/usr/bin/env python3
import twixtools
import numpy as np
import imageio
from collections import defaultdict


def homodyne_reconstruct_frame(
    mdb_list,
    full_phase=128,
    full_readout=200,
    crop_lower=8,
    crop_upper=136,
    full_target_readout=128,
    phase_offset=32,
    acquired_lines=40,
    return_kspace=False,
):
    """
    Reconstruct one frame using a basic homodyne method.

    Parameters:
      mdb_list : list of mdb blocks belonging to one frame (assumed image scans)
      full_phase : final number of phase encoding lines in k-space (e.g. 128)
      full_readout : original readout length (e.g. 200)
      crop_lower, crop_upper : indices for cropping the readout dimension (to 128)
      full_target_readout : final readout size (128)
      phase_offset : offset to place measured phase lines into full k-space
      acquired_lines : number of measured phase lines (e.g. 40)
      return_kspace : if True, also return the combined k-space magnitude

    Returns:
      img : reconstructed 2D image (normalized) using homodyne reconstruction
      [kspace] : (optional) combined k-space magnitude (root-sum-of-squares over channels)
    """
    # Determine number of channels from first mdb block
    sample_data = np.array(mdb_list[0].data)
    n_channels = sample_data.shape[0]

    # Create a full k-space array (zeros for missing data)
    kspace_full = np.zeros(
        (full_phase, full_target_readout, n_channels), dtype=np.complex64
    )

    # Insert each mdb block's cropped data into its proper k-space row.
    for mdb in mdb_list:
        data = np.array(mdb.data)  # shape: (channels, full_readout)
        data_cropped = data[:, crop_lower:crop_upper]  # shape: (channels, 128)
        measured_line = mdb.cLin  # assumed to be 0 to (acquired_lines-1)
        full_line = measured_line + phase_offset
        # If a line is already filled (e.g. if there are multiple entries for the same line), raise an error.
        if np.any(kspace_full[full_line]):
            raise ValueError(f"Duplicate data for line {full_line}")
        else:
            kspace_full[full_line] = data_cropped.T

    # Reconstruct each channel with homodyne correction
    imgs = []
    for ch in range(n_channels):
        k_ch = kspace_full[:, :, ch]
        # Standard IFFT reconstruction of the full (zero-filled) k-space:
        img_full = np.fft.fftshift(np.fft.ifft2(k_ch))

        # Create a low-pass filter to extract the phase information.
        # Here we use a simple rectangular window covering the acquired k-space region.
        # TODO: is this the low pass filter we want?
        lowpass_filter = np.zeros_like(k_ch)
        lowpass_filter[phase_offset : phase_offset + acquired_lines, :] = 1.0

        k_lowpass = k_ch * lowpass_filter
        img_lowres = np.fft.fftshift(np.fft.ifft2(k_lowpass))

        # Estimate the phase from the low-resolution image
        phase_est = np.angle(img_lowres)

        # Correct the full image phase by multiplying by exp(-i*phase_est)
        img_corrected = img_full * np.exp(-1j * phase_est)
        # Take the real part as the homodyne reconstructed image
        img_homodyne = np.real(img_corrected)
        imgs.append(img_homodyne)

    # Combine channels using a root-sum-of-squares
    img_combined = np.sqrt(np.sum(np.array(imgs) ** 2, axis=0))
    img_combined = img_combined / np.max(img_combined)  # normalize
    return img_combined


def main():
    filename = "Cine.dat"
    twix = twixtools.read_twix(filename)
    meas = twix[-1]

    # Get image mdb blocks from the measurement
    image_mdbs = []
    for mdb in meas["mdb"]:
        try:
            if mdb.is_image_scan():
                image_mdbs.append(mdb)
        except Exception:
            continue

    # Group the image mdb blocks by frame (using mdb.cRep)
    frames_dict = defaultdict(list)
    for mdb in image_mdbs:
        frames_dict[mdb.cRep].append(mdb)

    frame_numbers = sorted(frames_dict.keys())
    print("Reconstructing frames using homodyne for frames:", frame_numbers)

    recon_frames = []
    for rep in frame_numbers:
        mdb_list = frames_dict[rep]
        img = homodyne_reconstruct_frame(mdb_list, phase_offset=12, acquired_lines=40)
        recon_frames.append(img)

    # Convert frames to 8-bit grayscale images for GIF creation
    gif_frames = []
    for img in recon_frames:
        img_uint8 = (img * 255).astype(np.uint8)
        gif_frames.append(img_uint8)

    output_gif = "cine_homodyne.gif"
    imageio.mimsave(output_gif, gif_frames, duration=0.1)
    print(f"Saved homodyne cine GIF as {output_gif}")


if __name__ == "__main__":
    main()
