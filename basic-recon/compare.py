#!/usr/bin/env python3
import imageio
import numpy as np


def create_difference_gif(gif1, gif2, output_gif, duration=0.125):
    # Read the frames from the two GIFs
    frames1 = imageio.mimread(gif1)
    frames2 = imageio.mimread(gif2)

    # Ensure the two GIFs have the same number of frames
    assert len(frames1) == len(
        frames2
    ), "Warning: The GIFs have different number of frames!"

    diff_frames = []
    for f1, f2 in zip(frames1, frames2):
        # Convert to float for processing
        f1 = np.array(f1, dtype=np.float32)
        f2 = np.array(f2, dtype=np.float32)
        # Compute the absolute difference between frames
        diff = np.abs(f1 - f2)
        # Normalize the difference to span 0-1 (if the max is nonzero)
        max_val = np.max(diff)
        if max_val > 0:
            diff = diff / max_val
        # Convert back to 8-bit grayscale image
        diff_uint8 = (diff * 255).astype(np.uint8)
        diff_frames.append(diff_uint8)

    # Save the resulting difference frames as a GIF
    imageio.mimsave(output_gif, diff_frames, duration=duration)
    print(f"Saved magnitude difference GIF as {output_gif}")


def main():
    # Input the names of the two GIF files you want to compare.
    gif1 = "zero_filled_cine.gif"
    gif2 = "conjugate_symmetry_cine.gif"
    output_gif = "difference.gif"

    create_difference_gif(gif1, gif2, output_gif, duration=0.125)


if __name__ == "__main__":
    main()
