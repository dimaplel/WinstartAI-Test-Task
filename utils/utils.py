import numpy as np


# Helper function to decode an rl-encoded mask to flat indices
def decode_mask(mask, shape):
    mask_split = mask.split()
    starts, lengths = [np.array(x, dtype=int) for x in (mask_split[0::2], mask_split[1::2])]
    starts -= 1
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for start, length in zip(starts, lengths):
        img[start:start + length] = 1
    return img.reshape(shape).T


# Decode all masks of image
def decode_image(masks, shape):
    decoded = np.zeros(shape, dtype=np.uint8)
    for mask in masks:
        if isinstance(mask, str):
            decoded |= decode_mask(mask, shape)
    return decoded


# Function to randomly undersample categories of ships to make it more balanced
def undersample(series, sample_size=1000):
    if series["has_ships"].iloc[0] == 0.0:
        return series.sample(min(series.shape[0], sample_size // 2), replace=True)

    return series.sample(min(series.shape[0], sample_size), replace=True)

