import numpy as np


# convert run length encoding to image mask
def get_mask(img, metadata_df, img_id):
    h, w = img.shape[:2]
    mask = np.zeros(img.shape[:2]).astype(int)

    # need to get all corresponding rows because each row is a mask of one object
    encoded_pixels_list = list(
        metadata_df[metadata_df["ImageId"] == img_id]["EncodedPixels"]
    )
    for encoded_pixels in encoded_pixels_list:
        # check if it is float because otherwise throw exception
        if isinstance(encoded_pixels, float) and np.isnan(encoded_pixels):
            continue
        encoded_pixels = list(map(int, encoded_pixels.split(" ")))
        for i in range(int(len(encoded_pixels) / 2)):
            ran = np.arange(encoded_pixels[i * 2 + 1])
            ran = ran + encoded_pixels[i * 2]
            for num in ran:
                num = num - 1
                col = num // h
                row = num - h * col
                mask[row][col] = 1
    return mask
