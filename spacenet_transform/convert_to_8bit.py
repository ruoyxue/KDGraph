import numpy as np
from tqdm import tqdm
import os
import rasterio
import cv2


if __name__ == "__main__":
    dir_path_list = [
        r"/data/spacenet/AOI_3_Paris",
    ]
    other_save_path = r"/data/spacenet/image"

    for dir_path in dir_path_list:
        image_path = os.path.join(dir_path, "PS-RGB")
        output_dir = os.path.join(dir_path, "PS-RGB-8bit")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(16, 16))
        with tqdm(total=len(os.listdir(image_path))) as pbar:
            for image_name in os.listdir(image_path):
                with rasterio.open(os.path.join(image_path, image_name)) as f:
                    input_profile = f.profile.copy()
                    R = f.read(1).astype(np.float32)
                    G = f.read(2).astype(np.float32)
                    B = f.read(3).astype(np.float32)
                    R = 255.0 * ((R - np.min(R)) / (np.max(R) - np.min(R)))
                    G = 255.0 * ((G - np.min(G)) / (np.max(G) - np.min(G)))
                    B = 255.0 * ((B - np.min(B)) / (np.max(B) - np.min(B)))
                    R = clahe.apply(np.asarray(R, dtype=np.uint8))
                    G = clahe.apply(np.asarray(G, dtype=np.uint8))
                    B = clahe.apply(np.asarray(B, dtype=np.uint8))
                    R = R.astype(np.uint8)
                    G = G.astype(np.uint8)
                    B = B.astype(np.uint8)

                    output_profile = input_profile.copy()
                    output_profile["dtype"] = "uint8"
                    output_fn = os.path.join(output_dir, image_name)
                    output_fn2 = os.path.join(other_save_path, image_name)
                    with rasterio.open(output_fn, "w", **output_profile) as output:
                        output.write(R, 1)
                        output.write(G, 2)
                        output.write(B, 3)
                    
                    pbar.update()
