"""P1 step 0: convert the figshare .mat files to JPEG images and PNG masks.

Each braintumormri_figshare/brainTumorDataPublic_* folder gets a matching *_jpeg folder with
images/<label>/<n>.jpg and masks/<label>/<n>_mask.png (label 1=meningioma, 2=glioma, 3=pituitary).
"""
import h5py
import numpy as np
from PIL import Image
import os

project_root = os.path.dirname(os.path.abspath(__file__))
base_folder = os.path.join(project_root, 'braintumormri_figshare')

# Each brainTumorDataPublic_* folder of .mat files gets a matching *_jpeg output folder
batch_folders = sorted(
    d for d in os.listdir(base_folder)
    if d.startswith('brainTumorDataPublic_') and not d.endswith('_jpeg')
    and os.path.isdir(os.path.join(base_folder, d))
)

for batch in batch_folders:
    input_folder = os.path.join(base_folder, batch)            # folder with .mat files
    output_folder = os.path.join(base_folder, batch + '_jpeg') # folder to save converted files
    count = 0

    for fname in os.listdir(input_folder):
        if not fname.endswith('.mat'):
            continue

        path = os.path.join(input_folder, fname)
        # Files are MATLAB v7.3 (HDF5), so scipy.io.loadmat can't read them
        with h5py.File(path, 'r') as f:
            cjdata = f['cjdata']

            # Extract fields (transpose: MATLAB stores arrays column-major)
            image = np.array(cjdata['image']).T          # raw pixel array
            label = int(np.array(cjdata['label'])[0, 0]) # 1=meningioma, 2=glioma, 3=pituitary
            mask  = np.array(cjdata['tumorMask']).T      # binary mask

        # Normalize image to uint8
        img = image.astype(np.float64)
        img = (255 * (img - img.min()) / (img.max() - img.min())).astype(np.uint8)

        # Save image
        label_dir = os.path.join(output_folder, 'images', str(label))
        os.makedirs(label_dir, exist_ok=True)
        Image.fromarray(img).save(os.path.join(label_dir, fname.replace('.mat', '.jpg')))

        # Save mask as PNG (binary, so 0 or 255)
        mask_dir = os.path.join(output_folder, 'masks', str(label))
        os.makedirs(mask_dir, exist_ok=True)
        mask_img = (mask * 255).astype(np.uint8)
        Image.fromarray(mask_img).save(os.path.join(mask_dir, fname.replace('.mat', '_mask.png')))
        count += 1

    print(f"{batch}: converted {count} files")

print("Done converting.")
