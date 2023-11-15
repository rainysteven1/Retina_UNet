import h5py
import os, shutil
import numpy as np
from PIL import Image
from preprocessing import preprocess

HDF5_DATASET = "image"


def read_hdf5(input_file):
    with h5py.File(input_file, "r") as file:
        return file[HDF5_DATASET][()]


def write_hdf5(array, output_file):
    with h5py.File(output_file, "w") as file:
        file.create_dataset(HDF5_DATASET, data=array, dtype=array.dtype)


def gen_datasets(original_dir, groundTruth_dir, borderMask_dir):
    originals = np.stack(
        [
            np.asarray(Image.open(os.path.join(original_dir, original)))
            for original in sorted(os.listdir(original_dir))
        ]
    )
    originals = np.transpose(originals, (0, 3, 1, 2))
    Nimgs, _, height, width = originals.shape
    groundTruths = np.stack(
        [
            np.asarray(Image.open(os.path.join(groundTruth_dir, groundTruth)))
            for groundTruth in sorted(os.listdir(groundTruth_dir))
        ]
    ).reshape(Nimgs, 1, height, width)
    borderMasks = np.stack(
        [
            np.asarray(Image.open(os.path.join(borderMask_dir, borderMask)))
            for borderMask in sorted(os.listdir(borderMask_dir))
        ]
    ).reshape(Nimgs, 1, height, width)
    return {"imgs": originals, "groundTruth": groundTruths, "borderMask": borderMasks}


def write_datasets(
    current_dir, datasets_path, drive_path, category_list, imgs_dir_list
):
    datasets_dir = os.path.join(current_dir, datasets_path)
    shutil.rmtree(datasets_dir, ignore_errors=True)
    os.makedirs(datasets_dir, exist_ok=True)

    def get_imgs_category_list(category):
        category_path = os.path.join(current_dir, drive_path, category)
        return [os.path.join(category_path, dir) for dir in imgs_dir_list]

    category_dict = dict(
        [(category, get_imgs_category_list(category)) for category in category_list]
    )
    for category, dir_list in category_dict.items():
        os.makedirs(os.path.join(datasets_path, category), exist_ok=True)
        for key, value in gen_datasets(*dir_list).items():
            write_hdf5(value, os.path.join(datasets_path, category, f"{key}.hdf5"))


def is_patch_inside_FOV(x, y, img_w, img_h, patch_h):
    x_ = x - int(img_w / 2)
    y_ = y - int(img_h / 2)
    R_inside = 270 - int(patch_h * np.sqrt(2.0) / 2.0)
    radius = np.sqrt((x_**2) + (y_**2))
    return radius < R_inside


def extract_random(full_imgs, full_masks, patch_h, patch_w, N_patches, inside=True):
    patches_imgs = np.empty((N_patches, full_imgs.shape[1], patch_h, patch_w))
    patches_masks = np.empty((N_patches, full_masks.shape[1], patch_h, patch_w))
    img_h = full_imgs.shape[2]
    img_w = full_imgs.shape[3]
    patch_per_img = int(N_patches / full_imgs.shape[0])
    iter_total = 0
    for i in range(full_imgs.shape[0]):
        j = 0
        while j < patch_per_img:
            x_center = np.random.randint(0 + int(patch_w / 2), img_w - int(patch_w / 2))
            y_center = np.random.randint(0 + int(patch_h / 2), img_h - int(patch_h / 2))
            if inside and not is_patch_inside_FOV(
                x_center, y_center, img_w, img_h, patch_h
            ):
                continue
            patches_imgs[iter_total] = full_imgs[
                i,
                :,
                y_center - int(patch_h / 2) : y_center + int(patch_h / 2),
                x_center - int(patch_w / 2) : x_center + int(patch_w / 2),
            ]
            patches_masks[iter_total] = full_masks[
                i,
                :,
                y_center - int(patch_h / 2) : y_center + int(patch_h / 2),
                x_center - int(patch_w / 2) : x_center + int(patch_w / 2),
            ]
            iter_total += 1
            j += 1
    return patches_imgs, patches_masks


def masks_UNet(masks):
    mask_h = masks.shape[2]
    mask_w = masks.shape[3]
    masks = np.reshape(masks, (masks.shape[0], mask_h * mask_w))
    new_masks = np.empty((masks.shape[0], mask_h * mask_w, 2))
    new_masks[:, :, 0] = 1 - masks
    new_masks[:, :, 1] = masks
    return new_masks


def gen_training_data(datasets_path, patch_height, patch_width, N_subimgs, inside_FOV):
    imgs = read_hdf5(os.path.join(datasets_path, "training", "imgs.hdf5"))
    masks = read_hdf5(os.path.join(datasets_path, "training", "groundTruth.hdf5"))

    imgs = preprocess(imgs)
    masks = masks / 255.0

    cut_length = int((imgs.shape[2] - imgs.shape[3]) / 2)
    imgs = imgs[:, :, cut_length : -1 - cut_length, :]
    masks = masks[:, :, cut_length : -1 - cut_length, :]

    patches_imgs_train, patches_masks_train = extract_random(
        imgs, masks, patch_height, patch_width, N_subimgs, inside_FOV
    )
    patches_masks_train = masks_UNet(patches_masks_train)

    return patches_imgs_train, patches_masks_train
