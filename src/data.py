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


def gen_datasets(original_dir, groundTruth_dir, borderMasks_dir):
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
            np.asarray(Image.open(os.path.join(borderMasks_dir, borderMask)))
            for borderMask in sorted(os.listdir(borderMasks_dir))
        ]
    ).reshape(Nimgs, 1, height, width)
    return {"imgs": originals, "groundTruth": groundTruths, "borderMasks": borderMasks}


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


# TODO 矩阵形式
def group_images(data, per_row):
    data = np.transpose(data, (0, 2, 3, 1))
    all_stripe = []
    for i in range(int(data.shape[0] / per_row)):
        stripe = data[i * per_row]
        for k in range(i * per_row + 1, (i + 1) * per_row):
            stripe = np.concatenate((stripe, data[k]), axis=1)
        all_stripe.append(stripe)
    totimg = all_stripe[0]
    for i in range(1, len(all_stripe)):
        totimg = np.concatenate((totimg, all_stripe[i]), axis=0)
    return totimg


def paint_border_original(full_imgs, pad_h, pad_w):
    pad_width = [(0, 0)] * 4
    pad_width[2] = (0, pad_h)
    pad_width[3] = (0, pad_w)
    return np.pad(full_imgs, pad_width, mode="constant")


def paint_border(full_imgs, patch_h, patch_w):
    """
    Extend the full images because patch divison is not exact
    """
    pad_h = (
        patch_h - full_imgs.shape[2] % patch_h if full_imgs.shape[2] % patch_h else 0
    )
    pad_w = (
        patch_w - full_imgs.shape[3] % patch_w if full_imgs.shape[3] % patch_w else 0
    )
    return paint_border_original(full_imgs, pad_h, pad_w)


def paint_border_overlap(full_imgs, patch_h, patch_w, stride_h, stride_w):
    """
    Return only the pixels contained in the FOV, for both images and masks
    """
    pad_h = stride_h - (full_imgs.shape[2] - patch_h) % stride_h
    pad_w = stride_w - (full_imgs.shape[3] - patch_w) % stride_w
    return paint_border_original(full_imgs, pad_h, pad_w)


def extract_ordered_original(full_imgs, N_patches_img, patch_h, patch_w):
    patches = np.empty(
        (
            full_imgs.shape[0] * N_patches_img,
            full_imgs.shape[1],
            patch_h,
            patch_w,
        )
    )

    for i in range(full_imgs.shape[0]):
        patches[(i * N_patches_img) : ((i + 1) * N_patches_img)] = full_imgs[
            i, :, :patch_h, :patch_w
        ].reshape(-1, patch_h, patch_w)
    return patches


def extract_ordered(full_imgs, patch_h, patch_w):
    """
    Divide all the full_imgs in pacthes by order
    """
    N_patches_img = int(full_imgs.shape[2] / patch_h) * int(
        full_imgs.shape[3] / patch_w
    )
    return extract_ordered_original(full_imgs, N_patches_img, patch_h, patch_w)


def extract_ordered_overlap(full_imgs, patch_h, patch_w, stride_h, stride_w):
    N_patches_img = ((full_imgs.shape[2] - patch_h) // stride_h + 1) * (
        (full_imgs.shape[3] - patch_w) // stride_w + 1
    )
    return extract_ordered_original(full_imgs, N_patches_img, patch_h, patch_w)


def gen_training_data(
    datasets_path, hdf5_list, patch_height, patch_width, N_subimgs, inside_FOV
):
    imgs, masks = [
        read_hdf5(os.path.join(datasets_path, "training", f"{file_name}.hdf5"))
        for file_name in hdf5_list
    ]

    imgs = preprocess(imgs)
    masks = masks / 255.0

    cut_length = int((imgs.shape[2] - imgs.shape[3]) / 2)
    imgs = imgs[:, :, cut_length : -1 - cut_length, :]
    masks = masks[:, :, cut_length : -1 - cut_length, :]
    return extract_random(imgs, masks, patch_height, patch_width, N_subimgs, inside_FOV)


def gen_test_data(
    datasets_path,
    hdf5_list,
    patch_height,
    patch_width,
    full_imgs_to_test,
):
    """
    Load the original data and return the extracted patches for testing
    """
    imgs, masks = [
        read_hdf5(os.path.join(datasets_path, "test", f"{file_name}.hdf5"))
        for file_name in hdf5_list
    ]

    imgs = preprocess(imgs)
    masks = masks / 255.0

    imgs = imgs[:full_imgs_to_test, :, :, :]
    masks = masks[:full_imgs_to_test, :, :, :]
    imgs = paint_border(imgs, patch_height, patch_width)
    masks = paint_border(masks, patch_height, patch_width)

    patches_imgs_test = extract_ordered(imgs, patch_height, patch_width)
    patches_masks_test = extract_ordered(masks, patch_height, patch_width)
    return patches_imgs_test, patches_masks_test


def gen_test_data_overloap(
    datasets_path,
    hdf5_list,
    patch_height,
    patch_width,
    stride_height,
    stride_width,
    full_imgs_to_test,
):
    """
    Load the original data and return the extracted patches for testing
    return the ground truth in its original shape
    """
    imgs, masks = [
        read_hdf5(os.path.join(datasets_path, "test", f"{file_name}.hdf5"))
        for file_name in hdf5_list
    ]

    imgs = preprocess(imgs)
    masks = masks / 255.0

    imgs = imgs[:full_imgs_to_test, :, :, :]
    masks = masks[:full_imgs_to_test, :, :, :]
    imgs = paint_border_overlap(
        imgs, patch_height, patch_width, stride_height, stride_width
    )

    patches_imgs_test = extract_ordered_overlap(
        imgs, patch_height, patch_width, stride_height, stride_width
    )
    return patches_imgs_test, imgs.shape[2], imgs.shape[3], masks
