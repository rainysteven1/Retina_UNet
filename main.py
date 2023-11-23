import sys, os, argparse, pathlib, platform, shutil, time
import ujson

sys.path.insert(0, "./src")

from src.data import *
from src.logger import LoggerFactory, init_file_handler
from src.plot import visuliaze_sample_input
from src.preprocessing import preprocess
from src.process import Process

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(CURRENT_DIR, "resources/config/configuration.json")


def parse_args():
    parser = argparse.ArgumentParser(description="Allen Cahn")
    parser.add_argument(
        "--state",
        type=str,
        default="train",
        choices=["train", "predict"],
        help="running state",
    )
    parser.add_argument(
        "--load_model_dir",
        type=str,
        default="",
        help="model directory for loading",
    )
    return parser.parse_args()


def delete_pycache(folder_path):
    for root, dirs, _ in os.walk(folder_path):
        if "__pycache__" in dirs:
            shutil.rmtree(os.path.join(root, "__pycache__"), ignore_errors=True)


if __name__ == "__main__":
    args = parse_args()
    with open(CONFIG_PATH, "r") as file:
        config = ujson.load(file)
    data_path = config["data path"]
    data_attributes = config["data attributes"]
    training_settings = config["training settings"]
    test_settings = config["test settings"]
    result_settings = config["result settings"]

    start_time = time.time()
    str_start_time = time.strftime("%Z-%Y-%m-%d-%H%M%S", time.localtime(start_time))
    running_platform = platform.node()
    file_dir = os.path.abspath(os.path.dirname(__file__))
    if args.state == "train":
        working_dir = pathlib.Path(file_dir).joinpath("result")
        working_dir.mkdir(exist_ok=True)
        working_dir = working_dir.joinpath(str_start_time)
        working_dir.mkdir(exist_ok=True)
        for folder in result_settings["folder_list"]:
            working_dir.joinpath(folder).mkdir(exist_ok=True)
        working_dir = str(working_dir)
        shutil.copy(__file__, f"{working_dir}/code")
        shutil.copytree(f"{file_dir}/src", f"{working_dir}/code/src")
        shutil.copy(CONFIG_PATH, f"{working_dir}/config")
        delete_pycache(f"{working_dir}/code/src")
    elif args.state == "predict" and args.load_model_dir != "":
        working_dir = (
            pathlib.Path(file_dir).joinpath("result").joinpath(args.load_model_dir)
        )

    write_datasets(file_dir, **data_path)
    logger_factory = LoggerFactory()
    file_handler = init_file_handler("%s/log/log_%s.log" % (working_dir, args.state))
    logger_factory.add_file_handler(file_handler)

    def logger_basic(process):
        process.logger.info("Start time: " + str_start_time)
        process.logger.info("Using {} device".format(process.device))
        process.logger.info("GPU: {}".format(process.gpu_name))
        process.logger.info("Running platform: " + running_platform)
        process.logger.info("Running state: " + args.state)
        process.logger.info("File directory: " + file_dir)
        process.logger.info("Working directory: " + str(working_dir))
        process.logger.info(args)

    if args.state == "train":
        load_model_dir = f"{working_dir}/model"
        process = Process(
            1,
            2,
            logger=logger_factory.logger,
            load_model_dir=load_model_dir,
            **data_attributes,
        )
        logger_basic(process)

        loss_csv = os.path.join(working_dir, "log", "log_SGD.csv")
        patches_imgs_train, patches_masks_train = gen_training_data(
            os.path.join(CURRENT_DIR, data_path["datasets_path"]),
            training_settings["hdf5_list"],
            N_subimgs=training_settings["N_subimgs"],
            inside_FOV=training_settings["inside_FOV"],
            **data_attributes,
        )

        N_sample = min(patches_imgs_train.shape[0], 40)
        visuliaze_sample_input(
            group_images(patches_imgs_train[0:N_sample, :, :, :], 5),
            os.path.join(working_dir, "figure", "sample_input_imgs.png"),
        )
        visuliaze_sample_input(
            group_images(patches_masks_train[0:N_sample, :, :, :], 5),
            os.path.join(working_dir, "figure", "sample_input_masks.png"),
        )

        patches_masks_train = masks_UNet(patches_masks_train)
        process.train(
            (patches_imgs_train, patches_masks_train),
            training_settings["batch_size"],
            training_settings["N_epochs"],
            training_settings["validation_split"],
            loss_csv,
        )
    elif args.state == "predict":
        best_last = test_settings["best_last"]
        load_model_dir = "%s/model/%s.pth" % (
            working_dir,
            "checkpoint" if best_last == "best" else "model",
        )
        process = Process(
            1,
            2,
            logger=logger_factory.logger,
            load_model_dir=load_model_dir,
            **data_attributes,
        )
        logger_basic(process)

        if test_settings["average_mode"]:
            stride_height = test_settings["stride_height"]
            stride_width = test_settings["stride_width"]

            (
                patches_imgs_test,
                new_height,
                new_width,
                original_imgs,
                masks_test,
                borderMasks_test,
            ) = gen_test_data_overloap(
                os.path.join(CURRENT_DIR, data_path["datasets_path"]),
                test_settings["hdf5_list"],
                stride_height=stride_height,
                stride_width=stride_width,
                full_imgs_to_test=test_settings["full_images_to_test"],
                **data_attributes,
            )

            predictions = process.predict(
                patches_imgs_test, test_settings["batch_size"]
            )
            np.save(f"{working_dir}/tmp.npy", predictions)
            pred_patches = pred_to_imgs(predictions, mode="original", **data_attributes)
            pred_imgs = recompone_overlap(
                pred_patches, new_height, new_width, stride_height, stride_width
            )
            original_imgs = preprocess(original_imgs[0 : pred_imgs.shape[0], :, :, :])
            groundTruth_masks = masks_test
        else:
            (
                patches_imgs_test,
                patches_masks_test,
                original_imgs,
                borderMasks_test,
            ) = gen_test_data(
                os.path.join(CURRENT_DIR, data_path["datasets_path"]),
                test_settings["hdf5_list"],
                full_imgs_to_test=test_settings["full_images_to_test"],
                **data_attributes,
            )

            predictions = process.predict(
                patches_imgs_test, test_settings["batch_size"]
            )
            sys.exit()
            pred_patches = pred_to_imgs(predictions, mode="original", **data_attributes)
            pred_imgs = recompone(pred_patches, 13, 12)
            original_imgs = recompone(patches_imgs_test, 13, 12)
            groundTruth_masks = recompone(patches_masks_test, 13, 12)

        kill_border(pred_imgs, borderMasks_test)
        full_img_height = original_imgs.shape[2]
        full_img_width = original_imgs.shape[3]
        pred_imgs = pred_imgs[:, :, 0:full_img_height, 0:full_img_width]
        original_imgs = original_imgs[:, :, 0:full_img_height, 0:full_img_width]
        groundTruth_masks = groundTruth_masks[:, :, 0:full_img_height, 0:full_img_width]

        N_group_visual = test_settings["N_group_visual"]
        N_predicted = original_imgs.shape[0]
        for i in range(int(N_predicted / N_group_visual)):
            original_stripe = group_images(
                original_imgs[i * N_group_visual : (i + 1) * N_group_visual],
                N_group_visual,
            )
            masks_stripe = group_images(
                groundTruth_masks[i * N_group_visual : (i + 1) * N_group_visual],
                N_group_visual,
            )
            pred_stripe = group_images(
                pred_imgs[i * N_group_visual : (i + 1) * N_group_visual],
                N_group_visual,
            )
            total_img = np.concatenate(
                (original_stripe, masks_stripe, pred_stripe), axis=0
            )
            visuliaze_sample_input(
                total_img,
                os.path.join(
                    working_dir, "figure", f"Original_GroundTruth_Prediction_{i}.png"
                ),
            )
