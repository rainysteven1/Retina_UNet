import sys, os, argparse, pathlib, platform, shutil, time
import ujson

sys.path.insert(0, "./src")

from src.data import write_datasets, gen_training_data
from src.logger import LoggerFactory, init_file_handler
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
        load_model_dir = f"{working_dir}/model"
        scatter_dir = "%s/figure/scatter" % working_dir
        os.makedirs(scatter_dir, exist_ok=True)
        shutil.copy(__file__, f"{working_dir}/code")
        shutil.copytree(f"{file_dir}/src", f"{working_dir}/code/src")
        shutil.copy(CONFIG_PATH, f"{working_dir}/config")
        delete_pycache(f"{working_dir}/code/src")
    elif args.state == "predict" and args.load_model_dir != "":
        working_dir = (
            pathlib.Path(file_dir).joinpath("result").joinpath(args.load_model_dir)
        )
        load_model_dir = f"{working_dir}/model/model.pth"

    write_datasets(file_dir, **data_path)
    logger_factory = LoggerFactory()
    file_handler = init_file_handler("%s/log/log_%s.log" % (working_dir, args.state))
    logger_factory.add_file_handler(file_handler)
    process = Process(
        1,
        2,
        logger=logger_factory.logger,
        load_model_dir=load_model_dir,
        **data_attributes,
    )

    process.logger.info("Start time: " + str_start_time)
    process.logger.info("Using {} device".format(process.device))
    process.logger.info("GPU: {}".format(process.gpu_name))
    process.logger.info("Running platform: " + running_platform)
    process.logger.info("Running state: " + args.state)
    process.logger.info("File directory: " + file_dir)
    process.logger.info("Working directory: " + str(working_dir))
    process.logger.info(args)

    if args.state == "train":
        loss_csv = os.path.join(working_dir, "log", "log_SGD.csv")
        training_data = gen_training_data(
            os.path.join(CURRENT_DIR, data_path["datasets_path"]),
            N_subimgs=training_settings["N_subimgs"],
            inside_FOV=training_settings["inside_FOV"],
            **data_attributes,
        )
        process.train(
            training_data,
            training_settings["batch_size"],
            training_settings["N_epochs"],
            loss_csv,
        )
