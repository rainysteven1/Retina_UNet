import io, os, sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchsummary import summary
from sklearn.metrics import accuracy_score

from src import DEVICE, GPU_NAME
from dataset import DriverDataset
from model import UNet


class Process:
    def __init__(
        self,
        input_dim,
        output_dim,
        patch_height,
        patch_width,
        logger,
        load_model_dir="",
    ) -> None:
        self.device = DEVICE
        self.gpu_name = GPU_NAME
        self.logger = logger
        self.load_model_dir = load_model_dir
        self.mean_epoch_time = 0.0

        self.input_dim = input_dim
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.model = UNet(input_dim, output_dim, patch_height, patch_width).to(DEVICE)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=0.01,
            weight_decay=1e-6,
            momentum=0.3,
            nesterov=False,
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, "min", patience=5, verbose=True
        )
        self.criterion = nn.CrossEntropyLoss()

    def train(self, training_data, batch_size, epochs, validation_split, loss_csv):
        self.log_model_summary(batch_size)
        dataset = DriverDataset(*training_data)

        n_samples = len(dataset)
        n_val = int(n_samples * validation_split)
        indices_val = np.random.choice(n_samples, n_val, replace=False)
        dataset_train = Subset(dataset, np.delete(np.arange(n_samples), indices_val))
        dataset_val = Subset(dataset, indices_val)
        loader_train = DataLoader(dataset_train, batch_size, shuffle=True)
        loader_val = DataLoader(dataset_val, batch_size, shuffle=True)
        self.logger.info("number of training data: %d" % len(dataset_train))
        self.logger.info("number of validation data: %d" % len(dataset_val))

        best_loss = float("inf")
        loss_list = list()

        for epoch in range(1, epochs + 1):
            start_time = time.time()

            self.model.train()
            loss_train = 0.0
            score_train = 0.0
            for x, y in loader_train:
                x = x.to(self.device, dtype=torch.float32)
                y = y.reshape(-1, 2).to(self.device, dtype=torch.float32)

                self.optimizer.zero_grad()
                y_pred = self.model(x).reshape(-1, 2)
                loss = self.criterion(y_pred, y)
                loss.backward()
                self.optimizer.step()
                loss_train += loss.item()
                score_train += self.calculate_metrics(y, y_pred)
            loss_train /= len(loader_train)
            score_train /= len(loader_train)

            self.model.eval()
            loss_eval = 0.0
            score_eval = 0.0
            with torch.no_grad():
                for x, y in loader_val:
                    x = x.to(self.device, dtype=torch.float32)
                    y = y.reshape(-1, 2).to(self.device, dtype=torch.float32)
                    y_pred = self.model(x).reshape(-1, 2)
                    loss = self.criterion(y_pred, y)
                    loss_eval += loss.item()
                    score_eval += self.calculate_metrics(y, y_pred)
            loss_eval /= len(loader_val)
            score_eval /= len(loader_val)

            duration = time.time() - start_time
            self.mean_epoch_time = (
                0.0
                if epoch == 1
                else self.mean_epoch_time * (epoch - 2) / (epoch - 1)
                + duration / (epoch - 1)
            )
            lr = self.optimizer.param_groups[0]["lr"]
            self.logger.info(
                "Epoch: %5d/%d - %7.2fms - avg: %7.2fms - loss_train: %.4e - acc_train: %.4e - loss_eval: %.4e - acc_eval: %.4e - lr: %.4e"
                % (
                    epoch,
                    epochs,
                    duration * 1e3,
                    self.mean_epoch_time * 1e3,
                    loss_train,
                    score_train,
                    loss_eval,
                    score_eval,
                    lr,
                )
            )
            loss_list.append(
                dict(
                    [
                        ("epoch", epoch),
                        ("lr", lr),
                        ("loss_train", loss_train),
                        ("acc_train", score_train),
                        ("loss_eval", loss_eval),
                        ("acc_eval", score_eval),
                    ]
                )
            )

            if loss_eval < best_loss:
                checkpoint_path = os.path.join(self.load_model_dir, "checkpoint.pth")
                self.logger.info(
                    f"Valid loss improved from {best_loss:2.4f} to {loss_eval:2.4f}. Saving checkpoint: {checkpoint_path}"
                )
                best_loss = loss_eval
                torch.save(self.model.state_dict(), checkpoint_path)

        df = pd.DataFrame(loss_list)
        df.to_csv(loss_csv, encoding="utf8", index=False)
        torch.save(
            self.model.state_dict, os.path.join(self.load_model_dir, "model.pth")
        )

    def log_model_summary(self, batch_size):
        output = io.StringIO()
        sys.stdout = output
        summary(
            self.model,
            (self.input_dim, self.patch_height, self.patch_width),
            batch_size,
            self.device,
        )
        sys.stdout = sys.__stdout__
        summary_output = output.getvalue()
        self.logger.info("Model:\n{}".format(summary_output))

    def calculate_metrics(self, y_true, y_pred, threshold=0.5):
        y_true = y_true.reshape(-1).to(device=self.device, dtype=torch.uint8)
        y_true = y_true.cpu().numpy()

        y_pred = torch.softmax(y_pred, dim=1)
        y_pred = (
            (y_pred > threshold).reshape(-1).to(device=self.device, dtype=torch.uint8)
        )
        y_pred = y_pred.cpu().numpy()

        score_acc = accuracy_score(y_true, y_pred)
        return score_acc
