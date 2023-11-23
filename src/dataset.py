from torch.utils.data import Dataset


class TrainingDataset(Dataset):
    def __init__(self, imgs, masks) -> None:
        super().__init__()

        self.imgs = imgs
        self.masks = masks

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, index):
        return self.imgs[index], self.masks[index]


class PredictionDataset(Dataset):
    def __init__(self, imgs) -> None:
        super().__init__()

        self.imgs = imgs

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, index):
        return self.imgs[index]
