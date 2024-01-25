from torch.utils.data import Dataset


class CatalogDataset(Dataset):
    def __init__(self):
        super().__init__()

    def __getitem__(self, item):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()


class PredictorTestDataset(Dataset):
    def __init__(self):
        super().__init__()

    def __getitem__(self, item):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()