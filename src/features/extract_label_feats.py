import os
import open_clip
import pandas as pd
import torch
from src.utils import load_parameters
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from safetensors.torch import save_file


class LabelDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        self._data = pd.read_csv(data_path, header=None)
        self._tokenizer = tokenizer
        print(f'Having {len(self._data)} instances!')

    def __getitem__(self, item):
        name = self._data.iloc[item, 1]
        return name, self._tokenizer(name).squeeze()

    def __len__(self):
        return len(self._data)


def get_labels(in_dir, exceptions):
    return filter(lambda x: x[0] not in exceptions,
                  map(lambda x: (x.split('_')[0], x), os.listdir(in_dir)))


@torch.no_grad()
def main():
    parameters = load_parameters('../../configs/extract_label_feats.yaml')
    device = parameters['device']
    model, _, _ = open_clip.create_model_and_transforms(**parameters['model'])
    model = model.to(device)
    tokenizer = open_clip.get_tokenizer(parameters['tokenizer'])
    labels = get_labels(**parameters['labels'])
    out_dir = parameters['out_dir']
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    for lab, filename in labels:
        print(f'Doing {lab}')
        dataset = LabelDataset(data_path=f'{parameters["labels"]["in_dir"]}/{filename}',
                               tokenizer=tokenizer)
        loader = DataLoader(dataset, **parameters['dataset'])
        tensors = {}
        for names, tokens in tqdm(loader):
            tokens = tokens.to(device)
            embeddings = model.encode_text(tokens)
            tensors.update({name: embeddings[i] for i, name in enumerate(names)})
        save_file(tensors, f'{out_dir}/{lab}.safetensors')
        print('Done!')


if __name__ == '__main__':
    main()

