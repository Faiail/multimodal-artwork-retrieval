import open_clip
import utils
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import torch
from tqdm import tqdm
import pandas as pd
from safetensors.torch import save_file
from enum import IntEnum


class DataCaptionEnum(IntEnum):
    NAME = 0,
    PROMPT = 1,
    CAPTION = 2,


class ImageTextDataset(Dataset):
    def __init__(self, image_dir, data_path, tokenizer, preprocess=None, ext='.jpg'):
        self._image_dir = image_dir
        self._data = pd.read_csv(data_path, index_col=0)
        self._images = os.listdir(image_dir)
        self._tokenizer = tokenizer
        self._preprocess = preprocess
        self._ext = ext

    def __getitem__(self, item):
        image_entry = self._data.iloc[item]
        image_path = image_entry[DataCaptionEnum.NAME]
        image_caption = image_path[DataCaptionEnum.CAPTION]  # put the random choice between caption and prompt?
        image = Image.open(f'{self._image_dir}/{image_path}')
        image_filename = image_path[:-len(self._ext)]
        if self._preprocess:
            image = self._preprocess(image)
        tokens = self._tokenizer(image_caption)
        return image_filename, image, tokens.squeeze().contiguous()

    def __len__(self):
        return len(self._images)


def loop(model, dataloader, out_dir, device):
    for names, images, tokens in tqdm(dataloader):
        images = images.to(device)
        tokens = tokens.to(device)
        print(tokens.size())
        with torch.no_grad():
            image_embeddings = model.encode_image(images)
            text_embeddings = model.encode_text(tokens)
        for name, image, text in zip(names, image_embeddings, text_embeddings):
            data = {'image': image.squeeze(), 'text': text.squeeze()}
            save_file(data, f'{out_dir}/{name}.safetensors')


if __name__ == '__main__':
    parameters = utils.load_parameters('../configs/extract_images_text_feats.yaml')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _, preprocess = open_clip.create_model_and_transforms(parameters['model']['name'],
                                                                 pretrained=parameters['model']['pretrained'])
    model = model.to(device)
    tokenizer = open_clip.get_tokenizer(parameters['model']['tokenizer'])
    batch_size = parameters['model']['batch_size']

    image_dataset = ImageTextDataset(image_dir=parameters['data']['image_dir'],
                                     data_path=parameters['data']['path'],
                                     tokenizer=tokenizer,
                                     preprocess=preprocess)
    dataloader = DataLoader(image_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    out_dir = parameters['data']['out_dir']
    os.makedirs(out_dir, exist_ok=True)
    loop(model, dataloader, out_dir, device)