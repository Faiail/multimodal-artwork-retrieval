import pandas as pd
from src.models.competitors.basic_garcia.ranker import Ranker, TfidfEncoder
from src.models.competitors.basic_garcia.basic_ranker_dataset import BasicRankerDataset
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from torch.nn import CosineEmbeddingLoss
from torchvision.transforms import Compose
import torchvision.transforms as tr
from src.utils import load_ruamel
from src.models.utils import parse_args
from torch.optim import Adam
from src.models.EarlyStopping import EarlyStopping
import os
from torchvision.models import resnet152, ResNet152_Weights
import joblib


def train_model(
        model: Ranker,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer,
        criterion: CosineEmbeddingLoss,
        early_stop,
        num_epochs: int,
        device,
):
    loaders = {
        'train': train_loader,
        'val': val_loader
    }
    model = model.to(device)
    optimizer.zero_grad()
    for epoch in range(1, num_epochs + 1):
        for phase, loader in loaders.items():
            cumulated_loss = 0.0
            with torch.set_grad_enabled(phase == 'train'):
                for step, data_dict in enumerate(tqdm(loader)):
                    label = data_dict.get('gt')
                    images = data_dict.get('images').to(device)
                    comments = data_dict.get('comments').to(device)
                    titles = data_dict.get('titles').to(device)
                    image_embeddings, text_embeddings = model.forward(images, comments, titles)

                    loss = criterion(image_embeddings, text_embeddings, label)
                    cumulated_loss = cumulated_loss + loss.cpu().item()
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
            cumulated_loss = cumulated_loss / len(loader)
            print(f'Epoch {epoch:03d}: {phase} Loss: {cumulated_loss:.4f}')
            if phase == 'val':
                early_stop(cumulated_loss, model)
                if early_stop.early_stop:
                    print(f'early stopping at epoch {epoch:03d}')
                    return -early_stop.best_score
            return -early_stop.best_score


def get_encoder(vec_params: dict) -> TfidfEncoder:
    vectorizer = TfidfEncoder(**vec_params['params'])
    data = pd.read_csv(vec_params['train_data'])[vec_params['key']].tolist()
    vectorizer.fit(data)
    return vectorizer


def init_model(model_params: dict) -> dict:
    mapping_resnet = {'v1': ResNet152_Weights.IMAGENET1K_V1,
                      'v2': ResNet152_Weights.IMAGENET1K_V2}
    resnet_params = model_params.get('resnet')
    resnet_params['weights'] = mapping_resnet.get(resnet_params['weights'], ResNet152_Weights.IMAGENET1K_V2)
    resnet = resnet152(**resnet_params)
    comment_encoder = get_encoder(model_params['comment_tf_idf_vectorizer'])
    title_encoder = get_encoder(model_params['title_tf_idf_vectorizer'])
    model_params['resnet'] = resnet
    model_params['comment_tf_idf_vectorizer'] = comment_encoder
    model_params['title_tf_idf_vectorizer'] = title_encoder
    return model_params


def init_preprocess(preprocess_params: dict) -> dict:
    return Compose([
        tr.__dict__[k](**v) if type(v) == dict else tr.__dict__[k]()
        for k, v in preprocess_params.items()
    ])


def main():
    params_path = parse_args().params_path
    parameters = load_ruamel(params_path)
    parameters['model'] = init_model(parameters['model'])
    model = Ranker(**parameters['model'])

    parameters['dataset']['train']['preprocess'] = init_preprocess(parameters['dataset']['train']['preprocess'])
    parameters['dataset']['val']['preprocess'] = init_preprocess(parameters['dataset']['val']['preprocess'])

    train_data = BasicRankerDataset(**parameters['dataset']['train'])
    train_loader = DataLoader(dataset=train_data, **parameters['dataloader'])

    val_data = BasicRankerDataset(**parameters['dataset']['val'])
    val_loader = DataLoader(dataset=val_data, **parameters['dataloader'])

    optimizer = Adam(params=model.parameters(), **parameters['optimizer'])

    criterion = CosineEmbeddingLoss(**parameters['criterion'])

    model_dir = parameters["out_dir"]
    os.makedirs(model_dir, exist_ok=True)

    # saving tf-idf
    joblib.dump(model.title_tf_idf_vectorizer, f'{model_dir}/title_vectorizer.pkl')
    joblib.dump(model.comment_tf_idf_vectorizer, f'{model_dir}/comment_vectorizer.pkl')

    early_stop = EarlyStopping(path=f'{model_dir}/model_state_dict.pt', **parameters['early_stop'])

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        early_stop=early_stop,
        **parameters['general']
    )


if __name__ == '__main__':
    main()