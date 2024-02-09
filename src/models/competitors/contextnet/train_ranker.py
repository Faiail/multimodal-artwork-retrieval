import pandas as pd
from src.models.competitors.basic_garcia.ranker import TfidfEncoder
from src.models.competitors.contextnet.context_net_ranker import ContextNetRanker, ContextNet, OneHotAttributeEncoder
from src.data.context_net_dataset import ContextNetRankerDataset
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
from torchvision.models import resnet50, ResNet50_Weights
import joblib
from typing import Optional


def train_model(
        model: ContextNetRanker,
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
                    label = data_dict.get('score').to(device)
                    images = data_dict.get('image').to(device)
                    comments = data_dict.get('comment')
                    titles = data_dict.get('title')
                    attributes = data_dict.get('attributes')
                    image_embeddings, text_embeddings = model(images, comments, titles, attributes)

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


def get_tfidf_encoder(vec_params: dict) -> TfidfEncoder:
    vectorizer = TfidfEncoder(**vec_params['params'])
    data = pd.read_csv(vec_params['train_data'])[vec_params['key']].tolist()
    vectorizer.fit(data)
    return vectorizer


def get_onehot_encoder(enc_params: dict) -> OneHotAttributeEncoder:
    vectorizer = OneHotAttributeEncoder(**enc_params['params'])
    # get unique values
    data = pd.read_csv(enc_params['train_data']).drop('key', axis=1)
    vectorizer.fit(data)
    return vectorizer


def init_model(model_params: dict) -> dict:
    mapping_resnet = {'v1': ResNet50_Weights.IMAGENET1K_V1,
                      'v2': ResNet50_Weights.IMAGENET1K_V2}
    resnet_params = model_params.get('resnet')
    print('constructing resnet50...')
    resnet_params['weights'] = mapping_resnet.get(resnet_params['weights'], ResNet50_Weights.IMAGENET1K_V2)
    resnet = resnet50(**resnet_params)
    print('Done!')
    print('Constructing comment encoder...')
    comment_encoder = get_tfidf_encoder(model_params['comment_tf_idf_vectorizer'])
    print('Done!')
    print('Constructing title encoder...')
    title_encoder = get_tfidf_encoder(model_params['title_tf_idf_vectorizer'])
    print('Done!')
    print('Constructing attribute encoder...')
    attribute_encoder = get_onehot_encoder(model_params['attribute_one_hot_vectorizer'])
    print('Done!')
    model_params['resnet'] = resnet
    model_params['comment_tf_idf_vectorizer'] = comment_encoder
    model_params['title_tf_idf_vectorizer'] = title_encoder
    model_params['attribute_one_hot_encoder'] = attribute_encoder
    model_params['contextnet'] = init_contextnet(model_params['contextnet'])
    return model_params


def init_contextnet(contextnet_params: dict) -> ContextNet:
    contextnet = ContextNet(**contextnet_params['params'])
    state_dict = torch.load(contextnet_params['state_dict'])
    contextnet.load_state_dict(state_dict)
    return contextnet


def init_preprocess(preprocess_params: dict) -> Compose:
    return Compose([
        tr.__dict__[k](**v) if type(v) == dict else tr.__dict__[k]()
        for k, v in preprocess_params.items()
    ])


def main():
    params_path = parse_args().params_path
    parameters = load_ruamel(params_path)
    model_dir = parameters["out_dir"]
    os.makedirs(model_dir, exist_ok=True)

    parameters['model'] = init_model(parameters['model'])
    model = ContextNetRanker(**parameters['model'])
    for p in model.context_net.parameters():
        p.requires_grad = False
    for p in model.resnet.parameters():
        p.requires_grad = False

    parameters['dataset']['train']['preprocess'] = init_preprocess(parameters['dataset']['train']['preprocess'])
    parameters['dataset']['val']['preprocess'] = init_preprocess(parameters['dataset']['val']['preprocess'])

    train_data = ContextNetRankerDataset(**parameters['dataset']['train'])
    train_loader = DataLoader(dataset=train_data, **parameters['dataloader'])

    val_data = ContextNetRankerDataset(**parameters['dataset']['val'])
    val_loader = DataLoader(dataset=val_data, **parameters['dataloader'])

    optimizer = Adam(params=model.parameters(), **parameters['optimizer'])

    criterion = CosineEmbeddingLoss(**parameters['criterion'])


    # saving tf-idf and one-hot-encoder
    joblib.dump(model.title_tf_idf_vectorizer, f'{model_dir}/title_vectorizer.pkl')
    joblib.dump(model.comment_tf_idf_vectorizer, f'{model_dir}/comment_vectorizer.pkl')
    joblib.dump(model.attribute_vectorizer, f"{model_dir}/attribute_vectorizer.pkl")

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