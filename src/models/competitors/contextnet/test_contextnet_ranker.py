import json
from src.models.competitors.contextnet.context_net_ranker import ContextNetRanker, ContextNet
from src.utils import load_ruamel
from src.models.utils import parse_args
from torch.utils.data import DataLoader
from src.data.context_net_dataset import ContextNetRankerTestDataset, collate_fn
import torch
import joblib
from tqdm import tqdm
import torchmetrics
import os
from torchvision.transforms import Compose
import torchvision.transforms as tr
import warnings

warnings.filterwarnings('ignore')


def _init_model(params: dict):
    model_params = params.get('params')
    # create contextnet
    context_net = ContextNet(**model_params.get('context_net').get('params'))
    state_dict = torch.load(model_params.get('context_net').get('checkpoint'))
    context_net.load_state_dict(state_dict)
    model_params['context_net'] = context_net

    # create vectorizers
    title_vectorizer = joblib.load(model_params.get('title_vectorizer'))
    model_params['title_vectorizer'] = title_vectorizer
    comment_vectorizer = joblib.load(model_params.get('comment_vectorizer'))
    model_params['comment_vectorizer'] = comment_vectorizer
    attribute_vectorizer = joblib.load(model_params.get('attribute_vectorizer'))
    model_params['attribute_vectorizer'] = attribute_vectorizer

    # create final model
    model = ContextNetRanker(**model_params)
    state_dict = torch.load(params.get('checkpoint'))
    model.load_state_dict(state_dict)
    return model


def _init_preprocess(preprocess_params: dict) -> Compose:
    return Compose([
        tr.__dict__[k](**v) if type(v) == dict else tr.__dict__[k]()
        for k, v in preprocess_params.items()
    ])


def _init_datasets(datasets_params):
    cat_params = datasets_params.get('catalogue')
    if 'preprocess' in cat_params:
        cat_params['preprocess'] = _init_preprocess(cat_params.get('preprocess'))
    catalogue_dataset = ContextNetRankerTestDataset(**cat_params)

    test_params = datasets_params.get('test')
    if 'preprocess' in test_params:
        test_params['preprocess'] = _init_preprocess(test_params.get('preprocess'))
    test_dataset = ContextNetRankerTestDataset(**test_params)

    return catalogue_dataset, test_dataset


def _init_metrics(metrics_params):
    return {
        k: torchmetrics.retrieval.__dict__[v['type']](**v['params'])
        for k, v in metrics_params.items()
    }


def _get_indexes(cat_dim, b_s, device='cuda'):
    return torch.cat([torch.as_tensor([i]*cat_dim) for i in range(b_s)]).to(device)


def _prepare_for_metrics(
        predictions: torch.Tensor,
        target: torch.Tensor,
        device: str='cuda'
) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    flat_pred = predictions.view(-1).clone().cpu()
    flat_target = target.view(-1).clone().cpu()
    flat_indexes = _get_indexes(cat_dim=predictions.size(1), b_s=predictions.size(0), device=device)
    return flat_pred, flat_target, flat_indexes


@torch.no_grad
def encode_features(
        model: ContextNetRanker,
        catalogue: DataLoader,
        task: str,
        pbar: bool = True,
        device: str = 'cuda',
) -> torch.Tensor:
    ans = None
    bar = tqdm(enumerate(catalogue)) if pbar else enumerate(catalogue)
    model = model.to(device)
    for ix, data in bar:
        if task.startswith('text'):
            comments = data['caption']
            titles = data['title']
            attributes = data['attributes']
            feats = model.encode_text(comments, titles, attributes).cpu()
        else:
            images = data['image'].to(device)
            feats = model.encode_image(images).cpu()
        if ans is None:
            ans = torch.empty(len(catalogue.dataset), feats.size(1)).float()
        ans[(ix*catalogue.batch_size): ((ix+1) * catalogue.batch_size), :] = feats
    return ans


@torch.no_grad
def test_loop(
        model: ContextNetRanker,
        test_loader: DataLoader,
        catalogue: torch.Tensor,
        metrics: dict,
        pbar: bool,
        task: str,
        device: str = 'cuda',
) -> dict[str, float]:
    bar = tqdm(test_loader) if pbar else test_loader
    catalogue = catalogue.to(device)
    model = model.to(device)
    for batch in bar:

        label = batch.pop('score').to(device)
        if task.endswith('text'):
            comments = batch['caption']
            titles = batch['title']
            attributes = batch['attributes']
            feats = model.encode_text(comments, titles, attributes)
        else:
            images = batch['image'].to(device)
            feats = model.encode_image(images)
        preds = torch.mm(feats, catalogue.T)
        flat_pred, flat_target, flat_indexes = _prepare_for_metrics(preds, label, device='cpu')

        # update metrics
        for v in metrics.values():
            v.update(flat_pred, flat_target, flat_indexes)
    return {k: v.compute().detach().item() for k, v in metrics.items()}


def main():
    params_path = parse_args().params_path
    parameters = load_ruamel(params_path)
    model = _init_model(parameters['model'])
    catalogue_dataset, test_dataset = _init_datasets(parameters['dataset'])

    catalogue_loader = DataLoader(catalogue_dataset, collate_fn=collate_fn,  **parameters['dataloader'])
    test_loader = DataLoader(test_dataset,  collate_fn=collate_fn, **parameters['dataloader'])


    metrics = _init_metrics(parameters['metrics'])
    cat_feats = encode_features(model=model,
                                catalogue=catalogue_loader,
                                **parameters.get('encoding'))

    results = test_loop(
        model=model,
        test_loader=test_loader,
        catalogue=cat_feats,
        metrics=metrics,
        **parameters.get('test_loop')
    )

    out_dir = parameters.get('out_dir')
    os.makedirs(out_dir, exist_ok=True)
    print(f'saving to {out_dir}/test_metrics.json')
    with open(f'{out_dir}/test_metrics.json', 'w+') as f:
        json.dump(results, f)


if __name__ == '__main__':
    main()



