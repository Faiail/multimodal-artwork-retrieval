from src.utils import load_ruamel
from src.models.utils import parse_args
import src.models as mod
import torch
import torchmetrics.retrieval
import src.data
from torch.utils.data import DataLoader
from src.models.predictor import Predictor
from tqdm import tqdm
import json
import os
from torchvision.transforms import Compose
import torchvision.transforms as tr


"""
MODEL TAXONOMY
model:
- type: ClassName
- params: dict of parameters
- state_dict: state dict path
 
"""


def _init_model(model_params):
    model_cls = mod.__dict__[model_params.get('type')]
    model = model_cls(**model_params['params'])
    state_dict = torch.load(model_params['state_dict'])
    model.load_state_dict(state_dict)
    return model


"""
DATASET TAXONOMY
dataset:
- type: ClassName
- params: dict of dataset parameters
"""


def _init_preprocess(preprocess_params: dict) -> dict:
    return Compose([
        tr.__dict__[k](**v) if type(v) == dict else tr.__dict__[k]()
        for k, v in preprocess_params.items()
    ])


def _init_test_set(dataset_params, dataloader_params):
    if 'preprocess' in dataset_params.get('params').keys():
        dataset_params['params']['preprocess'] = _init_preprocess(dataset_params['params']['preprocess'])
    dataset = src.data.__dict__[dataset_params['type']](**dataset_params['params'])
    return DataLoader(dataset=dataset, **dataloader_params)


def _get_aux_models(aux_models_params):
    return None if aux_models_params is None else {k: _init_model(v) for k, v in aux_models_params.items()}


"""
METRIC TAXONOMY
metrics:
- recall_at_1:
    - type: retrieval.Recall
    - params:
        - top_k: k
"""


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
) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    flat_pred = predictions.view(-1).clone()
    flat_target = target.view(-1).clone()
    flat_indexes = _get_indexes(cat_dim=predictions.size(1), b_s=predictions.size(0))
    return flat_pred, flat_target, flat_indexes


def prepare_batch(batch, device):
    if isinstance(batch, list):
        return batch
    if isinstance(batch, dict):
        return {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
    if isinstance(batch, torch.Tensor):
        return batch.to(device)


@torch.no_grad
def test_loop(
        predictor: Predictor,
        test_loader: DataLoader,
        metrics: dict,
        pbar: bool,
        device: str = 'cuda',
        **kwargs,
) -> dict[str, float]:
    iterator = tqdm(test_loader) if pbar else test_loader
    predictor.model = predictor.model.to(device)
    for batch, label in iterator:
        batch = prepare_batch(batch, device)
        label = label.to(device)
        predictions = predictor.predict(batch)
        # flatten predictions and labels for computing metrics
        flat_pred, flat_target, flat_indexes = _prepare_for_metrics(predictions, label)
        # update metrics
        for v in metrics.values():
            v.update(flat_pred, flat_target, flat_indexes)

    # compute metrics and return
    return {k: v.compute().detach().item() for k, v in metrics.items()}


def main():
    params_path = parse_args().params_path
    params = load_ruamel(params_path)

    # data and model initialization
    model = _init_model(params['model'])
    aux_models_cat = _get_aux_models(params.get('auxiliary_models_cat', None))
    aux_models_query = _get_aux_models(params.get('aux_models_query', None))
    test_loader = _init_test_set(params['dataset'], params['dataloader'])
    assert isinstance(test_loader.dataset, src.data.PredictorTestDataset)
    test_catalogue = _init_test_set(params['catalogue'], params['dataloader'])
    assert isinstance(test_catalogue.dataset, src.data.CatalogDataset)
    metrics = _init_metrics(params['metrics'])

    predictor = Predictor(model=model,
                          auxiliary_models_cat=aux_models_cat,
                          auxiliary_models_query=aux_models_query,
                          **params['predictor'])

    predictor.encode_catalogue(catalogue=test_catalogue)

    # looping over test set
    metric_values = test_loop(predictor=predictor,
                              test_loader=test_loader,
                              metrics=metrics,
                              **params['test_loop'])
    out_dir = params['general']['out_dir']
    os.makedirs(out_dir, exist_ok=True)
    print(f'saving to {out_dir}/test_metrics.json')
    with open(f'{out_dir}/test_metrics.json', 'w+') as f:
        json.dump(metric_values, f)


if __name__ == '__main__':
    main()


