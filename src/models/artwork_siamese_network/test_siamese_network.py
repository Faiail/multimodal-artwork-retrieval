from src.models.artwork_siamese_network.fusion_module import FusionModule
from src.models.artwork_siamese_network.ArtworkSiameseNetwork import ArtworkSiameseNetwork
from src.models.feature_projector.FeatureProjector import FeatureProjector
from src.data.SiameseDataset import SiameseCatalogueDataset, SiameseTestDataset
from src.utils import load_ruamel
from src.models.utils import parse_args
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import torchmetrics
import os
from typing import Any
import json


def _init_model(params: dict) -> FusionModule:
    model = ArtworkSiameseNetwork(**params.get('params'))
    state_dict = torch.load(params.get('state_dict'))
    model.load_state_dict(state_dict)
    return model.fusion_module


def _init_aux_models(params: dict) -> dict[str, FeatureProjector]:
    out = {}
    for name, p_dict in params.items():
        model = FeatureProjector(**p_dict.get('params'))
        state_dict = torch.load(p_dict.get('state_dict'))
        model.load_state_dict(state_dict)
        out.update({name: model})
    return out


def _init_datasets(params: dict) -> (SiameseCatalogueDataset, SiameseTestDataset):
    catalogue_dataset = SiameseCatalogueDataset(**params.get('catalogue'))
    test_dataset = SiameseTestDataset(**params.get('test'))
    return catalogue_dataset, test_dataset


def _init_metrics(metrics_params) -> dict[str, Any]:
    return {
        k: torchmetrics.retrieval.__dict__[v['type']](**v['params'])
        for k, v in metrics_params.items()
    }


def _get_indexes(cat_dim, b_s, device='cuda') -> torch.Tensor:
    return torch.cat([torch.as_tensor([i]*cat_dim) for i in range(b_s)]).to(device)


def _prepare_for_metrics(
        predictions: torch.Tensor,
        target: torch.Tensor,
        device: str = 'cuda',
) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    flat_pred = predictions.view(-1).clone().cpu()
    flat_target = target.view(-1).clone().cpu()
    flat_indexes = _get_indexes(cat_dim=predictions.size(1), b_s=predictions.size(0), device=device)
    return flat_pred, flat_target, flat_indexes


@torch.no_grad
def encode_cat_features(
        catalogue: DataLoader,
        model: FusionModule,
        aux_models: dict[str, FeatureProjector],
        pbar: bool = True,
        device: str = 'cuda',
) -> torch.Tensor:
    ans = None
    bar = tqdm(enumerate(catalogue)) if pbar else enumerate(catalogue)
    model = model.to(device)

    # aux models to proper device
    for k in aux_models:
        aux_models[k] = aux_models[k].to(device)
    assert len(catalogue.dataset.modalities) == 1
    mod = catalogue.dataset.modalities[0]
    for ix, data in bar:
        x = data[mod].to(device)
        complete_data = {k: v(x) for k, v in aux_models.items()}
        complete_data.update({mod: x})
        _, bacth_data = zip(*sorted(complete_data.items()))
        feats = model(*bacth_data).cpu()
        if ans is None:
            ans = torch.zeros(size=(len(catalogue.dataset), feats.size(1)))
            ans[(ix*catalogue.batch_size): ((ix+1) * catalogue.batch_size), :] = feats
    return ans


@torch.no_grad
def test_loop(
        model: FusionModule,
        aux_models: dict[str, FeatureProjector],
        test_loader: DataLoader,
        catalogue: torch.Tensor,
        metrics: dict[str, Any],
        pbar: bool = True,
        device: str = 'cuda',
) -> dict[str, float]:
    bar = tqdm(test_loader) if pbar else test_loader
    catalogue = catalogue.to(device)
    model = model.to(device)
    # aux model to proper device
    for k in aux_models:
        aux_models[k] = aux_models[k].to(device)

    mod = test_loader.dataset.modalities[0]
    for data in bar:
        label = data.pop('score').to(device)
        x = data[mod].to(device)
        complete_data = {k: v(x) for k, v in aux_models.items()}
        complete_data.update({mod: x})
        _, bacth_data = zip(*sorted(complete_data.items()))
        feats = model(*bacth_data)
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

    aux_cat_models = _init_aux_models(parameters['auxiliary_models_cat'])
    aux_query_models = _init_aux_models(parameters['auxiliary_models_query'])

    catalogue_dataset, test_dataset = _init_datasets(parameters['dataset'])

    catalogue_loader = DataLoader(catalogue_dataset, **parameters['dataloader'])
    test_loader = DataLoader(test_dataset, **parameters['dataloader'])

    metrics = _init_metrics(parameters['metrics'])

    catalogue_features = encode_cat_features(
        catalogue=catalogue_loader,
        model=model,
        aux_models=aux_cat_models,
        **parameters.get('encoding')
    )

    results = test_loop(
        model=model,
        aux_models=aux_query_models,
        test_loader=test_loader,
        catalogue=catalogue_features,
        metrics=metrics,
        **parameters.get('test_loop')
    )

    out_dir = parameters.get('out_dir')
    os.makedirs(out_dir, exist_ok=True)
    print(f'saving to {out_dir}/test_metrics_text2img.json')
    with open(f'{out_dir}/test_metrics_text2img.json', 'w+') as f:
        json.dump(results, f)


if __name__ == '__main__':
    main()
