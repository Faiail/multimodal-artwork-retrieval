from src.models.competitors.contextnet.context_net import ContextNet
from torch.utils.data import DataLoader
import torch
from src.models.EarlyStopping import EarlyStopping
from tqdm import tqdm
from src.utils import load_ruamel
from src.models.utils import parse_args
import torchvision.transforms as tr
from src.data.context_net_dataset import ContextNetDataset
import os


def train_model(
        model: ContextNet,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion_kg: torch.nn.SmoothL1Loss,
        criterion_task: torch.nn.CrossEntropyLoss,
        early_stop: EarlyStopping,
        num_epochs: int,
        device: str,
        l: float,
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
                    label = data_dict.get('gt').to(device)
                    images = data_dict.get('images').to(device)
                    node2vec_embeddings = data_dict.get('embeddings').to(device)

                    preds, projs = model(images)

                    loss_kg = criterion_kg(projs, node2vec_embeddings)
                    loss_task = criterion_task(preds, label)
                    loss = (loss_task * l) + ((1 - l) * loss_kg)

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


def init_preprocess(preprocess_params: dict) -> tr.Compose:
    return tr.Compose([
        tr.__dict__[k](**v) if type(v) == dict else tr.__dict__[k]()
        for k, v in preprocess_params.items()
    ])


def main():
    params_path = parse_args().params_path
    parameters = load_ruamel(params_path)
    model = ContextNet(**parameters['model'])

    parameters['dataset']['train']['preprocess'] = init_preprocess(parameters['dataset']['train']['preprocess'])
    parameters['dataset']['val']['preprocess'] = init_preprocess(parameters['dataset']['val']['preprocess'])

    train_data = ContextNetDataset(**parameters['dataset']['train'])
    train_loader = DataLoader(dataset=train_data, **parameters['dataloader'])

    val_data = ContextNetDataset(**parameters['dataset']['val'])
    val_loader = DataLoader(dataset=val_data, **parameters['dataloader'])

    optimizer = torch.optim.SGD(params=model.parameters(), **parameters['optimizer'])

    criterion_kg = torch.nn.SmoothL1Loss(**parameters.get('criterion_kg', {}))
    criterion_task = torch.nn.CrossEntropyLoss(**parameters.get('criterion_task', {}))

    model_dir = parameters["out_dir"]
    os.makedirs(model_dir, exist_ok=True)

    early_stop = EarlyStopping(path=f'{model_dir}/model_state_dict.pt', **parameters['early_stop'])

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion_kg=criterion_kg,
        criterion_task=criterion_task,
        optimizer=optimizer,
        early_stop=early_stop,
        **parameters['general']
    )


if __name__ == '__main__':
    main()
