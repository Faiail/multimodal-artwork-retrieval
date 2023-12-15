import os

import numpy as np
import torch
from tqdm import tqdm
import open_clip
from src.data.FeatureProjectorDataset import DataModality, Mode


def train_model(
        model,
        train_loader,
        val_loader,
        num_epochs,
        optimizer,
        criterion,
        accum_iter,
        scheduler,
        early_stop,
        device,
        pbar,
        verbose,
):
    # setting iterator
    loaders = {
        'train': train_loader,
        'val': val_loader
    }
    backbone, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    del _
    backbone = backbone.to(device)
    for p in backbone.parameters():
        p.requires_grad = False

    model = model.to(device)
    for epoch in range(1, num_epochs + 1):
        for phase, loader in loaders.items():
            cumulated_loss = 0.0
            with torch.set_grad_enabled(phase == 'train'):
                iter_loader = tqdm(loader) if pbar else loader
                for step, (x, y) in enumerate(iter_loader):
                    x = x.to(device)
                    y = y.to(device)

                    with torch.no_grad():
                        if loader.dataset.mode == Mode.RAW.value:
                            if loader.dataset.source_modality == DataModality.IMAGE.value:
                                x = backbone.encode_image(x)
                            elif loader.dataset.source_modality == DataModality.TEXT.value:
                                x = backbone.encode_text(x)

                    out = model(x)
                    loss = criterion(out, y)

                    if torch.isnan(loss.cpu()).any():
                        os.makedirs('./test/', exists_ok=True)
                        torch.save(model.state_dict(), './test/model.pt')
                        torch.save(x, './test/x.pt')
                        torch.save(y, './test/y.pt')
                        torch.save(torch.as_tensor(cumulated_loss), './test/cumulated_loss.pt')
                        exit()


                    cumulated_loss = cumulated_loss + loss.cpu().item()
                    if phase == 'train':
                        loss = loss / accum_iter
                        loss.backward()
                        if ((step + 1) % accum_iter == 0) or (step + 1 == len(loader)):  # gradient accumulation
                            optimizer.step()
                            optimizer.zero_grad()

                cumulated_loss = cumulated_loss / len(loader)
                if verbose:
                    print(f'Epoch {epoch:03d}: {phase} Loss: {cumulated_loss:.4f}')
                    scheduler.step(cumulated_loss)
                if phase == 'val':
                    early_stop(cumulated_loss, model)
                    if early_stop.early_stop:
                        if verbose:
                            print(f'early stopping at epoch {epoch:03d}')
                        return -early_stop.best_score
    return -early_stop.best_score


def compute_loss(criterion, dataloader, model, device, pbar):
    running_loss = 0.0
    backbone, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    backbone = backbone.to(device)
    for p in backbone.parameters():
        p.requires_grad = False
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        out = model(x)
        running_loss = running_loss + criterion(out, y).item()
    return running_loss / len(dataloader)
