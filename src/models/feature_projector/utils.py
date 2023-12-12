import torch
from tqdm import tqdm
import open_clip
from src.data.FeatureProjectorDataset import DataModality


def feat_extract(x, y, backbone, source_modality, dest_modality):
    if source_modality == DataModality.IMAGE.value:
        x = backbone.encode_image(x)
    elif source_modality == DataModality.TEXT.value:
        x = backbone.encode_text(x)
    if dest_modality == DataModality.IMAGE.value:
        y = backbone.encode_image(y)
    elif dest_modality == DataModality.TEXT.value:
        y = backbone.encode_text(y)
    return x, y


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

                    x, y = feat_extract(
                        x=x,
                        y=y,
                        backbone=backbone,
                        source_modality=loader.dataset.source_modality,
                        dest_modality=loader.dataset.dest_modality,
                    )

                    out = model(x)
                    loss = criterion(out, y)

                    if phase == 'train':
                        loss /= accum_iter
                        loss.backward()
                        if ((step + 1) % accum_iter == 0) or (step + 1 == len(loader)):  # gradient accumulation
                            optimizer.step()
                            optimizer.zero_grad()
                    else:
                        cumulated_loss += loss.cpu().item()

                if phase == 'val':
                    cumulated_loss /= len(loader)
                    if verbose:
                        print(f'Epoch {epoch:03d}: Val Loss: {cumulated_loss:.4f}')
                    scheduler.step(cumulated_loss)
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

        x, y = feat_extract(
            x=x,
            y=y,
            backbone=backbone,
            source_modality=dataloader.dataset.source_modality,
            dest_modality=dataloader.dataset.dest_modality,
        )

        out = model(x)
        running_loss += criterion(out, y).item()
    return running_loss / len(dataloader)


