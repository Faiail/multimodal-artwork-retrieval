import torch
from tqdm import tqdm
from src.data.utils import Mode, DataModality


def train_model(
        model,
        backbone,
        tokenizer,
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
    backbone = backbone.to(device)
    for p in backbone.parameters():
        p.requires_grad = False

    model = model.to(device)
    for epoch in range(1, num_epochs + 1):
        for phase, loader in loaders.items():
            cumulated_loss = 0.0
            with torch.set_grad_enabled(phase == 'train'):
                iter_loader = tqdm(loader) if pbar else loader
                for step, data_dict in enumerate(iter_loader):
                    score = data_dict.pop("score")
                    score = score.to(device)

                    with torch.no_grad():
                        for k, d in data_dict.items():
                            for data_t, tensor in d.items():
                                if loader.dataset.mode[data_t] == Mode.RAW.value:
                                    if data_t == DataModality.IMAGE.value:
                                        data_dict[k][data_t] = backbone.encode_image(tensor)
                                    elif data_t == DataModality.TEXT.value:
                                        tokens = tokenizer.tokenize(tensor)
                                        data_dict[k][data_t] = backbone.encode_text(tokens)
                                data_dict[k][data_t] = data_dict[k][data_t].to(device)
                    x = tuple(data_dict["x"].items())
                    y = tuple(data_dict["y"].items())

                    out = model(x, y)
                    loss = criterion(out, score)

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
