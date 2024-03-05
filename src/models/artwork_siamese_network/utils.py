import torch
from tqdm import tqdm
from src.data.utils import Mode, DataModality
from accelerate import Accelerator, DistributedDataParallelKwargs
from src.models.artwork_siamese_network.ArtworkSiameseNetwork import ResultDict
from torch.utils.data import DataLoader


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
                    score = score.float().to(device)

                    with torch.no_grad():
                        for k, d in data_dict.items():
                            for data_t, tensor in d.items():
                                if loader.dataset.mode[data_t] == Mode.RAW.value:
                                    if data_t == DataModality.IMAGE.value:
                                        data_dict[k][data_t] = backbone.encode_image(tensor.to(device))
                                    elif data_t == DataModality.TEXT.value:
                                        tokens = tokenizer(tensor[0])
                                        data_dict[k][data_t] = backbone.encode_text(tokens.to(device))
                                data_dict[k][data_t] = data_dict[k][data_t].to(device)
                    x = tuple(data_dict["x"].values())
                    y = tuple(data_dict["y"].values())

                    out = model(x, y).squeeze()
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


class Run:
    def __init__(
            self,
            model,
            backbone,
            tokenizer,
            train_loader,
            val_loader,
            num_epochs,
            optimizer,
            criterion_out,
            criterion_emb,
            scheduler,
            early_stop,
            bar,
            accelerator=None,
    ):
        self.model = model
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.optimizer = optimizer
        self.criterion_out = criterion_out
        self.criterion_emb = criterion_emb
        self.scheduler = scheduler
        self.early_stop = early_stop
        self.bar = bar
        self.accelerator = accelerator
        if self.accelerator is None:
            kwargs = [
                DistributedDataParallelKwargs(find_unused_parameters=False),
            ]
            self.accelerator = Accelerator(kwargs_handlers=kwargs)
        (
            self.model,
            self.backbone,
            self.tokenizer,
            self.train_loader,
            self.val_loader,
            self.criterion_out,
            self.criterion_emb,
            self.scheduler,
            self.optimizer,
        ) = self.accelerator.prepare(
            self.model, self.backbone, self.tokenizer, self.train_loader,
            self.val_loader, self.criterion_out, self.criterion_emb,
            self.scheduler, self.optimizer,
        )

    def get_bar(self, loader: DataLoader, epoch: int):
        if not self.bar:
            return enumerate(loader)
        return tqdm(
            enumerate(loader),
            total=len(loader),
            postfix={"loss": 0},
            desc=f"Epoch {epoch}/{self.num_epochs}",
        )

    def update_bar(self, loss: float, bar):
        if not self.bar:
            return
        bar.set_postfix(
            {
                "loss": loss
            }
        )

    def train_epoch(self, epoch):
        cumulated_loss = torch.as_tensor([0.0]).to(self.accelerator.device)
        bar = self.get_bar(self.train_loader, epoch)
        # insert bar
        for ix, data_dict in bar:
            # print("Data fetched")
            score = data_dict.pop("score")
            score = score.float()

            # print(score.device)
            # input()

            with torch.no_grad():
                for k, d in data_dict.items():
                    for data_t, tensor in d.items():
                        if self.train_loader.dataset.mode[data_t] == Mode.RAW.value:
                            if data_t == DataModality.IMAGE.value:
                                data_dict[k][data_t] = self.backbone.encode_image(tensor)
                            elif data_t == DataModality.TEXT.value:
                                tokens = self.tokenizer(tensor[0]).to(self.accelerator.device)
                                # print(tokens.device)
                                # input()
                                data_dict[k][data_t] = self.backbone.encode_text(tokens)
                        data_dict[k][data_t] = data_dict[k][data_t]
            # update with new format here
            x = data_dict["x"]
            y = data_dict["y"]
            _, x = zip(*sorted(x.items()))
            _, y = zip(*sorted(y.items()))

            out = self.model(x, y, return_fused=True)
            preds = out[ResultDict.PRED].squeeze()
            fused_a, fused_b = out[ResultDict.FUSED]
            loss = 0.1 * self.criterion_out(preds, score) + 0.9 * self.criterion_emb(fused_a, fused_b, score)
            self.update_bar(loss=loss.cpu().item(), bar=bar)
            cumulated_loss = cumulated_loss + loss
            self.accelerator.backward(loss)
            self.optimizer.step()
            self.optimizer.zero_grad()
        cumulated_loss = self.accelerator.gather(cumulated_loss)
        if self.accelerator.is_main_process:
            cumulated_loss = cumulated_loss.cpu().mean().item() / len(self.train_loader)
            self.accelerator.print(f"Epoch {epoch}/{self.num_epochs}: training loss: {cumulated_loss:.4f}")

    def validate(self, epoch):
        cumulated_loss = torch.as_tensor([0.0]).to(self.accelerator.device)
        bar = self.get_bar(self.val_loader, epoch)
        # insert bar
        with torch.no_grad():
            for ix, data_dict in bar:
                score = data_dict.pop("score")
                score = score.float()

                # update with new format
                x = data_dict["x"]
                y = data_dict["y"]
                _, x = zip(*sorted(x.items()))
                _, y = zip(*sorted(y.items()))
                
                out = self.model(x, y, return_fused=True)
                preds = out[ResultDict.PRED].squeeze()
                fused_a, fused_b = out[ResultDict.FUSED]
                loss = 0.1 * self.criterion_out(preds, score) + 0.9 * self.criterion_emb(fused_a, fused_b, score)
                self.update_bar(loss=loss.cpu().item(), bar=bar)
                cumulated_loss = cumulated_loss + loss
            if self.accelerator.is_main_process:
                cumulated_loss = cumulated_loss.cpu().mean().item() / len(self.train_loader)
            self.early_stop(cumulated_loss, self.accelerator, self.model)
            self.scheduler.step(cumulated_loss)
            if self.early_stop.early_stop:
                self.accelerator.set_trigger()
                self.accelerator.print(f"Early stop at epoch {epoch}/{self.num_epochs}")
            self.accelerator.wait_for_everyone()
            return self.accelerator.check_trigger(), -self.early_stop.best_score

    def launch(self):
        self.accelerator.print("Start training loop")
        val_loss = None
        for epoch in range(self.num_epochs):
            self.accelerator.print(f"Epoch {epoch}/{self.num_epochs}")
            self.train_epoch(epoch)

            stop, val_loss = self.validate(epoch)
            # print and stop everything
            if stop:
                return val_loss
        return val_loss

