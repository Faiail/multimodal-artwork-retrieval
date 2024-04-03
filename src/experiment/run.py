import torch
from tqdm import tqdm
from src.data.utils import Mode, DataModality
from accelerate import Accelerator, DistributedDataParallelKwargs
from torch.utils.data import DataLoader
import accelerate
from src.models import ResultDict

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
            self.model,
            self.backbone,
            self.tokenizer,
            self.train_loader,
            self.val_loader,
            self.criterion_out,
            self.criterion_emb,
            self.scheduler,
            self.optimizer,
        )

    def get_test_bar(self, loader: DataLoader, desc: str):
        if not self.bar:
            return enumerate(loader)
        return tqdm(
            enumerate(loader),
            total=len(loader),
            desc=desc,
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
        bar.set_postfix({"loss": loss})

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
                                data_dict[k][data_t] = self.backbone.encode_image(
                                    tensor
                                )
                            elif data_t == DataModality.TEXT.value:
                                tokens = self.tokenizer(tensor[0]).to(
                                    self.accelerator.device
                                )
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
            loss = 0.1 * self.criterion_out(preds, score) + 0.9 * self.criterion_emb(
                fused_a, fused_b, score
            )
            cumulated_loss = cumulated_loss + loss
            self.update_bar(loss=cumulated_loss.cpu().item(), bar=bar)
            self.accelerator.backward(loss)
            self.optimizer.step()
            self.optimizer.zero_grad()
        cumulated_loss = self.accelerator.gather(cumulated_loss)
        cumulated_loss = cumulated_loss.cpu().sum().item() / len(self.train_loader)
        self.accelerator.print(
            f"Epoch {epoch}/{self.num_epochs}: training loss: {cumulated_loss:.4f}"
        )

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
                loss = 0.1 * self.criterion_out(
                    preds, score
                ) + 0.9 * self.criterion_emb(fused_a, fused_b, score)
                cumulated_loss = cumulated_loss + loss
                self.update_bar(loss=cumulated_loss.cpu().item(), bar=bar)
            cumulated_loss = self.accelerator.gather(
                cumulated_loss
            ).sum().cpu().item() / len(self.val_loader)
            self.accelerator.print(
                f"Epoch {epoch}/{self.num_epochs}: validation loss: {cumulated_loss:.4f}"
            )
            self.scheduler.step(cumulated_loss)
            best_loss = torch.empty(size=(1,), device=self.accelerator.device)
            if self.accelerator.is_main_process:
                self.early_stop(cumulated_loss, self.accelerator)
                if self.early_stop.early_stop:
                    self.accelerator.set_trigger()
                    self.accelerator.print(
                        f"Early stop at epoch {epoch}/{self.num_epochs}"
                    )
                best_loss = torch.as_tensor([-self.early_stop.best_score]).to(
                    self.accelerator.device
                )
            best_loss = accelerate.utils.broadcast(best_loss)
            self.accelerator.wait_for_everyone()
            return self.accelerator.check_trigger(), best_loss.cpu().item()

    def launch(self):
        self.accelerator.print("Start training loop")
        val_loss = None
        for epoch in range(1, self.num_epochs + 1):
            self.accelerator.print(f"Epoch {epoch}/{self.num_epochs}")
            self.train_epoch(epoch)

            stop, val_loss = self.validate(epoch)
            # print and stop everything
            if stop:
                return val_loss
        return val_loss
