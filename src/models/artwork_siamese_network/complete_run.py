from .utils import ResultDict, Run
import torch
from src.data import DataModality, Mode
import accelerate
import random


class CompleteRun(Run):
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
        super().__init__(
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
            accelerator,
        )
        self.source_modalities = None
        self.dest_modalities = None
        self.reset_modalities()

    def reset_modalities(self) -> None:
        self.source_modalities = [
            random.choice([DataModality.IMAGE.value, DataModality.TEXT.value])
        ]
        if random.choice(["graph", "no_graph"]) == "graph":
            self.source_modalities.append(DataModality.GRAPH.value)

        self.dest_modalities = (
            [DataModality.IMAGE.value]
            if DataModality.TEXT.value in self.source_modalities
            else [DataModality.TEXT.value]
        )
        if random.choice(["graph", "no_graph"]) == "graph":
            self.dest_modalities.append(DataModality.GRAPH.value)

    @torch.no_grad
    def encode_raw_info(self, data_dict: dict[str, dict]):
        for ix, (k, d) in enumerate(data_dict.items()):
            target = self.source_modalities if ix == 0 else self.dest_modalities
            for data_t, tensor in d.items():
                if (
                    self.train_loader.dataset.mode[data_t] == Mode.RAW.value
                    and data_t in target
                ):
                    if data_t == DataModality.IMAGE.value:
                        data_dict[k][data_t] = self.backbone.encode_image(tensor)
                    elif data_t == DataModality.TEXT.value:
                        tokens = self.tokenizer(tensor[0]).to(self.accelerator.device)
                        data_dict[k][data_t] = self.backbone.encode_text(tokens)
                data_dict[k][data_t] = data_dict[k][data_t]
        return data_dict

    def train_epoch(self, epoch):
        cumulated_loss = torch.as_tensor([0.0]).to(self.accelerator.device)
        bar = self.get_bar(self.train_loader, epoch)
        # insert bar
        for ix, data_dict in bar:
            score = data_dict.pop("score")
            score = score.float()

            data_dict = self.encode_raw_info(data_dict)
            # filter on source and dest modalities randomly chosen
            x = data_dict["x"]
            x = {k: v for k, v in x.items() if k in self.source_modalities}
            y = data_dict["y"]
            y = {k: v for k, v in y.items() if k in self.dest_modalities}

            self.reset_modalities()

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

                # filter on source and dest modalities randomly chosen
                x = data_dict["x"]
                x = {k: v for k, v in x.items() if k in self.source_modalities}
                y = data_dict["y"]
                y = {k: v for k, v in y.items() if k in self.dest_modalities}

                self.val_loader.dataset.reset_modalities()
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
