import torch
from src.utils import load_ruamel
import src.models.utils as ut
from src.models.optimizer import BaseOptimizer
from src.data.utils import DataModality
import open_clip
from src.models.artwork_siamese_network.ArtworkSiameseNetwork import (
    ArtworkSiameseNetwork,
)
from src.data.SiameseDataset import SiameseDataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from src.models.loss.loss import BinaryFocalLoss
from torch.nn import CosineEmbeddingLoss
from src.models.EarlyStopping import ParallelEarlyStopping
import os
from src.models.artwork_siamese_network import utils as tr_ut
from accelerate import Accelerator, DistributedDataParallelKwargs
import optuna
from src.models.utils import get_optuna_distribution
import json
import joblib


class Optimizer(BaseOptimizer):
    def __init__(
        self,
        params,
    ):
        super().__init__(params)
        self._init_augmentations()

    def _init_augmentations(self):
        for split in self.params["dataset"].keys():
            preprocess_parameters = self.params["dataset"][split].get("preprocess", {})
            for k, v in preprocess_parameters.items():
                if k == DataModality.IMAGE.value:
                    _, _, preprocess = open_clip.create_model_and_transforms(
                        **self.params["backbone"]
                    )
                    preprocess.transforms.pop(2)
                    external_augmentations = ut.convert_image_preprocess(v)
                    for val in external_augmentations:
                        preprocess.transforms.insert(-1, val)
                elif k == DataModality.TEXT.value:
                    preprocess = ut.convert_text_preprocess(v)
                preprocess_parameters[k] = preprocess
            self.params["dataset"][split]["preprocess"] = preprocess_parameters
        return self.params["dataset"]

    def objective(self, params):
        print(
            f'Making configuration {self.current_run + 1}/{self.params["n_trials"]} with parameters {params}'
        )
        parameters = self.apply_params(params)
        model = ArtworkSiameseNetwork(**parameters["model"])

        train_data = SiameseDataset(**parameters["dataset"]["train"])
        train_loader = DataLoader(dataset=train_data, **parameters["dataloader"])

        val_data = SiameseDataset(**parameters["dataset"]["val"])
        val_loader = DataLoader(dataset=val_data, **parameters["dataloader"])

        optimizer = AdamW(model.parameters(), **parameters["optimizer"])

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, **parameters["scheduler"]
        )

        # add different criteria
        criterion_out = BinaryFocalLoss(**parameters["criterion_out"])
        criterion_emb = CosineEmbeddingLoss(**parameters["criterion_emb"])

        model_dir = parameters["out_dir"]
        os.makedirs(model_dir, exist_ok=True)
        ut.save_params(f"{model_dir}/params_{self.current_run}.json", params)
        early_stop = ParallelEarlyStopping(
            out_dir=f"{model_dir}/{self.current_run}", **parameters["early_stop"]
        )

        backbone, _, _ = open_clip.create_model_and_transforms(**parameters["backbone"])
        for p in backbone.parameters():
            p.requires_grad = False
        del _
        tokenizer = open_clip.get_tokenizer(**parameters["tokenizer"])

        self.current_run += 1

        run = tr_ut.Run(
            model=model,
            backbone=backbone,
            tokenizer=tokenizer,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            early_stop=early_stop,
            criterion_out=criterion_out,
            criterion_emb=criterion_emb,
            num_epochs=parameters["num_epochs"],
            bar=parameters["pbar"],
        )

        return run.launch()

    def test_model(self):
        raise NotImplementedError()


class OptunaOptimizer(Optimizer):
    def __init__(self, params, accelerator: Accelerator):
        super().__init__(params)
        self.accelerator = accelerator
        self._get_space()
        # TODO: fix single study for different gpus
        self.study = self.create_or_load_study()

    def create_or_load_study(self):
        if self.accelerator.is_main_process:
            study = optuna.create_study(direction="minimize")
            joblib.dump(study, f'{self.params.get("out_dir")}/tmp_study.pkl')
        self.accelerator.wait_for_everyone()
        return joblib.load(f'{self.params.get("out_dir")}/tmp_study.pkl')

    def _get_space(self):
        params = self.params.get("optuna", {})
        self.space = {k: get_optuna_distribution(v) for k, v in params.items()}

    def optimize(self):
        self.accelerator.print("Start optimization")
        for current_run_id in range(self.params.get("n_trials")):
            self.current_run = current_run_id
            # do not ask multiple times the parameters
            if self.accelerator.is_main_process:
                stage_params = self.study.ask(self.space).params
                # save stage params to disk
                with open(f"{self.params.get('out_dir')}/tmp_params.json", "w+") as f:
                    json.dump(stage_params, f)

            # wait and load params generated
            self.accelerator.wait_for_everyone()
            with open(f"{self.params.get('out_dir')}/tmp_params.json", "r") as f:
                stage_params = json.load(f)
            parameters = self.apply_params(stage_params=stage_params)

            # preparing for the run
            self.accelerator.print(
                f'Making configuration {self.current_run + 1}/{self.params["n_trials"]} with parameters {stage_params}'
            )
            model = ArtworkSiameseNetwork(**parameters["model"])

            train_data = SiameseDataset(**parameters["dataset"]["train"])
            train_loader = DataLoader(dataset=train_data, **parameters["dataloader"])

            val_data = SiameseDataset(**parameters["dataset"]["val"])
            val_loader = DataLoader(dataset=val_data, **parameters["dataloader"])

            optimizer = AdamW(model.parameters(), **parameters["optimizer"])

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, **parameters["scheduler"]
            )

            # add different criteria
            criterion_out = BinaryFocalLoss(**parameters["criterion_out"])
            criterion_emb = CosineEmbeddingLoss(**parameters["criterion_emb"])

            model_dir = parameters["out_dir"]
            if self.accelerator.is_main_process:
                os.makedirs(model_dir, exist_ok=True)
                ut.save_params(
                    f"{model_dir}/params_{self.current_run}.json", stage_params
                )
            early_stop = ParallelEarlyStopping(
                out_dir=f"{model_dir}/{self.current_run}", **parameters["early_stop"]
            )

            backbone, _, _ = open_clip.create_model_and_transforms(
                **parameters["backbone"]
            )
            for p in backbone.parameters():
                p.requires_grad = False
            del _
            tokenizer = open_clip.get_tokenizer(**parameters["tokenizer"])

            self.current_run += 1

            run = tr_ut.Run(
                model=model,
                backbone=backbone,
                tokenizer=tokenizer,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                early_stop=early_stop,
                criterion_out=criterion_out,
                criterion_emb=criterion_emb,
                num_epochs=parameters["num_epochs"],
                bar=parameters["pbar"],
            )
            result = run.launch()
            self.study.tell(result)

    def test_model(self):
        raise NotImplementedError()


def get_accelerator() -> Accelerator:
    kwargs = [
        DistributedDataParallelKwargs(find_unused_parameters=False),
    ]
    return Accelerator(kwargs_handlers=kwargs)


def main_optuna():
    params_file = ut.parse_args().params_path
    params = load_ruamel(params_file)
    accelerator = get_accelerator()
    optimizer = OptunaOptimizer(params=params, accelerator=accelerator)
    optimizer.optimize()


if __name__ == "__main__":
    main_optuna()
