from .train_siamese_network import Optimizer
from accelerate import Accelerator, DistributedDataParallelKwargs
from src.models import model_registry, CompleteArtwrokSiameseNetwork
from src.models.train_siamese_network import Optimizer
from src.models.utils import get_optuna_distribution
import accelerate
import os
import optuna
from optuna import Study
from src.data import CompleteSiameseDataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch
from torch.nn import CosineEmbeddingLoss
from src.models.loss import BinaryFocalLoss
import src.models.utils as ut
from src.models.EarlyStopping import ParallelEarlyStopping
from src.models.artwork_siamese_network.complete_run import CompleteRun
import open_clip
from src.utils import load_ruamel
import joblib
from enum import Enum
import joblib
import warnings
from copy import deepcopy
import torchmetrics
from src.data import SiameseTestDataset, SiameseCatalogueDataset
import json

warnings.filterwarnings("ignore")


class ExtFname(Enum):
    TMP_STUDY = "tmp_study.pkl"
    TMP_TRIAL = "tmp_trial.pkl"
    TMP_BEST_TRIAL = "tmp_best_trial.pkl"


class CompleteOptunaOptimizer(Optimizer):
    def __init__(self, params, accelerator: Accelerator):
        self.params = params
        super()._init_augmentations()
        self.accelerator = accelerator
        self._get_space()
        self.accelerator.print("Space created")
        os.makedirs(params.get("out_dir"), exist_ok=True)
        self._create_study()
        self.accelerator.print("Study created")

    def _create_study(self) -> None:
        if self.accelerator.is_main_process:
            study = optuna.create_study(direction="minimize")
            joblib.dump(
                study, f'{self.params.get("out_dir")}/{ExtFname.TMP_STUDY.value}'
            )
        self.accelerator.wait_for_everyone()
        self.study = joblib.load(
            f'{self.params.get("out_dir")}/{ExtFname.TMP_STUDY.value}'
        )

    def _get_space(self):
        params = self.params.get("optuna", {})
        self.space = {k: get_optuna_distribution(v) for k, v in params.items()}

    def ask_params(self):
        if self.accelerator.is_main_process:
            trial = self.study.ask(self.space)
            joblib.dump(
                trial, f"{self.params.get('out_dir')}/{ExtFname.TMP_TRIAL.value}"
            )
        self.accelerator.wait_for_everyone()
        return joblib.load(f"{self.params.get('out_dir')}/{ExtFname.TMP_TRIAL.value}")

    def tell_result(self, trial, result):
        if self.accelerator.is_main_process:
            self.study.tell(trial, result)
            joblib.dump(
                self.study, f'{self.params.get("out_dir")}/{ExtFname.TMP_STUDY.value}'
            )
        self.accelerator.wait_for_everyone()
        self.study = joblib.load(
            f'{self.params.get("out_dir")}/{ExtFname.TMP_STUDY.value}'
        )

    def get_best_trial(self):
        if self.accelerator.is_main_process:
            best_trial = self.study.best_trial
            joblib.dump(
                best_trial, f'{self.params.get("out_dir")}/{ExtFname.TMP_BEST_TRIAL}'
            )
        self.accelerator.wait_for_everyone()
        best_trial = joblib.load(
            f'{self.params.get("out_dir")}/{ExtFname.TMP_BEST_TRIAL}'
        )
        return best_trial

    def remove_tmp(self):
        if self.accelerator.is_main_process:
            for f in os.listdir(self.params.get("out_dir")):
                if f.startswith("tmp"):
                    os.remove(f'{self.params.get("out_dir")}/{f}')

    def optimize(self):
        self.accelerator.print("Start optimization")
        for current_run_id in range(self.params.get("n_trials")):
            self.current_run = current_run_id

            trial = self.ask_params()
            stage_params = trial.params
            parameters = self.apply_params(stage_params=stage_params)

            self.accelerator.print(
                f'Making configuration {self.current_run + 1}/{self.params["n_trials"]} with parameters {stage_params}'
            )
            model_params = deepcopy(parameters["model"])
            model_name = model_params.pop("name")
            model = model_registry[model_name](model_params)

            train_data = CompleteSiameseDataset(**parameters["dataset"]["train"])
            train_loader = DataLoader(dataset=train_data, **parameters["dataloader"])

            val_data = CompleteSiameseDataset(**parameters["dataset"]["val"])
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

            run = CompleteRun(
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
                accelerator=self.accelerator,
            )
            result = run.launch()
            self.tell_result(trial, result)

    def _get_metrics(
        self,
    ):
        return {
            k: torchmetrics.retrieval.__dict__[v["type"]](**v["params"])
            for k, v in self.params.get("metrics", {}).items()
        }

    @torch.no_grad
    def test(self):
        self.accelerator.print("Start testing...")
        if self.accelerator.is_main_process:
            best_trial = joblib.load(f"{self.params['out_dir']}/best_trial.pkl")
            self.accelerator.print("Loading best parameters...")
            trial_id = best_trial._trial_id
            best_params = best_trial._params
            parameters = self.apply_params(best_params)
            parameters["dataloader"]["shuffle"] = False

            for task, task_params in parameters.get("dataset").get("test", {}).items():
                if not task_params:
                    self.accelerator.print(f"No task params for task {task}")
                    continue
                self.accelerator.print(f"Making test for task {task}")
                cat_data = SiameseCatalogueDataset(**task_params.get("catalogue"))
                cat_dataloader = DataLoader(cat_data, **parameters.get("dataloader"))
                query_data = SiameseTestDataset(**task_params.get("query"))
                query_dataloader = DataLoader(
                    query_data, **parameters.get("dataloader")
                )
                metrics = self._get_metrics()
                model_params = deepcopy(parameters["model"])
                model_name = model_params.pop("name")
                model = model_registry[model_name](model_params)

                train_data = CompleteSiameseDataset(**parameters["dataset"]["train"])
                train_loader = DataLoader(
                    dataset=train_data, **parameters["dataloader"]
                )

                val_data = CompleteSiameseDataset(**parameters["dataset"]["val"])
                val_loader = DataLoader(dataset=val_data, **parameters["dataloader"])

                optimizer = AdamW(model.parameters(), **parameters["optimizer"])

                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, **parameters["scheduler"]
                )

                # add different criteria
                criterion_out = BinaryFocalLoss(**parameters["criterion_out"])
                criterion_emb = CosineEmbeddingLoss(**parameters["criterion_emb"])

                model_dir = parameters["out_dir"]
                early_stop = ParallelEarlyStopping(
                    out_dir=f"{model_dir}/{self.current_run}",
                    **parameters["early_stop"],
                )

                backbone, _, _ = open_clip.create_model_and_transforms(
                    **parameters["backbone"]
                )
                for p in backbone.parameters():
                    p.requires_grad = False
                del _
                tokenizer = open_clip.get_tokenizer(**parameters["tokenizer"])

                criterion_out = BinaryFocalLoss(**parameters["criterion_out"])
                criterion_emb = CosineEmbeddingLoss(**parameters["criterion_emb"])

                metrics = CompleteRun(
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
                    accelerator=self.accelerator,
                    cat_loader=cat_dataloader,
                    query_loader=query_dataloader,
                    metrics=metrics,
                    task=task,
                    state_dict_dir=f"{parameters['out_dir']}/{trial_id}",
                ).test()
                os.makedirs(f'{parameters.get("out_dir")}/{task}', exist_ok=True)
                with open(
                    f'{parameters.get("out_dir")}/{task}/test_metrics.json', "w+"
                ) as f:
                    json.dump(metrics, f)


def get_accelerator() -> Accelerator:
    kwargs = [
        DistributedDataParallelKwargs(find_unused_parameters=True),
    ]
    return Accelerator(kwargs_handlers=kwargs)


def main():
    params_file = ut.parse_args().params_path
    params = load_ruamel(params_file)
    accelerator = get_accelerator()
    optimizer = CompleteOptunaOptimizer(params=params, accelerator=accelerator)
    optimizer.optimize()
    best_trial = optimizer.get_best_trial()
    optimizer.remove_tmp()
    accelerator.print(best_trial)
    if accelerator.is_main_process:
        joblib.dump(best_trial, f'{params.get("out_dir")}/best_trial.pkl')
    optimizer.test()


if __name__ == "__main__":
    main()
