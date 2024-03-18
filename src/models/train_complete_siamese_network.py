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


class CompleteOptunaOptimizer(Optimizer):
    def __init__(self, params, accelerator: Accelerator):
        self.params = params
        super()._init_augmentations()
        self.accelerator = accelerator
        self._get_space()
        os.makedirs(params.get("out_dir"), exist_ok=True)
        self._create_study()

    def _create_study(self) -> None:
        self.study = []
        if self.accelerator.is_main_process:
            self.study = optuna.create_study(direction="minimize")
            accelerate.utils.broadcast_object_list([self.study])
        self.accelerator.wait_for_everyone()
        self.study = self.study if isinstance(self.study, Study) else self.study[0]

    def _get_space(self) -> None:
        self.space = []
        if self.accelerator.is_main_process:
            params = self.params.get("optuna", {})
            self.space = {k: get_optuna_distribution(v) for k, v in params.items()}
            accelerate.utils.broadcast_object_list([self.space])
        self.accelerator.wait_for_everyone()
        self.space = self.space if isinstance(self.space, dict) else self.space[0]

    def ask_params(self) -> optuna.Trial:
        trial = []
        if self.accelerator.is_main_process:
            trial = self.study.ask(self.space)
            accelerate.utils.broadcast_object_list([trial])
        self.accelerator.wait_for_everyone()
        return trial[0] if isinstance(trial, list) else trial

    def tell_result(self, trial, result) -> None:
        if self.accelerator.is_main_process:
            self.study.tell(trial, result)
            accelerate.utils.broadcast_object_list([self.study])
        self.accelerator.wait_for_everyone()
        self.study = self.study if isinstance(self.study, Study) else self.study[0]

    def get_best_trial(self) -> optuna.Trial:
        best_trial = []
        if self.accelerator.is_main_process:
            best_trial = self.study.best_trial
            accelerate.utils.broadcast_object_list([best_trial])
        self.accelerator.wait_for_everyone()
        return best_trial[0] if isinstance(best_trial, list) else best_trial

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
            model_name = parameters["model"].pop("name")
            model = model_registry[model_name](parameters["model"])

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


def get_accelerator() -> Accelerator:
    kwargs = [
        DistributedDataParallelKwargs(find_unused_parameters=False),
    ]
    return Accelerator(kwargs_handlers=kwargs)


def main():
    params_file = ut.parse_args().params_path
    params = load_ruamel(params_file)
    accelerator = get_accelerator()
    optimizer = CompleteOptunaOptimizer(params=params, accelerator=accelerator)
    optimizer.optimize()
    best_trial = optimizer.get_best_trial()
    accelerator.print(best_trial)
    if accelerator.is_main_process:
        joblib.dump(best_trial, f'{params.get("out_dir")}/best_trial.pkl')


if __name__ == "__main__":
    main()
