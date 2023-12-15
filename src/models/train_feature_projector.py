import os
import torch
import hyperopt
from src.models.feature_projector.FeatureProjector import FeatureProjector
from src.data.FeatureProjectorDataset import FeatureProjectorDataset, DataModality
from src.models.EarlyStopping import EarlyStopping
from src.models.feature_projector.utils import train_model, compute_loss
from src.utils import load_ruamel
from torch.utils.data import DataLoader
import json
import argparse
from torchvision.transforms import v2
import nlpaug.augmenter.word as naw
import open_clip


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--params_path')
    return argparser.parse_args()


def get_distribution(key, value):
    if value['type'] == 'choice':
        return hyperopt.hp.choice(key, value['range'])
    if value['type'] == 'uniform':
        return hyperopt.hp.uniform(key, *value['range'])
    raise ValueError(f'{value["type"]} not supported')


def save_params(path, params):
    with open(path, 'w+') as f:
        json.dump(params, f)


def convert_image_preprocess(params):
    AUGMENTATIONS = [v2.RandomHorizontalFlip, v2.RandomPerspective, v2.RandomRotation]
    NAMES = list(map(lambda x: x.__name__, AUGMENTATIONS))
    out = []
    for entry, val in params.items():
        if entry in NAMES:
            out.append(
                AUGMENTATIONS[NAMES.index(entry)](**val)
            )
        else:
            raise ValueError(f'cannot encode {entry}. Possible data augmentation functions: {NAMES}')
    return out


def convert_text_preprocess(params):
    AUGMENTATIONS = [naw.SynonymAug, naw.SynonymAug, naw.RandomWordAug]
    NAMES = list(map(lambda x: x.__name__, AUGMENTATIONS))
    out = []
    for entry, val in params.items():
        if entry in NAMES:
            out.append(
                AUGMENTATIONS[NAMES.index(entry)](**val)
            )
        else:
            raise ValueError(f'cannot encode {entry}. Possible data augmentation functions: {NAMES}')
    return out


class Optimizer:
    def __init__(
            self,
            params,
    ):
        self.params = params
        self._init_augmentations()
        self.space = self._get_space()
        self.best_params = None
        self.current_run = 0
        self._clean_out_dir()

    def _clean_out_dir(self):
        for f in os.listdir(self.params['out_dir']):
            os.remove(f'{self.params["out_dir"]}/{f}')

    def _init_augmentations(self):
        for split in self.params['dataset'].keys():  # for each split
            dataset_parameters = self.params['dataset'][split]  # take params
            if 'preprocess_source' not in dataset_parameters:  # if no preprocess (val-test) continue
                assert dataset_parameters['mode'] == 'embedding'
                continue
            # handle image preprocessing
            if dataset_parameters[f'source_modality'] == DataModality.IMAGE.value:
                _, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
                preprocess.transforms.pop(2)  # remove rgb conversion
                if dataset_parameters[f'preprocess_source'] != 'auto':
                    external_augmentations = convert_image_preprocess(dataset_parameters[f'preprocess_source'])
                    for val in (external_augmentations):
                        preprocess.transforms.insert(-1, val)
                dataset_parameters[f'preprocess_source'] = preprocess
            # handle text preprocessing
            if dataset_parameters[f'source_modality'] == DataModality.TEXT.value:
                if dataset_parameters[f'preprocess_source'] == 'auto':
                    preprocess = None
                else:
                    preprocess = convert_text_preprocess(dataset_parameters[f'preprocess_source'])
                dataset_parameters[f'preprocess_source'] = {
                    'preprocess': preprocess,
                    'tokenizer': open_clip.get_tokenizer('ViT-B-32')
                }
            self.params['dataset'][split] = dataset_parameters
        return dataset_parameters

    def _get_space(self):
        to_optimize = self.params['hyperopt']
        return {
            k: get_distribution(k, v) for k, v in to_optimize.items()
        }

    def apply_params(self, stage_params):
        start_params = self.params
        for k, v in stage_params.items():
            key, value = k.split('__')
            start_params[key][value] = v
        return start_params

    def objective(self, params):
        print(f'Making configuration {self.current_run + 1}/{self.params["n_trials"]} with parameters {params}')
        parameters = self.apply_params(params)
        model = FeatureProjector(**parameters['model'])
        train_data = FeatureProjectorDataset(**parameters['dataset']['train'])
        train_loader = DataLoader(dataset=train_data, **parameters['dataloader'])

        val_data = FeatureProjectorDataset(**parameters['dataset']['val'])
        val_loader = DataLoader(dataset=val_data, **parameters['dataloader'])

        optimizer = torch.optim.Adam(model.parameters(), **parameters['optimizer'])

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **parameters['scheduler'])

        model_dir = parameters["out_dir"]
        os.makedirs(model_dir, exist_ok=True)
        save_params(f'{model_dir}/params_{self.current_run}.json', params)
        early_stop = EarlyStopping(path=f'{model_dir}/model_{self.current_run}.pt', **parameters['early_stop'])

        criterion = torch.nn.SmoothL1Loss()  # maybe add parameters for criterion

        self.current_run += 1
        return train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            early_stop=early_stop,
            criterion=criterion,
            **parameters['general']
        )

    def optimize(self):
        self.best_params = hyperopt.fmin(
            fn=self.objective,
            space=self.space,
            algo=hyperopt.tpe.suggest,
            max_evals=self.params['n_trials'],
            show_progressbar=self.params["progress_bar"],
        )
        return self.best_params

    def find_best_model(self):
        best_values = {
            k: self.params['hyperopt'][k]['range'][v] if self.params['hyperopt'][k]['type'] == 'choice' else v
            for k, v in self.best_params.items()
        }
        for f in filter(lambda x: x.endswith('json'), os.listdir(self.params['out_dir'])):
            with open(f"{self.params['out_dir']}/{f}") as json_f:
                data = json.load(json_f)
                if data == best_values:
                    return int(f"{self.params['out_dir']}/{f}"[-6])
        raise Exception('Bad. Cannot find a single configuration that is the best one.')

    @torch.no_grad
    def test_model(self):
        best_values = {
            k: self.params['hyperopt'][k]['range'][v] if self.params['hyperopt'][k]['type'] == 'choice' else v
            for k, v in self.best_params.items()
        }
        parameters = self.apply_params(best_values)
        test_data = FeatureProjectorDataset(**parameters['dataset']['train'])
        test_loader = DataLoader(dataset=test_data, **parameters['dataloader'])

        model = FeatureProjector(**parameters['model'])
        best_model_id = self.find_best_model()
        out_dir = f"{parameters['out_dir']}"
        model_name = f"{out_dir}/model_{best_model_id}.pt"
        state_dict = torch.load(f'{model_name}')
        model.load_state_dict(state_dict)
        device = parameters['general']['device']
        model = model.to(device)
        criterion = torch.nn.SmoothL1Loss()
        pbar = parameters['general']['pbar']

        test_loss = compute_loss(
            dataloader=test_loader,
            model=model,
            criterion=criterion,
            device=device,
            pbar=pbar,
        )

        for f in filter(lambda x: f'_{best_model_id}.' not in x, os.listdir(out_dir)):
            os.remove(f"{out_dir}/{f}")

        save_params(
            f'{out_dir}/test_scores.json',
            {
                'loss': test_loss,
            },
        )


def main():
    param_file = parse_args().params_path
    optimizer = Optimizer(params=load_ruamel(param_file))
    optimizer.optimize()
    optimizer.test_model()


if __name__ == '__main__':
    main()
