import os
import src.models.utils as ut
import json
import hyperopt
import torch
import numpy as np
from copy import deepcopy


class BaseOptimizer:
    def __init__(
            self,
            params,
    ):
        self.params = params
        self.space = self._get_space()
        self.best_params = None
        self.current_run = 0
        #self._clean_out_dir()

    def _clean_out_dir(self):
        os.makedirs(self.params['out_dir'], exist_ok=True)
        for f in os.listdir(self.params['out_dir']):
            os.remove(f'{self.params["out_dir"]}/{f}')

    def _init_augmentations(self):
        raise NotImplementedError()

    def _get_space(self):
        to_optimize = self.params['hyperopt']
        return {
            k: ut.get_distribution(k, v) for k, v in to_optimize.items()
        }

    def apply_params(self, stage_params):
        start_params = self.params
        for k, v in stage_params.items():
            key, value = k.split('__')
            start_params[key][value] = v
        return start_params

    def objective(self, params):
        raise NotImplementedError()

    def optimize(self):
        self.best_params = hyperopt.fmin(
            fn=self.objective,
            space=self.space,
            algo=hyperopt.tpe.suggest,
            max_evals=self.params['n_trials'],
            show_progressbar=self.params["progress_bar"],
            rstate=np.random.default_rng(42),
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
        raise NotImplementedError()
