import os
import torch
import src.utils as utils
import hyperopt
from copy import deepcopy
from gae.HeteroGAE import HeteroGAE
from EarlyStopping import EarlyStopping
import torch_geometric as pyg
from tqdm import tqdm
import json


class Optimizer:
    def __init__(self, params):
        self.params = params
        self.train_data = torch.load(params['data']['train'])
        self.train_data = pyg.transforms.ToUndirected()(self.train_data).to(params['device'])

        self.val_data = torch.load(params['data']['val'])
        self.val_data = pyg.transforms.ToUndirected()(self.val_data).to(params['device'])

        self.test_data = torch.load(params['data']['test'])
        self.test_data = pyg.transforms.ToUndirected()(self.test_data).to(params['device'])

        self.out_dir = params['out']
        os.makedirs(self.out_dir, exist_ok=True)

        # clear directory
        for f in os.listdir(self.out_dir):
            os.remove(f'{self.out_dir}/{f}')

        self.space = self._get_space()
        self.best_params = None

    def _get_space(self):
        return {
            k: hyperopt.hp.choice(k, v) for k, v in self.params['model'].items() if isinstance(v, list)
        }

    @torch.no_grad
    def test_model(self):
        best_values = {k: self.params['model'][k][v] for k, v in self.best_params.items()}
        print(f'best model parameters: {best_values}')
        best_model_name = stringfy_params(best_values, 'hgae_')
        model_state_dict = torch.load(f"{self.out_dir}/{best_model_name}")

        # delete other models
        for f in filter(lambda x: x != best_model_name, os.listdir(self.out_dir)):
            os.remove(f'{self.out_dir}/{f}')

        start_params = deepcopy(self.params)
        for k, v in best_values.items():
            start_params['model'][k] = v

        model = HeteroGAE(
            metadata=self.train_data.metadata(),
            **start_params['model']
        ).to('cuda')
        # lazy parameter initialization
        z = model.encode(self.test_data.x_dict, self.test_data.edge_index_dict)

        model.load_state_dict(model_state_dict)
        model = model.to(start_params['device'])

        z = model.encode(self.test_data.x_dict, self.test_data.edge_index_dict)
        auc, ap = model.test(z, self.test_data.edge_index_dict)

        print(f'AUC: {auc:.4f}, AP: {ap:.4f}')

        with open(f'{self.out_dir}/{best_model_name[:-3]}.json', 'w+') as f:
            json.dump(
                {
                    'params': best_values,
                    'metrics': {
                        'auc': auc,
                        'ap': ap,
                    },
                },
                f,
            )

    def objective(self, params):
        torch.cuda.empty_cache()
        start_params = deepcopy(self.params)
        for k, v in params.items():
            start_params['model'][k] = v

        model = HeteroGAE(
            metadata=self.train_data.metadata(),
            **start_params['model']
        ).to(start_params['device'])
        num_epochs = start_params['num_epochs']
        optimizer = torch.optim.Adam(params=model.parameters(), **start_params['optimizer'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, **start_params['scheduler']
        )
        model_name = stringfy_params(params, 'hgae_')
        print(f'saving to:  {model_name}')
        early_stop = EarlyStopping(path=f'{self.out_dir}/{model_name}', **start_params['early_stop'])
        try:
            train_model(
                model=model,
                train_data=self.train_data,
                val_data=self.val_data,
                num_epochs=num_epochs,
                scheduler=scheduler,
                early_stop=early_stop,
                optimizer=optimizer,
                verbose=start_params['verbose'],
                pbar=start_params['pbar'],
            )
        except torch.cuda.OutOfMemoryError:
            print('Experiment failed due to out of memory error')
            return {
                'loss': 1e15,
                'status': hyperopt.STATUS_FAIL,
            }

        # returning best model validation loss
        with torch.no_grad():
            model.load_state_dict(torch.load(early_stop.path))
            z = model.encode(self.val_data.x_dict, self.val_data.edge_index_dict)
            best_model_val_loss = model.recon_loss(z, self.val_data.edge_index_dict)
        return {
            'loss': best_model_val_loss.cpu().item(),
            'status': hyperopt.STATUS_OK,
        }

    def optimize(self):
        self.best_params = hyperopt.fmin(
            fn=self.objective,
            space=self.space,
            algo=hyperopt.tpe.suggest,
            max_evals=self.params['n_trials'],
        )
        return self.best_params


def stringfy_params(params, prefix='hgae'):
    return prefix + '_'.join([f'{k}-{str(v).replace(".", "__")}' for k, v in params.items()]) + '.pt'


def train_model(
        model,
        train_data,
        val_data,
        num_epochs,
        scheduler,
        early_stop,
        optimizer,
        verbose,
        pbar
):
    iter_epochs = range(num_epochs) if not pbar else tqdm(range(num_epochs))
    for epoch in iter_epochs:
        # train stage
        model.train()
        optimizer.zero_grad()
        z = model.encode(train_data.x_dict, train_data.edge_index_dict)
        loss = model.recon_loss(z, train_data.edge_index_dict)
        loss.backward()
        optimizer.step()

        # validation stage
        model.eval()
        with torch.no_grad():
            z = model.encode(val_data.x_dict, val_data.edge_index_dict)
            val_loss = model.recon_loss(z, val_data.edge_index_dict)
            auc, ap = model.test(z, val_data.edge_index_dict)
            if verbose:
                print(f'Epoch {epoch:03d} Loss: {val_loss:.4f}, AUC: {auc:.4f}, AP: {ap:.4f}')
            scheduler.step(val_loss)
            early_stop(val_loss, model)
            if early_stop.early_stop:
                print(f'Early stopping at epoch {epoch:03d}')
                return


def main():
    parameters = utils.load_ruamel('../../configs/train_hgae.yaml')
    optimizer = Optimizer(parameters)
    optimizer.optimize()
    optimizer.test_model()


if __name__ == '__main__':
    main()

