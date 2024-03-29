import torch
from typing import Optional, Dict
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class Predictor:
    MODALITIES = ['image', 'text, graph']

    def __init__(
            self,
            model: torch.nn.Module,
            task: str,
            hidden_size,
            auxiliary_models_cat: Optional[Dict[str, torch.nn.Module]] = None,
            auxiliary_models_query: Optional[Dict[str, torch.nn.Module]] = None,
            pbar: Optional[bool] = True,
            norm: Optional[bool] = True,
            device: str = 'cuda',
    ) -> None:
        self.model = model.to(device)
        self.task = task
        self.hidden_size = hidden_size
        self.auxiliary_models_cat = {k: v.to(device) for k, v in auxiliary_models_cat.items()} if isinstance(auxiliary_models_cat, dict) else None
        self.auxiliary_models_query = {k: v.to(device) for k, v in auxiliary_models_query} if isinstance(auxiliary_models_query, dict) else None
        self.catalogue_features = None
        self.pbar = pbar
        self.norm = norm
        self.device = device

        self._source, self._dest = task.split('2')

    @torch.no_grad
    def _encode_catalogue_batch(self, batch):
        if self.auxiliary_models_cat is None:
            return self.model.encode_image(**batch) if self._dest == 'img' else self.model.encode_text(**batch)
        aux_modalities = []
        for mod in self.MODALITIES:
            mod_model = self.auxiliary_models_cat.get(mod, None)
            if not mod_model:
                aux_modalities.append(batch)
            aux_modalities.append(mod_model(batch))
        return self.model(aux_modalities)

    @torch.no_grad
    def encode_catalogue(self, catalogue: DataLoader) -> torch.Tensor:
        # create empty tensor for catalogue features
        self.catalogue_features = torch.zeros(size=(len(catalogue.dataset), self.hidden_size))
        # setting iterator
        iterator = tqdm(catalogue) if self.pbar else catalogue
        batch_size = catalogue.batch_size
        # looping over iterator
        for ix, batch in enumerate(iterator):

            if isinstance(batch, dict):
                batch = {k: v.to(self.device) for k, v in batch.items()}
            else:
                batch = batch.to(self.device)

            self.catalogue_features[ix*batch_size: (ix+1)*batch_size] = self._encode_catalogue_batch(batch)
        if self.norm:
            self.catalogue_features = torch.nn.functional.normalize(self.catalogue_features, p=2, dim=1)
        self.catalogue_features = self.catalogue_features.to(self.device)
        return self.catalogue_features

    def _encode_query_batch(self, query: torch.Tensor) -> torch.Tensor:
        if self.auxiliary_models_query is None:
            return self.model.encode_image(**query) if self._source == 'img' else self.model.encode_text(**query)
        aux_modalities = []
        for mod in self.MODALITIES:
            mod_model = self.auxiliary_models_query.get(mod, None)
            if not mod_model:
                aux_modalities.append(query)
            aux_modalities.append(mod_model(query))
        return self.model(aux_modalities)

    def _compute_cossim(self, query):
        if self.norm:
            query = torch.nn.functional.normalize(query, p=2, dim=1).to(self.device)
        return torch.mm(query, self.catalogue_features.T)

    def predict(self, query: torch.Tensor) -> torch.Tensor:
        if self.catalogue_features is None:
            raise ValueError('Empty catalogue. Please encode your catalogue features by calling'
                             'encode_catalogue method.')
        query_tensor = self._encode_query_batch(query)
        return self._compute_cossim(query_tensor)
