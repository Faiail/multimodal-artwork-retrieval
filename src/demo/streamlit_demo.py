import streamlit as st
from src.models.artwork_siamese_network.test_siamese_network import (
    _init_model, _init_aux_models, _init_datasets, encode_cat_features
)
from src.models.artwork_siamese_network.fusion_module import FusionModule
from src.data.SiameseDataset import SiameseTestDataset
import torch
from torch.utils.data import DataLoader
from src.utils import load_ruamel
from src.models.feature_projector import FeatureProjector


def prepare(config_file: str) -> (FusionModule, torch.Tensor, SiameseTestDataset, dict[str, FeatureProjector]):
    parameters = load_ruamel(config_file)
    model = _init_model(parameters.get('model'))
    st.progress(20)
    aux_cat_models = _init_aux_models(parameters['auxiliary_models_cat'])
    st.progress(40)
    aux_query_models = _init_aux_models(parameters['auxiliary_models_query'])
    st.progress(60)
    catalogue_dataset, test_dataset = _init_datasets(parameters['dataset'])

    catalogue_loader = DataLoader(catalogue_dataset, **parameters['dataloader'])
    st.progress(80)
    catalogue_features = encode_cat_features(
        catalogue=catalogue_loader,
        model=model,
        aux_models=aux_cat_models,
        **parameters.get('encoding')
    )
    st.progress(100)
    return model, catalogue_features, test_dataset, aux_query_models


def main():
    st.set_page_config(layout="wide", page_title="Multimodal Artwork Retrieval")
    st.title("Multimodal Artwork Retrieval")
    st.sidebar.title("Settings")
    with st.sidebar:
        config_file = st.text_input("Config path", "configs_cineca/test_siamese_text2img.yaml")
        image_dir = st.text_input("Raw image directory", "data/raw/images-resized")
        captions_dir = st.text_input("Raw caption file path", "data/processed/proj/test.csv")
        num_items = st.slider("Number of items in result set", 1, 20, 1)
    with st.spinner("Preparing..."):
        model, catalogue_feats, test_dataset, aux_query_models = prepare(config_file)
        # TODO: riprendi di qua prendendo per bene immagini e testi
    data_point_idx = st.slider("Data point", 1, len(test_dataset), 1)
    if st.button('Predict'):
        print('Predicting...')

    # get dataset


if __name__ == '__main__':
    main()