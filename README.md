multimodal-artwork-retrieval
==============================

A short description of the project.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

# USEFUL COMMANDS
* IMAGE TEXT FEATURE EXTRACTION:
```
python -m src.features.extract_images_text_feats
```
* DOWNLOADING ARTGRAPH RAW DATA:
```
python -m src.data.download_artgraph
```

* EXTRACT LABEL FEATS
```
python -m src.features.extract_label_feats
```

* SPLIT ARTFGRAPH
```
python -m src.data.split_artgraph
```

* TRAIN HGAE
```
python -m src.models.train_hgae
```

* EXTRACT GRAPH FEATURES AND MERGE WITH IMAGE AND TEXT FEATURES
```
python -m src.features.extract_graph_and_merge
```

* TRAIN IMAGE-TEXT PROJECTOR
```
python -m src.models.train_feature_projector --params_path configs/train_img_text_proj.yaml
```
*N.B.* To train a different feature projector, please change the parameters file (e.g. see ./configs/ directory)

* PREPARE SIAMESE DATASET
```
python -m src.data.prepare_similarity_dataset --params_path configs/prepare_similarity_train_set.yaml
python -m src.data.prepare_similarity_dataset --params_path configs/prepare_similarity_val_set.yaml
python -m src.data.prepare_similarity_dataset --params_path configs/prepare_similarity_test_set.yaml
```

* TRAIN SIAMESE NETWORK
```
python -m src.models.train_siamese_network --params_path configs/train_siamese_network.yaml
```

* PREPROCESS BASIC GARCIA
```
python -m src.models.competitors.basic_garcia.preprocess --params_path configs/competitors/preprocess_garcia_basic.yaml
```
* TRAIN BASIC GARCIA
```
python -m src.models.competitors.basic_garcia.train_model --params_path configs/competitors/train_garcia_basic.yaml
```

* PREPROCESS CONTEXTNET
```
python -m src.models.competitors.contextnet.preprocess --params_path configs/competitors/preprocess_contextnet.yaml
```

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
