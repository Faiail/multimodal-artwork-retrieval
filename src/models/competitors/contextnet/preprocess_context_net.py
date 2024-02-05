import os

import torch
from neo4j import GraphDatabase, Driver
from typing import List
import pandas as pd
from src.utils import load_parameters
from src.models.utils import parse_args
from src.data.utils import save_embeddings
from tqdm import tqdm
tqdm.pandas()


def get_node_types(driver: Driver, db: str) -> List[str]:
    with driver.session(database=db) as session:
        q = 'CALL db.labels() YIELD label RETURN label;'
        nodes = pd.DataFrame(session.run(q).data()).label.tolist()
    return nodes


def get_edge_types(driver: Driver, db: str) -> List[str]:
    with driver.session(database=db) as session:
        q = 'CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType;'
        edges = pd.DataFrame(session.run(q).data()).relationshipType.tolist()
    return edges


def create_meta_graph(
        driver: Driver,
        db: str,
        name: str,
        node_types: List[str],
        edge_types: List[str],
) -> None:
    with driver.session(database=db) as session:
        q = f'call gds.graph.project("{name}", {node_types}, {edge_types})'
        session.run(q)


def get_node2vec_embeddings(driver: Driver, db: str, name: str) -> pd.DataFrame:
    with driver.session(database=db) as session:
        q = f'call gds.beta.node2vec.stream("{name}") yield nodeId, embedding return nodeId, embedding'
        data = pd.DataFrame(session.run(q).data())
    return data


def get_artworks(driver: Driver, db: str) -> pd.DataFrame:
    with driver.session(database=db) as session:
        q = 'match(a:Artwork) return id(a) as artwork_id, a.name as artwork_name'
        ids = pd.DataFrame(session.run(q).data())
    return ids


def get_attribute_data(driver: Driver, db: str) -> pd.DataFrame:
    with driver.session(database=db) as session:
        q = """
        match (a:Artwork)--(ar:Artist)
        match (g:Genre)--(a)--(s:Style)
        return a.name as artwork_name, ar.printed_name as artist, g.name as genre, s.name as style
        """
        attributes = pd.DataFrame(session.run(q).data())
    return attributes


def get_local_artwork_style(mapping_file: str, relation_file: str) -> pd.DataFrame:
    mapping = pd.read_csv(mapping_file, names=['idx', 'name'])
    relation = pd.read_csv(relation_file, names=['artwork', 'style'])
    return pd.merge(
        left=mapping,
        right=relation,
        left_on='idx',
        right_on='artwork'
    ).drop(['idx', 'artwork'], axis=1)


def save_node2vec_embeddings(embeddings: pd.DataFrame, out_dir: str) -> None:
    embeddings.progress_apply(
        lambda x: save_embeddings({'node2vec': torch.as_tensor(x['embedding'])}, f'{out_dir}/{x["artwork_name"]}'),
        axis=1
    )


def get_classification_split(split_file: str, local_data: pd.DataFrame) -> pd.DataFrame:
    split_data = pd.read_csv(split_file)['Columns.NAME']
    return pd.merge(
        left=split_data,
        right=local_data,
        left_on='Columns.NAME',
        right_on='name',
    ).drop(['Columns.NAME'], axis=1)


def get_attribute_split(split_file: str, attribute_data: pd.DataFrame) -> pd.DataFrame:
    split_data = pd.read_csv(split_file)['Columns.NAME']
    return pd.merge(
        left=split_data,
        right=attribute_data,
        left_on='Columns.NAME',
        right_on='artwork_name',
    ).drop(["Columns.NAME"], axis=1)


def main():
    # connection to db
    params = load_parameters(parse_args().params_path)
    params['driver']['auth'] = tuple(params['driver']['auth'])
    driver = GraphDatabase.driver(**params['driver'])
    db = params['db']
    name = params['name']
    emb_out_dir = params['emb_out_dir']
    os.makedirs(emb_out_dir, exist_ok=True)
    data_class_out_dir = params['data_class_out_dir']
    os.makedirs(data_class_out_dir, exist_ok=True)
    data_attr_out_dir = params['data_attr_out_dir']
    os.makedirs(data_attr_out_dir, exist_ok=True)

    # get node2vec features
    node_types = get_node_types(driver, db)
    edge_types = get_edge_types(driver, db)
    create_meta_graph(driver, db, name, node_types, edge_types)
    embeddings_data = get_node2vec_embeddings(driver, db, name)
    artwork_data = get_artworks(driver, db)
    complete_data = pd.merge(left=embeddings_data,
                             right=artwork_data,
                             left_on='nodeId',
                             right_on='artwork_id').drop(['nodeId'], axis=1)
    complete_data['artwork_name'] = complete_data['artwork_name'].map(lambda x: x[:-4])

    # saving embeddings
    save_node2vec_embeddings(complete_data, emb_out_dir)

    # making classification splits
    local_data = get_local_artwork_style(**params['local'])
    local_data['name'] = local_data['name'].map(lambda x: x.replace('.jpg', '.safetensors'))
    attribute_data = get_attribute_data(driver, db)
    attribute_data['artwork_name'] = attribute_data['artwork_name'].map(lambda x: x.replace('.jpg', '.safetensors'))
    splits = params['split_dir']
    for f in os.listdir(splits):
        data_class = get_classification_split(
            split_file=f'{splits}/{f}',
            local_data=local_data,
        )
        data_attr = get_attribute_split(
            split_file=f'{splits}/{f}',
            attribute_data=attribute_data,
        )
        data_class.to_csv(f'{data_class_out_dir}/{f}', index=False)
        data_attr.to_csv(f'{data_attr_out_dir}/{f}', index=False)


if __name__ == '__main__':
    main()