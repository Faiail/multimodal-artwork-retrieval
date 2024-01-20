import pandas as pd
from src.utils import load_ruamel
from src.models.utils import parse_args
from neo4j import GraphDatabase
import os


def load_titles(uri, username, password, database):
    driver = GraphDatabase.driver(uri=uri, auth=(username, password))
    query = 'match (a:Artwork) return a.name as name, a.title as title'
    with driver.session(database=database) as session:
        names_titles = pd.DataFrame(session.run(query).data())
    return names_titles


def load_captions(captions_path):
    captions = pd.read_csv(captions_path)
    captions.rename(columns=lambda x: x.split('.')[1].lower(), inplace=True)
    captions['name'] = captions['name'].map(lambda x: x.replace('.safetensors', '.jpg'))
    return captions


def main():
    params_path = parse_args().params_path
    params = load_ruamel(params_path)
    titles = load_titles(**params['neo4j'])
    in_dir = params['captions_dir']
    out_dir = params['out_dir']
    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(in_dir):
        captions = load_captions(f"{in_dir}/{f}")
        titles_captions = pd.merge(left=titles, right=captions, left_on='name', right_on='name')

        assert len(titles_captions) == len(captions), f"Having just {len(titles_captions)}/{len(captions)} items."
        titles_captions.to_csv(f"{out_dir}/{f}", index=False)


if __name__ == '__main__':
    main()



