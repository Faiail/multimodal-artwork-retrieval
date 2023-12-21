import os
import pandas as pd
from src.utils import load_ruamel
from src.data.node_similarity import NodeSimilarityScorer
from tqdm import tqdm
from argparse import ArgumentParser
import random
from itertools import combinations_with_replacement
from scipy.special import comb

tqdm.pandas()


class Couple:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __eq__(self, other):
        return (self.a == other.a and self.b == other.b) or (self.a == other.b and self.b == other.a)

    def __hash__(self):
        return hash(self.a + self.b) + hash(abs(self.a - self.b))

    def __repr__(self):
        return f'Couple({self.a}, {self.b})'


def generate_random_couples(len_data, num_couples, seed):
    # TODO: the couples must include all self couples e.g. (x, x)
    #  some couples (x, y) that are similar and others (x, y) that are not
    couples = set()
    random.seed(seed)
    for _ in tqdm(range(num_couples)):
        current_len = len(couples)
        while current_len == len(couples):
            x = Couple(random.randint(0, len_data), random.randint(0, len_data))
            couples.add(x)
    couples = list(couples)
    return list(map(lambda x: (x.a, x.b), couples))


def generate_complete_couples(len_data, step):
    num_max = comb(len_data + 1, 2)
    print(f'Number of yields is: {num_max // step}')
    exit()
    data = combinations_with_replacement(range(len_data), 2)
    step_data = []
    for ix, val in enumerate(data):
        step_data.append(val)
        if ((ix + 1) % step == 0) or (ix + 1) == num_max:
            yield step_data
            step_data = []


def generate_complete_scores(iterator, scorer, data):
    for sample in iterator:
        df = pd.DataFrame(sample, columns=['artwork_a', 'artwork_b'])
        df['score'] = df.progress_apply(lambda x: scorer.compute_score(data[x[0]], data[x[1]]), axis=1)
        yield df


def generate_random_scores(iterator, data, scorer):
    df = pd.DataFrame(iterator, columns=['artwork_a', 'artwork_b'])
    df['score'] = df.progress_apply(lambda x: scorer.compute_score(data[x[0]], data[x[1]]), axis=1)
    return df


def generate_scores(
        data,
        mode,
        scorer,
        split,
        out_dir,
        step=None,
        num_couples=None,
        seed=None,
):
    os.makedirs(f'{out_dir}/{split}', exist_ok=True)
    if mode == 'random':
        couples = generate_random_couples(len(data) - 1, num_couples, seed)
        scores = generate_random_scores(iterator=couples, scorer=scorer, data=data)
        scores.to_csv(f'{out_dir}/{split}/data.csv')
    elif mode == 'complete':
        couples = generate_complete_couples(len(data), step)
        scores = generate_complete_scores(couples, scorer, data)
        for ix, val in enumerate(scores):
            val.to_csv(f'{out_dir}/{split}/data_{ix:03d}.csv')


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--config_file')
    return parser.parse_args()


def main():
    param_file = parse_args().config_file
    params = load_ruamel(param_file)

    data = pd.read_csv(params['mapping_file'])['Columns.NAME'].tolist()
    data = list(map(lambda x: x.split('.')[0] + '.jpg', data))
    scorer = NodeSimilarityScorer(data=data, **params['scorer'])

    generate_scores(data=data, scorer=scorer, **params['general'])


if __name__ == '__main__':
    main()
