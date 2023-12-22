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
    def __init__(self, a, b, score):
        self.a = a
        self.b = b
        self.score = score

    def __eq__(self, other):
        return (self.a == other.a and self.b == other.b) or (self.a == other.b and self.b == other.a)

    def __hash__(self):
        return hash(self.a + self.b) + hash(abs(self.a - self.b))

    def __repr__(self):
        return f'Couple({self.a}, {self.b}, {self.score})'


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


def generate_random_scores(data, scorer, max_entries_per_category):
    # couples = {Couple(x, x, 1.0) for x in range(len(data))}
    couples = set()
    positive = {}
    for ix, artwork in enumerate(tqdm(data)):
        sim_len, dis_len = 0, 0
        for y in iter(random.sample(range(len(data)), len(data))):
            if sim_len == max_entries_per_category == dis_len:
                break
            cur_len = len(couples)
            score = scorer.compute_score(data[ix], data[y])
            if score < 0.5:
                if dis_len < max_entries_per_category:
                    couples.add(Couple(ix, y, score))
                    dis_len += len(couples) - cur_len
            else:
                if sim_len < max_entries_per_category:
                    couples.add(Couple(ix, y, score))
                    sim_len += len(couples) - cur_len
                    positive[ix] = positive.get(ix, 0) + (len(couples) - cur_len)
    return pd.DataFrame(list(map(lambda x: (x.a, x.b, x.score), couples)),
                        columns=['artwork_a', 'artwork_b', 'score'])


def generate_scores(
        data,
        mode,
        scorer,
        split,
        out_dir,
        step=None,
        seed=None,
        max_entries_per_category=None,
):
    os.makedirs(f'{out_dir}/{split}', exist_ok=True)
    if mode == 'random':
        random.seed(seed)
        scores = generate_random_scores(scorer=scorer, data=data, max_entries_per_category=max_entries_per_category)
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
