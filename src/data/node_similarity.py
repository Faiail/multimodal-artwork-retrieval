import json
import os
from neo4j import GraphDatabase
from enum import Enum
import numpy as np


class Files(Enum):
    LABELS = 'labels.json'
    STATISTICS = 'statistics.json'
    RESULT_SETS = 'results_set.json'


class Node:
    def __init__(self, node_t, name):
        self.node_t = node_t
        self.name = name

    def __eq__(self, other):
        return self.node_t == other.node_t and self.name == other.name

    def __hash__(self):
        return hash(self.node_t + self.name) + hash(self.name + self.node_t)

    def __repr__(self):
        return f'Node({self.node_t}, {self.name})'


class NodeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Node):
            out = obj.__dict__
            out.update({'__class__': 'Node'})
            return out
        return super().default(obj)


class NodeDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.dict_to_node, *args, **kwargs)

    def dict_to_node(self, d):
        if '__class__' in d and d['__class__'] == 'Node':
            return Node(d['node_t'], d['name'])
        return d


def get_neighborhood_map(neigh_list):
    return {
        x['artwork']: list({Node(list(y.labels)[0], y._properties['name']) for y in x['setN']})
        for x in neigh_list
    }


class NodeSimilarityScorer:
    def __init__(
            self,
            uri,
            user,
            pwd,
            db,
            node_t,
            data=None,
            out_dir=None,
            save=True,
    ):
        self.statistics = None
        self.result_sets = None
        self.uri = uri
        self.user = user
        self.pwd = pwd
        self.db = db
        self.node_t = node_t
        self.data = data
        self.out_dir = out_dir
        self.save = save

        if self.out_dir:
            os.makedirs(self.out_dir, exist_ok=True)

        self.connector = GraphDatabase.driver(uri=uri, auth=(self.user, self.pwd))

        self.get_statistics()

        self.get_result_sets()

    def get_data(self):
        with self.connector.session(database=self.db) as session:
            q = f'MATCH (a:{self.node_t}) return a.name as name'
            return session.run(q).data()

    def get_neighborhood_space(self):
        if self.out_dir:
            if Files.LABELS.value in os.listdir(self.out_dir):
                with open(f'{self.out_dir}/{Files.LABELS.value}') as f:
                    return json.load(f)
        with self.connector.session(database=self.db) as session:
            q = f'''
            MATCH (:{self.node_t})--(n) 
            return distinct labels(n)[0] as label
            '''
            ans = session.run(q).data()
            ans = list(map(lambda x: list(x.values())[0], ans))
        if self.save and self.out_dir:
            with open(f'{self.out_dir}/{Files.LABELS.value}', 'w') as f:
                json.dump(ans, f)
        return ans

    def get_stat_for_node_type(self, node_t):
        with self.connector.session(database=self.db) as session:
            q = f'''
            MATCH (:{self.node_t})-[r]-(n:{node_t})
            return n.name as name, count(distinct(r)) as num 
            '''
            ans = session.run(q).data()
        # TODO: SOMETHING LIKE A SCALER, TO COOL DOWN VALUES
        return {x["name"]: np.log10(x['num']) if x['num'] > 1 else x['num'] for x in ans}

    def get_statistics(self):
        if self.out_dir and Files.STATISTICS.value in os.listdir(self.out_dir):
            with open(f'{self.out_dir}/{Files.STATISTICS.value}') as f:
                self.statistics = json.load(f)
                return self.statistics
        nodes = self.get_neighborhood_space()
        self.statistics = {x: self.get_stat_for_node_type(x) for x in nodes}
        if self.out_dir and self.save:
            with open(f'{self.out_dir}/statistics.json', 'w') as f:
                json.dump(self.statistics, f)
        return self.statistics

    def get_first_order_neighborhood(self, source):
        with self.connector.session(database=self.db) as session:
            q = f"""
            match (a:{self.node_t})--(n) where a.name in {source}
            return a.name as {self.node_t.lower()}, collect(distinct n) as setN
            """
            return list(session.run(q))

    def compute_results_set(self, source):
        mid = self.get_first_order_neighborhood(source)
        return get_neighborhood_map(mid)

    def get_result_sets(self):
        if os.path.exists(f'{self.out_dir}/{Files.RESULT_SETS.value}'):
            with open(f'{self.out_dir}/{Files.RESULT_SETS.value}') as f:
                self.result_sets = json.load(f, cls=NodeDecoder)
        else:
            self.result_sets = self.compute_results_set(self.data)
            if self.save and self.out_dir:
                with open(f'{self.out_dir}/{Files.RESULT_SETS.value}', 'w') as f:
                    json.dump(self.result_sets, f, cls=NodeEncoder)
        return self.result_sets

    def compute_set_score(self, _set):
        return sum(map(lambda x: 1 / self.statistics[x['label']][x['name']], _set))

    def compute_score(self, node_a, node_b):
        res_a, res_b = set(self.result_sets[node_a]), set(self.result_sets[node_b])
        intersection, union = res_a.intersection(res_b), res_a.union(res_b)
        intersection = [
            {
                'label': x.node_t,
                'name': x.name,
            }
            for x in intersection
        ]

        union = [
            {
                'label': x.node_t,
                'name': x.name,
            }
            for x in union
        ]
        return self.compute_set_score(intersection) / self.compute_set_score(union)
