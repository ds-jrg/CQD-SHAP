from tqdm import tqdm
from symbolic import SymbolicReasoning
from graph import Dataset, Graph
from query import Query, QueryDataset, human_readable
from cqd import create_cqd_file
import os, argparse
import torch
from kbc.cqd_co_xcqa import main
import pandas as pd
import json, glob

def load_dataset(data_dir):

    dataset = Dataset()
    dataset.set_id2node(f'{data_dir}/ind2ent.pkl')
    dataset.set_id2rel(f'{data_dir}/ind2rel.pkl')
    dataset.set_node2title(f'{data_dir}/extra/entity2text.txt')

    graph_train = Graph(dataset)
    graph_train.load_triples(f'{data_dir}/train.txt', skip_missing=False, add_reverse=True)
    graph_train.get_num_nodes(), graph_train.get_num_edges()

    graph_valid = Graph(dataset)
    # add training edges to validation graph
    for edge in graph_train.get_edges():
        # we set add_reverse=False because it already exists in the training graph
        graph_valid.add_edge(edge.get_head().get_name(), edge.get_name(), edge.get_tail().get_name(), skip_missing=False, add_reverse=False)
    graph_valid.load_triples(f'{data_dir}/valid.txt', skip_missing=False, add_reverse=True)
    graph_valid.get_num_nodes(), graph_valid.get_num_edges()

    graph_test = Graph(dataset)
    # add training and validation edges to test graph (validation graph contains all training edges)
    for edge in graph_valid.get_edges():
        # we set add_reverse=False because it already exists in the validation graph
        graph_test.add_edge(edge.get_head().get_name(), edge.get_name(), edge.get_tail().get_name(), skip_missing=False, add_reverse=False)
    graph_test.load_triples(f'{data_dir}/test.txt', skip_missing=False, add_reverse=True)
    graph_test.get_num_nodes(), graph_test.get_num_edges()

    return dataset, graph_train, graph_valid, graph_test

def create_all_1p_queries(dataset):
    """
    Create all 1p queries for the given dataset.
    
    Args:
        dataset: The dataset containing nodes and relations.
        
    Returns:
        A list of 1p queries.
    """
    all_1p_queries = []
    for node in tqdm(dataset.id2node.values(), desc="Creating 1p queries"):
        for relation in dataset.id2rel.values():
            node_id = dataset.get_id_by_node(node)
            relation_id = dataset.get_id_by_relation(relation)
            if node_id is not None and relation_id is not None:
                query = Query('1p', (((node_id, (relation_id,)),), []))
                all_1p_queries.append(query)
    print(f'Number of all 1p queries: {len(all_1p_queries)}')
    return all_1p_queries

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="CQD for XCQA")
    dataset, graph_train, graph_valid, graph_test = load_dataset('data/FB15k-237')
    all_1p_queries = create_all_1p_queries(dataset)
    
    if not os.path.exists('data/FB15k-237/all_1p_queries'):
        os.makedirs('data/FB15k-237/all_1p_queries')
        os.system('cp data/FB15k-237/FB15k-237_test_complete.pkl data/FB15k-237/all_1p_queries/FB15k-237_test_complete.pkl')
        
    chunk_size = 100000

    for i in tqdm(range(0, len(all_1p_queries), chunk_size), desc="Creating CQD files for 1p queries"):
        batch_queries = all_1p_queries[i:i + chunk_size]
        create_cqd_file(batch_queries, output_file=f'data/FB15k-237/all_1p_queries/all_1p_queries_{i // chunk_size}.pkl')
        sample_path = f'data/FB15k-237/all_1p_queries/all_1p_queries_{i // chunk_size}.pkl'
        result_path = f'data/FB15k-237/all_1p_queries/results_{i // chunk_size}.json'
        # run the cqd_co_xcqa model
        args = argparse.Namespace(
            path = 'FB15k-237',
            sample_path = sample_path,
            model_path = 'models/FB15k-237-model-rank-1000-epoch-100-1602508358.pt',
            dataset = 'FB15k-237',
            mode = 'test',
            chain_type = '1_1', # '1_1', '1_2', '2_2', '2_2_disj', '1_3', '2_3', '3_3', '4_3', '4_3_disj', '1_3_joint'
            t_norm = 'prod', # 'min', 'prod'
            reg = None,
            lr = 0.1,
            optimizer='adam', # 'adam', 'adagrad', 'sgd'
            max_steps = 1000,
            sample = False,
            result_path = result_path,
            save_result = True,
            save_k = 25
        )
        main(args)

    # read all json files and merge them into a single one
    def merge_json_files(directory: str, output_file: str):
        all_data = []
        for filename in tqdm(glob.glob(os.path.join(directory, '*.json')), desc="Merging JSON files"):
            with open(filename, 'r') as f:
                data = json.load(f)
                all_data.extend(data)
        
        with open(output_file, 'w') as f:
            json.dump(all_data, f)
            
    merge_json_files('data/FB15k-237/all_1p_queries', 'data/FB15k-237/all_1p_queries.json')