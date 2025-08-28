
from tqdm import tqdm
from graph import Dataset, Graph
import time, json
from query import Query, QueryDataset, human_readable
from symbolic import SymbolicReasoning
from cqd import cqd_query, get_cache_prediction
import pandas as pd
from xcqa import XCQA
from shapley import shapley_value
from shapley import value_function

def average_shapley_value(xcqa, query, easy_answers, qoi='rank', k=10, mode='non_zero', t_norm='prod', t_conorm='prod'):

    num_atoms = 0
    if query.query_type == '2p':
        num_atoms = 2
    elif query.query_type == '3p':
        num_atoms = 3
    elif query.query_type == '2i':
        num_atoms = 2
    elif query.query_type == '2u':
        num_atoms = 2
    elif query.query_type == '3i':
        num_atoms = 3
    elif query.query_type == 'pi':
        num_atoms = 3
    elif query.query_type == 'up':
        num_atoms = 3
    elif query.query_type == 'ip':
        num_atoms = 3
    else:
        raise ValueError(f"Unsupported query type: {query.query_type}.")

    cqd_coalition = [1] * num_atoms
    cqd_results = xcqa.query_execution(query, k=k, coalition=cqd_coalition, t_norm=t_norm, t_conorm=t_conorm)

    if mode == 'non_zero':
        cqd_results = cqd_results[cqd_results['score'] > 0].index.tolist()
    elif mode == 'top_k':
        cqd_results = cqd_results.head(k).index.tolist()
    else:
        cqd_results = cqd_results.index.tolist()

    # remove easy answers from the results
    target_entities = [a for a in cqd_results if a not in easy_answers]
    
    shapley_values = {}
    for atom_idx in range(num_atoms):
        shapley_values[atom_idx] = []

    for target_entity in target_entities:
        for atom_idx in range(num_atoms):
            sv = shapley_value(xcqa, query, atom_idx=atom_idx, easy_answers=easy_answers, target_entity=target_entity, qoi=qoi, k=k, t_norm=t_norm, t_conorm=t_conorm)
            shapley_values[atom_idx].append(sv)

    average_shapley_values = {}
    for atom_idx in range(num_atoms):
        average_shapley_values[atom_idx] = sum(shapley_values[atom_idx]) / len(shapley_values[atom_idx]) if shapley_values[atom_idx] else 0.0

    return average_shapley_values

def average_shapley_values_for_query_type(xcqa, query_dataset, query_dataset_hard, query_type, qoi='rank', k=10, mode='non_zero', t_norm='prod', t_conorm='prod'):

    num_atoms = 0
    if query_type == '2p':
        num_atoms = 2
    elif query_type == '3p':
        num_atoms = 3
    elif query_type == '2i':
        num_atoms = 2
    elif query_type == '2u':
        num_atoms = 2
    elif query_type == '3i':
        num_atoms = 3
    elif query_type == 'pi':
        num_atoms = 3
    elif query_type == 'up':
        num_atoms = 3
    elif query_type == 'ip':
        num_atoms = 3
    else:
        raise ValueError(f"Unsupported query type: {query_type}.")
    
    average_shapley_values = {}

    for atom_idx in range(num_atoms):
        average_shapley_values[atom_idx] = []

    hard = query_dataset_hard.get_queries(query_type)
    complete = query_dataset.get_queries(query_type)
    for idx, query in enumerate(tqdm(complete, desc=f"Calculating average Shapley values for {query_type} queries")):
        easy_answers = query.get_answer()
        easy_answers = [a for a in easy_answers if a not in hard[idx].get_answer()]
        avg_shapley_values = average_shapley_value(xcqa, query, easy_answers, qoi=qoi, k=k, mode=mode, t_norm=t_norm, t_conorm=t_conorm)
        for atom_idx in range(num_atoms):
            average_shapley_values[atom_idx].append(avg_shapley_values[atom_idx])

    for atom_idx in range(num_atoms):
        average_shapley_values[atom_idx] = sum(average_shapley_values[atom_idx]) / len(average_shapley_values[atom_idx]) if average_shapley_values[atom_idx] else 0.0

    return average_shapley_values

if __name__ == "__main__":

    dataset = Dataset()

    data_dir = 'data/FB15k-237'
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
    
    dir_query_2p = 'data/FB15k-237/test_ans_2c.pkl'
    dir_query_3p = 'data/FB15k-237/test_ans_3c.pkl'
    dir_query_2i = 'data/FB15k-237/test_ans_2i.pkl'
    dir_query_2u = 'data/FB15k-237/test_ans_2u.pkl'
    dir_query_3i = 'data/FB15k-237/test_ans_3i.pkl'
    dir_query_pi = 'data/FB15k-237/test_ans_ci.pkl'
    dir_query_ip = 'data/FB15k-237/test_ans_ic.pkl'
    dir_query_up = 'data/FB15k-237/test_ans_uc.pkl'

    query_dataset = QueryDataset(dataset)
    query_dataset.load_queries_from_pkl(dir_query_2p, query_type='2p')
    query_dataset.load_queries_from_pkl(dir_query_3p, query_type='3p')
    query_dataset.load_queries_from_pkl(dir_query_2i, query_type='2i')
    query_dataset.load_queries_from_pkl(dir_query_2u, query_type='2u')
    query_dataset.load_queries_from_pkl(dir_query_3i, query_type='3i')
    query_dataset.load_queries_from_pkl(dir_query_pi, query_type='pi')
    query_dataset.load_queries_from_pkl(dir_query_ip, query_type='ip')
    query_dataset.load_queries_from_pkl(dir_query_up, query_type='up')
    query_dataset.get_num_queries()
    
    dir_query_2p = 'data/FB15k-237/test_ans_2c_hard.pkl'
    dir_query_3p = 'data/FB15k-237/test_ans_3c_hard.pkl'
    dir_query_2i = 'data/FB15k-237/test_ans_2i_hard.pkl'
    dir_query_2u = 'data/FB15k-237/test_ans_2u_hard.pkl'
    dir_query_3i = 'data/FB15k-237/test_ans_3i_hard.pkl'
    dir_query_pi = 'data/FB15k-237/test_ans_ci_hard.pkl'
    dir_query_ip = 'data/FB15k-237/test_ans_ic_hard.pkl'
    dir_query_up = 'data/FB15k-237/test_ans_uc_hard.pkl'

    query_dataset_hard = QueryDataset(dataset)
    query_dataset_hard.load_queries_from_pkl(dir_query_2p, query_type='2p')
    query_dataset_hard.load_queries_from_pkl(dir_query_3p, query_type='3p')
    query_dataset_hard.load_queries_from_pkl(dir_query_2i, query_type='2i')
    query_dataset_hard.load_queries_from_pkl(dir_query_2u, query_type='2u')
    query_dataset_hard.load_queries_from_pkl(dir_query_3i, query_type='3i')
    query_dataset_hard.load_queries_from_pkl(dir_query_pi, query_type='pi')
    query_dataset_hard.load_queries_from_pkl(dir_query_ip, query_type='ip')
    query_dataset_hard.load_queries_from_pkl(dir_query_up, query_type='up')
    query_dataset_hard.get_num_queries()
    
    reasoner_train = SymbolicReasoning(graph_train, logging=False)
    reasoner_valid = SymbolicReasoning(graph_valid, logging=False)
    reasoner_test = SymbolicReasoning(graph_test, logging=False)
    
    cqd_cache = pd.read_json('data/FB15k-237/all_1p_queries_top25.json', orient='records')
    
    cqd_cache = cqd_cache.set_index(['entity_id', 'relation_id'])  # Set index for faster access


    query_types = ['2p', '3p', '2i', '2u', '3i', 'pi', 'up', 'ip']
    
    xcqa = XCQA(symbolic=reasoner_train, dataset=dataset, cqd_cache=cqd_cache, logging=False)
    
    results = {}
    for query_type in query_types:
        start = time.time()
        print(f"Calculating average Shapley values for query type: {query_type}")
        avg_shapley_values = average_shapley_values_for_query_type(xcqa, query_dataset, query_dataset_hard, query_type, qoi='rank', k=10, mode='non_zero', t_norm='prod', t_conorm='prod')
        end = time.time()
        print(f"Average Shapley values for {query_type}: {avg_shapley_values}")
        print(f"Time taken: {end - start:.2f} seconds")
        results[query_type] = avg_shapley_values

    with open('average_shapley_values.json', 'w') as f:
        json.dump(results, f, indent=4)