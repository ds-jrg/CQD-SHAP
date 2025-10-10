from graph import Dataset, Graph
from query import Query, QueryDataset
from symbolic import SymbolicReasoning
from xcqa import XCQA
import pandas as pd
from tqdm import tqdm
import argparse
import json


hybrid_coalitions = {
    "2p": [1, 0],
    "3p": [0, 0, 1],
    "2i": [0, 0],
    "3i": [0, 0, 0],
    "ip": [0, 0, 1],
    # "pi": [0, 0, 1],
    "pi": [0, 0, 0],
    "2u": [0, 1],
    "up": [0, 0, 1]
}


def compute_metrics(result, answer_hard, answer_complete):
    """
    result: pd.Series or pd.DataFrame with index = candidate answers, sorted by predicted score (descending).
    answer_hard: set/list/array of gold answers (to score).
    answer_complete: set/list/array of all correct answers (to exclude during filtered ranking).
    """
    if not answer_hard:
        return 0.0

    mrr = 0.0
    hit_1 = 0
    hit_3 = 0
    hit_10 = 0
    # Convert to sets for fast exclusion
    answer_hard_set = set(answer_hard)
    answer_complete_set = set(answer_complete)

    # Get a filtered version of result for each a_hard
    result_index = list(result.index)
    answer_pos = {x: i for i, x in enumerate(result_index)}

    for a_hard in answer_hard:
        # Remove all other correct answers from ranking for this answer
        filtered_exclude = answer_complete_set - {a_hard}
        # Build mask once for speed
        filtered_index = [x for x in result_index if x not in filtered_exclude]
        if a_hard in filtered_index:
            rank = filtered_index.index(a_hard) + 1
            mrr += 1.0 / rank
            if rank == 1:
                hit_1 += 1
            if rank <= 3:
                hit_3 += 1
            if rank <= 10:
                hit_10 += 1
        # If not found, no contribution

    return mrr / len(answer_hard), hit_1 / len(answer_hard), hit_3 / len(answer_hard), hit_10 / len(answer_hard)

def get_num_atoms(query_type):
    """Get the number of atoms for a given query type."""
    atom_mapping = {
        '2p': 2, '3p': 3, '2i': 2, '2u': 2, 
        '3i': 3, 'pi': 3, 'up': 3, 'ip': 3
    }
    if query_type not in atom_mapping:
        raise ValueError(f"Unsupported query type: {query_type}.")
    return atom_mapping[query_type]

def get_query_file_paths(data_dir, query_type, hard=False, split='test'):
    """Get file paths for query data based on query type."""
    prefix = split + '_ans_'
    file_mapping = {
        '2p': prefix + '2c', '3p': prefix + '3c', '2i': prefix + '2i', 
        '2u': prefix + '2u', '3i': prefix + '3i', 'pi': prefix + 'ci',
        'ip': prefix + 'ic', 'up': prefix + 'uc'
    }
    
    if query_type not in file_mapping:
        raise ValueError(f"Unsupported query type: {query_type}.")
    
    suffix = '_hard' if hard else ''
    filename = f"{file_mapping[query_type]}{suffix}.pkl"
    return f"{data_dir}/{filename}"

def load_query_datasets(dataset, data_dir, query_types, split='test'):
    """Load query datasets for the specific query type."""
    # Load complete query dataset
    query_dataset = QueryDataset(dataset)
    for query_type in query_types:
        query_path = get_query_file_paths(data_dir, query_type, hard=False, split=split)
        query_dataset.load_queries_from_pkl(query_path, query_type=query_type)
    
    # Load hard query dataset
    query_dataset_hard = QueryDataset(dataset)
    for query_type in query_types:
        query_path_hard = get_query_file_paths(data_dir, query_type, hard=True, split=split)
        query_dataset_hard.load_queries_from_pkl(query_path_hard, query_type=query_type)
    
    return query_dataset, query_dataset_hard

def get_cqd_coalition(query_type):
    return [1] * get_num_atoms(query_type)

def main(args):
    print("Starting computation...")
    print(f"Using t-norm: {args.t_norm}, t-conorm: {args.t_conorm}, k: {args.k}, query types: {args.query_types}")

    mrr_results = {
        "cqd": {query_type: [] for query_type in args.query_types},
        "hybrid": {query_type: [] for query_type in args.query_types}
    }

    hit_1_results = {
        "cqd": {query_type: [] for query_type in args.query_types},
        "hybrid": {query_type: [] for query_type in args.query_types}
    }

    hit_3_results = {
        "cqd": {query_type: [] for query_type in args.query_types},
        "hybrid": {query_type: [] for query_type in args.query_types}
    }

    hit_10_results = {
        "cqd": {query_type: [] for query_type in args.query_types},
        "hybrid": {query_type: [] for query_type in args.query_types}
    }

    dataset = Dataset()

    data_dir = 'data/FB15k-237'
    dataset.set_id2node(f'{data_dir}/ind2ent.pkl')
    dataset.set_id2rel(f'{data_dir}/ind2rel.pkl')
    dataset.set_node2title(f'{data_dir}/extra/entity2text.txt')

    print("Loading graph...")
    graph_train = Graph(dataset)
    graph_train.load_triples(f'{data_dir}/train.txt', skip_missing=False, add_reverse=True)
    print(f"Training Graph loaded with {graph_train.get_num_nodes()} nodes and {graph_train.get_num_edges()} edges.")

    graph_valid = Graph(dataset)
    # add training edges to validation graph
    for edge in graph_train.get_edges():
        # we set add_reverse=False because it already exists in the training graph
        graph_valid.add_edge(edge.get_head().get_name(), edge.get_name(), edge.get_tail().get_name(), skip_missing=True, add_reverse=False)
    graph_valid.load_triples(f'{data_dir}/valid.txt', skip_missing=False, add_reverse=True)
    print(f"Validation Graph loaded with {graph_valid.get_num_nodes()} nodes and {graph_valid.get_num_edges()} edges.")

    graph_test = Graph(dataset)
    # add training and validation edges to test graph (validation graph contains all training edges)
    for edge in graph_valid.get_edges():
        # we set add_reverse=False because it already exists in the validation graph
        graph_test.add_edge(edge.get_head().get_name(), edge.get_name(), edge.get_tail().get_name(), skip_missing=True, add_reverse=False)
    graph_test.load_triples(f'{data_dir}/test.txt', skip_missing=False, add_reverse=True)
    print(f"Test Graph loaded with {graph_test.get_num_nodes()} nodes and {graph_test.get_num_edges()} edges.")
    print("Graph loaded.")

    print("Loading queries...")
    query_dataset, query_dataset_hard = load_query_datasets(dataset, data_dir, args.query_types, args.split)
    print(f"Loaded {query_dataset.get_num_queries()} complete queries and {query_dataset_hard.get_num_queries()} hard queries")
    print("Queries loaded.")

    print("Loading cached predictions...")
    import json

    with open('data/FB15k-237/all_1p_queries_top25.json') as f:
        cqd_cache = json.load(f)

    cqd_cache = {
        (rec['entity_id'], rec['relation_id']): (rec['top_k_entities'], rec['top_k_scores'])
        for rec in cqd_cache
    }
    #cqd_cache = pd.read_json('data/FB15k-237/all_1p_queries_128.json', orient='records')
    #cqd_cache = cqd_cache.set_index(['entity_id', 'relation_id'])  # Set index for faster access
    print("Cached predictions loaded.")

    if args.split == 'test':
        reasoner = SymbolicReasoning(graph_valid, logging=False)
    else:
        reasoner = SymbolicReasoning(graph_train, logging=False)

    xcqa = XCQA(symbolic = reasoner, dataset=dataset, cqd_cache=cqd_cache, logging=False)

    for sample_query_type in args.query_types:
        print(f"Processing query type: {sample_query_type}")
        queries = query_dataset.get_queries(sample_query_type)
        queries_hard = query_dataset_hard.get_queries(sample_query_type)
        cqd_coalition = get_cqd_coalition(sample_query_type)
        if args.coalition:
            # conver 0,0,0 to [0,0,0] for compatibility
            hybrid_coalition = [int(x) for x in args.coalition.split(',')]
            if len(hybrid_coalition) != get_num_atoms(sample_query_type):
                raise ValueError(f"Coalition length {len(args.coalition)} does not match number of atoms {get_num_atoms(sample_query_type)} for query type {sample_query_type}.")
        else:
            hybrid_coalition = hybrid_coalitions[sample_query_type]
        for idx in tqdm(range(len(queries_hard))):
            query = queries[idx]
            query_hard = queries_hard[idx]
            cqd_result = xcqa.query_execution(query, k=args.k, coalition=cqd_coalition, t_norm=args.t_norm, t_conorm=args.t_conorm)
            cqd_mrr, hit_1_cqd, hit_3_cqd, hit_10_cqd = compute_metrics(cqd_result, query_hard.get_answer(), query.get_answer())
            hybrid_result = xcqa.query_execution(query, k=args.k, coalition=hybrid_coalition, t_norm=args.t_norm, t_conorm=args.t_conorm)
            hybrid_mrr, hit_1_hybrid, hit_3_hybrid, hit_10_hybrid = compute_metrics(hybrid_result, query_hard.get_answer(), query.get_answer())
            mrr_results["cqd"][sample_query_type].append(cqd_mrr)
            mrr_results["hybrid"][sample_query_type].append(hybrid_mrr)
            hit_1_results["cqd"][sample_query_type].append(hit_1_cqd)
            hit_1_results["hybrid"][sample_query_type].append(hit_1_hybrid)
            hit_3_results["cqd"][sample_query_type].append(hit_3_cqd)
            hit_3_results["hybrid"][sample_query_type].append(hit_3_hybrid)
            hit_10_results["cqd"][sample_query_type].append(hit_10_cqd)
            hit_10_results["hybrid"][sample_query_type].append(hit_10_hybrid)

    # report the average of MRR, Hit@1, Hit@3, and Hit@10 for each coalition type and query type
    results = {
        "mrr": {query_type: {"cqd": sum(mrr_results["cqd"][query_type]) / len(mrr_results["cqd"][query_type]),
                            "hybrid": sum(mrr_results["hybrid"][query_type]) / len(mrr_results["hybrid"][query_type])}
                for query_type in args.query_types},
        "hit_1": {query_type: {"cqd": sum(hit_1_results["cqd"][query_type]) / len(hit_1_results["cqd"][query_type]),
                              "hybrid": sum(hit_1_results["hybrid"][query_type]) / len(hit_1_results["hybrid"][query_type])}
                  for query_type in args.query_types},
        "hit_3": {query_type: {"cqd": sum(hit_3_results["cqd"][query_type]) / len(hit_3_results["cqd"][query_type]),
                              "hybrid": sum(hit_3_results["hybrid"][query_type]) / len(hit_3_results["hybrid"][query_type])}
                  for query_type in args.query_types},
        "hit_10": {query_type: {"cqd": sum(hit_10_results["cqd"][query_type]) / len(hit_10_results["cqd"][query_type]),
                               "hybrid": sum(hit_10_results["hybrid"][query_type]) / len(hit_10_results["hybrid"][query_type])}
                   for query_type in args.query_types}
    }
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='xCQA MRR Computation')
    # t_norm, t_conorm, k, query types
    parser.add_argument('--t_norm', type=str, default='prod', help='T-norm to use (default: prod)')
    parser.add_argument('--t_conorm', type=str, default='prod', help='T-conorm to use (default: prod)')
    parser.add_argument('--k', type=int, default=10, help='Number of top results to consider (default: 10)')
    parser.add_argument('--query_types', type=str, nargs='+', default=list(hybrid_coalitions.keys()), help='Query types to process (default: all)')
    parser.add_argument('--split', type=str, default='test', help='Dataset split to use (default: test)')
    parser.add_argument('--coalition', type=str, default=None, help='Coalition to use for hybrid queries (default: None, uses predefined coalitions)')
    args = parser.parse_args()
    results = main(args)
    print("Computation finished. Results:")
    print(results)
    filename = f'results/metric_results_{args.t_norm}_{args.t_conorm}_k{args.k}_{"_".join(args.query_types)}.json'
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)