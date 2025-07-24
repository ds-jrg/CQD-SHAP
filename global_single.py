import argparse
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

def get_num_atoms(query_type):
    """Get the number of atoms for a given query type."""
    atom_mapping = {
        '2p': 2, '3p': 3, '2i': 2, '2u': 2, 
        '3i': 3, 'pi': 3, 'up': 3, 'ip': 3
    }
    if query_type not in atom_mapping:
        raise ValueError(f"Unsupported query type: {query_type}.")
    return atom_mapping[query_type]

def get_query_file_paths(data_dir, query_type, hard=False):
    """Get file paths for query data based on query type."""
    file_mapping = {
        '2p': 'test_ans_2c', '3p': 'test_ans_3c', '2i': 'test_ans_2i', 
        '2u': 'test_ans_2u', '3i': 'test_ans_3i', 'pi': 'test_ans_ci',
        'ip': 'test_ans_ic', 'up': 'test_ans_uc'
    }
    
    if query_type not in file_mapping:
        raise ValueError(f"Unsupported query type: {query_type}.")
    
    suffix = '_hard' if hard else ''
    filename = f"{file_mapping[query_type]}{suffix}.pkl"
    return f"{data_dir}/{filename}"

def average_shapley_value(xcqa, query, easy_answers, qoi='rank', k=10, mode='non_zero', t_norm='prod', t_conorm='prod'):
    num_atoms = get_num_atoms(query.query_type)

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
    num_atoms = get_num_atoms(query_type)
    
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

def setup_dataset_and_graphs(data_dir):
    """Setup dataset and graphs (train, valid, test)."""
    dataset = Dataset()
    dataset.set_id2node(f'{data_dir}/ind2ent.pkl')
    dataset.set_id2rel(f'{data_dir}/ind2rel.pkl')
    dataset.set_node2title(f'{data_dir}/extra/entity2text.txt')

    # Setup training graph
    graph_train = Graph(dataset)
    graph_train.load_triples(f'{data_dir}/train.txt', skip_missing=False, add_reverse=True)
    
    # Setup validation graph
    graph_valid = Graph(dataset)
    for edge in graph_train.get_edges():
        graph_valid.add_edge(edge.get_head().get_name(), edge.get_name(), edge.get_tail().get_name(), skip_missing=False, add_reverse=False)
    graph_valid.load_triples(f'{data_dir}/valid.txt', skip_missing=False, add_reverse=True)
    
    # Setup test graph
    graph_test = Graph(dataset)
    for edge in graph_valid.get_edges():
        graph_test.add_edge(edge.get_head().get_name(), edge.get_name(), edge.get_tail().get_name(), skip_missing=False, add_reverse=False)
    graph_test.load_triples(f'{data_dir}/test.txt', skip_missing=False, add_reverse=True)
    
    return dataset, graph_train, graph_valid, graph_test

def load_query_datasets(dataset, data_dir, query_type):
    """Load query datasets for the specific query type."""
    # Load complete query dataset
    query_dataset = QueryDataset(dataset)
    query_path = get_query_file_paths(data_dir, query_type, hard=False)
    query_dataset.load_queries_from_pkl(query_path, query_type=query_type)
    
    # Load hard query dataset
    query_dataset_hard = QueryDataset(dataset)
    query_path_hard = get_query_file_paths(data_dir, query_type, hard=True)
    query_dataset_hard.load_queries_from_pkl(query_path_hard, query_type=query_type)
    
    return query_dataset, query_dataset_hard

def main():
    parser = argparse.ArgumentParser(description='Calculate average Shapley values for a specific query type')
    parser.add_argument('query_type', choices=['2p', '3p', '2i', '2u', '3i', 'pi', 'up', 'ip'], 
                       help='Query type to process')
    parser.add_argument('--data_dir', default='data/FB15k-237', 
                       help='Directory containing the dataset (default: data/FB15k-237)')
    parser.add_argument('--cqd_cache', default='data/FB15k-237/all_1p_queries_top25.json',
                       help='Path to CQD cache file (default: data/FB15k-237/all_1p_queries_top25.json)')
    parser.add_argument('--qoi', default='rank', choices=['rank', 'score'],
                       help='Quantity of interest (default: rank)')
    parser.add_argument('--k', type=int, default=10,
                       help='Top-k parameter (default: 10)')
    parser.add_argument('--mode', default='non_zero', choices=['non_zero', 'top_k', 'all'],
                       help='Filtering mode for results (default: non_zero)')
    parser.add_argument('--t_norm', default='prod', 
                       help='T-norm parameter (default: prod)')
    parser.add_argument('--t_conorm', default='prod',
                       help='T-conorm parameter (default: prod)')
    parser.add_argument('--output_dir', default='.',
                       help='Output directory for results (default: current directory)')
    
    args = parser.parse_args()
    
    print(f"Processing query type: {args.query_type}")
    print(f"Data directory: {args.data_dir}")
    print(f"Parameters: qoi={args.qoi}, k={args.k}, mode={args.mode}, t_norm={args.t_norm}, t_conorm={args.t_conorm}")
    
    # Setup dataset and graphs
    print("Setting up dataset and graphs...")
    dataset, graph_train, graph_valid, graph_test = setup_dataset_and_graphs(args.data_dir)
    print(f"Train graph: {graph_train.get_num_nodes()} nodes, {graph_train.get_num_edges()} edges")
    print(f"Valid graph: {graph_valid.get_num_nodes()} nodes, {graph_valid.get_num_edges()} edges")
    print(f"Test graph: {graph_test.get_num_nodes()} nodes, {graph_test.get_num_edges()} edges")
    
    # Load query datasets for the specific query type
    print(f"Loading query datasets for {args.query_type}...")
    query_dataset, query_dataset_hard = load_query_datasets(dataset, args.data_dir, args.query_type)
    print(f"Loaded {query_dataset.get_num_queries()} complete queries and {query_dataset_hard.get_num_queries()} hard queries")
    
    # Setup reasoners
    print("Setting up reasoners...")
    reasoner_train = SymbolicReasoning(graph_train, logging=False)
    reasoner_valid = SymbolicReasoning(graph_valid, logging=False)
    reasoner_test = SymbolicReasoning(graph_test, logging=False)
    
    # Load CQD cache
    print(f"Loading CQD cache from {args.cqd_cache}...")
    cqd_cache = pd.read_json(args.cqd_cache, orient='records')
    
    # Setup XCQA
    xcqa = XCQA(symbolic=reasoner_train, dataset=dataset, cqd_cache=cqd_cache, logging=False)
    
    # Calculate average Shapley values
    print(f"Calculating average Shapley values for query type: {args.query_type}")
    start = time.time()
    
    avg_shapley_values = average_shapley_values_for_query_type(
        xcqa, query_dataset, query_dataset_hard, args.query_type, 
        qoi=args.qoi, k=args.k, mode=args.mode, 
        t_norm=args.t_norm, t_conorm=args.t_conorm
    )
    
    end = time.time()
    
    print(f"Average Shapley values for {args.query_type}: {avg_shapley_values}")
    print(f"Time taken: {end - start:.2f} seconds")
    
    # Save results
    results = {args.query_type: avg_shapley_values}
    output_filename = f"{args.output_dir}/average_shapley_values_{args.query_type}.json"
    
    with open(output_filename, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to: {output_filename}")

if __name__ == "__main__":
    main()