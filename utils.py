from graph import Dataset, Graph
import logging
from query import  QueryDataset

def get_num_atoms(query_type):
    """Get the number of atoms for a given query type."""
    atom_mapping = {
        '2p': 2, '3p': 3, '2i': 2, '2u': 2, 
        '3i': 3, 'pi': 3, 'up': 3, 'ip': 3
    }
    if query_type not in atom_mapping:
        raise ValueError(f"Unsupported query type: {query_type}.")
    return atom_mapping[query_type]

def get_first(query_type):
    """Get the number of atoms for a given query type."""
    atom_mapping = {
        '2p': [0], '3p': [0], '2i': [0, 1], '2u': [0, 1], 
        '3i': [0, 1, 2], 'pi': [0, 2], 'up': [0, 1], 'ip': [0, 1]
    }
    if query_type not in atom_mapping:
        raise ValueError(f"Unsupported query type: {query_type}.")
    return atom_mapping[query_type]

def get_last(query_type):
    """Get the last atom for a given query type."""
    atom_mapping = {
        '2p': [1], '3p': [2], '2i': [0, 1], '2u': [0, 1], 
        '3i': [0, 1, 2], 'pi': [1, 2], 'up': [2], 'ip': [2]
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

def setup_dataset_and_graphs(data_dir, logging: bool = True):
    """Setup dataset and graphs (train, valid, test)."""
    dataset = Dataset(logging=logging)
    dataset.set_id2node(f'{data_dir}/ind2ent.pkl')
    dataset.set_id2rel(f'{data_dir}/ind2rel.pkl')
    dataset.set_node2title(f'{data_dir}/extra/entity2text.txt')

    # Setup training graph
    graph_train = Graph(dataset, logging=logging)
    graph_train.load_triples(f'{data_dir}/train.txt', skip_missing=False, add_reverse=True)
    
    # Setup validation graph
    graph_valid = Graph(dataset, logging=logging)
    for edge in graph_train.get_edges():
        graph_valid.add_edge(edge.get_head().get_name(), edge.get_name(), edge.get_tail().get_name(), skip_missing=False, add_reverse=False)
    graph_valid.load_triples(f'{data_dir}/valid.txt', skip_missing=False, add_reverse=True)
    
    return dataset, graph_train, graph_valid


def load_query_datasets(dataset, data_dir, query_type, split='test'):
    """Load query datasets for the specific query type."""
    # Load complete query dataset
    query_dataset = QueryDataset(dataset)
    query_path = get_query_file_paths(data_dir, query_type, hard=False, split=split)
    query_dataset.load_queries_from_pkl(query_path, query_type=query_type)
    
    # Load hard query dataset
    query_dataset_hard = QueryDataset(dataset)
    query_path_hard = get_query_file_paths(data_dir, query_type, hard=True, split=split)
    query_dataset_hard.load_queries_from_pkl(query_path_hard, query_type=query_type)
    
    return query_dataset, query_dataset_hard

def load_all_queries(dataset, data_dir, split='test'):
    """Load all query datasets for all query types."""
    query_types = ['2p', '3p', '2i', '2u', '3i', 'pi', 'ip', 'up']
    query_dataset = QueryDataset(dataset)
    query_dataset_hard = QueryDataset(dataset)
    for query_type in query_types:
        # Load complete query dataset
        query_path = get_query_file_paths(data_dir, query_type, hard=False, split=split)
        query_dataset.load_queries_from_pkl(query_path, query_type=query_type)
        
        # Load hard query dataset
        query_path_hard = get_query_file_paths(data_dir, query_type, hard=True, split=split)
        query_dataset_hard.load_queries_from_pkl(query_path_hard, query_type=query_type)

    return query_dataset, query_dataset_hard

def setup_logger(filename):
    # Create a logger
    logger = logging.getLogger(filename + "evaluation")
    logger.setLevel(logging.DEBUG)  # Set the lowest level of log messages to capture
    
    # Formatter to control the log output format
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Console handler (for terminal output)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    
    # File handler (for saving logs to a file)
    file_handler = logging.FileHandler(filename + ".log", mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # Avoid adding multiple handlers if function called multiple times
    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
    
    return logger

def compute_rank(result, answer_complete, target_answer):
    # Convert to sets for fast exclusion
    answer_complete_set = set(answer_complete)

    # Get a filtered version of result for each a_hard
    result_index = list(result.index)

    # Remove all other correct answers from ranking for this answer
    filtered_exclude = answer_complete_set - {target_answer}
    # Build mask once for speed
    filtered_index = [x for x in result_index if x not in filtered_exclude]
    rank = filtered_index.index(target_answer) + 1
    
    return rank

def check_missing_link(reasoner, head, relation, target):
    preds = reasoner.predict(head, relation, True, -1)
    score = float(preds.loc[target]['score'])
    return score != 1.0

def compute_metrics(result, answer_complete, target_answer):
    """
    result: pd.Series or pd.DataFrame with index = candidate answers, sorted by predicted score (descending).
    answer_complete: set/list/array of all correct answers (to exclude during filtered ranking).
    target_answer: the specific answer to evaluate.
    """

    mrr = 0.0
    hit_1 = 0
    hit_3 = 0
    hit_10 = 0

    # Convert to sets for fast exclusion
    answer_complete_set = set(answer_complete)

    # Get a filtered version of result for each a_hard
    result_index = list(result.index)

    # Remove all other correct answers from ranking for this answer
    filtered_exclude = answer_complete_set - {target_answer}
    # Build mask once for speed
    filtered_index = [x for x in result_index if x not in filtered_exclude]
    if target_answer in filtered_index:
        rank = filtered_index.index(target_answer) + 1
        mrr += 1.0 / rank
        if rank == 1:
            hit_1 += 1
        if rank <= 3:
            hit_3 += 1
        if rank <= 10:
            hit_10 += 1

    return mrr, hit_1, hit_3, hit_10