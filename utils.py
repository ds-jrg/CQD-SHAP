import os
from graph import Dataset, Graph
import logging
from query import  QueryDataset, Query

def get_query_types(version=1):
    """Get the list of query types based on the version."""
    query_types = ['2p', '3p', '2i', '2u', '3i', 'pi', 'ip', 'up']
    if version == 2:
        query_types.extend(['4p', '4i'])
    return query_types

def get_num_atoms(query_type):
    """Get the number of atoms for a given query type."""
    atom_mapping = {
        '2p': 2, '3p': 3, '2i': 2, '2u': 2, 
        '3i': 3, 'pi': 3, 'up': 3, 'ip': 3,
        '4p': 4, '4i': 4
    }
    if query_type not in atom_mapping:
        raise ValueError(f"Unsupported query type: {query_type}.")
    return atom_mapping[query_type]

def get_first(query_type):
    """Get the number of atoms for a given query type."""
    atom_mapping = {
        '2p': [0], '3p': [0], '2i': [0, 1], '2u': [0, 1], 
        '3i': [0, 1, 2], 'pi': [0, 2], 'up': [0, 1], 'ip': [0, 1],
        '4p': [0], '4i': [0, 1, 2, 3]
    }
    if query_type not in atom_mapping:
        raise ValueError(f"Unsupported query type: {query_type}.")
    return atom_mapping[query_type]

def get_last(query_type):
    """Get the last atom for a given query type."""
    atom_mapping = {
        '2p': [1], '3p': [2], '2i': [0, 1], '2u': [0, 1], 
        '3i': [0, 1, 2], 'pi': [1, 2], 'up': [2], 'ip': [2],
        '4p': [3], '4i': [0, 1, 2, 3]
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

def setup_dataset_and_graphs(data_dir, logging: bool = True, add_reverse: bool = True):
    """Setup dataset and graphs (train, valid, test)."""
    dataset = Dataset(logging=logging)
    # check if ind2ent.pkl exist or id2ent.pkl
    if os.path.exists(f'{data_dir}/ind2ent.pkl'):
        dataset.set_id2node(f'{data_dir}/ind2ent.pkl')
    else:
        dataset.set_id2node(f'{data_dir}/id2ent.pkl')
    if os.path.exists(f'{data_dir}/ind2rel.pkl'):
        dataset.set_id2rel(f'{data_dir}/ind2rel.pkl')
    else:
        dataset.set_id2rel(f'{data_dir}/id2rel.pkl')
    if os.path.exists(f'{data_dir}/extra/entity2text.txt'):
        dataset.set_node2title(f'{data_dir}/extra/entity2text.txt')

    # Setup training graph
    graph_train = Graph(dataset, logging=logging)
    graph_train.load_triples(f'{data_dir}/train.txt', skip_missing=False, add_reverse=add_reverse)
    
    # Setup validation graph
    graph_valid = Graph(dataset, logging=logging)
    for edge in graph_train.get_edges():
        graph_valid.add_edge(edge.get_head().get_name(), edge.get_name(), edge.get_tail().get_name(), skip_missing=False, add_reverse=False)
    graph_valid.load_triples(f'{data_dir}/valid.txt', skip_missing=False, add_reverse=add_reverse)
    
    # Setup test graph
    graph_test = Graph(dataset)
    for edge in graph_valid.get_edges():
        graph_test.add_edge(edge.get_head().get_name(), edge.get_name(), edge.get_tail().get_name(), skip_missing=False, add_reverse=False)
    graph_test.load_triples(f'{data_dir}/test.txt', skip_missing=False, add_reverse=add_reverse)
    
    return dataset, graph_train, graph_valid, graph_test

def load_all_queries(dataset, data_dir, split='test', version=2):
    """Load all query datasets for all query types."""
    query_types = ['2p', '3p', '2i', '2u', '3i', 'pi', 'ip', 'up']
    if version == 2:
        query_types.extend(['4p', '4i'])
    query_dataset = QueryDataset(dataset, type='complete')
    query_dataset_hard = QueryDataset(dataset, type='hard')
    if version == 1:

        for query_type in query_types:
            # Load complete query dataset
            query_path = get_query_file_paths(data_dir, query_type, hard=False, split=split)
            query_dataset.load_queries_v1(query_path, query_type=query_type)
            
            # Load hard query dataset
            query_path_hard = get_query_file_paths(data_dir, query_type, hard=True, split=split)
            query_dataset_hard.load_queries_v1(query_path_hard, query_type=query_type)

    elif version == 2:
        print(f'Loading queries for the complete dataset ...')
        query_dataset.load_queries_v2(data_dir, split=split)
        print(f'Loading queries for the hard dataset ...')
        query_dataset_hard.load_queries_v2(data_dir, split=split)
        print(f'Finished loading queries.')

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

def format_atom(atom, dataset: Dataset, fol_format: bool = False):
    '''
    Format an atom for display.
    
    Args:
        atom: Dictionary with 'head', 'relation', 'tail' keys
        dataset: Dataset object for looking up names/titles
        fol_format: If True, format for FOL (e.g., "p_5(e_123, V1)")
                   If False, format for human-readable (e.g., "Alice (123) --[friend_of (5)]--> V1")
    
    Returns:
        Formatted string representation of the atom
    '''
    head_id = atom['head']
    relation_id = atom['relation']
    tail_id = atom['tail']
    
    if fol_format:
        # FOL format: friend_of(Alice, V1)
        if isinstance(head_id, int):
            head_title = dataset.get_title_by_id(head_id)
            head_str = head_title if head_title else f"e_{head_id}"
        else:
            head_str = head_id
        
        if isinstance(relation_id, int):
            relation_name = dataset.get_relation_by_id(relation_id)
            if relation_name:
                # Remove +/- prefix if present
                if relation_name.startswith(('+', '-')):
                    relation_name = relation_name[1:]
                # Extract the last part after the final /
                if '/' in relation_name:
                    rel_str = relation_name.split('/')[-1]
                else:
                    rel_str = relation_name
            else:
                rel_str = f"p_{relation_id}"
        else:
            rel_str = f"p_{relation_id}"
        
        if isinstance(tail_id, int):
            tail_title = dataset.get_title_by_id(tail_id)
            tail_str = tail_title if tail_title else f"e_{tail_id}"
        else:
            tail_str = tail_id
        
        return f"{rel_str}({head_str}, {tail_str})"
    else:
        # Human-readable format: Alice (123) --[friend_of (5)]--> V1
        if isinstance(head_id, int):
            head_title = dataset.get_title_by_id(head_id)
            head_str = f"{head_title} ({head_id})"
        else:
            head_str = head_id  # variable like V1, V2
        
        if isinstance(relation_id, int):
            relation_name = dataset.get_relation_by_id(relation_id)
            relation_str = f"{relation_name} ({relation_id})"
        else:
            relation_str = relation_id  # should not happen
        
        if isinstance(tail_id, int):
            tail_title = dataset.get_title_by_id(tail_id)
            tail_str = f"{tail_title} ({tail_id})"
        else:
            tail_str = tail_id  # variable like V1, V2
        
        return f"{head_str} --[{relation_str}]--> {tail_str}"


def human_readable(query: Query, dataset: Dataset, fol: bool = False):
    '''
    Convert a Query object to a human-readable string using the dataset's entity and relation mappings.
    For example, 2p query should look like:
    Alice (1234) --[friend_of (56)]--> V1
    V1 --[colleague_of (78)]--> V2
    ∃V2: friend_of(Alice, V1) ∧ colleague_of(V1, V2)
    
    If fol=True, returns the FOL representation instead.
    '''
    if fol:
        return _generate_fol(query, dataset)
    
    atoms = query.get_atoms()
    lines = []
    for index, atom in atoms.items():
        line = format_atom(atom, dataset, fol_format=False)
        lines.append(line)
    return "\n".join(lines)


def _generate_fol(query: Query, dataset: Dataset):
    '''
    Generate FOL (First-Order Logic) representation of a query.
    '''
    query_type = query.get_type()
    atoms = query.get_atoms()
    
    if query_type == '1p':
        # ?V1: p1(e1, V1)
        atom = format_atom(atoms[0], dataset, fol_format=True)
        return f"?V1: {atom}"
    
    elif query_type == '2p':
        # ?V2: ∃V1 · p1(e1, V1) ∧ p2(V1, V2)
        atom0 = format_atom(atoms[0], dataset, fol_format=True)
        atom1 = format_atom(atoms[1], dataset, fol_format=True)
        return f"?V2: ∃V1 · {atom0} ∧ {atom1}"
    
    elif query_type == '3p':
        # ?V3: ∃V1, V2 · p1(e1, V1) ∧ p2(V1, V2) ∧ p3(V2, V3)
        atom0 = format_atom(atoms[0], dataset, fol_format=True)
        atom1 = format_atom(atoms[1], dataset, fol_format=True)
        atom2 = format_atom(atoms[2], dataset, fol_format=True)
        return f"?V3: ∃V1, V2 · {atom0} ∧ {atom1} ∧ {atom2}"
    
    elif query_type == '4p':
        # ?V4: ∃V1, V2, V3 · p1(e1, V1) ∧ p2(V1, V2) ∧ p3(V2, V3) ∧ p4(V3, V4)
        atom0 = format_atom(atoms[0], dataset, fol_format=True)
        atom1 = format_atom(atoms[1], dataset, fol_format=True)
        atom2 = format_atom(atoms[2], dataset, fol_format=True)
        atom3 = format_atom(atoms[3], dataset, fol_format=True)
        return f"?V4: ∃V1, V2, V3 · {atom0} ∧ {atom1} ∧ {atom2} ∧ {atom3}"
    
    elif query_type == '2i':
        # ?V1: p1(e1, V1) ∧ p2(e2, V1)
        atom0 = format_atom(atoms[0], dataset, fol_format=True)
        atom1 = format_atom(atoms[1], dataset, fol_format=True)
        return f"?V1: {atom0} ∧ {atom1}"
    
    elif query_type == '3i':
        # ?V1: p1(e1, V1) ∧ p2(e2, V1) ∧ p3(e3, V1)
        atom0 = format_atom(atoms[0], dataset, fol_format=True)
        atom1 = format_atom(atoms[1], dataset, fol_format=True)
        atom2 = format_atom(atoms[2], dataset, fol_format=True)
        return f"?V1: {atom0} ∧ {atom1} ∧ {atom2}"
    
    elif query_type == '4i':
        # ?V1: p1(e1, V1) ∧ p2(e2, V1) ∧ p3(e3, V1) ∧ p4(e4, V1)
        atom0 = format_atom(atoms[0], dataset, fol_format=True)
        atom1 = format_atom(atoms[1], dataset, fol_format=True)
        atom2 = format_atom(atoms[2], dataset, fol_format=True)
        atom3 = format_atom(atoms[3], dataset, fol_format=True)
        return f"?V1: {atom0} ∧ {atom1} ∧ {atom2} ∧ {atom3}"
    
    elif query_type == '2u':
        # ?V1: p1(e1, V1) ∨ p2(e2, V1)
        atom0 = format_atom(atoms[0], dataset, fol_format=True)
        atom1 = format_atom(atoms[1], dataset, fol_format=True)
        return f"?V1: {atom0} ∨ {atom1}"
    
    elif query_type == 'up':  # 2u1p
        # ?V2: ∃V1 · (p1(e1, V1) ∨ p2(e2, V1)) ∧ p3(V1, V2)
        atom0 = format_atom(atoms[0], dataset, fol_format=True)
        atom1 = format_atom(atoms[1], dataset, fol_format=True)
        atom2 = format_atom(atoms[2], dataset, fol_format=True)
        return f"?V2: ∃V1 · ({atom0} ∨ {atom1}) ∧ {atom2}"
    
    elif query_type == 'ip':  # 2i1p
        # ?V2: ∃V1 · p1(e1, V1) ∧ p2(e2, V1) ∧ p3(V1, V2)
        atom0 = format_atom(atoms[0], dataset, fol_format=True)
        atom1 = format_atom(atoms[1], dataset, fol_format=True)
        atom2 = format_atom(atoms[2], dataset, fol_format=True)
        return f"?V2: ∃V1 · {atom0} ∧ {atom1} ∧ {atom2}"
    
    elif query_type == 'pi':  # 1p2i
        # ?V2: ∃V1 · p1(e1, V1) ∧ p2(V1, V2) ∧ p3(e2, V2)
        atom0 = format_atom(atoms[0], dataset, fol_format=True)
        atom1 = format_atom(atoms[1], dataset, fol_format=True)
        atom2 = format_atom(atoms[2], dataset, fol_format=True)
        return f"?V2: ∃V1 · {atom0} ∧ {atom1} ∧ {atom2}"
    
    else:
        raise ValueError(f"FOL generation not supported for query type: {query_type}")