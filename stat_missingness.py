from utils import setup_dataset_and_graphs
from utils import load_all_queries
from symbolic_torch import SymbolicReasoning
from xcqa_torch import XCQA
from collections import defaultdict
from utils import check_missing_link, get_num_atoms
from tqdm import tqdm
import argparse
import os
import logging
from pathlib import Path

def setup_logging(log_path="missingness.log"):
    """Configure logging to both console and file."""
    Path(os.path.dirname(log_path) or ".").mkdir(parents=True, exist_ok=True)
    
    # Create a custom root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers (important if running in notebooks)
    for h in logger.handlers[:]:
        logger.removeHandler(h)

    # File handler
    f_handler = logging.FileHandler(log_path, mode='w')
    f_format = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    f_handler.setFormatter(f_format)
    logger.addHandler(f_handler)

    # Console handler
    c_handler = logging.StreamHandler()
    c_format = logging.Formatter("%(message)s")  # cleaner for console
    c_handler.setFormatter(c_format)
    logger.addHandler(c_handler)

    logging.info(f"Logging to {log_path} and console.")

def report_missing_link_combinations(
    query_type,
    query_dataset_hard,
    xcqa,
    reasoner,
    t_norm="prod",
    t_conorm="prod",
    k=10
):
    """
    For a given query_type:
        - considers all queries
        - considers all hard answers
        - counts all combinations of missing links across atoms

    Returns:
        {
            total_qa_pairs,
            combination_counts,
            combination_frequencies
        }
    """

    queries = query_dataset_hard.get_queries(query_type)
    num_atoms = get_num_atoms(query_type)

    total_qa_pairs = 0
    combination_counts = defaultdict(int)
    per_atom_counts = [0] * num_atoms

    for query in tqdm(queries, desc=f"Processing queries ({query_type})"):

        query_atoms = query.get_atoms()
        hard_answers = query.get_answer()

        coalition_binary = [0] * num_atoms
        result_xcqa = xcqa.query_execution(
            query,
            k=k,
            coalition=coalition_binary,
            t_norm=t_norm,
            t_conorm=t_conorm
        )

        for hard_answer_idx in tqdm(
            hard_answers,
            desc="  Hard answers",
            leave=False
        ):

            if hard_answer_idx not in result_xcqa.index:
                continue

            hard_answer = result_xcqa.loc[hard_answer_idx]

            missing_pattern = []

            for idx, atom in query_atoms.items():

                # -------- Resolve head / relation / tail -------- #

                if query_type in ['2i', '3i', '4i', '2u']:
                    head = atom['head']
                    relation = atom['relation']
                    tail = hard_answer_idx

                else:
                    # HEAD
                    if isinstance(atom['head'], int):
                        head = atom['head']
                    elif isinstance(atom['head'], str) and atom['head'].startswith('V'):
                        variable_number = int(atom['head'][1:])
                        head = hard_answer[f'variable_{variable_number}']
                    else:
                        raise ValueError(f"Unexpected head: {atom['head']}")

                    # RELATION
                    if isinstance(atom['relation'], int):
                        relation = atom['relation']
                    else:
                        raise ValueError(f"Unexpected relation: {atom['relation']}")

                    # TAIL
                    if isinstance(atom['tail'], int):
                        tail = atom['tail']
                    elif isinstance(atom['tail'], str) and atom['tail'].startswith('V'):
                        variable_number = int(atom['tail'][1:])

                        if query_type in ['ip', 'pi', 'up']:
                            limit = num_atoms - 2
                        else:
                            limit = num_atoms - 1

                        if variable_number == limit:
                            tail = hard_answer_idx
                        else:
                            tail = hard_answer[f'variable_{variable_number}']
                    else:
                        raise ValueError(f"Unexpected tail: {atom['tail']}")

                head = int(head)
                relation = int(relation)
                tail = int(tail)

                is_missed = check_missing_link(reasoner, head, relation, tail)
                is_missed = bool(is_missed)
                missing_pattern.append(is_missed)

                if is_missed:
                    per_atom_counts[idx] += 1

            # Convert to tuple so it can be used as dict key
            missing_pattern = tuple(missing_pattern)

            combination_counts[missing_pattern] += 1
            total_qa_pairs += 1

    # -------- Reporting -------- #

    logging.info(f"\nQuery type: {query_type}")
    logging.info(f"Total QA pairs: {total_qa_pairs}\n")

    logging.info("Missing link combinations:\n")
    for pattern, count in sorted(combination_counts.items()):
        freq = count / total_qa_pairs if total_qa_pairs > 0 else 0.0
        logging.info(f"{pattern}  ->  Count: {count} | Frequency: {freq:.6f}")
    logging.info("\nPer-atom missing counts:\n")

    for atom_idx in range(num_atoms):
        count = per_atom_counts[atom_idx]
        freq = count / total_qa_pairs if total_qa_pairs > 0 else 0.0
        logging.info(
            f"Atom {atom_idx} missed -> Count: {count} | Frequency: {freq:.6f}"
        )
    return {
        "query_type": query_type,
        "total_qa_pairs": total_qa_pairs,
        "combination_counts": dict(combination_counts),
        "combination_frequencies": {
            k: v / total_qa_pairs if total_qa_pairs > 0 else 0.0
            for k, v in combination_counts.items()
        }
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Report missing link stats for datasets and query types')
    parser.add_argument('--kg', choices=['Freebase', 'NELL'], default='Freebase')
    parser.add_argument('--benchmark', choices=[1, 2], type=int, default=2)
    parser.add_argument('--query_type', choices=['2p', '3p', '4p', '2i', '3i', '4i', '2u', 'pi', 'up', 'ip'])
    parser.add_argument('--data_dir')
    parser.add_argument('--k', type=int, default=2048, help='Top-k filtering for XCQA (only applies to 3p and 4p queries when --use_topk is False)')
    parser.add_argument('--t_norm', default='prod')
    parser.add_argument('--t_conorm', default='prod')
    parser.add_argument('--split', default='test', choices=['train', 'valid', 'test'])
    parser.add_argument('--model_path')
    parser.add_argument('--output_path', default='missingness')
    parser.add_argument('--log_file')
    parser.add_argument('--normalize', action='store_true', help='Whether to normalize scores using sigmoid', default=False)
    parser.add_argument('--use_topk', action='store_true', help='Whether to use top-k filtering in XCQA', default=False)
    args = parser.parse_args()
    output_path = os.path.join(args.output_path, args.kg)
    os.makedirs(output_path, exist_ok=True)
    if args.benchmark == 1 and args.query_type in ['4i', '4p']:
        raise ValueError("Query types '4i' and '4p' are not supported in Benchmark 1.")
    if not args.log_file:
        if args.query_type:
            log_path = os.path.join(output_path, f"bench{args.benchmark}_{args.query_type}.log")
        else:
            log_path = os.path.join(output_path, f"bench{args.benchmark}_all.log")
    else:
        log_path = args.log_file
    setup_logging(log_path)
    # --- Restore KG-specific defaults if not explicitly given ---
    if not args.data_dir:
        if args.benchmark == 1:
            args.data_dir = {
                'Freebase': 'data/FB15k-237',
                'NELL': 'data/NELL995'
            }[args.kg]
        elif args.benchmark == 2:
            args.data_dir = {
                'Freebase': 'data/FB15k-237+H',
                'NELL': 'data/NELL995+H'
            }[args.kg]
        else:
            raise ValueError(f"Unknown benchmark: {args.benchmark}. Supported: 1 (Original CQD), 2 (Is Complex Query Answering Really Complex?)")

    if not args.model_path:
        args.model_path = {
            'Freebase': 'models/FB15k-237-model-rank-1000-epoch-100-1602508358.pt',
            'NELL': 'models/NELL-model-rank-1000-epoch-100-1602499096.pt'
        }[args.kg]

    logging.info(f"Computing stats for KG: {args.kg}, Benchmark: {args.benchmark}, Query type: {args.query_type or 'ALL'}")
    logging.info(f"Model path: {args.model_path}")

    dataset, graph_train, graph_valid, graph_test = setup_dataset_and_graphs(args.data_dir, logging=True, add_reverse=(args.benchmark==1))
    logging.info("Dataset and Graphs are set up.")
    query_dataset, query_dataset_hard = load_all_queries(dataset, args.data_dir, "test", version=args.benchmark)
    logging.info("Queries are loaded.")
    all_relations = list(dataset.id2rel.keys())
    logging.info(f"There are {len(all_relations)} relations in the dataset.")

    reasoner = SymbolicReasoning(graph_valid if args.split == "test" else graph_train, logging=False, gpu=False)
    reasoner_test = SymbolicReasoning(graph_test if args.split == "test" else graph_valid, logging=False, gpu=False)
    xcqa = XCQA(symbolic=reasoner, dataset=dataset, logging=False, model_path=args.model_path, normalize=args.normalize, use_topk=args.use_topk)
    xcqa_topk = XCQA(symbolic=reasoner, dataset=dataset, logging=False, model_path=args.model_path, normalize=args.normalize, use_topk=True)
    logging.info("XCQA model is initialized (normalize={})".format(args.normalize))

    if args.query_type:
        query_types = [args.query_type]
    else:
        if args.benchmark == 1:
            query_types =  ['2u', '2i', '3i', 'up', '2p', 'pi', 'ip', '3p']
        else:
            query_types =  ['2u', '2i', '3i', '4i', 'up', '2p', 'pi', 'ip', '3p', '4p']

    for query_type in query_types:
        logging.info(f"\n\n=== Processing query type: {query_type} ===")
        xcqa_choice = xcqa_topk if query_type in ['3p', '4p'] else xcqa
        if query_type in ['3p', '4p']:
            logging.info("Using XCQA with top-k filtering for this query type (k={})".format(args.k))
        result = report_missing_link_combinations(
            query_type=query_type,
            query_dataset_hard=query_dataset_hard,
            xcqa=xcqa_choice,
            reasoner=reasoner)