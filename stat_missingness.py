from utils import setup_dataset_and_graphs
from utils import load_all_queries
from symbolic_torch import SymbolicReasoning
from xcqa_torch_stat import XCQA
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
    t_norm="prod",
    t_conorm="prod",
    k=10,
    reachability_threshold=0.99
):
    """
    For a given query_type:
        - Counts ALL reachable missingness patterns (exact structural cases)
        - Separately computes minimal missingness statistics
    Returns:
        {
            total_qa_pairs,
            combination_counts,
            combination_frequencies,
            minimal_missingness_counts,
            minimal_missingness_frequencies
        }
    """

    queries = query_dataset_hard.get_queries(query_type)
    num_atoms = get_num_atoms(query_type)
    coalition = [0] * num_atoms
    # Generate all binary patterns (e.g., 2p -> [0,0], [0,1], [1,0], [1,1])
    patterns = [
        list(map(int, bin(i)[2:].zfill(num_atoms)))
        for i in range(2 ** num_atoms)
    ]

    combination_counts = {tuple(pattern): 0 for pattern in patterns}
    minimal_missingness_counts = {i: 0 for i in range(num_atoms + 1)}

    total_qa_pairs = 0

    for query in tqdm(queries, desc=f"Processing queries ({query_type})"):

        hard_answers = query.get_answer()

        # Execute query under every pattern once per query
        patterns_executions = [
            xcqa.query_execution(
                query,
                k=k,
                coalition=coalition,
                t_norm=t_norm,
                t_conorm=t_conorm,
                pattern = pattern
            )
            for pattern in patterns
        ]

        for hard_answer in hard_answers:
            total_qa_pairs += 1

            reachable_patterns = []

            for pattern, execution in zip(patterns, patterns_executions):
                if execution.loc[hard_answer, "final_score"] > reachability_threshold:
                    reachable_patterns.append(pattern)

            if not reachable_patterns:
                continue

            # ✅ Compute minimal missingness
            minimal_missingness = min(sum(p) for p in reachable_patterns)
            minimal_missingness_counts[minimal_missingness] += 1

            # ✅ Count ONLY patterns at minimal level
            for pattern in reachable_patterns:
                if sum(pattern) == minimal_missingness:
                    combination_counts[tuple(pattern)] += 1

    # Frequencies
    combination_frequencies = {
        pattern: count / total_qa_pairs if total_qa_pairs > 0 else 0.0
        for pattern, count in combination_counts.items()
    }

    minimal_missingness_frequencies = {
        m: count / total_qa_pairs if total_qa_pairs > 0 else 0.0
        for m, count in minimal_missingness_counts.items()
    }

    # Logging
    logging.info(f"\nTotal QA pairs: {total_qa_pairs}\n")

    logging.info("Exact Minimal Missingness Pattern Counts:")
    for pattern, count in sorted(combination_counts.items()):
        if count > 0:
            logging.info(
                f"  Pattern {pattern}: {count} "
                f"({combination_frequencies[pattern]:.6f})"
            )

    logging.info("\nMinimal Missingness Distribution:")
    for m, count in sorted(minimal_missingness_counts.items()):
        logging.info(
            f"  Missingness {m}: {count} "
            f"({minimal_missingness_frequencies[m]:.6f})"
        )

    return {
        "total_qa_pairs": total_qa_pairs,
        "combination_counts": combination_counts,
        "combination_frequencies": combination_frequencies,
        "minimal_missingness_counts": minimal_missingness_counts,
        "minimal_missingness_frequencies": minimal_missingness_frequencies
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
    xcqa = XCQA(reasoner=reasoner, reasoner_test=reasoner_test,dataset=dataset, logging=False, model_path=args.model_path, normalize=args.normalize, use_topk=args.use_topk)
    xcqa_topk = XCQA(reasoner=reasoner, reasoner_test=reasoner_test, dataset=dataset, logging=False, model_path=args.model_path, normalize=args.normalize, use_topk=True)
    
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
            xcqa=xcqa_choice)