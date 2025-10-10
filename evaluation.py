import logging
from symbolic_torch import SymbolicReasoning
from xcqa_torch import XCQA
from utils import get_num_atoms, setup_dataset_and_graphs, load_query_datasets, get_first, get_last, compute_metrics
import argparse
from tqdm import tqdm
from shapley import shapley_value
import random
import json
from statistics import mean
from pathlib import Path
import os
import pandas as pd
import time
import torch

# ------------------------------------------------------------
# Setup logging
# ------------------------------------------------------------
def setup_logging(log_path="evaluation.log"):
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

# ============================================================
# Utility functions
# ============================================================
random.seed(42)
torch.manual_seed(42)

def get_metric_dict():
    return {k: [] for k in ["mrr", "hit_1", "hit_3", "hit_10"]}

def average_metrics(metrics: dict) -> dict:
    return {k: mean(v) if v else 0.0 for k, v in metrics.items()}

def append_metrics(metrics: dict, values: tuple):
    for key, value in zip(["mrr", "hit_1", "hit_3", "hit_10"], values):
        metrics[key].append(value)

def report_metrics(query_type: str, metrics_original: dict, metrics_new: dict, satisfied_per_query: list):
    """Log summary for original vs new metrics in compact one-line format."""
    keys = ["mrr", "hit_1", "hit_3", "hit_10"]
    logging.info(f"\n=== Metrics Summary for query type: {query_type} ===")

    for key in keys:
        before = mean(metrics_original[key]) if metrics_original[key] else 0.0
        after = mean(metrics_new[key]) if metrics_new[key] else 0.0
        delta = after - before
        logging.info(f"{key.upper():<6}: before={before:.4f} | after={after:.4f} | Δ={delta:+.4f}")

    avg_satisfied = mean(satisfied_per_query) if satisfied_per_query else 0.0
    total_queries = len(satisfied_per_query)
    logging.info(f"Condition satisfaction: avg per-query = {avg_satisfied:.4f}, total satisfied = {sum(satisfied_per_query)}, total queries = {total_queries}")
    logging.info("=" * 80 + "\n")

# ============================================================
# Evaluation
# ============================================================
def evaluation(hard: list,
               complete: list,
               num_atoms: int,
               xcqa: "XCQA",
               k: int,
               t_norm: str,
               t_conorm: str,
               method: str = "shapley",
               mode: str = "necessary",
               records: list = None):
    
    assert mode in ("necessary", "sufficient"), "mode must be 'necessary' or 'sufficient'"
    flip_value = 0 if mode == "necessary" else 1
    keep_value = 1 - flip_value
    query_type = hard[0].query_type

    base_coalition = [keep_value] * num_atoms
    metrics_original = get_metric_dict()
    metrics_new = get_metric_dict()
    satisfied_per_query = []

    def choose_atom(query, result_ref, hard_answer, method):
        if method == "score":
            importance = {a: float(result_ref.loc[hard_answer][f"scores_{a}"]) for a in range(num_atoms)}
            if query.query_type == "2u":
                atom = max(importance, key=importance.get)
            elif query.query_type == "up":
                max_idx = max((0, 1), key=lambda k: importance[k])
                atom = min((max_idx, 2), key=lambda k: importance[k])
            else:
                atom = min(importance, key=importance.get)
            return atom, importance[atom]
        elif method == "shapley":
            shapley_vals = {
                a: shapley_value(
                    xcqa, query, a, filtered_exclude, hard_answer,
                    "rank", k, t_norm, t_conorm
                )
                for a in range(num_atoms)
            }
            atom = max(shapley_vals, key=shapley_vals.get)
            return atom, shapley_vals[atom]
        elif method == "random":
            return random.randint(0, num_atoms - 1), None
        elif method == "first":
            return random.choice(get_first(query.query_type)), None
        elif method == "last":
            return random.choice(get_last(query.query_type)), None
        else:
            raise ValueError(f"Unknown method: {method}")

    for i, (query_hard, query_complete) in enumerate(tqdm(zip(hard, complete), total=len(hard))):
        all_answers = set(query_complete.get_answer())
        result_original = xcqa.query_execution(query_hard, k=k, coalition=base_coalition, t_norm=t_norm, t_conorm=t_conorm)
        
        current_orig = get_metric_dict()
        current_new = get_metric_dict()
        satisfied_count = 0

        for hard_answer in query_hard.answer:
            global filtered_exclude
            filtered_exclude = all_answers - {hard_answer}
            start_time = time.time()
            mrr, hit_1, hit_3, hit_10 = compute_metrics(result_original, query_complete.answer, hard_answer)

            satisfied = (hit_1 == 1.0 and mrr == 1.0) if mode == "necessary" else (hit_1 != 1.0 and mrr != 1.0)
            if not satisfied:
                continue
            satisfied_count += 1
            append_metrics(current_orig, (mrr, hit_1, hit_3, hit_10))

            result_ref = (
                xcqa.query_execution(query_hard, k=k, coalition=[1]*num_atoms, t_norm=t_norm, t_conorm=t_conorm)
                if method == "score" and mode == "sufficient" else result_original
            )
            chosen_atom, best_value = choose_atom(query_hard, result_ref, hard_answer, method)
            new_coalition = base_coalition.copy()
            new_coalition[chosen_atom] = flip_value

            result_new = xcqa.query_execution(query_hard, k=k, coalition=new_coalition, t_norm=t_norm, t_conorm=t_conorm)
            mrr_new, hit_1_new, hit_3_new, hit_10_new = compute_metrics(result_new, query_complete.answer, hard_answer)
            runtime = time.time() - start_time
            append_metrics(current_new, (mrr_new, hit_1_new, hit_3_new, hit_10_new))

            if records is not None:
                records.append({
                    "query_type": query_type,
                    "query_idx": i,
                    "target_idx": hard_answer,
                    "best_atom": chosen_atom,
                    "best_value": best_value,
                    "runtime": runtime,
                    "delta_mrr": mrr_new - mrr,
                })
        if satisfied_count != 0:
            satisfied_per_query.append(satisfied_count)
        if current_orig["mrr"]:
            avg_o = average_metrics(current_orig)
            avg_n = average_metrics(current_new)
            for key in ["mrr", "hit_1", "hit_3", "hit_10"]:
                metrics_original[key].append(avg_o[key])
                metrics_new[key].append(avg_n[key])

    report_metrics(query_type, metrics_original, metrics_new, satisfied_per_query)
    return metrics_original, metrics_new, satisfied_per_query


# ============================================================
# Main entry point
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate necessary/sufficient explanations')
    parser.add_argument('--kg', choices=['Freebase', 'NELL'], default='Freebase')
    parser.add_argument('--query_type', choices=['2p', '3p', '2i', '2u', '3i', 'pi', 'up', 'ip'])
    parser.add_argument('--data_dir')
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--t_norm', default='prod')
    parser.add_argument('--t_conorm', default='prod')
    parser.add_argument('--split', default='test', choices=['train', 'valid', 'test'])
    parser.add_argument('--model_path')
    parser.add_argument('--method', default="shapley", choices=['shapley', 'score', 'random', 'first', 'last'])
    parser.add_argument('--explanation', default="necessary", choices=['necessary', 'sufficient'])
    parser.add_argument('--output_path', default='eval')
    parser.add_argument('--log_file', default='evaluation.log')
    args = parser.parse_args()
    setup_logging(args.log_file)
    # --- Restore KG-specific defaults if not explicitly given ---
    if not args.data_dir:
        args.data_dir = {
            'Freebase': 'data/FB15k-237',
            'NELL': 'data/NELL'
        }[args.kg]

    if not args.model_path:
        args.model_path = {
            'Freebase': 'models/FB15k-237-model-rank-1000-epoch-100-1602508358.pt',
            'NELL': 'models/NELL-model-rank-1000-epoch-100-1602499096.pt'
        }[args.kg]

    logging.info(f"Evaluating {args.explanation} explanations using {args.method} on {args.kg} KG")
    logging.info(f"Model path: {args.model_path}")

    dataset, graph_train, graph_valid = setup_dataset_and_graphs(args.data_dir)
    output_file = os.path.join(args.output_path, f"{args.explanation}_{args.query_type}_{args.method}.csv")

    reasoner = SymbolicReasoning(graph_valid if args.split == "test" else graph_train, logging=False)
    xcqa = XCQA(symbolic=reasoner, dataset=dataset, logging=False, model_path=args.model_path)

    query_types = [args.query_type] if args.query_type else ['2u', '2i', '3i', 'up', '2p', 'pi', 'ip', '3p']
    records = []

    for query_type in query_types:
        logging.info(f"Processing query type: {query_type}")
        query_dataset, query_dataset_hard = load_query_datasets(dataset, args.data_dir, query_type, args.split)
        num_atoms = get_num_atoms(query_type)
        hard = query_dataset_hard.get_queries(query_type)
        complete = query_dataset.get_queries(query_type)

        metrics_original, metrics_new, satisfied_per_query = evaluation(
            hard, complete, num_atoms, xcqa, args.k, args.t_norm, args.t_conorm, args.method, args.explanation, records
        )

        logging.info(f"Finished processing query type: {query_type}")
        logging.info("-" * 80)

    df = pd.DataFrame(records)
    csv_path = Path(output_file)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    logging.info(f"✅ Saved all intermediate steps to {csv_path}")