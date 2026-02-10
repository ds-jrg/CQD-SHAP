import logging
from symbolic_torch import SymbolicReasoning
from xcqa_torch import XCQA
from utils import get_num_atoms, setup_dataset_and_graphs, get_first, get_last, compute_metrics
import argparse
from tqdm import tqdm
from shapley import Shapley
import random
import json
from statistics import mean
from pathlib import Path
import os
import pandas as pd
import time
import torch
from utils import get_query_types
from utils import load_all_queries

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
def choose_atom(query, hard_answer, method, shapley: Shapley = None):
    num_atoms = get_num_atoms(query.query_type)
    if method == "score":
        result_ref = shapley.execution(query, coalition=[1]*num_atoms)
        importance = {a: float(result_ref.loc[hard_answer][f"scores_{a}"]) for a in range(num_atoms)}
        if query.query_type == "2u":
            atom = max(importance, key=importance.get)
        elif query.query_type == "up":
            max_idx = max((0, 1), key=lambda k: importance[k])
            atom = min((max_idx, 2), key=lambda k: importance[k])
        else:
            atom = min(importance, key=importance.get)
        return atom, importance
    elif method == "shapley":
        shapley_vals = shapley.shapley_values(query, filtered_exclude, hard_answer)
        atom = max(shapley_vals, key=shapley_vals.get)
        return atom, shapley_vals
    elif method == "random":
        selected_atom = random.randint(0, num_atoms - 1)
        scores = [0] * num_atoms
        scores[selected_atom] = 1
        return selected_atom, scores
    elif method == "first":
        selected_atom = random.choice(get_first(query.query_type))
        scores = [0] * num_atoms
        scores[selected_atom] = 1
        return selected_atom, scores
    elif method == "last":
        selected_atom = random.choice(get_last(query.query_type))
        scores = [0] * num_atoms
        scores[selected_atom] = 1
        return selected_atom, scores
    else:
        raise ValueError(f"Unknown method: {method}")


def evaluation(hard: list,
               complete: list,
               num_atoms: int,
               xcqa: "XCQA",
               k: int,
               t_norm: str,
               t_conorm: str,
               method: str = "shapley",
               records_necc: list = None,
               records_suff: list = None):
    
    flip_value = {'necessary': 0, 'sufficient': 1}
    keep_value = {'necessary': 1, 'sufficient': 0}
    query_type = hard[0].query_type

    base_coalition_necc = [keep_value['necessary']] * num_atoms
    base_coalition_suff = [keep_value['sufficient']] * num_atoms
    metrics_original_necc = get_metric_dict()
    metrics_new_necc = get_metric_dict()
    metrics_original_suff = get_metric_dict()
    metrics_new_suff = get_metric_dict()
    satisfied_per_query = []
    
    shapley = Shapley(xcqa, qoi='rank', k=k, t_norm=t_norm, t_conorm=t_conorm)

    for i, (query_hard, query_complete) in enumerate(tqdm(zip(hard, complete), total=len(hard), desc="Queries", position=0)):
        
            # shapley cache must be reset for each query
            shapley.reset_execution_cache()
            
            all_answers = set(query_complete.get_answer())
            result_original_necc = shapley.execution(query_hard, coalition=base_coalition_necc)
            if method == 'score':
                result_original_suff = shapley.execution(query_hard, coalition=base_coalition_suff) 
            result_original_suff = shapley.execution(query_hard, coalition=base_coalition_suff)
            
            current_orig_necc = get_metric_dict()
            current_new_necc = get_metric_dict()
            current_orig_suff = get_metric_dict()
            current_new_suff = get_metric_dict()
            satisfied_count = 0
            

            # Inner progress bar for hard answers
            for hard_answer in tqdm(query_hard.answer, desc=f"Query {i} Answers", position=1, leave=False):
                global filtered_exclude
                filtered_exclude = all_answers - {hard_answer}
                
                # necessary explanation
                start_time_necc = time.time()
                mrr_necc, hit_1_necc, hit_3_necc, hit_10_necc = compute_metrics(result_original_necc, query_complete.answer, hard_answer)

                # satisfied = (hit_1 == 1.0 and mrr == 1.0) if mode == "necessary" else (hit_1 != 1.0 and mrr != 1.0)
                # if not satisfied:
                #    continue
                satisfied_count += 1
                append_metrics(current_orig_necc, (mrr_necc, hit_1_necc, hit_3_necc, hit_10_necc))
                
                chosen_atom, values = choose_atom(query_hard, hard_answer, method, shapley=shapley)
                new_coalition_necc = base_coalition_necc.copy()
                new_coalition_necc[chosen_atom] = flip_value['necessary']
                
                result_new_necc = shapley.execution(query_hard, coalition=new_coalition_necc)
                mrr_new_necc, hit_1_new_necc, hit_3_new_necc, hit_10_new_necc = compute_metrics(result_new_necc, query_complete.answer, hard_answer)
                runtime_necc = time.time() - start_time_necc
                append_metrics(current_new_necc, (mrr_new_necc, hit_1_new_necc, hit_3_new_necc, hit_10_new_necc))

                if records_necc is not None:
                    records_necc.append({
                        "query_type": query_type,
                        "query_idx": i,
                        "target_idx": hard_answer,
                        "values": values,
                        "best_atom": chosen_atom,
                        "best_value": values[chosen_atom],
                        "runtime": runtime_necc,
                        "delta_mrr": mrr_new_necc - mrr_necc,
                        "delta_hit_1": hit_1_new_necc - hit_1_necc,
                        "delta_hit_3": hit_3_new_necc - hit_3_necc,
                        "delta_hit_10": hit_10_new_necc - hit_10_necc,
                        "mrr_before": mrr_necc,
                        "mrr_after": mrr_new_necc,
                    })
                
                # sufficient explanation
                start_time_suff = time.time()
                mrr_suff, hit_1_suff, hit_3_suff, hit_10_suff = compute_metrics(result_original_suff, query_complete.answer, hard_answer)
                append_metrics(current_orig_suff, (mrr_suff, hit_1_suff, hit_3_suff, hit_10_suff))
                new_coalition_suff = base_coalition_suff.copy()
                new_coalition_suff[chosen_atom] = flip_value['sufficient']
                result_new_suff = shapley.execution(query_hard, coalition=new_coalition_suff)
                mrr_new_suff, hit_1_new_suff, hit_3_new_suff, hit_10_new_suff = compute_metrics(result_new_suff, query_complete.answer, hard_answer)
                runtime_suff = time.time() - start_time_suff
                append_metrics(current_new_suff, (mrr_new_suff, hit_1_new_suff, hit_3_new_suff, hit_10_new_suff))
                
                if records_suff is not None:
                    records_suff.append({
                        "query_type": query_type,
                        "query_idx": i,
                        "target_idx": hard_answer,
                        "values": values,
                        "best_atom": chosen_atom,
                        "best_value": values[chosen_atom],
                        "runtime": runtime_suff,
                        "delta_mrr": mrr_new_suff - mrr_suff,
                        "delta_hit_1": hit_1_new_suff - hit_1_suff,
                        "delta_hit_3": hit_3_new_suff - hit_3_suff,
                        "delta_hit_10": hit_10_new_suff - hit_10_suff,
                        "mrr_before": mrr_suff,
                        "mrr_after": mrr_new_suff,
                    })
                    
            if satisfied_count != 0:
                satisfied_per_query.append(satisfied_count)
            if current_orig_necc["mrr"]:
                avg_o = average_metrics(current_orig_necc)
                avg_n = average_metrics(current_new_necc)
                for key in ["mrr", "hit_1", "hit_3", "hit_10"]:
                    metrics_original_necc[key].append(avg_o[key])
                    metrics_new_necc[key].append(avg_n[key])
            if current_orig_suff["mrr"]:
                avg_o = average_metrics(current_orig_suff)
                avg_n = average_metrics(current_new_suff)
                for key in ["mrr", "hit_1", "hit_3", "hit_10"]:
                    metrics_original_suff[key].append(avg_o[key])
                    metrics_new_suff[key].append(avg_n[key])
    print("NECESSARY EXPLANATIONS")
    print("=" * 80)
    report_metrics(query_type, metrics_original_necc, metrics_new_necc, satisfied_per_query)
    print("=" * 80)
    print("SUFFICIENT EXPLANATIONS")
    print("=" * 80)
    report_metrics(query_type, metrics_original_suff, metrics_new_suff, satisfied_per_query)
    print("=" * 80)
    return metrics_original_necc, metrics_new_necc, metrics_original_suff, metrics_new_suff, satisfied_per_query


# ============================================================
# Main entry point
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate necessary/sufficient explanations')
    parser.add_argument('--kg', choices=['Freebase', 'NELL'], default='Freebase')
    parser.add_argument('--benchmark', choices=[1, 2], type=int, default=2)
    parser.add_argument('--query_type', choices=['2p', '3p', '4p', '2i', '3i', '4i', '2u', 'pi', 'up', 'ip'])
    parser.add_argument('--data_dir')
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--t_norm', default='prod')
    parser.add_argument('--t_conorm', default='prod')
    parser.add_argument('--split', default='test', choices=['train', 'valid', 'test'])
    parser.add_argument('--model_path')
    parser.add_argument('--method', default="shapley", choices=['shapley', 'score', 'random', 'first', 'last'])
    # parser.add_argument('--explanation', default="necessary", choices=['necessary', 'sufficient'])
    parser.add_argument('--output_path', default='evaluation')
    parser.add_argument('--log_file')
    parser.add_argument('--normalize', action='store_true', help='Whether to normalize scores using sigmoid')
    args = parser.parse_args()
    output_path = args.output_path + "_benchmark" + str(args.benchmark)
    output_path = os.path.join(output_path, args.kg)
    os.makedirs(output_path, exist_ok=True)
    if args.benchmark == 1 and args.query_type in ['4i', '4p']:
        raise ValueError("Query types '4i' and '4p' are not supported in Benchmark 1.")
    if not args.log_file:
        if args.query_type:
            log_path = os.path.join(output_path, f"bench{args.benchmark}_{args.query_type}_{args.method}.log")
        else:
            log_path = os.path.join(output_path, f"bench{args.benchmark}_all_{args.method}.log")
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

    logging.info(f"Evaluating explanations using {args.method} on {args.kg} KG")
    logging.info(f"Model path: {args.model_path}")

    dataset, graph_train, graph_valid, graph_test = setup_dataset_and_graphs(args.data_dir, logging=True, add_reverse=(args.benchmark==1))
    logging.info("Dataset and Graphs are set up.")
    query_dataset, query_dataset_hard = load_all_queries(dataset, args.data_dir, "test", version=args.benchmark)
    logging.info("Queries are loaded.")
    all_relations = list(dataset.id2rel.keys())
    logging.info(f"There are {len(all_relations)} relations in the dataset.")

    reasoner = SymbolicReasoning(graph_valid if args.split == "test" else graph_train, logging=False)
    xcqa = XCQA(symbolic=reasoner, dataset=dataset, logging=False, model_path=args.model_path, normalize=args.normalize)
    logging.info("XCQA model is initialized (normalize={})".format(args.normalize))

    if args.query_type:
        query_types = [args.query_type]
    else:
        if args.benchmark == 1:
            query_types =  ['2u', '2i', '3i', 'up', '2p', 'pi', 'ip', '3p']
        else:
            query_types =  ['2u', '2i', '3i', '4i', 'up', '2p', 'pi', 'ip', '3p', '4p']
    records_necc = []
    records_suff = []
    for query_type in query_types:
        logging.info(f"Processing query type: {query_type}")
        num_atoms = get_num_atoms(query_type)
        hard = query_dataset_hard.get_queries(query_type)
        complete = query_dataset.get_queries(query_type)

        metrics_original_necc, metrics_new_necc, metrics_original_suff, metrics_new_suff, satisfied_per_query = evaluation(
            hard, complete, num_atoms, xcqa, args.k, args.t_norm, args.t_conorm, args.method, records_necc, records_suff
        )

        logging.info(f"Finished processing query type: {query_type}")
        logging.info("-" * 80)


    
    if args.query_type:
        output_file = os.path.join(output_path, f"bench{args.benchmark}_{args.query_type}_{args.method}_necessary.csv")
    else:
        output_file = os.path.join(output_path, f"bench{args.benchmark}_all_{args.method}_necessary.csv")
    df_necc = pd.DataFrame(records_necc)
    csv_path = Path(output_file)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_necc.to_csv(csv_path, index=False)
    logging.info(f"✅ Saved all intermediate steps to {csv_path}")
    
    if args.query_type:
        output_file = os.path.join(output_path, f"bench{args.benchmark}_{args.query_type}_{args.method}_sufficient.csv")
    else:
        output_file = os.path.join(output_path, f"bench{args.benchmark}_all_{args.method}_sufficient.csv")
    df_suff = pd.DataFrame(records_suff)
    csv_path = Path(output_file)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_suff.to_csv(csv_path, index=False)
    logging.info(f"✅ Saved all intermediate steps to {csv_path}")