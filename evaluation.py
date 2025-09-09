from symbolic_torch import SymbolicReasoning
from xcqa_torch import XCQA
from utils import get_num_atoms, get_query_file_paths, setup_dataset_and_graphs, load_query_datasets, setup_logger
import argparse
from tqdm import tqdm
from shapley import shapley_value
import random
import json
from statistics import mean
from tqdm import tqdm
from pathlib import Path

# set random seed
random.seed(42)

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

def get_metric_dict():
    return {
        "mrr": [],
        "hit_1": [],
        "hit_3": [],
        "hit_10": []
    }

from collections import defaultdict
from statistics import mean
from tqdm import tqdm


def get_metric_dict():
    """Return dict with empty lists for each metric."""
    return {k: [] for k in ["mrr", "hit_1", "hit_3", "hit_10"]}


def average_metrics(metrics: dict) -> dict:
    """Convert lists of metric values into their mean values."""
    return {k: mean(v) if v else 0.0 for k, v in metrics.items()}


def append_metrics(metrics: dict, values: tuple):
    """Append metrics (mrr, hit_1, hit_3, hit_10) into dict of lists."""
    keys = ["mrr", "hit_1", "hit_3", "hit_10"]
    for key, value in zip(keys, values):
        metrics[key].append(value)


def report_metrics(metrics_cqd: dict, metrics_new: dict):
    """Print average values before and after removing an atom."""
    keys = ["mrr", "hit_1", "hit_3", "hit_10"]
    print("\n=== Reduced Metrics Report ===")
    for key in keys:
        avg_before = mean(metrics_cqd[key]) if metrics_cqd[key] else 0.0
        avg_after = mean(metrics_new[key]) if metrics_new[key] else 0.0
        avg_diff = avg_after - avg_before

        print(f"{key.upper()}:")
        print(f"  Before:     {avg_before:.4f}")
        print(f"  After:      {avg_after:.4f}")
        print(f"  Difference: {avg_diff:+.4f}\n")

def get_metric_dict():
    """Return dict with empty lists for each metric."""
    return {k: [] for k in ["mrr", "hit_1", "hit_3", "hit_10"]}


def average_metrics(metrics: dict) -> dict:
    """Convert lists of metric values into their mean values."""
    return {k: mean(v) if v else 0.0 for k, v in metrics.items()}


def append_metrics(metrics: dict, values: tuple):
    """Append metrics (mrr, hit_1, hit_3, hit_10) into dict of lists."""
    keys = ["mrr", "hit_1", "hit_3", "hit_10"]
    for key, value in zip(keys, values):
        metrics[key].append(value)


def report_metrics(metrics_original: dict, metrics_new: dict, satisfied_per_query: list):
    """Print summary report for original vs new metrics + satisfied counts per query."""
    keys = ["mrr", "hit_1", "hit_3", "hit_10"]

    print("\n=== Reduced Metrics Report ===")
    for key in keys:
        values_before = metrics_original[key]
        values_after = metrics_new[key]

        avg_before = mean(values_before) if values_before else 0.0
        avg_after = mean(values_after) if values_after else 0.0
        avg_diff = avg_after - avg_before

        total_items = len(values_before)  # same as len(values_after)

        print(f"{key.upper()}:")
        print(f"  Before Avg: {avg_before:.4f}")
        print(f"  After Avg:  {avg_after:.4f}")
        print(f"  Difference: {avg_diff:+.4f}")

    # Report satisfied condition statistics (per query)
    avg_satisfied = mean(satisfied_per_query) if satisfied_per_query else 0.0
    print("=== Condition Satisfaction  ===")
    # print(f"  Per-query counts: {satisfied_per_query}")
    print(f"  Average count per query: {avg_satisfied:.4f}")
    print(f"  Total count (satisfied query-answer pairs): {sum(satisfied_per_query)}")
    print(f"  Total count (queries): {total_items}\n")


def necessary(hard: list, complete: list, num_atoms: int, xcqa: "XCQA", k: int,
              t_norm: str, t_conorm: str, output_path: str, method: str = 'shapley'):
    """
    Run evaluation and save metrics + satisfied counts to output_path.
    """

    original_coalition = [1] * num_atoms

    metrics_original = get_metric_dict()
    metrics_new = get_metric_dict()

    satisfied_per_query = []   # store per-query satisfied counts
    raw_metrics_per_query = [] # store raw pre-averaged metrics for saving

    for i in tqdm(range(len(hard))):
        query_hard = hard[i]
        query_complete = complete[i]
        all_answers = set(query_complete.get_answer())

        result_original = xcqa.query_execution(
            query_hard, k=k, coalition=original_coalition,
            t_norm=t_norm, t_conorm=t_conorm
        )

        current_metrics_original = get_metric_dict()
        current_metrics_new = get_metric_dict()

        satisfied_count = 0  # <- per query counter

        for hard_answer in query_hard.answer:
            mrr, hit_1, hit_3, hit_10 = compute_metrics(
                result_original, query_complete.answer, hard_answer
            )

            if hit_1 == 1.0 and mrr == 1.0:
                # all the other answers (easy or hard) except the current hard answer
                filtered_exclude = all_answers - {hard_answer}
                
                satisfied_count += 1

                # Collect metrics for current query
                append_metrics(current_metrics_original, (mrr, hit_1, hit_3, hit_10))
                
                # Zero out (symbolic execution) the atom with the lowest link prediction score
                if method == "score":

                    importance = {
                        atom: float(result_original.loc[hard_answer][f"scores_{atom}"])
                        for atom in range(num_atoms)
                    }
                    highest_atom = min(importance, key=importance.get)

                    new_coalition = original_coalition.copy()
                    new_coalition[highest_atom] = 0   # zero out most important atom (symbolic execution)
                # Zero out (symbolic execution) the atom with the lowest shapley value
                elif method == "shapley":
                    shapley = {}
                    for atom in range(num_atoms):
                        shapley[atom] = shapley_value(xcqa, query_hard, atom, filtered_exclude, hard_answer, "rank", k, t_norm, t_conorm)
                    lowest_atom = min(shapley, key=shapley.get)
                    new_coalition = original_coalition.copy()
                    new_coalition[lowest_atom] = 0
                # Randomly zero out (symbolic execution) one atom
                elif method == "random":
                    random_atom = random.randint(0, num_atoms - 1)
                    new_coalition = original_coalition.copy()
                    new_coalition[random_atom] = 0
                # Always zero out (symbolic execution) first atom
                elif method == "first":
                    new_coalition = original_coalition.copy()
                    first_atoms = get_first(query_hard.query_type)
                    # select a random atom from first atoms
                    random_first_atom = random.choice(first_atoms)
                    new_coalition[random_first_atom] = 0   # zero out most important atom (symbolic execution)
                # Always zero out (symbolic execution) last atom
                elif method == "last":
                    new_coalition = original_coalition.copy()
                    last_atoms = get_last(query_hard.query_type)
                    # select a random atom from last atoms
                    random_last_atom = random.choice(last_atoms)
                    new_coalition[random_last_atom] = 0   # zero out last projection atom (symbolic execution)

                result_new = xcqa.query_execution(
                    query_hard, k=k, coalition=new_coalition,
                    t_norm=t_norm, t_conorm=t_conorm
                )
                    
                mrr_new, hit_1_new, hit_3_new, hit_10_new = compute_metrics(
                    result_new, query_complete.answer, hard_answer
                )
                append_metrics(current_metrics_new,
                               (mrr_new, hit_1_new, hit_3_new, hit_10_new))

        # store per-query satisfied count
        satisfied_per_query.append(satisfied_count)

        # store raw (not averaged) metrics for later analysis
        raw_metrics_per_query.append({
            "original": current_metrics_original,
            "new": current_metrics_new,
            "satisfied_count": satisfied_count
        })

        # If we collected any data, average and store
        if current_metrics_original["mrr"]:
            averaged_original = average_metrics(current_metrics_original)
            averaged_new = average_metrics(current_metrics_new)

            for key in ["mrr", "hit_1", "hit_3", "hit_10"]:
                metrics_original[key].append(averaged_original[key])
                metrics_new[key].append(averaged_new[key])

    # --- final reporting ---
    report_metrics(metrics_original, metrics_new, satisfied_per_query)

    # --- save raw data to file ---
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    save_content = {
        "metrics_original": metrics_original,
        "metrics_new": metrics_new,
        "raw_metrics_per_query": raw_metrics_per_query,
        "satisfied_per_query": satisfied_per_query,
        "average_satisfied_per_query": mean(satisfied_per_query) if satisfied_per_query else 0.0,
    }

    with open(path, "w") as f:
        json.dump(save_content, f, indent=2)

    return metrics_original, metrics_new, satisfied_per_query

def sufficient(hard: list, complete: list, num_atoms: int, xcqa: "XCQA", k: int,
              t_norm: str, t_conorm: str, output_path: str, method: str = 'shapley'):
    """
    Run evaluation and save metrics + satisfied counts to output_path.
    """

    original_coalition = [0] * num_atoms

    metrics_original = get_metric_dict()
    metrics_new = get_metric_dict()

    satisfied_per_query = []   # store per-query satisfied counts
    raw_metrics_per_query = [] # store raw pre-averaged metrics for saving

    for i in tqdm(range(len(hard))):
        query_hard = hard[i]
        query_complete = complete[i]
        all_answers = set(query_complete.get_answer())

        result_original = xcqa.query_execution(
            query_hard, k=k, coalition=original_coalition,
            t_norm=t_norm, t_conorm=t_conorm
        )

        current_metrics_original = get_metric_dict()
        current_metrics_new = get_metric_dict()

        satisfied_count = 0  # <- per query counter

        for hard_answer in query_hard.answer:

            filtered_exclude = all_answers - {hard_answer}
            mrr, hit_1, hit_3, hit_10 = compute_metrics(
                result_original, query_complete.answer, hard_answer
            )

            if hit_1 != 1.0 and mrr != 1.0:
                satisfied_count += 1

                # Collect metrics for current query
                append_metrics(current_metrics_original, (mrr, hit_1, hit_3, hit_10))
                
                # Zero out (symbolic execution) the atom with the lowest link prediction score
                if method == "score":
                    cqd_coalition = [1] * num_atoms
                    result_cqd = xcqa.query_execution(
                        query_hard, k=k, coalition=cqd_coalition,
                        t_norm=t_norm, t_conorm=t_conorm
                    )

                    importance = {
                        atom: float(result_cqd.loc[hard_answer][f"scores_{atom}"])
                        for atom in range(num_atoms)
                    }
                    highest_atom = min(importance, key=importance.get)

                    new_coalition = original_coalition.copy()
                    new_coalition[highest_atom] = 1

                    result_new = xcqa.query_execution(
                        query_hard, k=k, coalition=new_coalition,
                        t_norm=t_norm, t_conorm=t_conorm
                    )
                # Zero out (symbolic execution) the atom with the lowest shapley value
                elif method == "shapley":
                    shapley = {}
                    for atom in range(num_atoms):
                        shapley[atom] = shapley_value(xcqa, query_hard, atom, filtered_exclude, hard_answer, "rank", k, t_norm, t_conorm)
                    lowest_atom = min(shapley, key=shapley.get)
                    new_coalition = original_coalition.copy()
                    new_coalition[lowest_atom] = 1
                # Randomly zero out (symbolic execution) one atom
                elif method == "random":
                    random_atom = random.randint(0, num_atoms - 1)
                    new_coalition = original_coalition.copy()
                    new_coalition[random_atom] = 1
                    
                    result_new = xcqa.query_execution(
                        query_hard, k=k, coalition=new_coalition,
                        t_norm=t_norm, t_conorm=t_conorm
                    )
                # Always zero out (symbolic execution) first atom
                elif method == "first":
                    new_coalition = original_coalition.copy()
                    first_atoms = get_first(query_hard.query_type)
                    # select a random atom from first atoms
                    random_first_atom = random.choice(first_atoms)
                    new_coalition[random_first_atom] = 1

                    result_new = xcqa.query_execution(
                        query_hard, k=k, coalition=new_coalition,
                        t_norm=t_norm, t_conorm=t_conorm
                    )
                # Always zero out (symbolic execution) last atom
                elif method == "last":
                    new_coalition = original_coalition.copy()
                    last_atoms = get_last(query_hard.query_type)
                    # select a random atom from last atoms
                    random_last_atom = random.choice(last_atoms)
                    new_coalition[random_last_atom] = 1

                result_new = xcqa.query_execution(
                    query_hard, k=k, coalition=new_coalition,
                    t_norm=t_norm, t_conorm=t_conorm
                )
                    
                mrr_new, hit_1_new, hit_3_new, hit_10_new = compute_metrics(
                    result_new, query_complete.answer, hard_answer
                )
                append_metrics(current_metrics_new,
                               (mrr_new, hit_1_new, hit_3_new, hit_10_new))

        # store per-query satisfied count
        satisfied_per_query.append(satisfied_count)

        # store raw (not averaged) metrics for later analysis
        raw_metrics_per_query.append({
            "original": current_metrics_original,
            "new": current_metrics_new,
            "satisfied_count": satisfied_count
        })

        # If we collected any data, average and store
        if current_metrics_original["mrr"]:
            averaged_original = average_metrics(current_metrics_original)
            averaged_new = average_metrics(current_metrics_new)

            for key in ["mrr", "hit_1", "hit_3", "hit_10"]:
                metrics_original[key].append(averaged_original[key])
                metrics_new[key].append(averaged_new[key])

    # --- final reporting ---
    report_metrics(metrics_original, metrics_new, satisfied_per_query)

    # --- save raw data to file ---
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    save_content = {
        "metrics_original": metrics_original,
        "metrics_new": metrics_new,
        "raw_metrics_per_query": raw_metrics_per_query,
        "satisfied_per_query": satisfied_per_query,
        "average_satisfied_per_query": mean(satisfied_per_query) if satisfied_per_query else 0.0,
    }

    with open(path, "w") as f:
        json.dump(save_content, f, indent=2)

    return metrics_original, metrics_new, satisfied_per_query

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Calculate necessary/sufficient explanations')
    parser.add_argument('query_type', choices=['2p', '3p', '2i', '2u', '3i', 'pi', 'up', 'ip'], help='Query type to process')
    parser.add_argument('--data_dir', default='data/FB15k-237', help='Directory containing the dataset (default: data/FB15k-237)')
    parser.add_argument('--k', type=int, default=10, help='Top-k parameter (default: 10)')
    parser.add_argument('--t_norm', default='prod', help='T-norm parameter (default: prod)')
    parser.add_argument('--t_conorm', default='prod', help='T-conorm parameter (default: prod)')
    parser.add_argument('--split', default='test', choices=['train', 'valid', 'test'], help='Dataset split to use (default: test)')
    parser.add_argument('--model_path', default='models/FB15k-237-model-rank-1000-epoch-100-1602508358.pt', help='Path to the model file (default: models/FB15k-237-model-rank-1000-epoch-100-1602508358.pt)')
    parser.add_argument('--method', default="shapley", choices=['shapley', 'score', 'random', 'first', 'last'], help='Method to select atom to zero out (default: shapley)')
    parser.add_argument('--explanation', default="necessary", choices=['necessary', 'sufficient'], help='Explanation type (default: necessary)')
    parser.add_argument('--output_path', default='output.json', help='Path to the output file (default: output.json)')
    args = parser.parse_args()
    
    dataset, graph_train, graph_valid = setup_dataset_and_graphs(args.data_dir)
    
    if args.split == "test":
        reasoner = SymbolicReasoning(graph_valid, logging=False)
    else:
        reasoner = SymbolicReasoning(graph_train, logging=False)
    
    query_dataset, query_dataset_hard = load_query_datasets(dataset, args.data_dir, args.query_type, args.split)
    
    xcqa = XCQA(symbolic=reasoner, dataset=dataset, logging=False, model_path=args.model_path)
    
    num_atoms = get_num_atoms(args.query_type)

    hard = query_dataset_hard.get_queries(args.query_type)
    complete = query_dataset.get_queries(args.query_type)
    
    if args.explanation == "necessary":
        metrics_original, metrics_new, satisfied_per_query = necessary(hard, complete, num_atoms, xcqa, args.k, args.t_norm, args.t_conorm, args.output_path, args.method)
    elif args.explanation == "sufficient":
        metrics_original, metrics_new, satisfied_per_query = sufficient(hard, complete, num_atoms, xcqa, args.k, args.t_norm, args.t_conorm, args.output_path, args.method)
    else:
        raise ValueError(f"Unsupported explanation type: {args.explanation}")