from query import Query
from xcqa_torch import XCQA
import pandas as pd
from graph import Dataset
import math
from utils import get_num_atoms

import numpy as np
np.random.seed(42)

def value_function(xcqa: XCQA, query: Query, filtered_nodes: list, target_entity: int, qoi: str = 'rank', k: int = 10,
                   coalition: list = None, t_norm: str = 'prod', t_conorm: str = 'min'):
    result = xcqa.query_execution(query, k=k, coalition=coalition, t_norm=t_norm, t_conorm=t_conorm)
    empty_coalition = xcqa.query_execution(query, k=k, coalition=[0]*len(coalition), t_norm=t_norm, t_conorm=t_conorm)
    empty_coalition = empty_coalition[~empty_coalition.index.isin(filtered_nodes)]
    empty_rank = empty_coalition.index.get_loc(target_entity) + 1

    # remove filtered nodes from the result
    result = result[~result.index.isin(filtered_nodes)]
    if qoi == 'rank':
        if target_entity in result.index:
            value = result.index.get_loc(target_entity) + 1  # ranks are 1-based
        else:
            raise ValueError(f"Target entity {target_entity} not found in the result")
    elif qoi == 'hit1':
        value = 1 if target_entity in result.index[:1] else 0
    elif qoi == 'hit3':
        value = 1 if target_entity in result.index[:3] else 0
    elif qoi == 'hit10':
        value = 1 if target_entity in result.index[:10] else 0
    else:
        raise ValueError(f"Unsupported QoI: {qoi}. Supported values are 'rank', 'hit1', 'hit3', 'hit10'.")
    value = - value + empty_rank  # we will add the empty coalition rank
    return value
    

def shapley_value(xcqa: XCQA, query: Query, atom_idx: int, filtered_nodes: list,
                  target_entity: int, qoi: str = 'rank', k: int = 10, t_norm: str = 'prod', t_conorm: str = 'prod'):
    num_atoms = get_num_atoms(query.query_type)

    shapley_value = 0.0
    
    num_remaining_atoms = num_atoms - 1
    for i in range(2**num_remaining_atoms):
        coalition = [int(x) for x in bin(i)[2:].zfill(num_remaining_atoms)]
        # create the coalition in the format of a list of 0s and 1s
        coalition_mask = [0] * num_atoms
        counter = 0
        for idx, _ in enumerate(coalition_mask):
            if idx== atom_idx:
                coalition_mask[idx] = 0
            else:
                coalition_mask[idx] = coalition[counter]
                counter += 1
        if xcqa.logging:
            print(f"Coalition: {coalition_mask}, Atom Index: {atom_idx}")

        # calculate the weight term (|S|! (p-|S|-1)! \ p!
        weight = (math.factorial(sum(coalition)) * math.factorial(num_atoms - sum(coalition) - 1)) / math.factorial(num_atoms)
        
        # calculate the value function for the current coalition
        value = value_function(xcqa, query, filtered_nodes, target_entity, qoi=qoi, k=k, coalition=coalition_mask, t_norm=t_norm, t_conorm=t_conorm)
        
        # calculate the contribution of the current coalition when the atom is added
        added_coalition_mask = coalition_mask.copy()
        added_coalition_mask[atom_idx] = 1
        added_value = value_function(xcqa, query, filtered_nodes, target_entity, qoi=qoi, k=k, coalition=added_coalition_mask, t_norm=t_norm, t_conorm=t_conorm)
        
        # compute the difference
        contribution = added_value - value
        if xcqa.logging:
            print(f"Coalition: {coalition_mask}, Contribution: {contribution} (before adding atom: {value}, after adding atom: {added_value}), weight: {weight})")
            
        # add the contribution to the shapley value
        shapley_value += contribution * weight
        
    if xcqa.logging:
        print(f"Shapley value for atom {atom_idx}: {shapley_value}")
    return shapley_value

def shapley_values(xcqa: XCQA, query: Query, filtered_nodes: list, target_entity: int, qoi: str = 'rank', k: int = 10,
                   t_norm: str = 'prod', t_conorm: str = 'prod'):
    num_atoms = get_num_atoms(query.query_type)
    shapley_values = []
    for atom_idx in range(num_atoms):
        sv = shapley_value(xcqa, query, atom_idx, filtered_nodes, target_entity, qoi=qoi, k=k, t_norm=t_norm, t_conorm=t_conorm)
        shapley_values.append(sv)
    return shapley_values