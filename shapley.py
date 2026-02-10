from query import Query
from xcqa_torch import XCQA
import pandas as pd
from graph import Dataset
import math
from utils import get_num_atoms

import numpy as np
np.random.seed(42)

class Shapley:
    def __init__(self, xcqa: XCQA, qoi: str = 'rank', k: int = 10, t_norm: str = 'prod', t_conorm: str = 'prod', execution_cache: dict = {}):
        self.xcqa = xcqa
        self.qoi = qoi
        self.k = k
        self.t_norm = t_norm
        self.t_conorm = t_conorm
        self.execution_cache = execution_cache
        
    def reset_execution_cache(self):
        self.execution_cache = {}
        
    def set_execution_cache(self, execution_cache: dict):
        self.execution_cache = execution_cache
        
    def execution(self, query: Query, coalition: list = None):
        coalition_key = tuple(coalition)
        if self.execution_cache.get(coalition_key) is not None:
            result = self.execution_cache[coalition_key]
        else:
            result = self.xcqa.query_execution(query, k=self.k, coalition=coalition, t_norm=self.t_norm, t_conorm=self.t_conorm)
            self.execution_cache[coalition_key] = result
        return result
    
    def value_function(self, query: Query, filtered_nodes: list, target_entity: int, coalition: list = None):
        result = self.execution(query, coalition=coalition)
        empty_coalition = [0]*len(coalition)
        empty_coalition = self.execution(query, coalition=empty_coalition)
        empty_coalition = empty_coalition[~empty_coalition.index.isin(filtered_nodes)]
        empty_rank = empty_coalition.index.get_loc(target_entity) + 1

        # remove filtered nodes from the result
        result = result[~result.index.isin(filtered_nodes)]
        if self.qoi == 'rank':
            if target_entity in result.index:
                value = result.index.get_loc(target_entity) + 1  # ranks are 1-based
            else:
                raise ValueError(f"Target entity {target_entity} not found in the result")
        elif self.qoi == 'hit1':
            value = 1 if target_entity in result.index[:1] else 0
        elif self.qoi == 'hit3':
            value = 1 if target_entity in result.index[:3] else 0
        elif self.qoi == 'hit10':
            value = 1 if target_entity in result.index[:10] else 0
        else:
            raise ValueError(f"Unsupported QoI: {self.qoi}. Supported values are 'rank', 'hit1', 'hit3', 'hit10'.")
        value = - value + empty_rank  # we will add the empty coalition rank
        return value
        

    def shapley_value(self, query: Query, atom_idx: int, filtered_nodes: list, target_entity: int):
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
            if self.xcqa.logging:
                print(f"Coalition: {coalition_mask}, Atom Index: {atom_idx}")

            # calculate the weight term (|S|! (p-|S|-1)! \ p!
            weight = (math.factorial(sum(coalition)) * math.factorial(num_atoms - sum(coalition) - 1)) / math.factorial(num_atoms)
            
            # calculate the value function for the current coalition
            value = self.value_function(query, filtered_nodes, target_entity, coalition=coalition_mask)
            
            # calculate the contribution of the current coalition when the atom is added
            added_coalition_mask = coalition_mask.copy()
            added_coalition_mask[atom_idx] = 1
            added_value = self.value_function(query, filtered_nodes, target_entity, coalition=added_coalition_mask)
            
            # compute the difference
            contribution = added_value - value
            if self.xcqa.logging:
                print(f"Coalition: {coalition_mask}, Contribution: {contribution} (before adding atom: {value}, after adding atom: {added_value}), weight: {weight})")
                
            # add the contribution to the shapley value
            shapley_value += contribution * weight
            
        if self.xcqa.logging:
            print(f"Shapley value for atom {atom_idx}: {shapley_value}")
        return shapley_value

    def shapley_values(self, query: Query, filtered_nodes: list, target_entity: int):
        num_atoms = get_num_atoms(query.query_type)
        shapley_values = {}
        for atom_idx in range(num_atoms):
            sv = self.shapley_value(query, atom_idx, filtered_nodes, target_entity)
            shapley_values[atom_idx] = sv
        return shapley_values