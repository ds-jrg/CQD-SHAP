import time
import pandas as pd
import torch
from query import Query
from symbolic_torch import SymbolicReasoning
from graph import Dataset
import numpy as np
from cqd_link_prediction import LinkPrediction

np.random.seed(42)
fillna_value = 0.01

class XCQA:
    def __init__(self, symbolic: SymbolicReasoning, dataset: Dataset, logging: bool = True, model_path: str = "models/FB15k-237-model-rank-1000-epoch-100-1602508358.pt"):

        self.symbolic = symbolic
        self.dataset = dataset
        self.logging = logging
        self.link_prediction = LinkPrediction(model_path=model_path)

    def get_num_atoms(query_type):
        """Get the number of atoms for a given query type."""
        atom_mapping = {
            '2p': 2, '3p': 3, '2i': 2, '2u': 2, 
            '3i': 3, 'pi': 3, 'up': 3, 'ip': 3
        }
        if query_type not in atom_mapping:
            raise ValueError(f"Unsupported query type: {query_type}.")
        return atom_mapping[query_type]
    
    def atom_predict(self, anchor: int, relation: int, mask: int, k: int = -1):
        if mask == 1:
            return self.link_prediction.predict(h_id=anchor, r_id=relation, return_df=False, k=k)
        else:
            return self.symbolic.predict(h_id=anchor, rel_id=relation, return_df=False, k=k)

    def atom_batch_predict(self, anchors: list, relations: list, mask: int, k: int = -1):
        if mask == 1:
            return self.link_prediction.predict_batch(anchors, relations, k=k)
        else:
            return self.symbolic.predict_batch(anchors, relations, k=k)

    def query_2p(self, h_id, r_ids, coalition, k=5):
        """
        lp_model: LinkPrediction object
        h_id: starting entity (int) -> e.g. 10645
        r_ids: list of relation ids for each hop -> e.g. [135, 94]
        k: beam size (top-k expansion at each layer) -> e.g. 10
        """

        # device = lp_model.device
        # num_entities = lp_model.model.sizes[2]  # depends on model definition

        scores_first = self.atom_predict(anchor=h_id, relation=r_ids[0], mask=coalition[0], k=-1) # [1, num_entities]
        scores_first = scores_first.squeeze(0)  # [num_entities]

        topk_scores, topk_indices = torch.topk(scores_first, k)   # [k]
        print(topk_scores)
        print(topk_indices)
        scores_second = self.atom_batch_predict(topk_indices, r_ids[1], mask=coalition[1], k=-1) # [k, num_entities]

        combined = scores_second * topk_scores.unsqueeze(1) # [k, num_entities]

        max_scores, max_parent = combined.max(dim=0) # [num_entities]
        print("max_scores:", max_scores)
        print("max_parent:", max_parent)
        col_idx = torch.arange(scores_second.size(1))  # [num_entities]
        df = pd.DataFrame({
                'scores_0': topk_scores[max_parent].detach().cpu().numpy(),
                'scores_1': scores_second[max_parent, col_idx].detach().cpu().numpy(),
                'variable_0': topk_indices[max_parent].detach().cpu().numpy(),
                'final_score': max_scores.detach().cpu().numpy()
        })
        df = df.sort_values(by="final_score", ascending=False)
        df['answer'] = df.index
        df = df.reset_index(drop=True)
        return df

    def query_3p(self, h_id, r_ids, coalition, k=5):
        """
        3p path query: h --r1--> ? --r2--> ? --r3--> t
        lp_model: LinkPrediction object
        h_id: starting entity (int)
        r_ids: list of 3 relation ids [r1, r2, r3]
        k: beam size for expansions
        """
        # Step 1: First hop
        scores_first = self.atom_predict(anchor=h_id, relation=r_ids[0], mask=coalition[0], k=k) # [1, num_entities]
        topk1_scores, topk1_indices = torch.topk(scores_first, k)   # [k]

        # Step 2: Second hop - expand all topk1
        scores_second = self.atom_batch_predict(topk1_indices, r_ids[1], mask=coalition[1], k=k) # [k, num_entities]
        combined2 = scores_second * topk1_scores.unsqueeze(1) # Broadcast multiplication
        max_scores2, max_parent2 = combined2.max(dim=0) # Aggregate best over k
        topk2_scores, topk2_indices = torch.topk(max_scores2, k)

        # Step 3: Third hop - expand topk2 results
        scores_third = self.atom_batch_predict(topk2_indices, r_ids[2], mask=coalition[2], k=k) # [k, num_entities]
        combined3 = scores_third * topk2_scores.unsqueeze(1) # [k, num_entities]
        max_scores3, max_parent3 = combined3.max(dim=0)

        # Index bookkeeping
        col_idx = torch.arange(scores_third.size(1))
        df = pd.DataFrame({
            'scores_0': topk1_scores[max_parent2[max_parent3]].detach().cpu().numpy(),
            'scores_1': scores_second[max_parent2[max_parent3], topk2_indices[max_parent3]].detach().cpu().numpy(),
            'scores_2': scores_third[max_parent3, col_idx].detach().cpu().numpy(),
            'variable_0': topk1_indices[max_parent2[max_parent3]].detach().cpu().numpy(),
            'variable_1': topk2_indices[max_parent3].detach().cpu().numpy(),
            'final_score': max_scores3.detach().cpu().numpy()
        })
        df = df.sort_values(by="final_score", ascending=False)
        df['answer'] = df.index
        df = df.reset_index(drop=True)
        return df

    def query_execution(self, query: Query, k: int = 10, coalition: list = None, 
                    t_norm: str = 'prod', t_conorm: str = 'min'):

        # Set default coalition if not provided (default settings would be CQD execution)
        if coalition is None:
            coalition = [1] * get_num_atoms(query.query_type)

        if query.query_type == '2p':
            anchor = query.get_query()[0][0]
            relations = query.get_query()[0][1]
            return self.query_2p(anchor, relations, coalition, k=k)

        elif query.query_type == '3p':
            anchor = query.get_query()[0][0]
            relations = query.get_query()[0][1] 
            return self.query_3p(anchor, relations, coalition, k=k)