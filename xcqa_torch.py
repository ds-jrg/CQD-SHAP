import time
import pandas as pd
import torch
from code.LitCQDrepo.LitCQD.models.discrete import query_3p
from query import Query
from symbolic import SymbolicReasoning
from graph import Dataset
import numpy as np
from cqd_link_prediction import LinkPrediction

np.random.seed(42)
fillna_value = 0.01

class XCQA:
    def __init__(self, symbolic: SymbolicReasoning, dataset: Dataset, inner_cache: dict = None,  logging: bool = True):

        self.symbolic = symbolic
        self.dataset = dataset
        self.inner_cache = inner_cache if inner_cache is not None else {'cqd': {}, 'symbolic': {}}
        self.logging = logging
        self.link_prediction = LinkPrediction(model_path="models/FB15k-237-model-rank-1000-epoch-100-1602508358.pt")

    def atom_execution(self, anchor: int, relation: int, mask: int, k: int = 10):
        """
        Executes a single atom query, either using CQD or symbolic reasoning based on the mask.
        """
        if mask == 1:  # CQD
            return self.link_prediction.predict(h_id = anchor, r_id = relation, return_df=True, k=k)
        else: # Symbolic Reasoning
            # print(anchor, relation)
            if self.inner_cache is not None and (anchor, relation) in self.inner_cache['symbolic']:
                result = self.inner_cache['symbolic'][(anchor, relation)]
                if k > 0:
                    result = result.head(k)
            else:
                result = self.symbolic.query_1p(anchor, relation)
                result = self.symbolic.fixed_size_answer(result, len(self.dataset.id2node))
                if self.inner_cache is not None:
                    self.inner_cache['symbolic'][(anchor, relation)] = result
                if k > 0:
                    result = result.head(k)
        return result.copy()

    import torch

    def query_2p(lp_model, h_id, r_ids, k=5):
        """
        lp_model: LinkPrediction object
        h_id: starting entity (int) -> e.g. 10645
        r_ids: list of relation ids for each hop -> e.g. [135, 94]
        k: beam size (top-k expansion at each layer) -> e.g. 10
        """

        # device = lp_model.device
        # num_entities = lp_model.model.sizes[2]  # depends on model definition

        scores_first = lp_model.predict(h_id, r_ids[0], return_df=False)  # [1, num_entities]
        scores_first = scores_first.squeeze(0)  # [num_entities]

        topk_scores, topk_indices = torch.topk(scores_first, k)   # [k]

        scores_second = lp_model.predict_batch(topk_indices, r_ids[1], return_df=False) # [k, num_entities]

        combined = scores_second * topk_scores.unsqueeze(1) # [k, num_entities]

        max_scores, max_parent = combined.max(dim=0) # [num_entities]

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

    def query_3p(lp_model, h_id, r_ids, k=5):
        """
        3p path query: h --r1--> ? --r2--> ? --r3--> t
        lp_model: LinkPrediction object
        h_id: starting entity (int)
        r_ids: list of 3 relation ids [r1, r2, r3]
        k: beam size for expansions
        """
        # Step 1: First hop
        scores_first = lp_model.predict(h_id, r_ids[0], return_df=False).squeeze(0)  # [num_entities]
        topk1_scores, topk1_indices = torch.topk(scores_first, k)   # [k]

        # Step 2: Second hop - expand all topk1
        scores_second = lp_model.predict_batch(topk1_indices, r_ids[1], return_df=False) # [k, num_entities]
        combined2 = scores_second * topk1_scores.unsqueeze(1) # Broadcast multiplication
        max_scores2, max_parent2 = combined2.max(dim=0) # Aggregate best over k
        topk2_scores, topk2_indices = torch.topk(max_scores2, k)

        # Step 3: Third hop - expand topk2 results
        scores_third = lp_model.predict_batch(topk2_indices, r_ids[2], return_df=False) # [k, num_entities]
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
        if query.query_type == '2p':
            anchor = query.get_query()[0][0]
            relations = query.get_query()[0][1]
            return self.link_prediction.query_2p(self.link_prediction, anchor, relations, k=k)

        elif query.query_type == '3p':
            anchor = query.get_query()[0][0]
            relations = query.get_query()[0][1] 
            return query_3p(self.link_prediction, anchor, relations, k=k)