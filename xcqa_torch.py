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

    def query_2p(self, h_id, r_ids, coalition, k=5, t_norm: str = "prod"):
        """
        2p query: (h, r1, VAR) AND (VAR, r2, ?)
        """
        # First hop
        scores_first = self.atom_predict(anchor=h_id, relation=r_ids[0], mask=coalition[0], k=-1)
        scores_first = scores_first.squeeze(0)  # [num_entities]

        # Top-k expansion
        topk_scores, topk_indices = torch.topk(scores_first, k)   # [k]
        scores_second = self.atom_batch_predict(
            topk_indices, r_ids[1], mask=coalition[1], k=-1
        )  # [k, num_entities]

        # Combine depending on t_norm
        if t_norm == "prod":
            combined = scores_second * topk_scores.unsqueeze(1)
        elif t_norm == "min":
            combined = torch.min(scores_second, topk_scores.unsqueeze(1))
        elif t_norm == "max":
            combined = torch.max(scores_second, topk_scores.unsqueeze(1))
        else:
            raise ValueError(f"Unsupported t_norm: {t_norm}")

        # Existential aggregation across intermediate vars
        max_scores, max_parent = combined.max(dim=0)

        col_idx = torch.arange(scores_second.size(1))
        df = pd.DataFrame({
                'scores_0': topk_scores[max_parent].detach().cpu().numpy(),
                'scores_1': scores_second[max_parent, col_idx].detach().cpu().numpy(),
                'variable_0': topk_indices[max_parent].detach().cpu().numpy(),
                'final_score': max_scores.detach().cpu().numpy()
        })

        df = df.sort_values(by="final_score", ascending=False)
        return df


    def query_2i(self, h_ids, r_ids, coalition, k=5, t_norm: str = "prod"):
        """
        2i query: (h1, r1, ?) AND (h2, r2, ?)
        Intersection via t_norm (prod/min/max).
        """
        assert len(h_ids) == 2 and len(r_ids) == 2 and len(coalition) == 2

        scores_0 = self.atom_predict(anchor=h_ids[0], relation=r_ids[0], mask=coalition[0], k=-1)
        scores_1 = self.atom_predict(anchor=h_ids[1], relation=r_ids[1], mask=coalition[1], k=-1)

        scores_0 = scores_0.squeeze(0)
        scores_1 = scores_1.squeeze(0)

        # Apply intersection operator depending on t_norm
        if t_norm == "prod":
            combined = scores_0 * scores_1
        elif t_norm == "min":
            combined = torch.min(scores_0, scores_1)
        elif t_norm == "max":
            combined = torch.max(scores_0, scores_1)
        else:
            raise ValueError(f"Unsupported t_norm: {t_norm}")

        df = pd.DataFrame({
            'scores_0': scores_0.detach().cpu().numpy(),
            'scores_1': scores_1.detach().cpu().numpy(),
            'final_score': combined.detach().cpu().numpy()
        })

        df = df.sort_values(by="final_score", ascending=False)
        return df
    
    def query_2u(self, h_ids, r_ids, coalition, k=5, t_conorm: str = "prod"):
        """
        2u query: (h1, r1, ?) OR (h2, r2, ?)
        Union via t_conorm (prod/min/max).
        """
        assert len(h_ids) == 2 and len(r_ids) == 2 and len(coalition) == 2

        scores_0 = self.atom_predict(anchor=h_ids[0], relation=r_ids[0], mask=coalition[0], k=-1)
        scores_1 = self.atom_predict(anchor=h_ids[1], relation=r_ids[1], mask=coalition[1], k=-1)

        scores_0 = scores_0.squeeze(0)
        scores_1 = scores_1.squeeze(0)

        # Apply union operator depending on t_conorm
        if t_conorm == "prod":
            combined = scores_0 + scores_1 - (scores_0 * scores_1)
        elif t_conorm == "min":
            combined = torch.min(scores_0, scores_1)
        elif t_conorm == "max":
            combined = torch.max(scores_0, scores_1)
        else:
            raise ValueError(f"Unsupported t_conorm: {t_conorm}")

        df = pd.DataFrame({
            'scores_0': scores_0.detach().cpu().numpy(),
            'scores_1': scores_1.detach().cpu().numpy(),
            'final_score': combined.detach().cpu().numpy()
        })

        df = df.sort_values(by="final_score", ascending=False)
        return df
    
    def query_3i(self, h_ids, r_ids, coalition, k=5, t_norm: str = "prod"):
        """
        3i query: (h1, r1, ?) AND (h2, r2, ?) AND (h3, r3, ?)
        Intersection of 3 projections via t_norm (prod/min/max).
        """
        assert len(h_ids) == 3 and len(r_ids) == 3 and len(coalition) == 3

        # Get branch scores
        scores_0 = self.atom_predict(anchor=h_ids[0], relation=r_ids[0], mask=coalition[0], k=-1).squeeze(0)
        scores_1 = self.atom_predict(anchor=h_ids[1], relation=r_ids[1], mask=coalition[1], k=-1).squeeze(0)
        scores_2 = self.atom_predict(anchor=h_ids[2], relation=r_ids[2], mask=coalition[2], k=-1).squeeze(0)

        # Intersection operator
        if t_norm == "prod":
            combined = scores_0 * scores_1 * scores_2
        elif t_norm == "min":
            combined = torch.min(torch.min(scores_0, scores_1), scores_2)
        elif t_norm == "max":
            combined = torch.max(torch.max(scores_0, scores_1), scores_2)
        else:
            raise ValueError(f"Unsupported t_norm: {t_norm}")

        # Build dataframe
        df = pd.DataFrame({
            'scores_0': scores_0.detach().cpu().numpy(),
            'scores_1': scores_1.detach().cpu().numpy(),
            'scores_2': scores_2.detach().cpu().numpy(),
            'final_score': combined.detach().cpu().numpy()
        })

        df = df.sort_values(by="final_score", ascending=False)
        return df

    def query_execution(self, query: Query, k: int = 10, coalition: list = None, 
                        t_norm: str = 'prod', t_conorm: str = 'max'):
        if coalition is None:
            coalition = [1] * self.get_num_atoms(query.query_type)

        if query.query_type == '2p':
            anchor = query.get_query()[0][0]
            relations = query.get_query()[0][1]
            return self.query_2p(anchor, relations, coalition, k=k, t_norm=t_norm)

        elif query.query_type == '2i':
            anchors = [q[0] for q in query.get_query()]
            relations = [q[1][0] for q in query.get_query()]
            return self.query_2i(anchors, relations, coalition, k=k, t_norm=t_norm)

        elif query.query_type == '2u':
            anchors = [q[0] for q in query.get_query()]
            relations = [q[1][0] for q in query.get_query()]
            return self.query_2u(anchors, relations, coalition, k=k, t_conorm=t_conorm)

        elif query.query_type == '3i':
            anchors = [q[0] for q in query.get_query()]
            relations = [q[1][0] for q in query.get_query()]
            return self.query_3i(anchors, relations, coalition, k=k, t_norm=t_norm)