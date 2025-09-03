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
    
    def query_3p(self, h_id, r_ids, coalition, k=5, t_norm: str = "prod"):
        """ 
        3p query: (h, r1, VAR1) AND (VAR1, r2, VAR2) AND (VAR2, r3, ?)
        Three-hop projection with intermediate k expansions.
        """
        # First hop
        scores_first = self.atom_predict(anchor=h_id, relation=r_ids[0], mask=coalition[0], k=-1)
        scores_first = scores_first.squeeze(0)  # [num_entities]
        
        # Get device for tensor operations
        device = scores_first.device
        
        # Top-k expansion for first hop
        topk_scores_first, topk_indices_first = torch.topk(scores_first, k)  # [k]
        
        # Second hop - batch predict for all k nodes from first hop
        scores_second = self.atom_batch_predict(
            topk_indices_first, r_ids[1], mask=coalition[1], k=-1
        )  # [k, num_entities]
        
        # Find top-k for each of the k nodes from first hop
        topk_scores_second, topk_indices_second = torch.topk(scores_second, k, dim=1)  # [k, k]
        
        # Combine first and second hop scores
        if t_norm == "prod":
            combined_second = topk_scores_second * topk_scores_first.unsqueeze(1)  # [k, k]
        elif t_norm == "min":
            combined_second = torch.min(topk_scores_second, topk_scores_first.unsqueeze(1))
        elif t_norm == "max":
            combined_second = torch.max(topk_scores_second, topk_scores_first.unsqueeze(1))
        else:
            raise ValueError(f"Unsupported t_norm: {t_norm}")
        
        # Flatten for third hop processing
        # We now have k^2 intermediate nodes
        flattened_scores = combined_second.view(-1)  # [k*k]
        flattened_indices = topk_indices_second.view(-1)  # [k*k]
        
        # Create parent tracking for intermediate results - ensure on same device
        parent_first = torch.arange(k, device=device).unsqueeze(1).expand(k, k).contiguous().view(-1)  # [k*k]
        parent_second = torch.arange(k, device=device).unsqueeze(0).expand(k, k).contiguous().view(-1)  # [k*k]
        
        # Third hop - batch predict for all k^2 nodes
        scores_third = self.atom_batch_predict(
            flattened_indices, r_ids[2], mask=coalition[2], k=-1
        )  # [k*k, num_entities]
        
        # Combine with intermediate scores
        if t_norm == "prod":
            combined_final = scores_third * flattened_scores.unsqueeze(1)  # [k*k, num_entities]
        elif t_norm == "min":
            combined_final = torch.min(scores_third, flattened_scores.unsqueeze(1))
        elif t_norm == "max":
            combined_final = torch.max(scores_third, flattened_scores.unsqueeze(1))
        else:
            raise ValueError(f"Unsupported t_norm: {t_norm}")
        
        # Existential aggregation across all intermediate paths
        max_scores, max_path_idx = combined_final.max(dim=0)  # [num_entities]
        
        # Create detailed results DataFrame - ensure col_idx is on same device
        col_idx = torch.arange(scores_third.size(1), device=device)
        
        # Get the best path information for each entity
        best_parent_first = parent_first[max_path_idx]
        best_parent_second = parent_second[max_path_idx] 
        best_intermediate_first = topk_indices_first[best_parent_first]
        best_intermediate_second = flattened_indices[max_path_idx]
        
        df = pd.DataFrame({
            'scores_0': topk_scores_first[best_parent_first].detach().cpu().numpy(),
            'scores_1': topk_scores_second[best_parent_first, best_parent_second].detach().cpu().numpy(),
            'scores_2': scores_third[max_path_idx, col_idx].detach().cpu().numpy(),
            'variable_0': best_intermediate_first.detach().cpu().numpy(),
            'variable_1': best_intermediate_second.detach().cpu().numpy(),
            'final_score': max_scores.detach().cpu().numpy()
        })
        
        df = df.sort_values(by="final_score", ascending=False)
        return df

    # 2u1p
    def query_up(self, h_ids, r_ids, coalition, k=5, 
               t_conorm: str = "prod", t_norm: str = "prod"):
        """
        2u1p query:
        First: (h1, r1, ?) OR (h2, r2, ?)
        Then: (VAR, r3, ?)
        Keep top-k from union, then project over relation r3.
        """

        assert len(h_ids) == 2 and len(r_ids) == 3 and len(coalition) == 3, \
            "Expected 2 anchors, 3 relations, and 3 coalition values."

        # --- First two projections (union operands) ---
        scores_0 = self.atom_predict(anchor=h_ids[0], relation=r_ids[0], 
                                    mask=coalition[0], k=-1).squeeze(0)
        scores_1 = self.atom_predict(anchor=h_ids[1], relation=r_ids[1], 
                                    mask=coalition[1], k=-1).squeeze(0)

        # Apply union t-conorm
        if t_conorm == "prod":
            combined_union = scores_0 + scores_1 - (scores_0 * scores_1)
        elif t_conorm == "min":
            combined_union = torch.min(scores_0, scores_1)
        elif t_conorm == "max":
            combined_union = torch.max(scores_0, scores_1)
        else:
            raise ValueError(f"Unsupported t_conorm: {t_conorm}")

        # --- Top-k union results ---
        topk_scores, topk_indices = torch.topk(combined_union, k)  # [k]

        # --- Projection step with r3 over these candidates ---
        scores_proj = self.atom_batch_predict(
            topk_indices, r_ids[2], mask=coalition[2], k=-1
        )  # [k, num_entities]

        # Combine union score with projection
        if t_norm == "prod":
            combined_final = scores_proj * topk_scores.unsqueeze(1)
        elif t_norm == "min":
            combined_final = torch.min(scores_proj, topk_scores.unsqueeze(1))
        elif t_norm == "max":
            combined_final = torch.max(scores_proj, topk_scores.unsqueeze(1))
        else:
            raise ValueError(f"Unsupported t_norm: {t_norm}")

        # Existential aggregation (best across top-k paths)
        max_scores, max_parent = combined_final.max(dim=0)  # [num_entities]

        # Column index for gathering
        col_idx = torch.arange(scores_proj.size(1))

        # Best chosen parent variable for each final entity
        best_intermediate = topk_indices[max_parent]

        # Now we also need to get scores_0 and scores_1 **for that intermediate**
        scores_0_best = scores_0[best_intermediate]
        scores_1_best = scores_1[best_intermediate]
        scores_2_best = scores_proj[max_parent, col_idx]

        df = pd.DataFrame({
            'scores_0': scores_0_best.detach().cpu().numpy(),
            'scores_1': scores_1_best.detach().cpu().numpy(),
            'scores_2': scores_2_best.detach().cpu().numpy(),
            'variable_0': best_intermediate.detach().cpu().numpy(),
            'final_score': max_scores.detach().cpu().numpy()
        })

        df = df.sort_values(by="final_score", ascending=False)
        return df
    
    # 2i1p
    def query_ip(self, h_ids, r_ids, coalition, k=5, t_norm: str = "prod"):
        """
        2i1p query:
        First: (h1, r1, ?) AND (h2, r2, ?)
        Then: (VAR, r3, ?)
        Keep top-k from intersection, then project over relation r3.
        """

        assert len(h_ids) == 2 and len(r_ids) == 3 and len(coalition) == 3, \
            "Expected 2 anchors, 3 relations, and 3 coalition values."

        # --- Two projections (the AND part) ---
        scores_0 = self.atom_predict(anchor=h_ids[0], relation=r_ids[0],
                                    mask=coalition[0], k=-1).squeeze(0)
        scores_1 = self.atom_predict(anchor=h_ids[1], relation=r_ids[1],
                                    mask=coalition[1], k=-1).squeeze(0)

        # Apply intersection t-norm
        if t_norm == "prod":
            combined_intersection = scores_0 * scores_1
        elif t_norm == "min":
            combined_intersection = torch.min(scores_0, scores_1)
        elif t_norm == "max":
            combined_intersection = torch.max(scores_0, scores_1)
        else:
            raise ValueError(f"Unsupported t_norm: {t_norm}")

        # --- Top-k from intersection ---
        topk_scores, topk_indices = torch.topk(combined_intersection, k)  # [k]

        # --- Projection with r3 ---
        scores_proj = self.atom_batch_predict(
            topk_indices, r_ids[2], mask=coalition[2], k=-1
        )  # [k, num_entities]

        # Combine with intersection scores
        if t_norm == "prod":
            combined_final = scores_proj * topk_scores.unsqueeze(1)
        elif t_norm == "min":
            combined_final = torch.min(scores_proj, topk_scores.unsqueeze(1))
        elif t_norm == "max":
            combined_final = torch.max(scores_proj, topk_scores.unsqueeze(1))
        else:
            raise ValueError(f"Unsupported t_norm: {t_norm}")

        # Existential aggregation
        max_scores, max_parent = combined_final.max(dim=0)  # [num_entities]

        # Column index for gathering
        col_idx = torch.arange(scores_proj.size(1))

        # Best chosen parent variable for each final entity
        best_intermediate = topk_indices[max_parent]

        # Now we also need to get scores_0 and scores_1 **for that intermediate**
        scores_0_best = scores_0[best_intermediate]
        scores_1_best = scores_1[best_intermediate]
        scores_2_best = scores_proj[max_parent, col_idx]

        df = pd.DataFrame({
            'scores_0': scores_0_best.detach().cpu().numpy(),
            'scores_1': scores_1_best.detach().cpu().numpy(),
            'scores_2': scores_2_best.detach().cpu().numpy(),
            'variable_0': best_intermediate.detach().cpu().numpy(),
            'final_score': max_scores.detach().cpu().numpy()
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
        
        elif query.query_type == '3p':
            anchor = query.get_query()[0][0]
            relations = query.get_query()[0][1]
            return self.query_3p(anchor, relations, coalition, k=k, t_norm=t_norm)
        
        elif query.query_type == 'up':
            query1 = query.query[0]
            anchor1 = query1[0]
            relation1 = query1[1][0]
            query2 = query.query[1]
            anchor2 = query2[0]
            relation2 = query2[1][0]
            relation3 = query.query[2]
            anchors = [anchor1, anchor2]
            relations = [relation1, relation2, relation3]
            return self.query_up(anchors, relations, coalition, k=k, t_norm=t_norm)

        elif query.query_type == 'ip':
            query1 = query.get_query()[0]
            anchor1 = query1[0]
            relation1 = query1[1][0]
            query2 = query.get_query()[1]
            anchor2 = query2[0]
            relation2 = query2[1][0]
            relation3 = query.get_query()[2]
            anchors = [anchor1, anchor2]
            relations = [relation1, relation2, relation3]
            return self.query_ip(anchors, relations, coalition, k=k, t_norm=t_norm)