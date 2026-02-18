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
    def __init__(self, symbolic: SymbolicReasoning, dataset: Dataset, logging: bool = True,
                 model_path: str = "models/FB15k-237-model-rank-1000-epoch-100-1602508358.pt",
                 normalize: bool = False, use_topk: bool = True):
        """
        Args:
            ...
            use_topk: If True, intermediate variable expansion keeps exactly the top-k
                      candidates at each hop. If False, keeps all candidates whose score
                      is strictly greater than fillna_value (threshold-based expansion).
        """
        self.symbolic = symbolic
        self.dataset = dataset
        self.logging = logging
        self.link_prediction = LinkPrediction(model_path=model_path, normalize=normalize)
        self.use_topk = use_topk

    # ------------------------------------------------------------------
    # Internal helper: select intermediate candidates from a 1-D score tensor
    # ------------------------------------------------------------------
    def _select_intermediate(self, scores_1d: torch.Tensor, k: int):
        """
        Returns (selected_scores, selected_indices) for an intermediate variable.
        
        If self.use_topk is True  → top-k by score.
        If self.use_topk is False → all entries with score > fillna_value.
                                    Falls back to top-1 if nothing passes the threshold.
        """
        if self.use_topk:
            return torch.topk(scores_1d, k)
        else:
            mask = scores_1d > fillna_value
            if mask.any():
                indices = mask.nonzero(as_tuple=True)[0]
                return scores_1d[indices], indices
            else:
                # Fallback: at least keep the single best candidate
                best_score, best_idx = scores_1d.max(dim=0)
                return best_score.unsqueeze(0), best_idx.unsqueeze(0)

    # ------------------------------------------------------------------
    # Internal helper: select intermediate candidates from a 2-D score matrix
    # (one row per upstream candidate, columns = entities)
    # Returns topk_scores [rows, k_out], topk_indices [rows, k_out] when use_topk,
    # but a *ragged* structure when not — so callers handle both cases.
    # ------------------------------------------------------------------
    def _select_intermediate_2d(self, scores_2d: torch.Tensor, k: int):
        """
        If use_topk: standard torch.topk over dim=1 → (scores [R,k], indices [R,k]).
        If not use_topk: for every row, keep all cols > fillna_value; pad/pack into
                         a uniform tensor by taking the union of qualifying columns
                         (i.e. any column that qualifies in *at least one* row).
                         Returns (scores [R, C'], indices [C']) where C' = qualifying cols.
        """
        if self.use_topk:
            return torch.topk(scores_2d, k, dim=1)
        else:
            col_mask = (scores_2d > fillna_value).any(dim=0)   # [num_entities]
            if not col_mask.any():
                # Fallback: best column per row → union of those
                col_mask = torch.zeros(scores_2d.size(1), dtype=torch.bool, device=scores_2d.device)
                col_mask[scores_2d.max(dim=0).indices] = True

            indices = col_mask.nonzero(as_tuple=True)[0]       # [C']
            selected_scores = scores_2d[:, indices]             # [R, C']
            return selected_scores, indices

    def get_num_atoms(query_type):
        """Get the number of atoms for a given query type."""
        atom_mapping = {
            '2p': 2, '3p': 3, '2i': 2, '2u': 2,
            '3i': 3, 'pi': 3, 'up': 3, 'ip': 3
        }
        if query_type not in atom_mapping:
            raise ValueError(f"Unsupported query type: {query_type}.")
        return atom_mapping[query_type]

    def atom_predict(self, anchor: int, relation: int, mask: int, k: int = -1, disj: bool = False):
        if mask == 1:
            return self.link_prediction.predict(h_id=anchor, r_id=relation, return_df=False, k=k, score_normalize=disj)
        else:
            return self.symbolic.predict(h_id=anchor, rel_id=relation, return_df=False, k=k)

    def atom_batch_predict(self, anchors: list, relations: list, mask: int, k: int = -1, disj: bool = False):
        if mask == 1:
            return self.link_prediction.predict_batch(anchors, relations, k=k, score_normalize=disj)
        else:
            return self.symbolic.predict_batch(anchors, relations, k=k)

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def query_2p(self, h_id, r_ids, coalition, k=5, t_norm: str = "prod"):
        """
        2p query: (h, r1, VAR) AND (VAR, r2, ?)
        """
        # First hop
        scores_first = self.atom_predict(anchor=h_id, relation=r_ids[0], mask=coalition[0], k=-1)
        scores_first = scores_first.squeeze(0)  # [num_entities]

        # Intermediate selection (top-k or threshold)
        topk_scores, topk_indices = self._select_intermediate(scores_first, k)

        scores_second = self.atom_batch_predict(
            topk_indices, r_ids[1], mask=coalition[1], k=-1
        )  # [k', num_entities]

        if t_norm == "prod":
            combined = scores_second * topk_scores.unsqueeze(1)
        elif t_norm == "min":
            combined = torch.min(scores_second, topk_scores.unsqueeze(1))
        elif t_norm == "max":
            combined = torch.max(scores_second, topk_scores.unsqueeze(1))
        else:
            raise ValueError(f"Unsupported t_norm: {t_norm}")

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
            'variable_0': np.arange(scores_0.size(0)),
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

        scores_0 = self.atom_predict(anchor=h_ids[0], relation=r_ids[0], mask=coalition[0], k=-1, disj=True)
        scores_1 = self.atom_predict(anchor=h_ids[1], relation=r_ids[1], mask=coalition[1], k=-1, disj=True)

        scores_0 = scores_0.squeeze(0)
        scores_1 = scores_1.squeeze(0)

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
            'variable_0': np.arange(scores_0.size(0)),
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

        scores_0 = self.atom_predict(anchor=h_ids[0], relation=r_ids[0], mask=coalition[0], k=-1).squeeze(0)
        scores_1 = self.atom_predict(anchor=h_ids[1], relation=r_ids[1], mask=coalition[1], k=-1).squeeze(0)
        scores_2 = self.atom_predict(anchor=h_ids[2], relation=r_ids[2], mask=coalition[2], k=-1).squeeze(0)

        if t_norm == "prod":
            combined = scores_0 * scores_1 * scores_2
        elif t_norm == "min":
            combined = torch.min(torch.min(scores_0, scores_1), scores_2)
        elif t_norm == "max":
            combined = torch.max(torch.max(scores_0, scores_1), scores_2)
        else:
            raise ValueError(f"Unsupported t_norm: {t_norm}")

        df = pd.DataFrame({
            'scores_0': scores_0.detach().cpu().numpy(),
            'scores_1': scores_1.detach().cpu().numpy(),
            'scores_2': scores_2.detach().cpu().numpy(),
            'variable_0': np.arange(scores_0.size(0)),
            'final_score': combined.detach().cpu().numpy()
        })

        df = df.sort_values(by="final_score", ascending=False)
        return df

    def query_4i(self, h_ids, r_ids, coalition, k=5, t_norm: str = "prod"):
        """
        4i query: (h1, r1, ?) AND (h2, r2, ?) AND (h3, r3, ?) AND (h4, r4, ?)
        Intersection of 4 projections via t_norm (prod/min/max).
        """
        assert len(h_ids) == 4 and len(r_ids) == 4 and len(coalition) == 4

        scores_0 = self.atom_predict(anchor=h_ids[0], relation=r_ids[0], mask=coalition[0], k=-1).squeeze(0)
        scores_1 = self.atom_predict(anchor=h_ids[1], relation=r_ids[1], mask=coalition[1], k=-1).squeeze(0)
        scores_2 = self.atom_predict(anchor=h_ids[2], relation=r_ids[2], mask=coalition[2], k=-1).squeeze(0)
        scores_3 = self.atom_predict(anchor=h_ids[3], relation=r_ids[3], mask=coalition[3], k=-1).squeeze(0)

        if t_norm == "prod":
            combined = scores_0 * scores_1 * scores_2 * scores_3
        elif t_norm == "min":
            combined = torch.min(torch.min(torch.min(scores_0, scores_1), scores_2), scores_3)
        elif t_norm == "max":
            combined = torch.max(torch.max(torch.max(scores_0, scores_1), scores_2), scores_3)
        else:
            raise ValueError(f"Unsupported t_norm: {t_norm}")

        df = pd.DataFrame({
            'scores_0': scores_0.detach().cpu().numpy(),
            'scores_1': scores_1.detach().cpu().numpy(),
            'scores_2': scores_2.detach().cpu().numpy(),
            'scores_3': scores_3.detach().cpu().numpy(),
            'final_score': combined.detach().cpu().numpy()
        })

        df = df.sort_values(by="final_score", ascending=False)
        return df

    def query_3p(self, h_id, r_ids, coalition, k=5, t_norm: str = "prod"):
        """
        3p query: (h, r1, VAR1) AND (VAR1, r2, VAR2) AND (VAR2, r3, ?)
        Three-hop projection with intermediate expansions (top-k or threshold).
        """
        # First hop
        scores_first = self.atom_predict(anchor=h_id, relation=r_ids[0], mask=coalition[0], k=-1)
        scores_first = scores_first.squeeze(0)  # [num_entities]

        device = scores_first.device

        # Select intermediate VAR1 candidates
        topk_scores_first, topk_indices_first = self._select_intermediate(scores_first, k)
        k1 = topk_indices_first.size(0)  # actual number of candidates (≥1)

        # Second hop — [k1, num_entities]
        scores_second = self.atom_batch_predict(
            topk_indices_first, r_ids[1], mask=coalition[1], k=-1
        )

        # Select intermediate VAR2 candidates (per-row union for threshold mode)
        topk_scores_second, topk_indices_second = self._select_intermediate_2d(scores_second, k)
        # topk_scores_second: [k1, k2]   topk_indices_second: [k2]  (threshold)
        #                     [k1, k]    [k1, k]                     (top-k)
        use_topk = self.use_topk

        if use_topk:
            k2 = topk_scores_second.size(1)  # == k

            if t_norm == "prod":
                combined_second = topk_scores_second * topk_scores_first.unsqueeze(1)
            elif t_norm == "min":
                combined_second = torch.min(topk_scores_second, topk_scores_first.unsqueeze(1))
            elif t_norm == "max":
                combined_second = torch.max(topk_scores_second, topk_scores_first.unsqueeze(1))
            else:
                raise ValueError(f"Unsupported t_norm: {t_norm}")

            # Flatten [k1, k2] → [k1*k2]
            flattened_scores  = combined_second.view(-1)
            flattened_indices = topk_indices_second.view(-1)

            parent_first  = torch.arange(k1, device=device).unsqueeze(1).expand(k1, k2).contiguous().view(-1)
            parent_second = torch.arange(k2, device=device).unsqueeze(0).expand(k1, k2).contiguous().view(-1)

        else:
            # topk_scores_second: [k1, k2]  topk_indices_second: [k2]  (shared column set)
            k2 = topk_scores_second.size(1)

            if t_norm == "prod":
                combined_second = topk_scores_second * topk_scores_first.unsqueeze(1)
            elif t_norm == "min":
                combined_second = torch.min(topk_scores_second, topk_scores_first.unsqueeze(1))
            elif t_norm == "max":
                combined_second = torch.max(topk_scores_second, topk_scores_first.unsqueeze(1))
            else:
                raise ValueError(f"Unsupported t_norm: {t_norm}")

            flattened_scores  = combined_second.view(-1)                                         # [k1*k2]
            # Expand the shared column indices to match every row
            flattened_indices = topk_indices_second.unsqueeze(0).expand(k1, k2).contiguous().view(-1)  # [k1*k2]

            parent_first  = torch.arange(k1, device=device).unsqueeze(1).expand(k1, k2).contiguous().view(-1)
            parent_second = torch.arange(k2, device=device).unsqueeze(0).expand(k1, k2).contiguous().view(-1)

        # Third hop — [k1*k2, num_entities]
        scores_third = self.atom_batch_predict(
            flattened_indices, r_ids[2], mask=coalition[2], k=-1
        )

        if t_norm == "prod":
            combined_final = scores_third * flattened_scores.unsqueeze(1)
        elif t_norm == "min":
            combined_final = torch.min(scores_third, flattened_scores.unsqueeze(1))
        elif t_norm == "max":
            combined_final = torch.max(scores_third, flattened_scores.unsqueeze(1))
        else:
            raise ValueError(f"Unsupported t_norm: {t_norm}")

        max_scores, max_path_idx = combined_final.max(dim=0)

        col_idx = torch.arange(scores_third.size(1), device=device)

        best_parent_first     = parent_first[max_path_idx]
        best_parent_second    = parent_second[max_path_idx]
        best_intermediate_first  = topk_indices_first[best_parent_first]
        best_intermediate_second = flattened_indices[max_path_idx]

        if use_topk:
            score_1_vals = topk_scores_second[best_parent_first, best_parent_second]
        else:
            score_1_vals = topk_scores_second[best_parent_first, best_parent_second]

        df = pd.DataFrame({
            'scores_0': topk_scores_first[best_parent_first].detach().cpu().numpy(),
            'scores_1': score_1_vals.detach().cpu().numpy(),
            'scores_2': scores_third[max_path_idx, col_idx].detach().cpu().numpy(),
            'variable_0': best_intermediate_first.detach().cpu().numpy(),
            'variable_1': best_intermediate_second.detach().cpu().numpy(),
            'final_score': max_scores.detach().cpu().numpy()
        })

        df = df.sort_values(by="final_score", ascending=False)
        return df

    def query_4p(self, h_id, r_ids, coalition, k=5, t_norm: str = "prod"):
        """
        4p query: (h, r1, VAR1) AND (VAR1, r2, VAR2) AND (VAR2, r3, VAR3) AND (VAR3, r4, ?)
        Four-hop projection with intermediate expansions (top-k or threshold).
        """
        # First hop
        scores_first = self.atom_predict(anchor=h_id, relation=r_ids[0], mask=coalition[0], k=-1)
        scores_first = scores_first.squeeze(0)

        device = scores_first.device

        # VAR1 candidates
        topk_scores_first, topk_indices_first = self._select_intermediate(scores_first, k)
        k1 = topk_indices_first.size(0)

        # Second hop — [k1, num_entities]
        scores_second = self.atom_batch_predict(
            topk_indices_first, r_ids[1], mask=coalition[1], k=-1
        )

        # VAR2 candidates
        topk_scores_second, topk_indices_second = self._select_intermediate_2d(scores_second, k)

        use_topk = self.use_topk

        if use_topk:
            k2 = topk_scores_second.size(1)
            if t_norm == "prod":
                combined_second = topk_scores_second * topk_scores_first.unsqueeze(1)
            elif t_norm == "min":
                combined_second = torch.min(topk_scores_second, topk_scores_first.unsqueeze(1))
            elif t_norm == "max":
                combined_second = torch.max(topk_scores_second, topk_scores_first.unsqueeze(1))
            else:
                raise ValueError(f"Unsupported t_norm: {t_norm}")

            flattened_scores_2  = combined_second.view(-1)
            flattened_indices_2 = topk_indices_second.view(-1)
            parent_first_2  = torch.arange(k1, device=device).unsqueeze(1).expand(k1, k2).contiguous().view(-1)
            parent_second_2 = torch.arange(k2, device=device).unsqueeze(0).expand(k1, k2).contiguous().view(-1)
        else:
            k2 = topk_scores_second.size(1)
            if t_norm == "prod":
                combined_second = topk_scores_second * topk_scores_first.unsqueeze(1)
            elif t_norm == "min":
                combined_second = torch.min(topk_scores_second, topk_scores_first.unsqueeze(1))
            elif t_norm == "max":
                combined_second = torch.max(topk_scores_second, topk_scores_first.unsqueeze(1))
            else:
                raise ValueError(f"Unsupported t_norm: {t_norm}")

            flattened_scores_2  = combined_second.view(-1)
            flattened_indices_2 = topk_indices_second.unsqueeze(0).expand(k1, k2).contiguous().view(-1)
            parent_first_2  = torch.arange(k1, device=device).unsqueeze(1).expand(k1, k2).contiguous().view(-1)
            parent_second_2 = torch.arange(k2, device=device).unsqueeze(0).expand(k1, k2).contiguous().view(-1)

        n2 = flattened_indices_2.size(0)  # k1*k2

        # Third hop — [n2, num_entities]
        scores_third = self.atom_batch_predict(
            flattened_indices_2, r_ids[2], mask=coalition[2], k=-1
        )

        # VAR3 candidates
        topk_scores_third, topk_indices_third = self._select_intermediate_2d(scores_third, k)

        if use_topk:
            k3 = topk_scores_third.size(1)
            if t_norm == "prod":
                combined_third = topk_scores_third * flattened_scores_2.unsqueeze(1)
            elif t_norm == "min":
                combined_third = torch.min(topk_scores_third, flattened_scores_2.unsqueeze(1))
            elif t_norm == "max":
                combined_third = torch.max(topk_scores_third, flattened_scores_2.unsqueeze(1))
            else:
                raise ValueError(f"Unsupported t_norm: {t_norm}")

            flattened_scores_3  = combined_third.view(-1)
            flattened_indices_3 = topk_indices_third.view(-1)
            parent_idx_from_second = torch.arange(n2, device=device).unsqueeze(1).expand(n2, k3).contiguous().view(-1)
            parent_third_pos       = torch.arange(k3, device=device).unsqueeze(0).expand(n2, k3).contiguous().view(-1)
        else:
            k3 = topk_scores_third.size(1)
            if t_norm == "prod":
                combined_third = topk_scores_third * flattened_scores_2.unsqueeze(1)
            elif t_norm == "min":
                combined_third = torch.min(topk_scores_third, flattened_scores_2.unsqueeze(1))
            elif t_norm == "max":
                combined_third = torch.max(topk_scores_third, flattened_scores_2.unsqueeze(1))
            else:
                raise ValueError(f"Unsupported t_norm: {t_norm}")

            flattened_scores_3  = combined_third.view(-1)
            flattened_indices_3 = topk_indices_third.unsqueeze(0).expand(n2, k3).contiguous().view(-1)
            parent_idx_from_second = torch.arange(n2, device=device).unsqueeze(1).expand(n2, k3).contiguous().view(-1)
            parent_third_pos       = torch.arange(k3, device=device).unsqueeze(0).expand(n2, k3).contiguous().view(-1)

        # Fourth hop
        scores_fourth = self.atom_batch_predict(
            flattened_indices_3, r_ids[3], mask=coalition[3], k=-1
        )

        if t_norm == "prod":
            combined_final = scores_fourth * flattened_scores_3.unsqueeze(1)
        elif t_norm == "min":
            combined_final = torch.min(scores_fourth, flattened_scores_3.unsqueeze(1))
        elif t_norm == "max":
            combined_final = torch.max(scores_fourth, flattened_scores_3.unsqueeze(1))
        else:
            raise ValueError(f"Unsupported t_norm: {t_norm}")

        max_scores, max_path_idx = combined_final.max(dim=0)

        col_idx = torch.arange(scores_fourth.size(1), device=device)

        best_path_in_third_level = parent_idx_from_second[max_path_idx]
        best_path_third_pos      = parent_third_pos[max_path_idx]

        best_parent_first  = parent_first_2[best_path_in_third_level]
        best_parent_second = parent_second_2[best_path_in_third_level]

        best_intermediate_first  = topk_indices_first[best_parent_first]
        best_intermediate_second = flattened_indices_2[best_path_in_third_level]
        best_intermediate_third  = flattened_indices_3[max_path_idx]

        df = pd.DataFrame({
            'scores_0': topk_scores_first[best_parent_first].detach().cpu().numpy(),
            'scores_1': topk_scores_second[best_parent_first, best_parent_second].detach().cpu().numpy(),
            'scores_2': topk_scores_third[best_path_in_third_level, best_path_third_pos].detach().cpu().numpy(),
            'scores_3': scores_fourth[max_path_idx, col_idx].detach().cpu().numpy(),
            'variable_0': best_intermediate_first.detach().cpu().numpy(),
            'variable_1': best_intermediate_second.detach().cpu().numpy(),
            'variable_2': best_intermediate_third.detach().cpu().numpy(),
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
        Keep top-k (or threshold) from union, then project over relation r3.
        """
        assert len(h_ids) == 2 and len(r_ids) == 3 and len(coalition) == 3, \
            "Expected 2 anchors, 3 relations, and 3 coalition values."

        scores_0 = self.atom_predict(anchor=h_ids[0], relation=r_ids[0],
                                     mask=coalition[0], k=-1, disj=True).squeeze(0)
        scores_1 = self.atom_predict(anchor=h_ids[1], relation=r_ids[1],
                                     mask=coalition[1], k=-1, disj=True).squeeze(0)

        if t_conorm == "prod":
            combined_union = scores_0 + scores_1 - (scores_0 * scores_1)
        elif t_conorm == "min":
            combined_union = torch.min(scores_0, scores_1)
        elif t_conorm == "max":
            combined_union = torch.max(scores_0, scores_1)
        else:
            raise ValueError(f"Unsupported t_conorm: {t_conorm}")

        # Intermediate selection (top-k or threshold)
        topk_scores, topk_indices = self._select_intermediate(combined_union, k)

        scores_proj = self.atom_batch_predict(
            topk_indices, r_ids[2], mask=coalition[2], k=-1
        )  # [k', num_entities]

        if t_norm == "prod":
            combined_final = scores_proj * topk_scores.unsqueeze(1)
        elif t_norm == "min":
            combined_final = torch.min(scores_proj, topk_scores.unsqueeze(1))
        elif t_norm == "max":
            combined_final = torch.max(scores_proj, topk_scores.unsqueeze(1))
        else:
            raise ValueError(f"Unsupported t_norm: {t_norm}")

        max_scores, max_parent = combined_final.max(dim=0)

        col_idx = torch.arange(scores_proj.size(1))

        best_intermediate = topk_indices[max_parent]
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
        Keep top-k (or threshold) from intersection, then project over relation r3.
        """
        assert len(h_ids) == 2 and len(r_ids) == 3 and len(coalition) == 3, \
            "Expected 2 anchors, 3 relations, and 3 coalition values."

        scores_0 = self.atom_predict(anchor=h_ids[0], relation=r_ids[0],
                                     mask=coalition[0], k=-1).squeeze(0)
        scores_1 = self.atom_predict(anchor=h_ids[1], relation=r_ids[1],
                                     mask=coalition[1], k=-1).squeeze(0)

        if t_norm == "prod":
            combined_intersection = scores_0 * scores_1
        elif t_norm == "min":
            combined_intersection = torch.min(scores_0, scores_1)
        elif t_norm == "max":
            combined_intersection = torch.max(scores_0, scores_1)
        else:
            raise ValueError(f"Unsupported t_norm: {t_norm}")

        # Intermediate selection (top-k or threshold)
        topk_scores, topk_indices = self._select_intermediate(combined_intersection, k)

        scores_proj = self.atom_batch_predict(
            topk_indices, r_ids[2], mask=coalition[2], k=-1
        )  # [k', num_entities]

        if t_norm == "prod":
            combined_final = scores_proj * topk_scores.unsqueeze(1)
        elif t_norm == "min":
            combined_final = torch.min(scores_proj, topk_scores.unsqueeze(1))
        elif t_norm == "max":
            combined_final = torch.max(scores_proj, topk_scores.unsqueeze(1))
        else:
            raise ValueError(f"Unsupported t_norm: {t_norm}")

        max_scores, max_parent = combined_final.max(dim=0)

        col_idx = torch.arange(scores_proj.size(1))

        best_intermediate = topk_indices[max_parent]
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

    def query_pi(self, h_ids, r_ids, coalition, k=5, t_norm: str = "prod"):
        """
        pi / 1p2i query:
        Branch A: (h1, r1, VAR) AND (VAR, r2, ?), i.e. 2p query
        Branch B: (h2, r3, ?), i.e. single projection
        Final: intersection of Branch A and Branch B results.
        """
        assert len(h_ids) == 2 and len(r_ids) == 3 and len(coalition) == 3, \
            "Expected 2 anchors, 3 relations, 3 coalition values."

        h_A   = h_ids[0]
        rA    = [r_ids[0], r_ids[1]]
        coalA = [coalition[0], coalition[1]]

        # First hop
        scores_first = self.atom_predict(anchor=h_A, relation=rA[0], mask=coalA[0], k=-1)
        scores_first = scores_first.squeeze(0)
        device = scores_first.device

        # Intermediate selection (top-k or threshold)
        topk_scores, topk_indices = self._select_intermediate(scores_first, k)

        scores_second = self.atom_batch_predict(topk_indices, rA[1], mask=coalA[1], k=-1)

        if t_norm == "prod":
            combined = scores_second * topk_scores.unsqueeze(1)
        elif t_norm == "min":
            combined = torch.min(scores_second, topk_scores.unsqueeze(1))
        elif t_norm == "max":
            combined = torch.max(scores_second, topk_scores.unsqueeze(1))
        else:
            raise ValueError(f"Unsupported t_norm: {t_norm}")

        max_scores_A, max_parent = combined.max(dim=0)
        col_idx = torch.arange(scores_second.size(1), device=device)

        scores_0   = topk_scores[max_parent]
        scores_1   = scores_second[max_parent, col_idx]
        variable_0 = topk_indices[max_parent]
        scores_branchA = max_scores_A

        # Branch B
        h_B  = h_ids[1]
        rB   = r_ids[2]
        coalB = coalition[2]

        scores_branchB = self.atom_predict(anchor=h_B, relation=rB,
                                           mask=coalB, k=-1).squeeze(0).to(device)

        if t_norm == "prod":
            combined_final = scores_branchA * scores_branchB
        elif t_norm == "min":
            combined_final = torch.min(scores_branchA, scores_branchB)
        elif t_norm == "max":
            combined_final = torch.max(scores_branchA, scores_branchB)
        else:
            raise ValueError(f"Unsupported t_norm: {t_norm}")

        df = pd.DataFrame({
            'scores_0': scores_0.detach().cpu().numpy(),
            'scores_1': scores_1.detach().cpu().numpy(),
            'scores_2': scores_branchB.detach().cpu().numpy(),
            'variable_0': variable_0.detach().cpu().numpy(),
            'final_score': combined_final.detach().cpu().numpy()
        })

        df = df.sort_values(by="final_score", ascending=False)
        return df

    def query_execution(self, query: Query, k: int = 10, coalition: list = None,
                        t_norm: str = 'prod', t_conorm: str = 'max'):
        if coalition is None:
            coalition = [1] * self.get_num_atoms(query.query_type)

        anchors, relations = extract_anchors_relations(query)
        anchor = anchors[0]

        if query.query_type == '2p':
            return self.query_2p(anchor, relations, coalition, k=k, t_norm=t_norm)
        elif query.query_type == '2i':
            return self.query_2i(anchors, relations, coalition, k=k, t_norm=t_norm)
        elif query.query_type == '2u':
            return self.query_2u(anchors, relations, coalition, k=k, t_conorm=t_conorm)
        elif query.query_type == '3i':
            return self.query_3i(anchors, relations, coalition, k=k, t_norm=t_norm)
        elif query.query_type == '4i':
            return self.query_4i(anchors, relations, coalition, k=k, t_norm=t_norm)
        elif query.query_type == '3p':
            return self.query_3p(anchor, relations, coalition, k=k, t_norm=t_norm)
        elif query.query_type == '4p':
            return self.query_4p(anchor, relations, coalition, k=k, t_norm=t_norm)
        elif query.query_type == 'up':
            return self.query_up(anchors, relations, coalition, k=k, t_norm=t_norm)
        elif query.query_type == 'ip':
            return self.query_ip(anchors, relations, coalition, k=k, t_norm=t_norm)
        elif query.query_type == 'pi':
            return self.query_pi(anchors, relations, coalition, k=k, t_norm=t_norm)
        else:
            raise ValueError(f"Unsupported query type: {query.query_type}.\n"
                             "Supported types: 2p, 2i, 2u, 3i, 3p, up (for 2u1p), ip (for 2i1p), pi (for 1p2i).")


def extract_anchors_relations(query: Query):
    anchors = []
    relations = []
    atoms = query.get_atoms()
    for atom in atoms.values():
        if isinstance(atom['head'], int):
            anchors.append(atom['head'])
        if isinstance(atom['tail'], int):
            anchors.append(atom['tail'])
        relations.append(atom['relation'])
    return anchors, relations