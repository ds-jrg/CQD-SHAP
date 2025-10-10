import pandas as pd
import numpy as np
import torch

np.random.seed(42)
FILLNA_VALUE = 0.0001
MAX_VALUE = 1.0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SymbolicReasoning:
    def __init__(self, graph, logging=True):
        self.graph = graph
        self.logging = logging
        # Build a fast (head, relation) => [tail_id, ...] lookup for quick 1-hop queries
        self.edge_index = {}  # (head, relation) -> set of tail_ids
        for edge in self.graph.get_edges():
            key = (edge.get_head().get_id(), edge.get_id())
            tail_id = edge.get_tail().get_id()
            if tail_id is None:
                if self.logging:
                    print(f"Warning: Edge {edge} has no tail_id, skipping.")
                    print(f"Edge: {edge}, Head ID: {edge.get_head().get_id()}, Relation ID: {edge.get_id()}")
            else:
                if key not in self.edge_index:
                    self.edge_index[key] = {tail_id}
                else:
                    self.edge_index[key].add(tail_id)

    def predict(self, h_id, rel_id, return_df=True, k=-1):
        results = self.edge_index.get((h_id, rel_id), set())
        scores = torch.full((len(self.graph.dataset.id2node),), FILLNA_VALUE, device=device)
        scores[list(results)] = MAX_VALUE
        if return_df:
            df = pd.DataFrame(scores.cpu().detach().numpy(), columns=["score"])
            df = df.sort_values(by="score", ascending=False)
            if k > 0:
                df = df.head(k)
            return df
        else:
            if k > 0:
                scores = scores.topk(k)
        return scores

    def predict_batch(self, h_ids, rel_id, k=-1):
        if k < 0:
            batch_result = torch.zeros((len(h_ids), len(self.graph.dataset.id2node)), dtype=torch.float32, device=device)
            for i, h_id in enumerate(h_ids):
                result = self.predict(h_id.item(), rel_id, return_df=False, k=k)
                batch_result[i] = result
            return batch_result
        else:
            scores = torch.empty((len(h_ids), len(self.graph.dataset.id2node)), dtype=torch.float32, device=device) # [batch_size, num_entities]
            for i, h_id in enumerate(h_ids):
                scores[i] = self.predict(h_id.item(), rel_id, return_df=False, k=-1)
            scores, indices = torch.topk(scores, k, dim=1)
            return scores, indices