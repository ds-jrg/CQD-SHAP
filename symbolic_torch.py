import pandas as pd
import numpy as np
import torch

np.random.seed(42)
FILLNA_VALUE = 0.0001
MAX_VALUE = 1.0

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
                print(f"Warning: Edge {edge} has no tail_id, skipping.")
                print(f"Edge: {edge}, Head ID: {edge.get_head().get_id()}, Relation ID: {edge.get_id()}")
            else:
                if key not in self.edge_index:
                    self.edge_index[key] = {tail_id}
                else:
                    self.edge_index[key].add(tail_id)

    def predict(self, h_id, rel_id, return_df=True, k=-1):
        results = self.edge_index.get((h_id, rel_id), set())
        scores = torch.full((len(self.graph.dataset.id2node),), FILLNA_VALUE)
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

    def predict_batch(self, h_ids, rel_ids, k=-1):
        # the result should be a tensor with shape batch_size (here h_ids) to num_entities (final result of predict)
        batch_result = torch.zeros((len(h_ids), len(self.graph.dataset.id2node)), dtype=torch.float32)
        for i, (h_id, rel_id) in enumerate(zip(h_ids, rel_ids)):
            result = self.predict(h_id, rel_id, return_df=False, k=k)
            batch_result[i] = result
        return batch_result