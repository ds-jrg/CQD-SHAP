import pandas as pd
import numpy as np

np.random.seed(42)
fillna_value = 0.0001

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

    def query_1p(self, head, relation):
        if self.logging:
            h_id = self.graph.dataset.get_node_by_id(head)
            h_title = self.graph.dataset.get_title_by_node(h_id)
            r_name = self.graph.dataset.get_relation_by_id(relation)
            print(f"Querying for head: {h_title} ({head} | {h_id}) and relation: {r_name} ({relation})")
        answers = self.edge_index.get((head, relation), set())
        if self.logging:
            for tail_id in answers:
                t_title = self.graph.dataset.get_title_by_node(self.graph.dataset.get_node_by_id(tail_id))
                r_name = self.graph.dataset.get_relation_by_id(relation)
                print(f"Found edge: {h_title} --{r_name}--> {t_title} ({tail_id})")
            print("-" * 50)
        # Return list (not set) for compatibility
        return list(answers)

    def query_2p(self, head, relations):
        first_level = self.query_1p(head, relations[0])
        second_level_answers = {}
        answers_set = set()
        for answer in first_level:
            res = self.query_1p(answer, relations[1])
            if res:
                second_level_answers[answer] = res
                answers_set.update(res)
        return second_level_answers, list(answers_set)

    def query_3p(self, head, relations):
        second_level_answers1, second_answers_flat = self.query_2p(head, relations[:2])
        third_level_answers = {}
        answers_set = set()
        for answer in second_answers_flat:
            res = self.query_1p(answer, relations[2])
            if res:
                third_level_answers[answer] = res
                answers_set.update(res)
        return third_level_answers, list(answers_set)

    def fixed_size_answer(self, answers, size):
        try:
            if answers is None:
                answers = []
            # Efficient DataFrame creation
            answers = np.asarray(answers, dtype=int)
            n_current = len(answers)
            df = pd.DataFrame({'score': np.ones(n_current, dtype=int)}, index=answers)
            if n_current < size:
                all_nodes = np.array(list(self.graph.dataset.id2node.keys()), dtype=int)
                # only pick nodes not in answers
                mask = ~np.isin(all_nodes, answers)
                additional_nodes = np.random.choice(all_nodes[mask], size - n_current, replace=False)
                # additional_df = pd.DataFrame({'score': np.zeros(len(additional_nodes), dtype=int)}, index=additional_nodes)
                # fill with 0.01
                additional_df = pd.DataFrame({'score': np.full(len(additional_nodes), fillna_value, dtype=float)}, index=additional_nodes)
                df = pd.concat([df, additional_df])
            elif n_current > size:
                df = df.sample(n=size, replace=False, random_state=None)
            return df
        except Exception as e:
            print(f"Error in fixed_size_answer: {e}")
            print(answers)