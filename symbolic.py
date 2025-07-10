import pandas as pd
from graph import Graph
import numpy as np

class SymbolicReasoning:
    def __init__(self, graph: Graph, logging: bool = True):
        self.graph = graph
        self.logging = logging

    def query_1p(self, head: int, relation: int):
        if self.logging:
            print(f"Querying for head: {self.graph.dataset.get_title_by_node(self.graph.dataset.get_node_by_id(head))} ({head} | {self.graph.dataset.get_node_by_id(head)}) and relation: {self.graph.dataset.get_relation_by_id(relation)} ({relation})")
        answers = []
        edges = self.graph.get_edges()
        for edge in edges:
            if edge.get_head().get_id() == head and edge.get_id() == relation:
                if self.logging:
                    print(f"Found edge: {edge.get_head().get_title()} --{edge.get_name()}--> {edge.get_tail().get_title()} ({edge.get_tail().get_id()})")
                answers.append(edge.get_tail().get_id())
        if self.logging:
            print("-" * 50)
        return list(set(answers))
    
    def query_2p(self, head: int, relations: tuple):
        first_level_answers = self.query_1p(head, relations[0])
        second_level_answers = {}
        for answer in first_level_answers:
            second_level_answers[answer] = self.query_1p(answer, relations[1])
        answers_set = set()
        for answer, second_level in second_level_answers.items():
            for item in second_level:
                answers_set.add(item)
        return second_level_answers, list(answers_set)
    
    def query_3p(self, head: int, relations: tuple):
        first_level_answers = self.query_2p(head, (relations[0], relations[1]))
        second_level_answers = {}
        for answer, second_level in first_level_answers[0].items():
            second_level_answers[answer] = self.query_1p(answer, relations[2])
        answers_set = set()
        for answer, second_level in second_level_answers.items():
            for item in second_level:
                answers_set.add(item)
        return second_level_answers, list(answers_set)
    
    def fixed_size_answer(self, answers: list, size: int):
        # make a dataframe which the index are answers and there is a column called score which the value is 1 for all answers
        array = np.full((len(answers), 1), 1)
        answers = np.array(answers)
        df = pd.DataFrame(array, index=answers, columns=['score'])

        if len(df) < size:
            # add random nodes to fill the size
            all_nodes = list(self.graph.dataset.id2node.keys())
            all_nodes_remaining = [node for node in all_nodes if node not in df.index]
            additional_nodes = np.random.choice(all_nodes_remaining, size - len(df), replace=False)
            additional_nodes = [int(node) for node in additional_nodes]
            # add them with score 0
            additional_df = pd.DataFrame(np.zeros((len(additional_nodes), 1)), index=additional_nodes, columns=['score'])
            if len(df) == 0:
                df = additional_df
            else:
                df = pd.concat([df, additional_df])
        elif len(df) > size:
            # truncate the dataframe to the size
            df = df.sample(size, replace=False)
        return df