import time
import pandas as pd
from query import Query
from cqd import cqd_query
from symbolic import SymbolicReasoning
from graph import Dataset
import numpy as np

np.random.seed(42)
fillna_value = 0.01

class XCQA:
    def __init__(self, symbolic: SymbolicReasoning, dataset: Dataset, cqd_cache: pd.DataFrame = None, inner_cache: dict = None,  logging: bool = True):

        self.symbolic = symbolic
        self.dataset = dataset
        self.cqd_cache = cqd_cache
        self.inner_cache = inner_cache if inner_cache is not None else {'cqd': {}, 'symbolic': {}}
        self.logging = logging

    def atom_execution(self, anchor: int, relation: int, mask: int, k: int = 10):
        """
        Executes a single atom query, either using CQD or symbolic reasoning based on the mask.
        """
        if mask == 1:  # CQD
            if self.inner_cache is not None and (anchor, relation) in self.inner_cache['cqd']:
                result = self.inner_cache['cqd'][(anchor, relation)]
            else:
                current_query = Query('1p', (((anchor, (relation,)),), []))
                result = cqd_query(current_query, k=k, cqd_cache=self.cqd_cache)
                if self.inner_cache is not None:
                    self.inner_cache['cqd'][(anchor, relation)] = result
        else: # Symbolic Reasoning
            if self.inner_cache is not None and (anchor, relation) in self.inner_cache['symbolic']:
                result = self.inner_cache['symbolic'][(anchor, relation)]
            else:
                result = self.symbolic.query_1p(anchor, relation)
                result = self.symbolic.fixed_size_answer(result, k)
                if self.inner_cache is not None:
                    self.inner_cache['symbolic'][(anchor, relation)] = result
        return result.copy()

    def keep_top_k(self, df: pd.DataFrame, k: int):
        """
        Keeps only the top k rows of the dataframe based on the 'score' column.
        """
        if df.empty:
            return df
        return df.nlargest(k, 'score')

    def query_execution(self, query: Query, k: int = 10, coalition: list = None, t_norm: str = 'prod', t_conorm: str = 'min'):
        # =============================================== 2p query =========================================================
        if query.query_type == '2p':
            anchor = query.get_query()[0][0]
            relations = query.get_query()[0][1]

            if self.logging:
                t0 = time.time()
            first_level_answers = self.atom_execution(anchor, relations[0], coalition[0], k)
            if self.logging:
                print(f"Time taken for first level query: {time.time() - t0:.2f} seconds")

            if self.logging:
                t1 = time.time()

            results = []
            for answer_idx, row in first_level_answers.iterrows():
                second_level = self.atom_execution(answer_idx, relations[1], coalition[1], k)
                if t_norm == 'prod':
                    second_level['score'] *= row['score']
                elif t_norm == 'min':
                    second_level['score'] = np.minimum(second_level['score'], row['score'])
                second_level['path'] = f"{anchor}--{relations[0]}-->{answer_idx}--{relations[1]}-->"
                results.append(second_level)

            # Concatenate all at once (much faster than in-loop)
            final_answers = pd.concat(results, axis=0) if results else None

            if self.logging:
                print(f"Time taken for second level query: {time.time() - t1:.2f} seconds")
        
        # =============================================== 2u query =========================================================
        elif query.query_type == '2u':
            query1 = query.get_query()[0]
            query2 = query.get_query()[1]
            anchor1 = query1[0]
            relation1 = query1[1][0]
            anchor2 = query2[0]
            relation2 = query2[1][0]

            first_branch_answers = None

            time_start = time.time()
            
            first_branch_answers = self.atom_execution(anchor1, relation1, coalition[0], k)
            first_branch_answers['path'] = str(anchor1) + f'--{relation1}-->'
            time_end = time.time()

            if self.logging:
                print(f"Time taken for first level query: {time_end - time_start:.2f} seconds")
                
            second_branch_answers = None

            time_start = time.time()
            second_branch_answers = self.atom_execution(anchor2, relation2, coalition[1], k)
            second_branch_answers['path'] = str(anchor2) + f'--{relation2}-->'

            if t_conorm == 'min':
                final_answers = pd.merge(first_branch_answers, second_branch_answers, left_index=True, right_index=True, how='outer', suffixes=('_1', '_2'))
                final_answers[['score_1', 'score_2']] = final_answers[['score_1', 'score_2']].fillna(fillna_value)
                final_answers.fillna(0, inplace=True)
                final_answers['score'] = final_answers[['score_1', 'score_2']].min(axis=1)
                final_answers = final_answers.drop(columns=['score_1', 'score_2'])
            elif t_conorm == 'prod':
                # if t-norm is prod, then t-conorm is sum - product
                final_answers = pd.merge(first_branch_answers, second_branch_answers, left_index=True, right_index=True, how='outer', suffixes=('_1', '_2'))
                final_answers[['score_1', 'score_2']] = final_answers[['score_1', 'score_2']].fillna(fillna_value)
                final_answers.fillna(0, inplace=True)
                # sum of score_1 and score_2 and then subtract their product (based on CQD implementation)
                final_answers['score'] = final_answers[['score_1', 'score_2']].sum(axis=1) - final_answers[['score_1', 'score_2']].prod(axis=1)
                final_answers = final_answers.drop(columns=['score_1', 'score_2'])
            elif t_conorm == 'max':
                # if t-norm is min, then t-conorm is max
                final_answers = pd.merge(first_branch_answers, second_branch_answers, left_index=True, right_index=True, how='outer', suffixes=('_1', '_2'))
                final_answers[['score_1', 'score_2']] = final_answers[['score_1', 'score_2']].fillna(fillna_value)
                final_answers.fillna(0, inplace=True)
                final_answers['score'] = final_answers[['score_1', 'score_2']].max(axis=1)
                final_answers = final_answers.drop(columns=['score_1', 'score_2'])
            else:
                raise ValueError(f"Unknown t_conorm: {t_conorm}. Supported values are 'min', 'prod', 'max'.")
            final_answers = final_answers.sort_values(by='score', ascending=False)
            time_end = time.time() 
            if self.logging:
                print(f"Time taken for second level query: {time_end - time_start:.2f} seconds")

        # =============================================== 2i query =========================================================
        elif query.query_type == '2i':
            query1 = query.get_query()[0]
            query2 = query.get_query()[1]
            anchor1 = query1[0]
            relation1 = query1[1][0]
            anchor2 = query2[0]
            relation2 = query2[1][0]

            first_branch_answers = None

            time_start = time.time()
            first_branch_answers = self.atom_execution(anchor1, relation1, coalition[0], k)
            first_branch_answers['path'] = str(anchor1) + f'--{relation1}-->'
            time_end = time.time()

            if self.logging:
                print(f"Time taken for first level query: {time_end - time_start:.2f} seconds")
                
            second_branch_answers = None

            time_start = time.time()
            second_branch_answers = self.atom_execution(anchor2, relation2, coalition[1], k)
            second_branch_answers['path'] = str(anchor2) + f'--{relation2}-->'

            if t_norm == 'min':
                final_answers = pd.merge(first_branch_answers, second_branch_answers, left_index=True, right_index=True, how='outer', suffixes=('_1', '_2'))
                final_answers[['score_1', 'score_2']] = final_answers[['score_1', 'score_2']].fillna(fillna_value)
                final_answers.fillna(0, inplace=True)
                final_answers['score'] = final_answers[['score_1', 'score_2']].min(axis=1)
                final_answers = final_answers.drop(columns=['score_1', 'score_2'])
            elif t_norm == 'prod':
                final_answers = pd.merge(first_branch_answers, second_branch_answers, left_index=True, right_index=True, how='outer', suffixes=('_1', '_2'))
                final_answers[['score_1', 'score_2']] = final_answers[['score_1', 'score_2']].fillna(fillna_value)
                final_answers.fillna(0, inplace=True)
                final_answers['score'] = final_answers[['score_1', 'score_2']].prod(axis=1)
                final_answers = final_answers.drop(columns=['score_1', 'score_2'])
            elif t_norm == 'max':
                final_answers = pd.merge(first_branch_answers, second_branch_answers, left_index=True, right_index=True, how='outer', suffixes=('_1', '_2'))
                final_answers[['score_1', 'score_2']] = final_answers[['score_1', 'score_2']].fillna(fillna_value)
                final_answers.fillna(0, inplace=True)
                final_answers['score'] = final_answers[['score_1', 'score_2']].max(axis=1)
                final_answers = final_answers.drop(columns=['score_1', 'score_2'])
            else:
                raise ValueError(f"Unknown t_conorm: {t_conorm}. Supported values are 'min', 'prod', 'max'.")
            final_answers = final_answers.sort_values(by='score', ascending=False)
            time_end = time.time() 
            if self.logging:
                print(f"Time taken for second level query: {time_end - time_start:.2f} seconds")
        
        # =============================================== 3p query =========================================================
        elif query.query_type == '3p':
            anchor = query.get_query()[0][0]
            relations = query.get_query()[0][1]

            first_level_answers = None
            final_answers = None

            time_start = time.time()
            first_level_answers = self.atom_execution(anchor, relations[0], coalition[0], k)
            time_end = time.time()

            if self.logging:
                print(f"Time taken for first level query: {time_end - time_start:.2f} seconds")

            start_time = time.time()
            second_level_answers = None
            for answer_idx, row in first_level_answers.iterrows():
                second_level_answer = self.atom_execution(answer_idx, relations[1], coalition[1], k)
                if t_norm == 'prod':  
                        second_level_answer['score'] = second_level_answer['score'] * row['score']
                elif t_norm == 'min':
                        second_level_answer['score'] = second_level_answer['score'].apply(lambda x: min(x, row['score']))
                second_level_answer['path'] = str(anchor) + f'--{relations[0]}-->{answer_idx}' + f'--{relations[1]}-->'
                if second_level_answers is None:
                    second_level_answers = second_level_answer
                else:
                    second_level_answers = pd.concat([second_level_answers, second_level_answer], axis=0)
            
            for second_answer_idx, second_row in second_level_answers.iterrows():
                third_level_answers = self.atom_execution(second_answer_idx, relations[2], coalition[2], k)
                if t_norm == 'prod':  
                    third_level_answers['score'] = third_level_answers['score'] * second_row['score']
                elif t_norm == 'min':
                    third_level_answers['score'] = third_level_answers['score'].apply(lambda x: min(x, second_row['score']))
                third_level_answers['path'] = second_row['path'] + str(second_answer_idx) + f'--{relations[2]}-->'
                if final_answers is None:
                    final_answers = third_level_answers
                else:
                    final_answers = pd.concat([final_answers, third_level_answers], axis=0)
            time_end = time.time()
            if self.logging:
                print(f"Time taken for second level query: {time_end - start_time:.2f} seconds")
                
        # =============================================== 3i query =========================================================
        elif query.query_type == '3i':
            query1 = query.get_query()[0]
            query2 = query.get_query()[1]
            query3 = query.get_query()[2]
            anchor1 = query1[0]
            relation1 = query1[1][0]
            anchor2 = query2[0]
            relation2 = query2[1][0]
            anchor3 = query3[0]
            relation3 = query3[1][0]

            first_branch_answers = None

            time_start = time.time()
            first_branch_answers = self.atom_execution(anchor1, relation1, coalition[0], k)
            first_branch_answers['path'] = str(anchor1) + f'--{relation1}-->'
            time_end = time.time()

            if self.logging:
                print(f"Time taken for first level query: {time_end - time_start:.2f} seconds")
                
            second_branch_answers = None

            time_start = time.time()
            second_branch_answers = self.atom_execution(anchor2, relation2, coalition[1], k)
            second_branch_answers['path'] = str(anchor2) + f'--{relation2}-->'
            time_end = time.time()
            if self.logging:
                print(f"Time taken for second level query: {time_end - time_start:.2f} seconds")
            
            time_start = time.time()
            third_branch_answers = None
            third_branch_answers = self.atom_execution(anchor3, relation3, coalition[2], k)
            third_branch_answers['path'] = str(anchor3) + f'--{relation3}-->'

            if t_norm == 'min':
                final_answers = pd.merge(first_branch_answers, second_branch_answers, left_index=True, right_index=True, how='outer', suffixes=('_1', '_2'))
                final_answers = pd.merge(final_answers, third_branch_answers, left_index=True, right_index=True, how='outer')
                final_answers[['score_1', 'score_2', 'score']] = final_answers[['score_1', 'score_2', 'score']].fillna(fillna_value)
                final_answers.fillna(0, inplace=True)
                final_answers['score'] = final_answers[['score_1', 'score_2', 'score']].min(axis=1)
                final_answers = final_answers.drop(columns=['score_1', 'score_2'])
            elif t_norm == 'prod':
                final_answers = pd.merge(first_branch_answers, second_branch_answers, left_index=True, right_index=True, how='outer', suffixes=('_1', '_2'))
                final_answers = pd.merge(final_answers, third_branch_answers, left_index=True, right_index=True, how='outer')
                final_answers[['score_1', 'score_2', 'score']] = final_answers[['score_1', 'score_2', 'score']].fillna(fillna_value)
                final_answers.fillna(0, inplace=True)
                final_answers['score'] = final_answers[['score_1', 'score_2', 'score']].prod(axis=1)
                final_answers = final_answers.drop(columns=['score_1', 'score_2'])
            elif t_norm == 'max':
                final_answers = pd.merge(first_branch_answers, second_branch_answers, left_index=True, right_index=True, how='outer', suffixes=('_1', '_2'))
                final_answers = pd.merge(final_answers, third_branch_answers, left_index=True, right_index=True, how='outer')
                final_answers[['score_1', 'score_2', 'score']] = final_answers[['score_1', 'score_2', 'score']].fillna(fillna_value)
                final_answers.fillna(0, inplace=True)
                final_answers['score'] = final_answers[['score_1', 'score_2', 'score']].max(axis=1)
                final_answers = final_answers.drop(columns=['score_1', 'score_2'])
            final_answers['path'] = (
                final_answers['path_1'].fillna('').astype(str) + "\n" +
                final_answers['path_2'].fillna('').astype(str) + "\n" +
                final_answers['path'].fillna('').astype(str)
            )
            final_answers = final_answers.drop(columns=['path_1', 'path_2'])
            final_answers = final_answers.sort_values(by='score', ascending=False)
            time_end = time.time()
            if self.logging:
                print(f"Time taken for third level query: {time_end - time_start:.2f} seconds")
        # =============================================== up query =========================================================
        elif query.query_type == 'up':
            query1 = query.query[0]
            anchor1 = query1[0]
            relation1 = query1[1][0]
            query2 = query.query[1]
            anchor2 = query2[0]
            relation2 = query2[1][0]
            relation3 = query.query[2]
            # first similar like 2u, then we need to do another projection for each final entity
            first_branch_answers = None
            time_start = time.time()
            first_branch_answers = self.atom_execution(anchor1, relation1, coalition[0], k)
            first_branch_answers['path'] = str(anchor1) + f'--{relation1}-->'
            time_end = time.time()
            if self.logging:
                print(f"Time taken for first level query: {time_end - time_start:.2f} seconds")
            second_branch_answers = None
            time_start = time.time()
            second_branch_answers = self.atom_execution(anchor2, relation2, coalition[1], k)
            second_branch_answers['path'] = str(anchor2) + f'--{relation2}-->'
            if t_conorm == 'min':
                union_answers = pd.merge(first_branch_answers, second_branch_answers, left_index=True, right_index=True, how='outer', suffixes=('_1', '_2'))
                union_answers[['score_1', 'score_2']] = union_answers[['score_1', 'score_2']].fillna(fillna_value)
                union_answers.fillna(0, inplace=True)
                union_answers['score'] = union_answers[['score_1', 'score_2']].min(axis=1)
                union_answers = union_answers.drop(columns=['score_1', 'score_2'])
            elif t_conorm == 'prod':
                union_answers = pd.merge(first_branch_answers, second_branch_answers, left_index=True, right_index=True, how='outer', suffixes=('_1', '_2'))
                union_answers[['score_1', 'score_2']] = union_answers[['score_1', 'score_2']].fillna(fillna_value)
                union_answers.fillna(0, inplace=True)
                union_answers['score'] = union_answers[['score_1', 'score_2']].sum(axis=1) - union_answers[['score_1', 'score_2']].prod(axis=1)
                union_answers = union_answers.drop(columns=['score_1', 'score_2'])
            elif t_conorm == 'max':
                union_answers = pd.merge(first_branch_answers, second_branch_answers, left_index=True, right_index=True, how='outer', suffixes=('_1', '_2'))
                union_answers[['score_1', 'score_2']] = union_answers[['score_1', 'score_2']].fillna(fillna_value)
                union_answers.fillna(0, inplace=True)
                union_answers['score'] = union_answers[['score_1', 'score_2']].max(axis=1)
                union_answers = union_answers.drop(columns=['score_1', 'score_2'])
            union_answers['path'] = (
                union_answers['path_1'].fillna('').astype(str) + "\n" +
                union_answers['path_2'].fillna('').astype(str)
            )
            union_answers = union_answers.drop(columns=['path_1', 'path_2'])
            union_answers = union_answers.sort_values(by='score', ascending=False)
            time_end = time.time()
            if self.logging:
                print(f"Time taken for second level query: {time_end - time_start:.2f} seconds")
            time_start = time.time()
            final_answers = None
            for answer_idx, row in union_answers.iterrows():
                third_level_answers = self.atom_execution(answer_idx, relation3, coalition[2], k)
                if t_norm == 'prod':  
                    third_level_answers['score'] = third_level_answers['score'] * row['score']
                elif t_norm == 'min':
                    third_level_answers['score'] = third_level_answers['score'].apply(lambda x: min(x, row['score']))
                third_level_answers['path'] = row['path'] + str(answer_idx) + f'--{relation3}-->'
                if final_answers is None:
                    final_answers = third_level_answers
                else:
                    final_answers = pd.concat([final_answers, third_level_answers], axis=0)
            time_end = time.time()
            if self.logging:
                print(f"Time taken for third level query: {time_end - time_start:.2f} seconds")
        # =============================================== ip query =========================================================
        elif query.query_type == 'ip':
            query1 = query.query[0]
            anchor1 = query1[0]
            relation1 = query1[1][0]
            query2 = query.query[1]
            anchor2 = query2[0]
            relation2 = query2[1][0]
            relation3 = query.query[2]
            # first similar like 2i, then we need to do another intersection for each final entity
            first_branch_answers = None
            time_start = time.time()
            first_branch_answers = self.atom_execution(anchor1, relation1, coalition[0], k)
            first_branch_answers['path'] = str(anchor1) + f'--{relation1}-->'
            time_end = time.time()
            if self.logging:
                print(f"Time taken for first level query: {time_end - time_start:.2f} seconds")
            second_branch_answers = None
            time_start = time.time()
            second_branch_answers = self.atom_execution(anchor2, relation2, coalition[1], k)
            second_branch_answers['path'] = str(anchor2) + f'--{relation2}-->'
            if t_norm == 'min':
                intersection_answers = pd.merge(first_branch_answers, second_branch_answers, left_index=True, right_index=True, how='outer', suffixes=('_1', '_2'))
                intersection_answers[['score_1', 'score_2']] = intersection_answers[['score_1', 'score_2']].fillna(fillna_value)
                intersection_answers.fillna(0, inplace=True)
                intersection_answers['score'] = intersection_answers[['score_1', 'score_2']].min(axis=1)
                intersection_answers = intersection_answers.drop(columns=['score_1', 'score_2'])
            elif t_norm == 'prod':
                intersection_answers = pd.merge(first_branch_answers, second_branch_answers, left_index=True, right_index=True, how='outer', suffixes=('_1', '_2'))
                intersection_answers[['score_1', 'score_2']] = intersection_answers[['score_1', 'score_2']].fillna(fillna_value)
                intersection_answers.fillna(0, inplace=True)
                intersection_answers['score'] = intersection_answers[['score_1', 'score_2']].prod(axis=1)
                intersection_answers = intersection_answers.drop(columns=['score_1', 'score_2'])
            elif t_norm == 'max':
                intersection_answers = pd.merge(first_branch_answers, second_branch_answers, left_index=True, right_index=True, how='outer', suffixes=('_1', '_2'))
                intersection_answers[['score_1', 'score_2']] = intersection_answers[['score_1', 'score_2']].fillna(fillna_value)
                intersection_answers.fillna(0, inplace=True)
                intersection_answers['score'] = intersection_answers[['score_1', 'score_2']].max(axis=1)
                intersection_answers = intersection_answers.drop(columns=['score_1', 'score_2'])
            intersection_answers['path'] = (
                intersection_answers['path_1'].fillna('').astype(str) + "\n" +
                intersection_answers['path_2'].fillna('').astype(str)
            )
            intersection_answers = intersection_answers.drop(columns=['path_1', 'path_2'])
            intersection_answers = intersection_answers.sort_values(by='score', ascending=False)
            time_end = time.time()
            if self.logging:
                print(f"Time taken for second level query: {time_end - time_start:.2f} seconds")
            time_start = time.time()
            final_answers = None
            for answer_idx, row in intersection_answers.iterrows():
                third_level_answers = self.atom_execution(answer_idx, relation3, coalition[2], k)
                if t_norm == 'prod':  
                    third_level_answers['score'] = third_level_answers['score'] * row['score']
                elif t_norm == 'min':
                    third_level_answers['score'] = third_level_answers['score'].apply(lambda x: min(x, row['score']))
                third_level_answers['path'] = row['path'] + str(answer_idx) + f'--{relation3}-->'
                if final_answers is None:
                    final_answers = third_level_answers
                else:
                    final_answers = pd.concat([final_answers, third_level_answers], axis=0)
            time_end = time.time()
            if self.logging:
                print(f"Time taken for third level query: {time_end - time_start:.2f} seconds")
        # =============================================== pi query =========================================================
        elif query.query_type == 'pi':
            branch1 = query.query[0]
            branch2 = query.query[1]
            anchor1 = branch1[0]
            relation1 = branch1[1][0]
            relation2 = branch1[1][1]
            anchor2 = branch2[0]
            relation3 = branch2[1][0]
            
            first_branch_first_level_answers = None
            time_start = time.time()
            first_branch_first_level_answers = self.atom_execution(anchor1, relation1, coalition[0], k)
            first_branch_first_level_answers['path'] = str(anchor1) + f'--{relation1}-->'
            time_end = time.time()
            if self.logging:
                print(f"Time taken for first level query: {time_end - time_start:.2f} seconds")
            results = []
            time_start = time.time()
            for answer_idx, row in first_branch_first_level_answers.iterrows():
                first_branch_second_level_answer = self.atom_execution(answer_idx, relation2, coalition[1], k)
                if t_norm == 'prod':  
                    first_branch_second_level_answer['score'] *= row['score']
                elif t_norm == 'min':
                    first_branch_second_level_answer['score'] = np.minimum(first_branch_second_level_answer['score'], row['score'])
                    # first_branch_second_level_answer['score'].apply(lambda x: min(x, row['score']))
                first_branch_second_level_answer['path'] = row['path'] + str(answer_idx) + f'--{relation2}-->'
                results.append(first_branch_second_level_answer)
            first_branch_second_level_answers = pd.concat(results, axis=0) if results else None

            time_end = time.time()
            if self.logging:
                print(f"Time taken for second level query: {time_end - time_start:.2f} seconds")

            second_branch_answers = None
            time_start = time.time()
            second_branch_answers = self.atom_execution(anchor2, relation3, coalition[2], k)
            second_branch_answers['path'] = str(anchor2) + f'--{relation3}-->'
            if t_norm == 'min':
                final_answers = pd.merge(first_branch_second_level_answers, second_branch_answers, left_index=True, right_index=True, how='outer', suffixes=('_1', '_2'))
                final_answers[['score_1', 'score_2']] = final_answers[['score_1', 'score_2']].fillna(fillna_value)
                final_answers.fillna(0, inplace=True)
                final_answers['score'] = final_answers[['score_1', 'score_2']].min(axis=1)
                final_answers = final_answers.drop(columns=['score_1', 'score_2'])
            elif t_norm == 'prod':
                final_answers = pd.merge(first_branch_second_level_answers, second_branch_answers, left_index=True, right_index=True, how='outer', suffixes=('_1', '_2'))
                final_answers[['score_1', 'score_2']] = final_answers[['score_1', 'score_2']].fillna(fillna_value)
                final_answers.fillna(0, inplace=True)
                final_answers['score'] = final_answers[['score_1', 'score_2']].prod(axis=1)
                final_answers = final_answers.drop(columns=['score_1', 'score_2'])
            elif t_norm == 'max':
                final_answers = pd.merge(first_branch_second_level_answers, second_branch_answers, left_index=True, right_index=True, how='outer', suffixes=('_1', '_2'))
                final_answers[['score_1', 'score_2']] = final_answers[['score_1', 'score_2']].fillna(fillna_value)
                final_answers.fillna(0, inplace=True)
                final_answers['score'] = final_answers[['score_1', 'score_2']].max(axis=1)
                final_answers = final_answers.drop(columns=['score_1', 'score_2'])
            final_answers['path'] = (
                final_answers['path_1'].fillna('').astype(str) + "\n" +
                final_answers['path_2'].fillna('').astype(str) + "\n"
            )
            final_answers = final_answers.drop(columns=['path_1', 'path_2'])
            final_answers = final_answers.sort_values(by='score', ascending=False)
            time_end = time.time()
            if self.logging:
                print(f"Time taken for third level query: {time_end - time_start:.2f} seconds")
            
        else:
            raise ValueError(f"Unsupported query type: {query.query_type}. Only '2p' queries are supported.")
        final_answers = final_answers.sort_values(by='score', ascending=False)

        # if we have duplicate answers, we need to keep only the one with the highest score
        # as the final_answers dataframe is already sorted by score, we can just keep the first occurrence of each index which means that we keep the highest score
        final_answers = final_answers[~final_answers.index.duplicated(keep='first')]

        # the output should be a dataframe of scores for each possible node in the graph
        df = pd.DataFrame(index=self.dataset.id2node.keys(), columns=['score', 'path'])
        df['score'] = 0.0
        df = df[~df.index.isin(final_answers.index)]
        df = pd.concat([final_answers, df])
        # shuffle the data to have a random order of answers and a fair measurement of performance
        # df = df.sample(frac=1)
        # sort by index and score at the same time to make sure that the order is consistent
        # df.index.name = 'entity_id'
        # df = df.sort_values(by=['score', 'entity_id'], ascending=[False, True])
        return df