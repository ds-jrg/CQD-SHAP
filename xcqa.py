import time
import pandas as pd
from query import Query
from cqd import cqd_query
from symbolic import SymbolicReasoning
from graph import Dataset

def query_execution(query: Query, symbolic: SymbolicReasoning, dataset: Dataset, k: int = 10, coalition: list = None, t_norm: str = 'prod', t_conorm: str = 'min', logging: bool = True, cqd_cache: pd.DataFrame = None, inner_cache: dict = None):
    # =============================================== 2p query =========================================================
    if query.query_type == '2p':
        anchor = query.get_query()[0][0]
        relations = query.get_query()[0][1]

        first_level_answers = None
        final_answers = None

        time_start = time.time()
        if coalition[0] == 1:
            if inner_cache is not None and (anchor, relations[0]) in inner_cache['cqd']:
                # print(f"Using cached CQD results for anchor: {anchor}, relation: {relations[0]}")
                first_level_answers = inner_cache['cqd'][(anchor, relations[0])]
            else:
                current_query = Query('1p', (((anchor, (relations[0],)),), []))
                first_level_answers = cqd_query(current_query, k=k, cqd_cache=cqd_cache)
                if inner_cache is not None:
                    inner_cache['cqd'][(anchor, relations[0])] = first_level_answers
        elif coalition[0] == 0:
            if inner_cache is not None and (anchor, relations[0]) in inner_cache['symbolic']:
                # print(f"Using cached Symbolic results for anchor: {anchor}, relation: {relations[0]}")
                first_level_answers = inner_cache['symbolic'][(anchor, relations[0])]
            else:
                first_level_answers = symbolic.query_1p(anchor, relations[0])
                first_level_answers = symbolic.fixed_size_answer(first_level_answers, k)
                if inner_cache is not None:
                    inner_cache['symbolic'][(anchor, relations[0])] = first_level_answers
        time_end = time.time()

        if logging:
            print(f"Time taken for first level query: {time_end - time_start:.2f} seconds")

        start_time = time.time()
        for answer_idx, row in first_level_answers.iterrows():
            if coalition[1] == 1:
                if inner_cache is not None and (answer_idx, relations[1]) in inner_cache['cqd']:
                    # print(f"Using cached CQD results for answer: {answer_idx}, relation: {relations[1]}")
                    second_level_answers = inner_cache['cqd'][(answer_idx, relations[1])].copy()
                else:
                    current_query = Query('1p', (((answer_idx, (relations[1],)),), []))
                    second_level_answers = cqd_query(current_query, k=k, cqd_cache=cqd_cache)
                    if inner_cache is not None:
                        inner_cache['cqd'][(answer_idx, relations[1])] = second_level_answers.copy()
            elif coalition[1] == 0:
                if inner_cache is not None and (answer_idx, relations[1]) in inner_cache['symbolic']:
                    # print(f"Using cached symbolic results for answer: {answer_idx}, relation: {relations[1]}")
                    second_level_answers = inner_cache['symbolic'][(answer_idx, relations[1])].copy()
                else:
                    second_level_answers = symbolic.query_1p(answer_idx, relations[1])
                    second_level_answers = symbolic.fixed_size_answer(second_level_answers, k)
                    if inner_cache is not None:
                        inner_cache['symbolic'][(answer_idx, relations[1])] = second_level_answers.copy()
            if t_norm == 'prod':  
                    second_level_answers['score'] = second_level_answers['score'] * row['score']
            elif t_norm == 'min':
                    second_level_answers['score'] = second_level_answers['score'].apply(lambda x: min(x, row['score']))
            # second_level_answers['path'] = str((anchor, relations[0], answer_idx, relations[1]))
            # second_row['path'] + f'--{relations[2]}-->{second_answer_idx}'
            second_level_answers['path'] = str(anchor) + f'--{relations[0]}-->{answer_idx}' + f'--{relations[1]}-->'
            if final_answers is None:
                final_answers = second_level_answers
            else:
                final_answers = pd.concat([final_answers, second_level_answers], axis=0)
        time_end = time.time()
        if logging:
            print(f"Time taken for second level query: {time_end - start_time:.2f} seconds")
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
        if coalition[0] == 1:
            if inner_cache is not None and (anchor1, relation1) in inner_cache['cqd']:
                # print(f"Using cached CQD results for anchor: {anchor1}, relation: {relation1}")
                first_branch_answers = inner_cache['cqd'][(anchor1, relation1)]
            else:
                current_query = Query('1p', (((anchor1, (relation1,)),), []))
                first_branch_answers = cqd_query(current_query, k=k, cqd_cache=cqd_cache)
                if inner_cache is not None:
                    inner_cache['cqd'][(anchor1, relation1)] = first_branch_answers
        elif coalition[0] == 0:
            if inner_cache is not None and (anchor1, relation1) in inner_cache['symbolic']:
                # print(f"Using cached Symbolic results for anchor: {anchor1}, relation: {relation1}")
                first_branch_answers = inner_cache['symbolic'][(anchor1, relation1)]
            else:
                first_branch_answers = symbolic.query_1p(anchor1, relation1)
                first_branch_answers = symbolic.fixed_size_answer(first_branch_answers, k)
                if inner_cache is not None:
                    inner_cache['symbolic'][(anchor1, relation1)] = first_branch_answers
        first_branch_answers['path'] = str(anchor1) + f'--{relation1}-->'
        time_end = time.time()

        if logging:
            print(f"Time taken for first level query: {time_end - time_start:.2f} seconds")
            
        second_branch_answers = None

        time_start = time.time()
        if coalition[1] == 1:
            if inner_cache is not None and (anchor2, relation2) in inner_cache['cqd']:
                # print(f"Using cached CQD results for anchor: {anchor2}, relation: {relation2}")
                second_branch_answers = inner_cache['cqd'][(anchor2, relation2)]
            else:
                current_query = Query('1p', (((anchor2, (relation2,)),), []))
                second_branch_answers = cqd_query(current_query, k=k, cqd_cache=cqd_cache)
                if inner_cache is not None:
                    inner_cache['cqd'][(anchor2, relation2)] = second_branch_answers
        elif coalition[1] == 0:
            if inner_cache is not None and (anchor2, relation2) in inner_cache['symbolic']:
                # print(f"Using cached Symbolic results for anchor: {anchor2}, relation: {relation2}")
                second_branch_answers = inner_cache['symbolic'][(anchor2, relation2)]
            else:
                second_branch_answers = symbolic.query_1p(anchor2, relation2)
                second_branch_answers = symbolic.fixed_size_answer(second_branch_answers, k)
                if inner_cache is not None:
                    inner_cache['symbolic'][(anchor2, relation2)] = second_branch_answers
        second_branch_answers['path'] = str(anchor2) + f'--{relation2}-->'

        if t_conorm == 'min':
            final_answers = pd.merge(first_branch_answers, second_branch_answers, left_index=True, right_index=True, how='outer', suffixes=('_1', '_2'))
            final_answers.fillna(0, inplace=True)
            final_answers['score'] = final_answers[['score_1', 'score_2']].min(axis=1)
            final_answers = final_answers.drop(columns=['score_1', 'score_2'])
        elif t_conorm == 'prod':
            # if t-norm is prod, then t-conorm is sum - product
            final_answers = pd.merge(first_branch_answers, second_branch_answers, left_index=True, right_index=True, how='outer', suffixes=('_1', '_2'))
            final_answers.fillna(0, inplace=True)
            # sum of score_1 and score_2 and then subtract their product (based on CQD implementation)
            final_answers['score'] = final_answers[['score_1', 'score_2']].sum(axis=1) - final_answers[['score_1', 'score_2']].prod(axis=1)
            final_answers = final_answers.drop(columns=['score_1', 'score_2'])
        elif t_conorm == 'max':
            # if t-norm is min, then t-conorm is max
            final_answers = pd.merge(first_branch_answers, second_branch_answers, left_index=True, right_index=True, how='outer', suffixes=('_1', '_2'))
            final_answers.fillna(0, inplace=True)
            final_answers['score'] = final_answers[['score_1', 'score_2']].max(axis=1)
            final_answers = final_answers.drop(columns=['score_1', 'score_2'])
        else:
            raise ValueError(f"Unknown t_conorm: {t_conorm}. Supported values are 'min', 'prod', 'max'.")
        time_end = time.time() 
        if logging:
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
        if coalition[0] == 1:
            if inner_cache is not None and (anchor1, relation1) in inner_cache['cqd']:
                # print(f"Using cached CQD results for anchor: {anchor1}, relation: {relation1}")
                first_branch_answers = inner_cache['cqd'][(anchor1, relation1)]
            else:
                current_query = Query('1p', (((anchor1, (relation1,)),), []))
                first_branch_answers = cqd_query(current_query, k=k, cqd_cache=cqd_cache)
                if inner_cache is not None:
                    inner_cache['cqd'][(anchor1, relation1)] = first_branch_answers
        elif coalition[0] == 0:
            if inner_cache is not None and (anchor1, relation1) in inner_cache['symbolic']:
                # print(f"Using cached Symbolic results for anchor: {anchor1}, relation: {relation1}")
                first_branch_answers = inner_cache['symbolic'][(anchor1, relation1)]
            else:
                first_branch_answers = symbolic.query_1p(anchor1, relation1)
                first_branch_answers = symbolic.fixed_size_answer(first_branch_answers, k)
                if inner_cache is not None:
                    inner_cache['symbolic'][(anchor1, relation1)] = first_branch_answers
        first_branch_answers['path'] = str(anchor1) + f'--{relation1}-->'
        time_end = time.time()

        if logging:
            print(f"Time taken for first level query: {time_end - time_start:.2f} seconds")
            
        second_branch_answers = None

        time_start = time.time()
        if coalition[1] == 1:
            if inner_cache is not None and (anchor2, relation2) in inner_cache['cqd']:
                # print(f"Using cached CQD results for anchor: {anchor2}, relation: {relation2}")
                second_branch_answers = inner_cache['cqd'][(anchor2, relation2)]
            else:
                current_query = Query('1p', (((anchor2, (relation2,)),), []))
                second_branch_answers = cqd_query(current_query, k=k, cqd_cache=cqd_cache)
                if inner_cache is not None:
                    inner_cache['cqd'][(anchor2, relation2)] = second_branch_answers
        elif coalition[1] == 0:
            if inner_cache is not None and (anchor2, relation2) in inner_cache['symbolic']:
                # print(f"Using cached Symbolic results for anchor: {anchor2}, relation: {relation2}")
                second_branch_answers = inner_cache['symbolic'][(anchor2, relation2)]
            else:
                second_branch_answers = symbolic.query_1p(anchor2, relation2)
                second_branch_answers = symbolic.fixed_size_answer(second_branch_answers, k)
                if inner_cache is not None:
                    inner_cache['symbolic'][(anchor2, relation2)] = second_branch_answers
        second_branch_answers['path'] = str(anchor2) + f'--{relation2}-->'

        if t_norm == 'min':
            final_answers = pd.merge(first_branch_answers, second_branch_answers, left_index=True, right_index=True, how='outer', suffixes=('_1', '_2'))
            final_answers.fillna(0, inplace=True)
            final_answers['score'] = final_answers[['score_1', 'score_2']].min(axis=1)
            final_answers = final_answers.drop(columns=['score_1', 'score_2'])
        elif t_norm == 'prod':
            final_answers = pd.merge(first_branch_answers, second_branch_answers, left_index=True, right_index=True, how='outer', suffixes=('_1', '_2'))
            final_answers.fillna(0, inplace=True)
            final_answers['score'] = final_answers[['score_1', 'score_2']].prod(axis=1)
            final_answers = final_answers.drop(columns=['score_1', 'score_2'])
        elif t_norm == 'max':
            final_answers = pd.merge(first_branch_answers, second_branch_answers, left_index=True, right_index=True, how='outer', suffixes=('_1', '_2'))
            final_answers.fillna(0, inplace=True)
            final_answers['score'] = final_answers[['score_1', 'score_2']].max(axis=1)
            final_answers = final_answers.drop(columns=['score_1', 'score_2'])
        else:
            raise ValueError(f"Unknown t_conorm: {t_conorm}. Supported values are 'min', 'prod', 'max'.")
        time_end = time.time() 
        if logging:
            print(f"Time taken for second level query: {time_end - time_start:.2f} seconds")
    # =============================================== 3p query =========================================================
    elif query.query_type == '3p':
        anchor = query.get_query()[0][0]
        relations = query.get_query()[0][1]

        first_level_answers = None
        final_answers = None

        time_start = time.time()
        if coalition[0] == 1:
            if inner_cache is not None and (anchor, relations[0]) in inner_cache['cqd']:
                # print(f"Using cached CQD results for anchor: {anchor}, relation: {relations[0]}")
                first_level_answers = inner_cache['cqd'][(anchor, relations[0])]
            else:
                current_query = Query('1p', (((anchor, (relations[0],)),), []))
                first_level_answers = cqd_query(current_query, k=k, cqd_cache=cqd_cache)
                if inner_cache is not None:
                    inner_cache['cqd'][(anchor, relations[0])] = first_level_answers
        elif coalition[0] == 0:
            if inner_cache is not None and (anchor, relations[0]) in inner_cache['symbolic']:
                # print(f"Using cached Symbolic results for anchor: {anchor}, relation: {relations[0]}")
                first_level_answers = inner_cache['symbolic'][(anchor, relations[0])]
            else:
                first_level_answers = symbolic.query_1p(anchor, relations[0])
                first_level_answers = symbolic.fixed_size_answer(first_level_answers, k)
                if inner_cache is not None:
                    inner_cache['symbolic'][(anchor, relations[0])] = first_level_answers
        time_end = time.time()

        if logging:
            print(f"Time taken for first level query: {time_end - time_start:.2f} seconds")

        start_time = time.time()
        second_level_answers = None
        for answer_idx, row in first_level_answers.iterrows():
            if coalition[1] == 1:
                if inner_cache is not None and (answer_idx, relations[1]) in inner_cache['cqd']:
                    # print(f"Using cached CQD results for answer: {answer_idx}, relation: {relations[1]}")
                    second_level_answer = inner_cache['cqd'][(answer_idx, relations[1])].copy()
                else:
                    current_query = Query('1p', (((answer_idx, (relations[1],)),), []))
                    second_level_answer = cqd_query(current_query, k=k, cqd_cache=cqd_cache)
                    if inner_cache is not None:
                        inner_cache['cqd'][(answer_idx, relations[1])] = second_level_answer.copy()
            elif coalition[1] == 0:
                if inner_cache is not None and (answer_idx, relations[1]) in inner_cache['symbolic']:
                    # print(f"Using cached symbolic results for answer: {answer_idx}, relation: {relations[1]}")
                    second_level_answer = inner_cache['symbolic'][(answer_idx, relations[1])].copy()
                else:
                    second_level_answer = symbolic.query_1p(answer_idx, relations[1])
                    second_level_answer = symbolic.fixed_size_answer(second_level_answer, k)
                    if inner_cache is not None:
                        inner_cache['symbolic'][(answer_idx, relations[1])] = second_level_answer.copy()
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
            if coalition[2] == 1:
                if inner_cache is not None and (second_answer_idx, relations[2]) in inner_cache['cqd']:
                    # print(f"Using cached CQD results for second answer: {second_answer_idx}, relation: {relations[2]}")
                    third_level_answers = inner_cache['cqd'][(second_answer_idx, relations[2])].copy()
                else:
                    current_query = Query('1p', (((second_answer_idx, (relations[2],)),), []))
                    third_level_answers = cqd_query(current_query, k=k, cqd_cache=cqd_cache)
                    if inner_cache is not None:
                        inner_cache['cqd'][(second_answer_idx, relations[2])] = third_level_answers.copy()
            elif coalition[2] == 0:
                if inner_cache is not None and (second_answer_idx, relations[2]) in inner_cache['symbolic']:
                    # print(f"Using cached symbolic results for second answer: {second_answer_idx}, relation: {relations[2]}")
                    third_level_answers = inner_cache['symbolic'][(second_answer_idx, relations[2])].copy()
                else:
                    third_level_answers = symbolic.query_1p(second_answer_idx, relations[2])
                    third_level_answers = symbolic.fixed_size_answer(third_level_answers, k)
                    if inner_cache is not None:
                        inner_cache['symbolic'][(second_answer_idx, relations[2])] = third_level_answers.copy()
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
        if logging:
            print(f"Time taken for second level query: {time_end - start_time:.2f} seconds")
            
    else:
        raise ValueError(f"Unsupported query type: {query.query_type}. Only '2p' queries are supported.")
    final_answers = final_answers.sort_values(by='score', ascending=False)

    # if we have duplicate answers, we need to keep only the one with the highest score
    # as the final_answers dataframe is already sorted by score, we can just keep the first occurrence of each index which means that we keep the highest score
    final_answers = final_answers[~final_answers.index.duplicated(keep='first')]

    # the output should be a dataframe of scores for each possible node in the graph
    df = pd.DataFrame(index=dataset.id2node.keys(), columns=['score', 'path'])
    df['score'] = 0.0
    for answer in final_answers.index:
        df.loc[answer, 'score'] = final_answers.loc[answer, 'score']
        if 'path' in final_answers.columns:
            # if the path column exists, we can add it to the dataframe
            df.loc[answer, 'path'] = final_answers.loc[answer, 'path']
        elif 'path_1' in final_answers.columns and 'path_2' in final_answers.columns:
            # if the path_1 and path_2 columns exist, we can concatenate them to create the path
            df.loc[answer, 'path'] = str(final_answers.loc[answer, 'path_1']) + "\n" + str(final_answers.loc[answer, 'path_2'])
        else:
            df.loc[answer, 'path'] = None
    # shuffle the data to have a random order of answers and a fair measurement of performance
    # df = df.sample(frac=1)
    # sort by index and score at the same time to make sure that the order is consistent
    df.index.name = 'entity_id'
    df = df.sort_values(by=['score', 'entity_id'], ascending=[False, True])
    return df