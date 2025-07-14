import pickle
from tqdm import tqdm
from kbc.chain_dataset import Chain, ChaineDataset
import argparse
import torch
from kbc.cqd_co_xcqa import main
import pandas as pd
import json
from query import Query

def create_cqd_file(queries: list, output_file: str = 'data/FB15k-237/FB15k-237_test_hard_sample.pkl'):
    """
    Create a file in CQD desired format. The CQD will make its predictions based on the queries provided in this file.
    Args:
        queries (list): List of Query objects.
        output_file (str): Path to the output CQD file.
    """
    
    query_chains = []
    for query in tqdm(queries, desc="Creating CQD file"):
        query_chain = Chain()
        # The last element in the query is the target, which is not important for our work, as it will be used only for metrics calculation.
        query_chain.data['raw_chain'] = [query.get_query()[0][0], query.get_query()[0][1][0], [0]]
        query_chain.data['anchors'] = [query.get_query()[0][0]]
        query_chain.data['optimisable'] = [-1, 0]
        query_chain.data['targets'] = [0]
        query_chains.append(query_chain)
        
    chains = ChaineDataset(None)
    chains.type1_1chain = query_chains
    
    with open(output_file, 'wb') as f:
        pickle.dump(chains, f)
        

def get_cache_prediction(predictions_df: pd.DataFrame, entity: int, relation: int):
    """
    Get the prediction for a specific entity and relation from the predictions DataFrame.
    
    Args:
        predictions_df (pd.DataFrame): DataFrame containing the predictions.
        entity (int): The ID of the entity to get the prediction for.
        relation (int): The ID of the relation to get the prediction for.
        
    Returns:
        pd.DataFrame: A DataFrame containing the predicted entities and their scores, sorted by score in descending order.
    """
    filtered_df = predictions_df[(predictions_df['entity_id'] == entity) & (predictions_df['relation_id'] == relation)]
    if filtered_df.empty:
        return [], []
    
    predicted_entities = filtered_df['top_k_entities'].tolist()[0]
    scores = filtered_df['top_k_scores'].tolist()[0]
    
    # make a df which index is predicted_entities and column is scores
    predictions = pd.DataFrame(scores, index=predicted_entities, columns=['score'])
    predictions = predictions.sort_values(by='score', ascending=False)
    
    return predictions


def cqd_query(query: Query, sample_path: str = 'data/FB15k-237/FB15k-237_test_hard_sample.pkl', result_path: str = 'scores.json', k: int = 5, cqd_cache: pd.DataFrame = None):
    
    if cqd_cache is not None:
        # If a cache DataFrame is provided, use it to get the predictions
        entity = query.get_query()[0][0]
        relation = query.get_query()[0][1][0]
        
        if entity is None or relation is None:
            raise ValueError("Entity or relation not found in the dataset.")
        
        # Get cached predictions
        predictions = get_cache_prediction(cqd_cache, entity, relation)
        
        # Get the top k answers
        top_k_answers = predictions.head(k)
        
        return top_k_answers
    
    else:

        # Create a CQD file with the query
        create_cqd_file([query], output_file=sample_path)

        # Set up the arguments for the CQD model (cqd_co_xcqa)
        args = argparse.Namespace(
            path = 'FB15k-237',
            sample_path = sample_path,
            model_path = 'models/FB15k-237-model-rank-1000-epoch-100-1602508358.pt',
            dataset = 'FB15k-237',
            mode = 'test',
            chain_type = '1_1', # '1_1', '1_2', '2_2', '2_2_disj', '1_3', '2_3', '3_3', '4_3', '4_3_disj', '1_3_joint'
            t_norm = 'prod', # 'min', 'prod'
            reg = None,
            lr = 0.1,
            optimizer='adam', # 'adam', 'adagrad', 'sgd'
            max_steps = 1000,
            sample = True,
            result_path = result_path,
            save_result = True,
            save_k = 5,
        )

        # Run the CQD model
        main(args)

        # Load the scores
        scores = None
        with open(result_path, 'rb') as f:
            scores = json.load(f)

        tmp_df = pd.read_json(result_path)
        tmp_df = pd.DataFrame(tmp_df.loc[0]['top_k_scores'], index=tmp_df.loc[0]['top_k_entities'], columns=['score'])
        tmp_df = tmp_df.sort_values(by='score', ascending=False)
        
        # Get the top k answers
        top_k_answers = tmp_df.head(k)
        
        return top_k_answers