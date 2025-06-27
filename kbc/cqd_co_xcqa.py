#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pickle
import os.path as osp
import json

from tqdm import tqdm
import torch

from kbc.utils import QuerDAG
from kbc.utils import preload_env
from kbc.metrics import evaluation


def score_queries(args):
    mode = args.mode

    dataset = osp.basename(args.path)
    if args.sample:
        data_hard_path = args.sample_path if args.sample_path else osp.join(args.path, f'{dataset}_{mode}_hard_sample.pkl')
    else:
        data_hard_path = osp.join(args.path, f'{dataset}_{mode}_hard.pkl')
    origin_path = osp.dirname(data_hard_path)
    data_complete_path = osp.join(origin_path, f'{dataset}_{mode}_complete.pkl')

    data_hard = pickle.load(open(data_hard_path, 'rb'))
    data_complete = pickle.load(open(data_complete_path, 'rb'))

    # Instantiate singleton KBC object
    preload_env(args.model_path, data_hard, args.chain_type, mode='hard')
    env = preload_env(args.model_path, data_complete, args.chain_type,
                      mode='complete')

    queries = env.keys_hard
    test_ans_hard = env.target_ids_hard
    test_ans = env.target_ids_complete
    chains = env.chains
    kbc = env.kbc

    if args.reg is not None:
        env.kbc.regularizer.weight = args.reg

    disjunctive = args.chain_type in (QuerDAG.TYPE2_2_disj.value,
                                      QuerDAG.TYPE4_3_disj.value)

    if args.chain_type == QuerDAG.TYPE1_1.value:
        # scores = kbc.model.link_prediction(chains)

        s_emb = chains[0][0]
        p_emb = chains[0][1]

        scores_lst = []
        nb_queries = s_emb.shape[0]
        #for i in tqdm(range(nb_queries)):
        if args.sample:
            i = 0
            batch_s_emb = s_emb[i, :].view(1, -1)
            batch_p_emb = p_emb[i, :].view(1, -1)
            batch_chains = [(batch_s_emb, batch_p_emb, None)]
            batch_scores = kbc.model.link_prediction(batch_chains)
            scores_lst += [batch_scores]

            scores = torch.cat(scores_lst, 0)
            # save scores into a text file
            torch.save(scores, args.result_path)
        else:
            for i in tqdm(range(nb_queries)):
                batch_s_emb = s_emb[i, :].view(1, -1)
                batch_p_emb = p_emb[i, :].view(1, -1)
                batch_chains = [(batch_s_emb, batch_p_emb, None)]
                batch_scores = kbc.model.link_prediction(batch_chains)
                scores_lst += [batch_scores]

            scores = torch.cat(scores_lst, 0)

    elif args.chain_type in (QuerDAG.TYPE1_2.value, QuerDAG.TYPE1_3.value):
        scores = kbc.model.optimize_chains(chains, kbc.regularizer,
                                           max_steps=args.max_steps,
                                           lr=args.lr,
                                           optimizer=args.optimizer,
                                           norm_type=args.t_norm)

    elif args.chain_type in (QuerDAG.TYPE2_2.value, QuerDAG.TYPE2_2_disj.value,
                             QuerDAG.TYPE2_3.value):
        scores = kbc.model.optimize_intersections(chains, kbc.regularizer,
                                                  max_steps=args.max_steps,
                                                  lr=args.lr,
                                                  optimizer=args.optimizer,
                                                  norm_type=args.t_norm,
                                                  disjunctive=disjunctive)

    elif args.chain_type == QuerDAG.TYPE3_3.value:
        scores = kbc.model.optimize_3_3(chains, kbc.regularizer,
                                        max_steps=args.max_steps,
                                        lr=args.lr,
                                        optimizer=args.optimizer,
                                        norm_type=args.t_norm)

    elif args.chain_type in (QuerDAG.TYPE4_3.value,
                             QuerDAG.TYPE4_3_disj.value):
        scores = kbc.model.optimize_4_3(chains, kbc.regularizer,
                                        max_steps=args.max_steps,
                                        lr=args.lr,
                                        optimizer=args.optimizer,
                                        norm_type=args.t_norm,
                                        disjunctive=disjunctive)
    else:
        raise ValueError(f'Uknown query type {args.chain_type}')

    return scores, queries, test_ans, test_ans_hard


def main(args):
    # print the current full path
    # print(f'Current path: {osp.abspath(__file__)}')
    scores, queries, test_ans, test_ans_hard = score_queries(args)
