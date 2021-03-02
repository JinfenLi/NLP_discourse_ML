#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Jinfen Li


import argparse
import gzip
import pickle
import numpy as np
import os
import scipy
from data_helper import DataHelper
from eval.evaluation import Evaluator
from models.classifiers import ActionClassifier, RelationClassifier
from models.parser import RstParser
import preprocess


def train_model(data_helper):

    action_clf = ActionClassifier(data_helper.action_feat_template, data_helper.action_map)
    relation_clf = RelationClassifier(data_helper.relation_feat_template_level_0,
                                      data_helper.relation_feat_template_level_1,
                                      data_helper.relation_feat_template_level_2,
                                      data_helper.relation_map)
    rst_parser = RstParser(action_clf, relation_clf)
    print('extracting action feature')

    action_fvs, action_labels,dependency_feats = list(zip(*data_helper.gen_action_train_data()))
    # action_fvs, bert_feats, action_labels, dependency_feats = list(zip(*data_helper.gen_action_train_data()))
    # with open('saved_model/feat.bin','wb') as file:
    #     pickle.dump((action_fvs, action_labels,dependency_feats),file)
    # with open('saved_model/feat.bin','rb') as file:
    #     action_fvs, action_labels,dependency_feats = pickle.load(file)

    # rst_parser.action_clf.train(np.concatenate([scipy.sparse.vstack(action_fvs).toarray(),dependency_feats],axis=1), action_labels)
    rst_parser.action_clf.train(
        np.concatenate([scipy.sparse.vstack(action_fvs).toarray(), dependency_feats], axis=1),
        action_labels)
    rst_parser.save(model_dir='saved_model')
    rst_parser.load(model_dir='saved_model')

    # train relation classifier
    for level in [0, 1,2]:
        relation_fvs, relation_labels, dependency_feat = list(
            zip(*data_helper.gen_relation_train_data(level)))
        print(np.concatenate(
            [scipy.sparse.vstack(relation_fvs).toarray(), np.array(dependency_feat)],
            axis=1).shape)

        print('{} relation samples at level {}.'.format(len(relation_labels), level))
        # rst_parser.relation_clf.train(scipy.sparse.vstack(relation_fvs), relation_labels, level)
        rst_parser.relation_clf.train(
            np.concatenate(
                [scipy.sparse.vstack(relation_fvs).toarray(), np.array(dependency_feat)], axis=1),
            relation_labels, level)
    rst_parser.save(model_dir='saved_model')



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default="../data/TEST", help='path to data directory.')
    parser.add_argument('--output_dir', default="saved_model", help='path to data directory.')
    parser.add_argument('--corenlp_dir', default='D:\\workspace\\stageDP\\stanford-corenlp',
                        help='path to Stanford Corenlp directory.')
    parser.add_argument('--parse_type', default='Treebank',
                        help='whether parse treebank or normal data')
    parser.add_argument('--isFlat', default= False,
                        help='use RN~ if isFlat')
    parser.add_argument('--preprocess', default=False,
                        help='preprocess data')
    parser.add_argument('--prepare', default=False,
                        help='whether to extract feature templates, action maps and relation maps')
    parser.add_argument('--train', default=False,
                        help='whether to train new models')
    parser.add_argument('--eval', default=True,
                        help='whether to do evaluation')
    parser.add_argument('--pred', default=False,
                        help='whether to do prediction')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    # Use brown clusters
    with gzip.open("features/bc3200.pickle.gz") as fin:
        print('Load Brown clusters for creating features ...')
        brown_clusters = pickle.load(fin)
    # print(brown_clusters)
    data_helper = DataHelper(max_action_feat_num=330000, max_relation_feat_num=300000,
                             min_action_feat_occur=1, min_relation_feat_occur=1,
                             brown_clusters=brown_clusters)
    if args.preprocess:
        preprocess.preprocess_data(args)
    if args.prepare:
        # Create training data
        data_helper.create_data_helper(data_dir=args.data_dir, output_dir=args.output_dir, parse_type=args.parse_type, isFlat=args.isFlat)
        data_helper.save_data_helper(os.path.join(args.output_dir, args.parse_type,"TRAINING", "data_helper.bin"))
    if args.train:
        data_helper.load_data_helper(os.path.join(args.output_dir, args.parse_type, "TRAINING", "data_helper.bin"))
        data_helper.load_train_data(data_dir=args.data_dir, output_dir=args.output_dir, parse_type=args.parse_type, isFlat=args.isFlat)
        train_model(data_helper)
    if args.eval:
        # Evaluate models on the RST-DT test set
        if args.isFlat:
            evaluator = Evaluator(isFlat=args.isFlat, model_dir=os.path.join(args.output_dir, "RN~model"))
        else:
            evaluator = Evaluator(isFlat=args.isFlat, model_dir=os.path.join(args.output_dir, "N~model"))
        evaluator.eval_parser(data_dir=args.data_dir, output_dir=args.output_dir, report=True, bcvocab=brown_clusters, draw=False, isFlat=args.isFlat)
    
    if args.pred:
        if args.isFlat:
            evaluator = Evaluator(isFlat=args.isFlat, model_dir=os.path.join(args.output_dir, "RN~model"))
        else:
            evaluator = Evaluator(isFlat=args.isFlat, model_dir=os.path.join(args.output_dir, "N~model"))
        print("predicting")
        evaluator.pred_parser(output_dir=args.output_dir, parse_type=args.parse_type, bcvocab=brown_clusters, draw=True)
