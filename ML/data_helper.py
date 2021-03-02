#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Yizhong
# created_at: 16-11-28 下午2:47
import gzip
import os
import pickle
from collections import defaultdict

import numpy as np
from features.selection import FeatureSelector
from models.tree import RstTree
from utils.other import vectorize


class DataHelper(object):
    def __init__(self, max_action_feat_num=-1, max_relation_feat_num=-1,
                 min_action_feat_occur=1, min_relation_feat_occur=1, brown_clusters=None):
        # number of features, feature selection will be triggered if feature num is larger than this
        self.max_action_feat_num = max_action_feat_num
        self.min_action_feat_occur = min_action_feat_occur
        self.max_relation_feat_num = max_relation_feat_num
        self.min_relation_feat_occur = min_relation_feat_occur
        self.brown_clusters = brown_clusters
        self.action_feat_template = {}
        self.relation_feat_template_level_0 = {}
        self.relation_feat_template_level_1 = {}
        self.relation_feat_template_level_2 = {}
        self.action_map, self.relation_map = {}, {}
        self.action_cnt, self.relation_cnt = {}, {}
        # train rst trees
        self.rst_tree_instances = []

    def create_data_helper(self, data_dir, output_dir, parse_type, isFlat):
        # read train data
        self.rst_tree_instances = self.read_rst_trees(data_dir=data_dir, output_dir=output_dir, parse_type=parse_type, isFlat=isFlat)
        print("extracting action samples ... ")
        action_samples = [sample for rst_tree in self.rst_tree_instances for sample in
                          rst_tree.generate_action_samples(self.brown_clusters)]
        print("extracting relation samples ... ")
        relation_samples_level_0 = [sample for rst_tree in self.rst_tree_instances for sample in
                                    rst_tree.generate_relation_samples(self.brown_clusters, level=0)]
        relation_samples_level_1 = [sample for rst_tree in self.rst_tree_instances for sample in
                                    rst_tree.generate_relation_samples(self.brown_clusters, level=1)]
        relation_samples_level_2 = [sample for rst_tree in self.rst_tree_instances for sample in
                                    rst_tree.generate_relation_samples(self.brown_clusters, level=2)]
        print("building action map ... ")
        self._build_action_map(action_samples)
        print("building relation map ... ")
        self._build_relation_map(relation_samples_level_0 + relation_samples_level_1 + relation_samples_level_2)
        print("building action feature map ... ")
        self.action_feat_template = self._build_action_feat_template(action_samples, topn=self.max_action_feat_num,
                                                                     thresh=self.min_action_feat_occur)
        print("building relation feature map ... ")
        self.relation_feat_template_level_0 = self._build_relation_feat_template(relation_samples_level_0, level=0,
                                                                                 topn=self.max_relation_feat_num,
                                                                                 thresh=self.min_relation_feat_occur)
        self.relation_feat_template_level_1 = self._build_relation_feat_template(relation_samples_level_1, level=1,
                                                                                 topn=self.max_relation_feat_num,
                                                                                 thresh=self.min_relation_feat_occur)
        self.relation_feat_template_level_2 = self._build_relation_feat_template(relation_samples_level_2, level=2,
                                                                                 topn=self.max_relation_feat_num,
                                                                                 thresh=self.min_relation_feat_occur)

    def save_data_helper(self, fname):
        print('Save data helper...')
        data_info = {
            'action_feat_template': self.action_feat_template,
            'relation_feat_template_level_0': self.relation_feat_template_level_0,
            'relation_feat_template_level_1': self.relation_feat_template_level_1,
            'relation_feat_template_level_2': self.relation_feat_template_level_2,
            'action_map': self.action_map,
            'relation_map': self.relation_map
        }
        with open(fname, 'wb') as fout:
            pickle.dump(data_info, fout)

    def load_data_helper(self, fname):
        print('Load data helper ...')
        with open(fname, 'rb') as fin:
            data_info = pickle.load(fin)
        self.action_feat_template = data_info['action_feat_template']
        self.relation_feat_template_level_0 = data_info['relation_feat_template_level_0']
        self.relation_feat_template_level_1 = data_info['relation_feat_template_level_1']
        self.relation_feat_template_level_2 = data_info['relation_feat_template_level_2']
        self.action_map = data_info['action_map']
        self.relation_map = data_info['relation_map']

    def load_train_data(self, data_dir, output_dir, parse_type, isFlat):
        self.rst_tree_instances = self.read_rst_trees(data_dir=data_dir, output_dir=output_dir, parse_type=parse_type, isFlat=isFlat)

    def gen_action_train_data(self):
        tree=[]
        for rst_tree in self.rst_tree_instances:
            print("gen_action_train_data ...")
            for feats, dependency_feats, action in rst_tree.generate_action_samples(self.brown_clusters):
                yield vectorize(feats, self.action_feat_template),  self.action_map[action],dependency_feats[0]
            tree.append(rst_tree)
        # with open('../data/model/action/tree.bin','wb') as file:
        #     pickle.dump(tree,file)

    def gen_relation_train_data(self, level):
        # trees = pickle.load(open('../data/model/action/tree.bin','rb'))
        for rst_tree in self.rst_tree_instances:
            print("gen_relation_train_data ...")
        # for rst_tree in trees:
            for feats, dependency_feat, relation in rst_tree.generate_relation_samples(self.brown_clusters, level):
                if level == 0:

                    yield vectorize(feats, self.relation_feat_template_level_0), self.relation_map[relation],dependency_feat[0]
                if level == 1:
                    yield vectorize(feats, self.relation_feat_template_level_1), self.relation_map[relation],dependency_feat[0]
                if level == 2:
                    yield vectorize(feats, self.relation_feat_template_level_2), self.relation_map[relation],dependency_feat[0]

    def _build_action_feat_template(self, action_samples, topn=-1, thresh=1):
        action_feat_template = {}
        action_feat_counts = {}
        for feats, dependency, action in action_samples:
            for feat in feats:
                try:
                    fidx = action_feat_template[feat]
                except KeyError:
                    action_feat_counts[feat] = defaultdict(float)
                    nfeats = len(action_feat_template)
                    action_feat_template[feat] = nfeats
                action_feat_counts[feat][action] += 1.0
        # print(self.action_map)
        if 0 < topn < len(action_feat_template) or self.min_action_feat_occur > 1:
            # Construct freq_table
            nrows, ncols = len(action_feat_counts), len(self.action_map)
            freq_table = np.zeros((nrows, ncols))
            for (feat, nrow) in action_feat_template.items():
                for (action, ncol) in self.action_map.items():
                    freq_table[nrow, ncol] = action_feat_counts[feat][action]
            # Feature selection
            fs = FeatureSelector(topn=topn, thresh=thresh, method='frequency')
            print('Original action_feat_template size: {}'.format(len(action_feat_template)))
            action_feat_template = fs.select(action_feat_template, freq_table)
            print('After feature selection, action_feat_template size: {}'.format(len(self.action_feat_template)))
        else:
            print('Action_feat_template size: {}'.format(len(action_feat_template)))
        return action_feat_template

    def _build_relation_feat_template(self, relation_samples, level, topn=-1, thresh=1):
        relation_feat_template = {}
        relation_feat_counts = {}
        for feats, dependency, relation in relation_samples:
            for feat in feats:
                try:
                    fidx = relation_feat_template[feat]
                except KeyError:
                    relation_feat_counts[feat] = defaultdict(float)
                    nfeats = len(relation_feat_template)
                    relation_feat_template[feat] = nfeats
                relation_feat_counts[feat][relation] += 1.0
        if 0 < topn < len(relation_feat_template) or self.min_relation_feat_occur > 1:
            # Construct freq_table
            nrows, ncols = len(relation_feat_counts), len(self.relation_map)
            freq_table = np.zeros((nrows, ncols))
            for (feat, nrow) in relation_feat_template.items():
                for (relation, ncol) in self.relation_map.items():
                    freq_table[nrow, ncol] = relation_feat_counts[feat][relation]
            # Feature selection
            fs = FeatureSelector(topn=topn, thresh=thresh, method='frequency')
            print('Original relation_feat_template size at level {}: {}'.format(level, len(relation_feat_template)))
            relation_feat_template = fs.select(relation_feat_template, freq_table)
            print('After feature selection, relation_feat_template size at level {}: {}'.format(level, len(
                relation_feat_template)))
        else:
            print('Relation_feat_template size at level {}: {}'.format(level, len(relation_feat_template)))
        return relation_feat_template

    def _build_action_map(self, action_samples):
        for  _, _,  action in action_samples:
            try:
                aidx = self.action_map[action]
                self.action_cnt[action] += 1
            except KeyError:
                naction = len(self.action_map)
                self.action_map[action] = naction
                self.action_cnt[action] = 1
        print('{} types of actions: {}'.format(len(self.action_map), self.action_map.keys()))
        for action, cnt in self.action_cnt.items():
            print('{}\t{}'.format(action, cnt))

    def _build_relation_map(self, relation_samples):

        for feats, dependency, relation in relation_samples:
            try:
                ridx = self.relation_map[relation]
                self.relation_cnt[relation] += 1
            except KeyError:
                nrelation = len(self.relation_map)
                self.relation_map[relation] = nrelation
                self.relation_cnt[relation] = 1
        print('{} types of relations: {}'.format(len(self.relation_map), self.relation_map.keys()))
        for relation, cnt in self.relation_cnt.items():
            print('{}\t{}'.format(relation, cnt))

    @staticmethod
    def save_feature_template(feature_template, fname):
        if not fname.endswith('.gz'):
            fname += '.gz'
        with gzip.open(fname, 'wb') as fout:
            pickle.dump(feature_template, fout)
        print('Save feature template into file: {}'.format(fname))

    @staticmethod
    def save_map(map, fname):
        if not fname.endswith('.gz'):
            fname += '.gz'
        with gzip.open(fname, 'wb') as fout:
            pickle.dump(map, fout)
        print('Save map into file: {}'.format(fname))

    @staticmethod
    def read_rst_trees(data_dir, output_dir, parse_type, isFlat):
        # Read RST tree file
        files = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if fname.endswith('.dis')]
        with open(os.path.join(output_dir, parse_type, "processed_data.p"), 'rb') as file:
            fmerges = pickle.load(file)
        rst_trees = []
        for fdis, fmerge in zip(files, fmerges):
            print("building tree " + fdis)
            rst_tree = RstTree(fdis, fmerge, isFlat)
            rst_tree.build()
            rst_trees.append(rst_tree)
        return rst_trees
