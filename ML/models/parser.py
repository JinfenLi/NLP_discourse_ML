#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Yizhong
# created_at: 10/27/2016 下午7:32
import os
import numpy as np
from ML.features.extraction import ActionFeatureGenerator, RelationFeatureGenerator
from ML.models.classifiers import ActionClassifier, RelationClassifier
from ML.models.state import ParsingState
from ML.models.tree import RstTree


class RstParser(object):
    def __init__(self, action_clf=None, relation_clf=None):
        self.action_clf = action_clf if action_clf is not None else ActionClassifier()
        self.relation_clf = relation_clf if relation_clf is not None else RelationClassifier()

    def save(self, model_dir):
        """Save models
        """
        self.action_clf.save(os.path.join(model_dir, 'model.action.gz'))
        self.relation_clf.save(os.path.join(model_dir, 'model.relation.gz'))

    def load(self, model_dir):
        """ Load models
        """
        self.action_clf.load(os.path.join(model_dir, 'model.action.gz'))
        self.relation_clf.load(os.path.join(model_dir, 'model.relation.gz'))

    def sr_parse(self, doc, isFlat, bcvocab=None):
        """ Shift-reduce RST parsing based on models prediction

        :type doc: Doc
        :param doc: the document instance

        :type bcvocab: dict
        :param bcvocab: brown clusters
        """
        # use transition-based parsing to build tree structure
        conf = ParsingState([], [])
        # initialize queue
        conf.init(doc)
        action_hist = []
        while not conf.end_parsing():
            stack, queue = conf.get_status()
            fg = ActionFeatureGenerator(stack, queue, action_hist, doc, bcvocab)
            action_feat, dependency_feat = fg.gen_features()
            action_probs = self.action_clf.predict_probs(action_feat, np.array(dependency_feat))
            # print(action_feats)
            # print(np.array(dependency_feat).shape)
            # action_probs = self.action_clf.predict_probs(np.concatenate([scipy.sparse.vstack(action_feats).toarray(),dependency_feat],axis=1))
            for action, cur_prob in action_probs:
                if conf.is_action_allowed(action):
                    conf.operate(action)
                    action_hist.append(action)
                    break
        tree = conf.get_parse_tree()
        # RstTree.down_prop(tree)
        # assign the node to rst_tree
        rst_tree = RstTree(isFlat)
        rst_tree.assign_tree(tree)
        rst_tree.assign_doc(doc)
        if isFlat:
            RstTree.down_flat_prop(tree)
            rst_tree.flat_back_prop(tree, doc)
            post_nodelist = RstTree.postorder_flat_DFT(rst_tree.tree, [])
        else:
            RstTree.down_prop(tree)
            rst_tree.back_prop(tree, doc)
            post_nodelist = RstTree.postorder_DFT(rst_tree.tree, [])


        for node in post_nodelist:

            if (node.lnode is not None) and (node.rnode is not None):
                fg = RelationFeatureGenerator(node, rst_tree, node.level, bcvocab)
                relation_feats,dependency_feat= fg.gen_features()
                # relation = self.relation_clf.predict(relation_feats, node.level)
                relation = self.relation_clf.predict(relation_feats,np.array(dependency_feat), node.level)
                # node.assign_relation(relation)
                node.assignRelation = relation
        return rst_tree
