#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Yizhong
# created_at: 10/27/2016 下午8:04
import os

from eval.metrics import Metrics
from ML.models.parser import RstParser
from ML.models.tree import RstTree
from utils.document import Doc
import pickle
from utils import other
import collections

class Evaluator(object):
    def __init__(self, isFlat, model_dir='../data/model'):
        print('Load parsing models ...')
        self.parser = RstParser()
        self.parser.load(model_dir)
        self.isFlat = isFlat

    def parse(self, doc):
        """ Parse one document using the given parsing models"""
        pred_rst = self.parser.sr_parse(doc, isFlat=self.isFlat)
        return pred_rst

    @staticmethod
    def writebrackets(fname, brackets):
        """ Write the bracketing results into file"""
        # print('Writing parsing results into file: {}'.format(fname))
        with open(fname, 'w') as fout:
            for item in brackets:
                fout.write(str(item) + '\n')

    def eval_parser(self, data_dir, output_dir='./examples', report=False, bcvocab=None, draw=True, isFlat=False):
        """ Test the parsing performance"""
        # Evaluation
        met = Metrics(levels=['span', 'nuclearity', 'relation'])
        # ----------------------------------------
        # Read all files from the given path
        with open(os.path.join(output_dir, "Treebank/TEST", "processed_data.p"), 'rb') as file:
            doclist = pickle.load(file)
        fnames = [fn for fn in os.listdir(data_dir) if fn.endswith(".out")]

        pred_forms = []
        gold_forms = []
        depth_per_relation = {}
        for lines, fname in zip(doclist, fnames):
            # ----------------------------------------
            # Read *.merge file
            doc = Doc()
            doc.read_from_fmerge(lines)
            fout = os.path.join(data_dir, fname)
            print(fout)
            # ----------------------------------------
            # Parsing
            print("************************ predict rst ************************")
            pred_rst = self.parser.sr_parse(doc, self.isFlat,bcvocab)
            if draw:
                pred_rst.draw_rst(fout+'.ps')
            # Get brackets from parsing results
            pred_brackets = pred_rst.bracketing(self.isFlat)
            fbrackets = fout+'.brackets'
            # Write brackets into file
            Evaluator.writebrackets(fbrackets, pred_brackets)
            # ----------------------------------------
            # Evaluate with gold RST tree
            if report:
                print("************************ gold rst ************************")
                fdis = fout+'.dis'
                gold_rst = RstTree(fdis, lines, isFlat)
                gold_rst.build()
                met.eval(gold_rst, pred_rst, self.isFlat)
                if isFlat:
                    for node in pred_rst.postorder_flat_DFT(pred_rst.tree, []):
                        pred_forms.append(node.form)
                    for node in gold_rst.postorder_flat_DFT(gold_rst.tree, []):
                        gold_forms.append(node.form)
                    nodes = gold_rst.postorder_flat_DFT(gold_rst.tree, [])
                else:
                    for node in pred_rst.postorder_DFT(pred_rst.tree, []):
                        pred_forms.append(node.form)
                    for node in gold_rst.postorder_DFT(gold_rst.tree, []):
                        gold_forms.append(node.form)
                    nodes = gold_rst.postorder_DFT(gold_rst.tree, [])
                inner_nodes = [node for node in nodes if node.lnode is not None and node.rnode is not None]
                for idx, node in enumerate(inner_nodes):
                    relation = node.rnode.relation if node.form == 'NS' else node.lnode.relation
                    rela_class = RstTree.extract_relation(relation)
                    if rela_class in depth_per_relation:
                        depth_per_relation[rela_class].append(node.depth)
                    else:
                        depth_per_relation[rela_class] = [node.depth]


        if report:
            met.report()


    def pred_parser(self, output_dir='./examples', parse_type=None, bcvocab=None, draw=True):
        """ Test the parsing performance"""
        # Evaluation
        # met = Metrics(levels=['span', 'nuclearity', 'relation'])
        # ----------------------------------------
        # Read all files from the given path
        with open(os.path.join(output_dir, parse_type, "processed_data.p"), 'rb') as file:
            doclist = pickle.load(file)
        relations = list(other.class2rel.keys())
        results = []
        for lines in doclist:
            # ----------------------------------------
            # Read *.merge file
            doc = Doc()
            relation_d = {rel:0.0 for rel in relations}
            if len(lines) >= 2:
                doc.read_from_fmerge(lines)
                # ----------------------------------------
                # Parsing
                pred_rst = self.parser.sr_parse(doc, bcvocab)

                # if draw:
                #     pred_rst.draw_rst(fmerge.replace(".merge", ".ps"))
                # Get brackets from parsing results

                pred_brackets = pred_rst.bracketing(self.isFlat)
                for brack in pred_brackets:
                    relation_d[brack[2]] +=1
            if sum(relation_d.values()):
                relation_d = {k: str(v) +"/"+ str(sum(relation_d.values())) for k, v in relation_d.items()}
            print(relation_d)
            results.append(relation_d)
        with open(os.path.join(output_dir, parse_type,"result.p"), 'wb') as file:
            pickle.dump(results, file)


