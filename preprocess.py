#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Yizhong
# created_at: 16-11-28 下午2:48

import argparse
import os
from utils.xmlreader import combine, parseXMLString
from stanfordnlp.server import CoreNLPClient
import pickle
import pandas as pd
import re


def join_edus(fedu):
    with open(fedu, 'r') as fin:
        lines = ' '.join([l.strip() for l in fin if l.strip()])
        return lines


def extract(XMLString):
    sent_list, const_list = parseXMLString(XMLString)
    sent_list = combine(sent_list, const_list)
    return sent_list


def merge_treebank(XMLString, fname):
    sent_list = extract(XMLString)
    fedu = fname
    fpara = fname.replace('.edus', '')
    with open(fedu, 'r') as fin1, open(fpara, 'r') as fin2:
        edus = [l.strip() for l in fin1 if l.strip()]
        paras = []
        para_cache = ''
        for line in fin2:
            if line.strip():
                para_cache += line.strip() + ' '
            else:
                paras.append(para_cache.strip())
                para_cache = ''
        if para_cache:
            paras.append(para_cache)
        edu_idx = 0
        para_idx = 0
        cur_edu_offset = len(edus[edu_idx]) - 1 + 1  # plus 1 for one blank space
        edu_cache = ''
        lines = []
        for sent in sent_list:
            for token in sent.tokenlist:

                token_end_offset = token.end_offset
                lines.append(str(sent.idx) + '\t' + str(token.idx) + '\t' + token.word + '\t' + token.lemma \
                       + '\t' + str(token.pos) + '\t' + str(token.deptype) + '\t' + str(token.headidx) \
                       + '\t' + str(token.nertype) + '\t' + str(token.partialparse)+'\t'+str(edu_idx + 1)+'\t'+str(para_idx + 1))

                if token_end_offset == cur_edu_offset:
                    edu_cache += edus[edu_idx] + ' '
                    if len(edu_cache) == len(paras[para_idx]) + 1:
                        edu_cache = ''
                        para_idx += 1
                    edu_idx += 1
                    if edu_idx < len(edus):
                        cur_edu_offset += len(edus[edu_idx]) + 1
                elif token_end_offset > cur_edu_offset:
                    print("Error while merging token \"{}\" in file {} with edu : {}.".format(token.word, fname,
                                                                                              edus[edu_idx]))
                    edu_idx += 1
                    if edu_idx < len(edus):
                        cur_edu_offset += len(edus[edu_idx]) + 1

        return lines


def merge(XMLString):
    sent_list = extract(XMLString)
    pre_end = -1
    paragraph_id = 1
    edu_par_d = {}
    lines = []
    for sent in sent_list:
        for token in sent.tokenlist:

            if not (
                    token.begin_offset == pre_end + 1 or token.begin_offset == pre_end ):
                paragraph_id += 1
            if sent.idx not in edu_par_d:
                edu_par_d[sent.idx] = paragraph_id

            line = str(sent.idx) + '\t' + str(token.idx) + '\t' + token.word + '\t' + token.lemma \
                   + '\t' + str(token.pos) + '\t' + str(token.deptype) + '\t' + str(token.headidx) \
                   + '\t' + str(token.nertype) + '\t' + str(token.partialparse) + '\t' \
                   + str(sent.idx + 1) + '\t' + str(edu_par_d[sent.idx]) + '\n'
            pre_end = token.end_offset
            lines.append(line)
    return lines


def preprocess_data(args):

    os.environ['CORENLP_HOME'] = args.corenlp_dir
    texts = []
    if "Treebank" in args.parse_type:
        print('Join the separated edus in *.edus file into *.text file with a single line...')
        texts = [(join_edus(fedu),fname) for fedu, fname in [(os.path.join(args.data_dir, fname),fname) for fname in os.listdir(args.data_dir) if fname.endswith('.edus')]]
    elif args.parse_type == "Wiki":
        data = pd.read_excel(os.path.join(args.data_dir, "Wikipedia_afd_persuasive.xlsx"))
        texts = [(text, '') for text in data['rationale'].values]
    file_list = []
    corenlp_list = []
    save_path = os.path.join(args.output_dir, args.parse_type)
    if "Treebank" in args.parse_type:
        save_path = os.path.join(args.output_dir, args.parse_type, args.data_dir.split("/")[2])
    if not os.path.exists(os.path.join(args.output_dir, args.parse_type, "corenlp_data.p")) or not os.path.getsize(os.path.join(args.output_dir, args.parse_type, "corenlp_data.p")):
        with CoreNLPClient(annotators=['tokenize', 'ssplit', 'pos', 'lemma', 'ner', 'parse'], timeout=30000, memory='16G',
                           output_format='xml') as client:
            for text, fname in texts:
                print(text)
                if text and not pd.isna(text):
                    if args.parse_type == "Wiki":
                        regular = re.compile(r"\[http.*]")
                        stop = ["Keep ", "*<sKeep or </s", "Delete - ", "Weak Keep - ", "<s>Delete</s -", "Keep - ", "Delete ",
                                "Keep - <s>", "Keep, ", "Strong Keep - ", "Keep both - ",
                                "Keep per [[WP:NEXIST]]. ", "Keep per [[WP:SUSTAINED]]. ", "Keep,", "*<sWeak Keep.",
                                "*<sDelete", "Keep<br>", "<sKeep", "Keep&mdash;", "Delete, ",
                                "*<sDelete: ", "delete ", "*<sKeep. ", "**<delDelete. ", "::<sKeep ", "Keep--", "Keep - <s>"]
                        for s in stop:
                            text = text.replace(s, "")
                        re_list = re.findall(regular, text)
                        for r in re_list:
                            text = text.replace(r, "link")

                    ann = client.annotate(text)
                else:
                    ann = ''
                corenlp_list.append((ann, fname))

        with open(os.path.join(save_path, "corenlp_data.p"), 'wb') as file:
            pickle.dump(corenlp_list, file)

    with open(os.path.join(save_path, "corenlp_data.p"), 'rb') as file:
        corenlp_list = pickle.load(file)
        for ann, fname in corenlp_list:
            # print(ann)
            if "Treebank" in args.parse_type:
                lines = merge_treebank(ann, os.path.join(args.data_dir, fname))
            elif args.parse_type == "Wiki":
                if not ann:
                    lines = []
                else:
                    lines = merge(ann)
            file_list.append(lines)

    with open(os.path.join(save_path, "processed_data.p"), 'wb') as file:
        pickle.dump(file_list, file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', default="data/Wiki", help='path to data directory.')
    parser.add_argument('--output_dir', default="ML/saved_model", help='path to data directory.')
    parser.add_argument('--corenlp_dir', default='D:\\workspace\\stageDP\\stanford-corenlp',
                        help='path to Stanford Corenlp directory.')
    parser.add_argument('--parse_type', default='Wiki',
                        help='whether parse treebank or normal data')
    args = parser.parse_args()
    preprocess_data(args)
