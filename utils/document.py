#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Jinfen Li
# created_at: 10/26/2016 下午9:06

from utils.token import Token


class Doc(object):
    """ Build one doc instance from *.merge file
    """

    def __init__(self):
        """
        """
        self.token_dict = None
        self.edu_dict = None
        self.rel_paris = None
        # self.data_file = None

    def read_from_fmerge(self, lines):
        """ Read information from the merge file, and create an Doc instance
        :type fmerge: string
        :param fmerge: merge file name
        """
        # self.data_file = data_file
        # if not isfile(data_file):
        #     raise IOError("File doesn't exist: {}".format(data_file))
        gidx, self.token_dict = 0, {}
        # with open(data_file, 'rb') as fin:
        for line in lines:
            line = line.strip()
            if len(line) == 0:
                continue
            tok = self._parse_file_line(line)
            self.token_dict[gidx] = tok
            gidx += 1
        # Get EDUs from tokendict
        self.edu_dict = self._recover_edus(self.token_dict)


    def init_from_tokens(self, token_list):
        self.token_dict = {idx: token for idx, token in enumerate(token_list)}
        self.edu_dict = self._recover_edus(self.token_dict)

    @staticmethod
    def _parse_file_line(line):
        """ Parse one line from *.merge file
        """
        items = line.split("\t")
        tok = Token()
        tok.pidx, tok.sidx, tok.tidx = int(items[-1]), int(items[0]), int(items[1])
        # Without changing the case
        tok.word, tok.lemma = items[2], items[3]
        tok.pos = items[4]
        tok.dep_label = items[5]
        try:
            tok.hidx = int(items[6])
        except ValueError:
            pass
        tok.ner, tok.partial_parse = items[7], items[8]
        try:
            tok.eduidx = int(items[9])
        except ValueError:
            print("EDU index for {} is missing in fmerge file".format(tok.word))
            # sys.exit()
            pass
        return tok


    @staticmethod
    def _recover_edus(token_dict):
        """ Recover EDUs from token_dict
        """
        N, edu_dict = len(token_dict), {}
        for gidx in range(N):
            token = token_dict[gidx]
            eidx = token.eduidx
            try:
                val = edu_dict[eidx]
                edu_dict[eidx].append(gidx)
            except KeyError:
                edu_dict[eidx] = [gidx]
        return edu_dict


