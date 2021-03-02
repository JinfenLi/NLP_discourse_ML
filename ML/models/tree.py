#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Yizhong
# created_at: 10/26/2016 下午8:37

import sys
from os.path import isfile

from ML.features.extraction import ActionFeatureGenerator, RelationFeatureGenerator
from ML.models.state import ParsingState
from nltk import Tree
from nltk.draw import TreeWidget
from nltk.draw.util import CanvasFrame
from utils.document import Doc
from utils.other import rel2class
from utils.span import SpanNode


class RstTree(object):
    count = set()

    def __init__(self, fdis=None, fmerge=None, isFlat=False):
        self.fdis = fdis
        self.fmerge = fmerge
        self.binary = True
        self.tree, self.doc = None, None
        self.isFlat = isFlat

    def assign_tree(self, tree):
        """ Assign a tree instance from external resource
        """
        self.tree = tree

    def assign_doc(self, doc):
        """ Assign a doc instance from external resource
        """
        self.doc = doc

    def build(self):
        """ Build BINARY RST tree
        """
        with open(self.fdis) as fin:
            text = fin.read()
        # Build RST as annotation
        self.tree = RstTree.build_tree(text)
        # Binarize it
        if self.isFlat:
            self.tree = RstTree.flat_tree(self.tree)
        else:
            self.tree = RstTree.binarize_tree(self.tree)
        # Read doc file
        if self.fmerge:
            doc = Doc()
            doc.read_from_fmerge(self.fmerge)
            self.doc = doc
        else:
            raise IOError("File doesn't exist: {}".format(self.fmerge))
        if self.isFlat:
            RstTree.down_flat_prop(self.tree)
            RstTree.flat_back_prop(self.tree, self.doc)
        else:
            RstTree.down_prop(self.tree)
            RstTree.back_prop(self.tree, self.doc)


    def generate_action_samples(self, bcvocab):
        """ Generate action samples from an binary RST tree
        :type bcvocab: dict
        :param bcvocab: brown clusters of words
        """
        # Parsing actions and relations
        if self.isFlat:
            actions, relations = self.decode_flat_rst_tree()
        else:
            actions, relations = self.decode_rst_tree()

        queue = RstTree.get_edu_node(self.tree)
        stack = []
        # Start simulating the shift-reduce parsing
        sr_parser = ParsingState(stack, queue)
        for idx, action in enumerate(actions):
            stack, queue = sr_parser.get_status()
            # Generate features
            fg = ActionFeatureGenerator(stack, queue, actions[:idx], self.doc, bcvocab)
            action_feats, dependency_feats = fg.gen_features()
            # print(action)
            # print(action_feats)

            yield action_feats, dependency_feats, action
            # Change status of stack/queue
            # action and relation are necessary here to avoid change rst_trees
            sr_parser.operate(action)


    def generate_relation_samples(self, bcvocab, level):
        """ Generate relation samples from an binary RST tree
        :type bcvocab: dict
        :param bcvocab: brown clusters of words
        """
        if self.isFlat:
            post_nodelist = RstTree.postorder_flat_DFT(self.tree, [])
        else:
            post_nodelist = RstTree.postorder_DFT(self.tree, [])
        for node in post_nodelist:
            if node.level == level and (node.lnode is not None) and (node.rnode is not None):
                fg = RelationFeatureGenerator(node, self, node.level, bcvocab)
                # print(node.edu_span)
                # print(node.lnode.dependency)
                # print(node.rnode.dependency)
                relation_feats, dependency_feat = fg.gen_features()
                form = node.form
                if (form == 'NN') or (form == 'NS'):
                    relation = RstTree.extract_relation(node.rnode.relation)
                else:
                    relation = RstTree.extract_relation(node.lnode.relation)
                yield relation_feats, dependency_feat, relation

    def decode_rst_tree(self):
        """ Decoding Shift-reduce actions and span relations from an binary RST tree
        """
        # Start decoding
        post_nodelist = RstTree.postorder_DFT(self.tree, [])
        action_list = []
        relation_list = []
        for node in post_nodelist:
            if (node.lnode is None) and (node.rnode is None):
                action_list.append(('Shift', None))
                relation_list.append(None)
            elif (node.lnode is not None) and (node.rnode is not None):
                form = node.form
                if (form == 'NN') or (form == 'NS'):
                    relation = RstTree.extract_relation(node.rnode.relation)
                else:
                    relation = RstTree.extract_relation(node.lnode.relation)
                action_list.append(('Reduce', form))
                relation_list.append(relation)
            else:
                raise ValueError("Can not decode Shift-Reduce action")
        return action_list, relation_list

    def decode_flat_rst_tree(self):
        """ Decoding Shift-reduce actions and span relations from an binary RST tree
        """
        # Start decoding
        post_nodelist = RstTree.postorder_flat_DFT(self.tree, [])
        action_list = []
        relation_list = []
        for node in post_nodelist:
            if not node.nodelist:
                action_list.append(('Shift', None))
                relation_list.append(None)
            elif len(node.nodelist) == 2:
                form = node.form
                if (form == 'NN') or (form == 'NS'):
                    relation = RstTree.extract_relation(node.rnode.relation)
                else:
                    relation = RstTree.extract_relation(node.lnode.relation)
                action_list.append(('Reduce', form))
                relation_list.append(relation)
            elif len(node.nodelist) > 2:
                action_list.extend([('R~', 'N~')] * (len(node.nodelist)-1))
                relation_list.extend([RstTree.extract_relation(node.lnode.relation)] * (len(node.nodelist)-1))
            else:
                raise ValueError("Can not decode Shift-Reduce action")
        return action_list, relation_list

    def convert_node_to_str(self, node, sep=' '):
        text = node.text
        words = [self.doc.token_dict[tidx].word for tidx in text]
        return sep.join(words)

    @staticmethod
    def get_edu_node(tree):
        """ Get all leaf nodes. It can be used for generating training
            examples from gold RST tree

        :type tree: SpanNode instance
        :param tree: an binary RST tree
        """
        # Post-order depth-first traversal
        post_nodelist = RstTree.postorder_DFT(tree, [])
        # EDU list
        edulist = []
        for node in post_nodelist:
            if (node.lnode is None) and (node.rnode is None):
                edulist.append(node)
        return edulist

    @staticmethod
    def build_tree(text):
        """ Build tree from *.dis file

        :type text: string
        :param text: RST tree read from a *.dis file
        """
        tokens = text.strip().replace('//TT_ERR', '').replace('\n', '').replace('(', ' ( ').replace(')', ' ) ').split()
        # print(tokens)

        queue = RstTree.process_text(tokens)
        # print('tokens = {}'.format(queue))
        # print 'queue = {}'.format(queue)
        stack = []
        while queue:
            token = queue.pop(0)
            if token == ')':
                # If ')', start processing
                content = []  # Content in the stack
                while stack:
                    cont = stack.pop()
                    if cont == '(':
                        break
                    else:
                        content.append(cont)
                content.reverse()  # Reverse to the original order
                # Parse according to the first content word
                if len(content) < 2:
                    raise ValueError("content = {}".format(content))
                label = content.pop(0)
                if label == 'Root':
                    node = SpanNode(prop=label)
                    node.create_node(content)
                    stack.append(node)
                elif label == 'Nucleus':
                    node = SpanNode(prop=label)
                    node.create_node(content)
                    stack.append(node)
                elif label == 'Satellite':
                    node = SpanNode(prop=label)
                    node.create_node(content)
                    stack.append(node)
                elif label == 'span':
                    # Merge
                    beginindex = int(content.pop(0))
                    endindex = int(content.pop(0))
                    stack.append(('span', beginindex, endindex))
                elif label == 'leaf':
                    # Merge
                    eduindex = int(content.pop(0))
                    RstTree.check_content(label, content)
                    stack.append(('leaf', eduindex, eduindex))
                elif label == 'rel2par':
                    # Merge
                    relation = content.pop(0)
                    RstTree.check_content(label, content)
                    stack.append(('relation', relation))
                elif label == 'text':
                    # Merge
                    txt = RstTree.create_text(content)
                    stack.append(('text', txt))
                else:
                    raise ValueError(
                        "Unrecognized parsing label: {} \n\twith content = {}\n\tstack={}\n\tqueue={}".format(label,
                                                                                                              content,
                                                                                                              stack,
                                                                                                              queue))
            else:
                # else, keep push into the stack
                stack.append(token)
        # print(stack)
        return stack[-1]

    @staticmethod
    def process_text(tokens):
        """ Preprocessing token list for filtering '(' and ')' in text
        :type tokens: list
        :param tokens: list of tokens
        """
        identifier = '_!'
        within_text = False
        for (idx, tok) in enumerate(tokens):
            if identifier in tok:
                for _ in range(tok.count(identifier)):
                    within_text = not within_text
            if ('(' in tok) and within_text:
                tok = tok.replace('(', '-LB-')
            if (')' in tok) and within_text:
                tok = tok.replace(')', '-RB-')
            tokens[idx] = tok
        return tokens

    @staticmethod
    def create_text(lst):
        """ Create text from a list of tokens

        :type lst: list
        :param lst: list of tokens
        """
        newlst = []
        for item in lst:
            item = item.replace("_!", "")
            newlst.append(item)
        text = ' '.join(newlst)
        # Lower-casing
        return text.lower()

    @staticmethod
    def check_content(label, c):
        """ Check whether the content is legal

        :type label: string
        :param label: parsing label, such 'span', 'leaf'

        :type c: list
        :param c: list of tokens
        """

        if len(c) > 0:
            raise ValueError("{} with content={}".format(label, c))

    @staticmethod
    def binarize_tree(tree):
        """ Convert a general RST tree to a binary RST tree

        :type tree: instance of SpanNode
        :param tree: a general RST tree
        """
        queue = [tree]
        while queue:
            node = queue.pop(0)
            queue += node.nodelist
            # Construct binary tree
            if len(node.nodelist) == 2:
                node.lnode = node.nodelist[0]
                node.rnode = node.nodelist[1]
                # Parent node
                node.lnode.pnode = node
                node.rnode.pnode = node


            elif len(node.nodelist) > 2:
                # Remove one node from the nodelist
                lc = node.nodelist[0].prop
                mark = 1
                for nl in node.nodelist:
                    mark &= nl.visited
                # if not mark:
                # # if not [0^nl.visited for nl in node.nodelist]:
                #     if node.relation:
                #         print(RstTree.extract_relation(node.relation))
                #     print([RstTree.extract_relation(l.relation) for l in node.nodelist])
                if len(set([l.prop for l in node.nodelist]))==1:
                    node.visited = True
                    # for nl in node.nodelist:
                    #     nl.visited = True

                node.lnode = node.nodelist.pop(0)
                newnode = SpanNode(node.nodelist[0].prop)
                newnode.nodelist += node.nodelist
                # Right-branching
                node.rnode = newnode
                # Parent node
                node.lnode.pnode = node
                node.rnode.pnode = node
                if node.visited:
                    newnode.visited = True

                queue.insert(0, newnode)
            # Clear nodelist for the current node
            node.nodelist = []
        return tree

    @staticmethod
    def flat_tree(tree):
        queue = [tree]
        while queue:
            node = queue.pop(0)
            queue += node.nodelist
            # Construct binary tree
            if len(node.nodelist) == 2:
                node.lnode = node.nodelist[0]
                node.rnode = node.nodelist[1]
                # Parent node
                node.lnode.pnode = node
                node.rnode.pnode = node


            elif len(node.nodelist) > 2:

                if len(set([l.prop for l in node.nodelist])) != 1:
                    node.lnode = node.nodelist.pop(0)
                    newnode = SpanNode(node.nodelist[0].prop)
                    newnode.nodelist += node.nodelist
                    # Right-branching
                    node.rnode = newnode
                    # Parent node
                    node.lnode.pnode = node
                    node.rnode.pnode = node

                    queue.insert(0, newnode)
                    # reset nodelist for the current node
                    node.nodelist = [node.lnode, node.rnode]
        return tree

    @staticmethod
    def back_prop(tree, doc):
        """ Starting from leaf node, propagating node
            information back to root node

        :type tree: SpanNode instance
        :param tree: an binary RST tree
        """
        tree_nodes = RstTree.BFTbin(tree)
        tree_nodes.reverse()
        for node in tree_nodes:
            if (node.lnode is not None) and (node.rnode is not None):
                # Non-leaf node
                node.edu_span = RstTree.__getspaninfo(node.lnode, node.rnode)
                node.text = RstTree.__gettextinfo(doc.edu_dict, node.edu_span)
                node.puretext = ' '.join([doc.token_dict[nt].word for nt in node.text])

                if node.relation is None:
                    # If it is a new node created by binarization
                    # if node.prop == 'Root':
                    #     pass
                    # else:
                    node.relation = RstTree.__getrelationinfo(node)
                node.assignRelation = RstTree.__getrelationinfo(node)
                if not node.form:
                    node.form, node.nuc_span, node.nuc_edu = RstTree.__getforminfo(node)
                node.height = max(node.lnode.height, node.rnode.height) + 1
                # print("gold height: ", node.height)
                node.max_depth = max(node.lnode.max_depth, node.rnode.max_depth)
                if node.form == 'NS':
                    node.child_relation = node.rnode.relation
                else:
                    node.child_relation = node.lnode.relation
                if doc.token_dict[node.lnode.text[0]].sidx == doc.token_dict[node.rnode.text[-1]].sidx:
                    node.level = 0
                elif doc.token_dict[node.lnode.text[0]].pidx == doc.token_dict[node.rnode.text[-1]].pidx:
                    node.level = 1
                else:
                    node.level = 2
            elif (node.lnode is None) and (node.rnode is not None):
                raise ValueError("Unexpected left node")
            elif (node.lnode is not None) and (node.rnode is None):
                raise ValueError("Unexpected right node")
            else:
                # Leaf node
                node.text = RstTree.__gettextinfo(doc.edu_dict, node.edu_span)
                node.height = 0
                node.max_depth = node.depth
                node.level = 0
        # print()

    @staticmethod
    def flat_back_prop(tree, doc):
        """ Starting from leaf node, propagating node
            information back to root node

        :type tree: SpanNode instance
        :param tree: an binary RST tree
        """
        tree_nodes = RstTree.flatBFTbin(tree)
        tree_nodes.reverse()
        for node in tree_nodes:
            if node.nodelist:
                node.lnode, node.rnode = node.nodelist[0], node.nodelist[-1]

                node.edu_span = RstTree.__getflatspaninfo(node)
                node.text = RstTree.__gettextinfo(doc.edu_dict, node.edu_span)
                node.puretext = ' '.join([doc.token_dict[nt].word for nt in node.text])
                # if node.relation is None and node.nodelist:
                node.assignRelation = RstTree.__getflatrelationinfo(node)
                if not node.form and node.nodelist:
                    node.form, node.nuc_span, node.nuc_edu = RstTree.__getflatforminfo(node)
                node.height = max([n.height for n in node.nodelist]) + 1
                node.max_depth = max([n.max_depth for n in node.nodelist]) + 1

                if node.form == 'NS':
                    node.child_relation = node.rnode.relation
                else:
                    node.child_relation = node.lnode.relation
                if doc.token_dict[node.lnode.text[0]].sidx == doc.token_dict[node.rnode.text[-1]].sidx:
                    node.level = 0
                elif doc.token_dict[node.lnode.text[0]].pidx == doc.token_dict[node.rnode.text[-1]].pidx:
                    node.level = 1
                else:
                    node.level = 2

            elif len(node.nodelist) == 1:
                raise ValueError("Unexpected nodelist")
            else:
                # Leaf node
                node.text = RstTree.__gettextinfo(doc.edu_dict, node.edu_span)
                node.height = 0
                node.max_depth = node.depth
                node.level = 0
                # node.bert = RstTree.__getbertinfo(doc, node.text)

    @staticmethod
    def down_prop(tree):
        """
        Starting from root node, propagating node information down to leaf nodes
        :param tree: SpanNode instance
        :param doc: Doc instance
        :return: root node
        """
        tree_nodes = RstTree.BFTbin(tree)
        root_node = tree_nodes.pop(0)
        root_node.depth = 0
        for node in tree_nodes:
            assert node.pnode.depth >= 0
            node.depth = node.pnode.depth + 1
            # for node in tree_nodes:
            #     if node.lnode is not None and node.rnode is not None:
            #         node.lnode.pnode = node
            #         node.rnode.pnode = node
            #         if node.form == 'NN':
            #             node.lnode.prop = "Nucleus"
            #             node.lnode.relation = node.child_relation
            #             node.rnode.prop = "Nucleus"
            #             node.rnode.relation = node.child_relation
            #         elif node.form == 'NS':
            #             node.lnode.prop = "Nucleus"
            #             node.lnode.relation = "span"
            #             node.rnode.prop = "Satellite"
            #             node.rnode.relation = node.child_relation
            #         elif node.form == 'SN':
            #             node.lnode.prop = "Satellite"
            #             node.lnode.relation = node.child_relation
            #             node.rnode.prop = "Nucleus"
            #             node.rnode.relation = "span"
            #         else:
            #             raise ValueError("Unrecognized form: {}".format(node.form))

    @staticmethod
    def down_flat_prop(tree):
        """
        Starting from root node, propagating node information down to leaf nodes
        :param tree: SpanNode instance
        :param doc: Doc instance
        :return: root node
        """

        queue = [tree]
        while queue:
            node = queue.pop(0)
            if not node.pnode:
                node.depth = 0
            else:
                node.depth = node.pnode.depth + 1
            queue.extend(node.nodelist)
        return tree

    @staticmethod
    def BFT(tree):
        """ Breadth-first treavsal on general RST tree

        :type tree: SpanNode instance
        :param tree: an general RST tree
        """
        queue = [tree]
        bft_nodelist = []
        while queue:
            node = queue.pop(0)
            bft_nodelist.append(node)
            queue += node.nodelist
        return bft_nodelist

    @staticmethod
    def BFTbin(tree):
        """ Breadth-first treavsal on binary RST tree

        :type tree: SpanNode instance
        :param tree: an binary RST tree
        """
        queue = [tree]
        bft_nodelist = []
        while queue:
            node = queue.pop(0)
            bft_nodelist.append(node)
            if node.lnode is not None:
                queue.append(node.lnode)
            if node.rnode is not None:
                queue.append(node.rnode)
        return bft_nodelist

    @staticmethod
    def flatBFTbin(tree):
        """ Breadth-first treavsal on binary RST tree

        :type tree: SpanNode instance
        :param tree: an binary RST tree
        """
        queue = [tree]
        bft_nodelist = []
        while queue:
            node = queue.pop(0)
            bft_nodelist.append(node)
            queue.extend(node.nodelist)
            # if node.lnode is not None:
            #     queue.append(node.lnode)
            # if node.rnode is not None:
            #     queue.append(node.rnode)
        return bft_nodelist



    @staticmethod
    def postorder_DFT(tree, nodelist):
        """ Post order traversal on binary RST tree

        :type tree: SpanNode instance
        :param tree: an binary RST tree

        :type nodelist: list
        :param nodelist: list of node in post order
        """
        if tree.lnode is not None:
            RstTree.postorder_DFT(tree.lnode, nodelist)
        if tree.rnode is not None:
            RstTree.postorder_DFT(tree.rnode, nodelist)
        nodelist.append(tree)
        return nodelist

    @staticmethod
    def postorder_flat_DFT(tree, nodelist):
        """ Post order traversal on binary RST tree
        right first, deep first

        :type tree: SpanNode instance
        :param tree: an binary RST tree

        :type nodelist: list
        :param nodelist: list of node in post order
        """
        stack = [tree]
        while stack:
            node = stack.pop()
            nodelist.append(node)
            stack.extend(node.nodelist)
        nodelist.reverse()
        return nodelist

    @staticmethod
    def __getspaninfo(lnode, rnode):
        """ Get span size for parent node

        :type lnode,rnode: SpanNode instance
        :param lnode,rnode: Left/Right children nodes
        """
        try:
            edu_span = (lnode.edu_span[0], rnode.edu_span[1])
            return edu_span
        except TypeError:
            print(lnode.prop, rnode.prop)
            print(lnode.nuc_span, rnode.nuc_span)
            sys.exit()

    @staticmethod
    def __getflatspaninfo(node):
        """ Get span size for parent node

        :type lnode,rnode: SpanNode instance
        :param lnode,rnode: Left/Right children nodes
        """

        edu_span = (node.nodelist[0].edu_span[0], node.nodelist[-1].edu_span[1])
        return edu_span

    @staticmethod
    def __getforminfo(node):
        """ Get Nucleus/Satellite form and Nucleus span

        :type lnode,rnode: SpanNode instance
        :param lnode,rnode: Left/Right children nodes
        """
        lnode = node.lnode
        rnode= node.rnode
        if lnode.prop == rnode.prop and lnode.pnode.visited:
            nuc_span = (lnode.edu_span[0], rnode.edu_span[1])
            nuc_edu = lnode.nuc_edu
            form = 'N~'
        elif (lnode.prop == 'Nucleus') and (rnode.prop == 'Satellite'):
            nuc_span = lnode.edu_span
            nuc_edu = lnode.nuc_edu
            form = 'NS'
        elif (lnode.prop == 'Satellite') and (rnode.prop == 'Nucleus'):
            nuc_span = rnode.edu_span
            nuc_edu = rnode.nuc_edu
            form = 'SN'
        elif (lnode.prop == 'Nucleus') and (rnode.prop == 'Nucleus'):
            nuc_span = (lnode.edu_span[0], rnode.edu_span[1])
            nuc_edu = lnode.nuc_edu
            form = 'NN'
        else:
            raise ValueError("")
        return form, nuc_span, nuc_edu

    @staticmethod
    def __getflatforminfo(node):
        """ Get Nucleus/Satellite form and Nucleus span

        :type lnode,rnode: SpanNode instance
        :param lnode,rnode: Left/Right children nodes
        """
        lnode, rnode = node.nodelist[0], node.nodelist[-1]
        if len(node.nodelist) == 2:
            if (lnode.prop == 'Nucleus') and (rnode.prop == 'Satellite'):
                nuc_span = lnode.edu_span
                nuc_edu = lnode.nuc_edu
                form = 'NS'
            elif (lnode.prop == 'Satellite') and (rnode.prop == 'Nucleus'):
                nuc_span = rnode.edu_span
                nuc_edu = rnode.nuc_edu
                form = 'SN'
            elif (lnode.prop == 'Nucleus') and (rnode.prop == 'Nucleus'):
                nuc_span = (lnode.edu_span[0], rnode.edu_span[1])
                nuc_edu = lnode.nuc_edu
                form = 'NN'
            else:
                raise ValueError("")
        else:
            nuc_span = (lnode.edu_span[0], rnode.edu_span[1])
            nuc_edu = lnode.nuc_edu
            form = 'N~'

        return form, nuc_span, nuc_edu

    @staticmethod
    def __getrelationinfo(node):
        """ Get relation information

        :type lnode,rnode: SpanNode instance
        :param lnode,rnode: Left/Right children nodes
        """
        lnode = node.lnode
        rnode = node.rnode
        if (lnode.prop == 'Nucleus') and (rnode.prop == 'Nucleus'):
            relation = lnode.relation
        elif (lnode.prop == 'Nucleus') and (rnode.prop == 'Satellite'):
            relation = rnode.relation
        elif (lnode.prop == 'Satellite') and (rnode.prop == 'Nucleus'):
            relation = lnode.relation
        else:
            print('lnode.prop = {}, lnode.edu_span = {}'.format(lnode.prop, lnode.edu_span))
            print('rnode.prop = {}, lnode.edu_span = {}'.format(rnode.prop, rnode.edu_span))
            raise ValueError("Error when find relation for new node")
        return relation

    @staticmethod
    def __getflatrelationinfo(node):
        """ Get relation information

        :type lnode,rnode: SpanNode instance
        :param lnode,rnode: Left/Right children nodes
        """
        if len(node.nodelist) == 2:
            lnode, rnode = node.nodelist[0], node.nodelist[1]
            if (lnode.prop == 'Nucleus') and (rnode.prop == 'Nucleus'):
                relation = lnode.relation
            elif (lnode.prop == 'Nucleus') and (rnode.prop == 'Satellite'):
                relation = rnode.relation
            elif (lnode.prop == 'Satellite') and (rnode.prop == 'Nucleus'):
                relation = lnode.relation
            else:
                print('lnode.prop = {}, lnode.edu_span = {}'.format(lnode.prop, lnode.edu_span))
                print('rnode.prop = {}, lnode.edu_span = {}'.format(rnode.prop, rnode.edu_span))
                raise ValueError("Error when find relation for new node")
        else:
            relation = node.nodelist[0].relation
        return relation

    @staticmethod
    def __gettextinfo(edu_dict, edu_span):
        """ Get text span for parent node

        :type edu_dict: dict of list
        :param edu_dict: EDU from this document

        :type edu_span: tuple with two elements
        :param edu_span: start/end of EDU IN this span
        """
        # text = lnode.text + " " + rnode.text
        text = []
        for idx in range(edu_span[0], edu_span[1] + 1, 1):
            text += edu_dict[idx]
        # Return: A list of token indices
        return text

    @staticmethod
    def extract_relation(s, level=0):
        """ Extract discourse relation on different level
        """
        return rel2class[s.lower()]

    def get_parse(self):
        """ Get parse tree

        :type tree: SpanNode instance
        :param tree: an binary RST tree
        """
        parse = []
        node_list = [self.tree]
        while node_list:
            node = node_list.pop()
            if node == ' ) ':
                parse.append(' ) ')
                continue
            if (node.lnode is None) and (node.rnode is None):
                # parse.append(" ( EDU " + str(node.nuc_edu))
                parse.append(" ( EDU " + '_!' + self.convert_node_to_str(node, sep='_') + '!_')
            else:
                parse.append(" ( " + node.form)
                # get the relation from its satellite node
                if node.form == 'NN':
                    parse += "-" + RstTree.extract_relation(node.rnode.relation)
                elif node.form == 'N~':
                    parse += "-" + RstTree.extract_relation(node.rnode.relation)
                elif node.form == 'NS':
                    parse += "-" + RstTree.extract_relation(node.rnode.relation)
                elif node.form == 'SN':
                    parse += "-" + RstTree.extract_relation(node.lnode.relation)
                else:
                    raise ValueError("Unrecognized N-S form")
            node_list.append(' ) ')
            if node.rnode is not None:
                node_list.append(node.rnode)
            if node.lnode is not None:
                node_list.append(node.lnode)
        return ''.join(parse)

    def draw_rst(self, fname):
        """ Draw RST tree into a file
        """
        tree_str = self.get_parse()
        if not fname.endswith(".ps"):
            fname += ".ps"
        cf = CanvasFrame()
        t = Tree.fromstring(tree_str)
        tc = TreeWidget(cf.canvas(), t)
        cf.add_widget(tc, 10, 10)  # (10,10) offsets
        cf.print_to_file(fname)
        cf.destroy()

    def isSymetric(self, lnode, rnode, root):
        if not lnode or not rnode or not root:
            return False
        if lnode.edu_span[0] == lnode.edu_span[1] or rnode.edu_span[0] == rnode.edu_span[1]:
            return False
        if RstTree.extract_relation(lnode.relation) == RstTree.extract_relation(rnode.relation):
            return True
        # if not root.relation and RstTree.extract_relation(lnode.relation) == RstTree.extract_relation(rnode.relation):
        #     return True
        # if RstTree.extract_relation(lnode.relation) == RstTree.extract_relation(rnode.relation) == RstTree.extract_relation(root.relation):
        #     return True
        return False

    def bracketing(self, isFlat):
        """ Generate brackets according an Binary RST tree
        """
        if isFlat:
            nodelist = RstTree.postorder_flat_DFT(self.tree, [])
        else:
            nodelist = RstTree.postorder_DFT(self.tree, [])
        # nodelist.pop()  # Remove the root node
        brackets = []
        for node in nodelist:
            if node.form:
                relation = RstTree.extract_relation(node.assignRelation)
                if node.form == "N~":
                    b = (node.edu_span, "NN", relation, node.level)
                else:
                    b = (node.edu_span, node.form, relation, node.level)
            # if not node.relation:
            #     print()
            # relation = RstTree.extract_relation(node.relation)
            # b = (node.edu_span, node.prop, relation)
            # parent_node = node.pnode
            # if parent_node and parent_node.rnode == node:
            #     if self.isSymetric(node.lnode, node.rnode, node) and self.isSymetric(parent_node.lnode, node,
            #                                                                          parent_node):
            #         continue

                brackets.append(b)
        return brackets


if __name__ == '__main__':
    tree = RstTree(fdis='', fmerge='')
    tree.build()
