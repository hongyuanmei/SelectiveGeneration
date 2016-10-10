# -*- coding: utf-8 -*-
import pickle
import time
import numpy
import theano
from theano import sandbox
import theano.tensor as tensor
import os
import scipy.io
import nltk

dtype = theano.config.floatX

'''
To be worked on !!!
Bleu score with Java can be found in dialog project !!!
We use NLTK for standard BLEU score computation
'''

class F1Compute(object):
    def __init__(self):
        print "F1 Compute built ... "
        self.list_golds = []
        self.list_aligns = []

    def reset_aligns(self):
        self.list_aligns = []

    def set_golds(self, list_golds):
        self.list_golds = list_golds

    def add_align(self, align):
        self.list_aligns.append(
            [
                idx for idx in align
            ]
        )

    def evaluate(self):
        assert(len(self.list_golds) == len(self.list_aligns))
        base_1 = 0.0
        base_2 = 0.0
        up = 0.0
        for gold, align in zip(self.list_golds, self.list_aligns):
            base_1 += 1.0 * len(gold)
            base_2 += 1.0 * len(align)
            up_current = 0.0
            for idx in align:
                if idx in gold:
                    up_current += 1.0
            up += 1.0 * up_current
        pc = up / base_2
        rc = up / base_1
        f1 = 2.0 * pc * rc / (pc + rc)
        f1 *= 100.0
        return f1 


class BleuScoreNLTK(object):
    def __init__(self):
        print "Bleu Score built ... "
        self.list_gens = []
        self.list_refs = []


    def reset_gens(self):
        self.list_gens = []

    def set_refs(self, list_refs):
        self.list_refs = list_refs

    def add_gen(self, text_gen):
        self.list_gens.append(text_gen)

    def evaluate(self):
        hypotheses = []
        list_of_references = []
        assert(len(self.list_refs) == len(self.list_gens) )
        for text_gen, text_ref in zip(self.list_gens, self.list_refs):
            hypotheses.append(
                [
                    token for token in text_gen.split()
                ]
            )
            list_of_references.append(
                [
                    [
                        token for token in text_ref.split()
                    ]
                ]
            )
        value_score = nltk.translate.bleu_score.corpus_bleu(
            list_of_references = list_of_references,
            hypotheses = hypotheses
        )
        value_score *= 100.0
        return value_score


class BleuScorePerl(object):
    '''
    This class calls Moses Perl BLEU score code
    to be filed
    '''
    #
