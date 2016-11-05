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
import jpype

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

class BleuScoreAngeli(object):
    #
    def __init__(self, settings):
        print "Bleu Score (Gabor Angeli wrapper) built ... "
        self.path_jvm = settings['path_jvm']
        self.path_jar = settings['path_jar']
        self.max_diff = settings['max_diff']
        self.djavapath = "-Djava.class.path=%s"%os.path.abspath(
            self.path_jar
        )
        jpype.startJVM(
            self.path_jvm, '-ea', self.djavapath
        )
        self.BleuScorerClass = jpype.JClass(
            "cortex.BleuScorer"
        )
        self.BleuScorer = self.BleuScorerClass()
        self.BleuScorer.setThreshold(
            numpy.int(self.max_diff)
        )
        # int32 causes errors in Angeli code
        self.list_gens = []
        self.list_refs = []
        #
    #
    def reset_gens(self):
        self.list_gens = []
    #
    def set_refs(self, list_refs):
        self.list_refs = list_refs
    #
    def add_gen(self, text_gen):
        self.list_gens.append(text_gen)
    #
    def evaluate(self):
        #
        assert(len(self.list_refs) == len(self.list_gens) )
        #
        #print "preparing refs ... "
        refSets = jpype.java.util.ArrayList()
        refSet = jpype.java.util.ArrayList()
        for ref0 in self.list_refs:
            reference = jpype.java.util.ArrayList()
            for s in ref0.split():
                #print s
                reference.add(s)
            refSet.add(reference)
        refSets.add(refSet)
        #print refSets
        #
        #print "preparing gens ... "
        tests = jpype.java.util.ArrayList()
        for test0 in self.list_gens:
            test = jpype.java.util.ArrayList()
            for s in test0.split():
                #print s
                test.add(s)
            tests.add(test)
        #
        #print tests
        value_eval = self.BleuScorer.evaluateBleu(
            tests, refSets
        )
        #print "computing bleu score ... "
        value_score = value_eval.getScore()
        value_score *= 100.0
        return value_score
        #
    #
    def shutdownJVM(self):
        jpype.shutdownJVM()
    #
    #

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

def main():
    bleu_scorer = BleuScoreAngeli(
        {
            'path_jvm': '/Library/Java/JavaVirtualMachines/jdk1.7.0_79.jdk/Contents/Home/jre/lib/server/libjvm.dylib',
            'path_jar': '../dist/generation.jar',
            'max_diff': 0
        }
    )
    bleu_scorer.set_refs(
        [
            'hello world in C++', 'hello world in Java'
        ]
    )
    #
    bleu_scorer.reset_gens()
    bleu_scorer.add_gen('hello world in C++')
    bleu_scorer.add_gen('hello world in Java')
    print "refs are : ", bleu_scorer.list_refs
    print "gens are : ", bleu_scorer.list_gens
    bleu_score = bleu_scorer.evaluate()
    #
    print "bleu score is : ", bleu_score
    #print "bleu score is : ", round(bleu_score, 2)
    #
if __name__ == "__main__": main()
