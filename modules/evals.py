# -*- coding: utf-8 -*-
import pickle
import time
import numpy
import theano
from theano import sandbox
import theano.tensor as tensor
import os
import scipy.io

dtype = theano.config.floatX

'''
To be worked on !!!
Bleu score with Java can be found in dialog project !!!
'''

class BleuScore(object):
    def __init__(self):
        pass
        '''
        borrow code from dialog project ...
        '''

class BleuScorePerl(object):
    '''
    This class calls Moses Perl BLEU score code
    '''
    #
    def __init__(self, bleuscore_dict):
        print "ready to compute Bleu using Moses ... "
        #self.path_program = '//mgalley1/BLEU/bin/CommonShell.exe'
        self.path_program = bleuscore_dict['path_program']
        self.path_bleu = bleuscore_dict['path_bleu']
        # temp folder save the intermediate files
        assert(self.path_bleu != None)
        if self.path_program == None:
            self.path_program = './modules/'
        #
        owd = os.getcwd()
        # in case it is run on Windows
        #if '\\' in owd:
        #    owd.replace('\\', '/')
        #if 'D:' in owd:
        #    owd.replace('D:', '/d')
        ##
        print owd
        #
        self.file_gens = owd+self.path_bleu[1:]+'outs.txt'
        self.file_refs = owd+self.path_bleu[1:]+'refs.txt'
        self.file_bleu = owd+self.path_bleu[1:]+'bleu.txt'
        #
        print self.file_gens
        #self.command = 'nohup ./multi-bleu.perl '+self.file_refs+' < '+self.file_gens+' > '+self.file_bleu

    def write_file(self, lists):

        os.system('mkdir -p '+self.path_bleu)
        open(self.file_gens,'w').write('\n'.join(lists['list_gens']))
        open(self.file_refs,'w').write(''.join(lists['list_refs']))
        # worth to note: ref ends with '\n' so does not need another '\n'
        # and also this does not affect correct results
        # because '\n' is not counted into ref
        # but i am really a bit fuzzy how this '\n' comes into existence

    def remove_file(self):
        os.system('rm -rf '+self.path_bleu)

    def read_score(self):
        with open(self.file_bleu,'r') as f:
            text = f.read()
        #print text
        self.all_numbers = text
        if 'BLEU = ' in text:
            sec_0 = text.split('BLEU = ')[1]
            #print sec_0
            sec_1 = sec_0.split(', ')[0]
            sec_2 = sec_0.split(', ')[1]
            # 19.9/5.6/2.1/1.2 (BP=1.000 -- like this
            #print sec_1
            self.bs = numpy.float32(sec_1)
            #
            sec_3 = sec_2.split(' (BP=')[0]
            sec_4 = sec_2.split(' (BP=')[1]
            self.bp = numpy.float32(sec_4)
            bs_list = sec_3.split('/')
            self.b1 = numpy.float32(bs_list[0])
            self.b2 = numpy.float32(bs_list[1])
            self.b3 = numpy.float32(bs_list[2])
            self.b4 = numpy.float32(bs_list[3])
        else:
            self.bs = numpy.float32(0.0)
            self.b1 = numpy.float32(0.0)
            self.b2 = numpy.float32(0.0)
            self.b3 = numpy.float32(0.0)
            self.b4 = numpy.float32(0.0)
            self.bp = numpy.float32(1.0)
        self.bs_arithmetic = self.bp * (self.b1+self.b2+self.b3+self.b4) / numpy.float32(4.0)
        self.bs_arithmetic = round(self.bs_arithmetic, 2)

    def get_bleu(self, lists):
        # use_special labels whether we want to use the special tokens
        # <s> and </s>
        self.write_file(lists)
        #
        owd = os.getcwd()
        # in case it is run on Windows
        #if '\\' in owd:
        #    owd.replace('\\', '/')
        #if 'D:' in owd:
        #    owd.replace('D:', '/d')
        #
        #self.file_gens = owd + self.file_gens[1:]
        #self.file_refs = owd + self.file_refs[1:]
        #self.file_bleu = owd + self.file_bleu[1:]
        self.command = 'nohup ./multi-bleu.perl '+self.file_refs+' < '+self.file_gens+' > '+self.file_bleu
        #
        os.chdir(self.path_program)
        os.system(self.command)
        os.chdir(owd)
        #
        #self.file_gens = self.path_bleu+'outs.txt'
        #self.file_refs = self.path_bleu+'refs.txt'
        #self.file_bleu = self.path_bleu+'bleu.txt'
        #
        self.read_score()
        #
        self.remove_file()
        #
        return self.bs


class BleuScoreMSR(object):
    '''
    This class calls MSR BLEU score code provided by Michel Galley
    '''
    #
    def __init__(self):
        print "ready to compute Bleu using MSR exe program ... "
        #self.path_program = '//mgalley1/BLEU/bin/CommonShell.exe'
        self.path_program = './BLEU/bin/CommonShell.exe'
        # temp folder save the intermediate files
        self.file_bleu = './bleu.txt'
        self.file_gens = './hyps.txt'
        self.file_refs = './refs.txt'
        #self.command = 'nohup '+self.path_program+' cmd BLEUALL 4 false Closest 1 '+self.file_refs+' '+self.file_gens+' "" > '+self.file_bleu+' & disown'
        self.command = 'nohup '+self.path_program+' cmd BLEUALL 4 false Closest 1 '+self.file_refs+' '+self.file_gens+' "" > '+self.file_bleu+' & disown'

    def write_file(self, lists):
        open(self.file_gens,'w').write('\n'.join(lists['list_gens']))
        open(self.file_refs,'w').write(''.join(lists['list_refs']))
        # worth to note: ref ends with '\n' so does not need another '\n'
        # and also this does not affect correct results
        # because '\n' is not counted into ref

    def remove_file(self):
        os.system('rm '+self.file_gens)
        os.system('rm '+self.file_refs)
        os.system('rm '+self.file_bleu)

    def read_score(self):
        with open(self.file_bleu,'r') as f:
            text = f.read()
        #print text
        self.all_numbers = text
        sec_0 = text.split('BLEU=')[1]
        #print sec_0
        sec_1 = sec_0.split(' %')[0]
        #print sec_1
        self.bs = numpy.float32(sec_1)

    def get_bleu(self, lists):
        # use_special labels whether we want to use the special tokens
        # <s> and </s>
        self.write_file(lists)
        os.system(self.command)
        #
        self.read_score()
        #
        self.remove_file()
        #
        return self.bs
